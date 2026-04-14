"""
STEP 3 — FastAPI Backend
=========================
Loads the trained XGBoost model and serves predictions via a REST API.
Includes offline sync support and SMS alerts via Africa's Talking.

HOW TO RUN:
    python 3_api.py

    To enable SMS alerts, set these env vars before running:
        AT_USERNAME=sandbox
        AT_API_KEY=your_api_key
        (See .env.example for full list)

API ENDPOINTS:
    POST /predict   — single prediction + optional SMS alert
    POST /sync      — batch sync of offline-queued assessments
    GET  /health    — health check
    GET  /sw.js     — service worker (served with correct scope header)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import os
import numpy as np
import pandas as pd
import shap
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Load .env file if python-dotenv is installed (optional convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sms_service  # Africa's Talking SMS — graceful no-op if not configured

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


# ── FEATURE ENGINEERING (must match 2_model_training.py exactly) ─────────────
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X['age_sq']        = X['child_age_months'] ** 2
    X['age_x_poverty'] = X['child_age_months'] * (6 - X['wealth_index'])
    X['age_x_low_edu'] = X['child_age_months'] * (3 - X['mother_education'])
    X['mother_risk']   = (
        (X['mother_education'] == 0).astype(float) +
        (X['mother_bmi'] < 18.5).astype(float)
    )
    X['env_risk'] = X['water_source'] + X['sanitation_type']
    return X


# ── LOAD MODEL ARTIFACTS ──────────────────────────────────────────────────────
def load_artifacts():
    artifacts = {}
    try:
        with open(os.path.join(MODELS_DIR, 'xgboost_model.pkl'), 'rb') as f:
            artifacts['model'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'shap_explainer.pkl'), 'rb') as f:
            artifacts['explainer'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'rb') as f:
            artifacts['features'] = pickle.load(f)

        # New in v2: encoder for region/country + optimal decision threshold
        encoder_path   = os.path.join(MODELS_DIR, 'encoder.pkl')
        threshold_path = os.path.join(MODELS_DIR, 'threshold.pkl')

        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                artifacts['encoder'] = pickle.load(f)
        else:
            artifacts['encoder'] = None   # v1 model — no encoding

        if os.path.exists(threshold_path):
            with open(threshold_path, 'rb') as f:
                artifacts['threshold'] = pickle.load(f)
        else:
            artifacts['threshold'] = 0.5  # v1 default

        print(f"[OK] Model artifacts loaded  "
              f"(threshold={artifacts['threshold']:.3f}, "
              f"encoder={'yes' if artifacts['encoder'] else 'no'})")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Run 2_model_training.py first to generate model files.")
        artifacts['model'] = None
    return artifacts

artifacts = load_artifacts()


# ── FASTAPI APP ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CHW Stunting Prediction API",
    description="Predicts child stunting risk for community health workers",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (manifest.json, icons, etc.)
_static_dir = os.path.join(BASE_DIR, 'frontend', 'static')
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── INPUT SCHEMA ──────────────────────────────────────────────────────────────
class ChildInput(BaseModel):
    # Country
    country: int = Field(..., description="1=Nigeria, 2=Rwanda, 3=Ethiopia")

    # Child
    child_age_months:   float = Field(..., ge=0,  le=59,  description="Age in months")
    child_sex:          int   = Field(..., ge=1,  le=2,   description="1=Male, 2=Female")
    birth_weight_grams: float = Field(..., ge=500, le=6000, description="Birth weight in grams")

    # Maternal
    mother_age:       float = Field(..., ge=15, le=49, description="Mother's age in years")
    mother_education: int   = Field(..., ge=0,  le=3,  description="0=None,1=Primary,2=Secondary,3=Higher")
    mother_bmi:       float = Field(..., ge=10, le=60, description="Mother's BMI")

    # Household
    wealth_index:    int   = Field(..., ge=1, le=5, description="1=Poorest to 5=Richest")
    water_source:    int   = Field(..., ge=0, le=2, description="0=Piped,1=Well,2=Surface")
    sanitation_type: int   = Field(..., ge=0, le=2, description="0=Flush,1=Pit latrine,2=None")
    household_size:  float = Field(..., ge=1, le=30, description="Number of people in household")

    # Geographic
    urban_rural: int = Field(..., ge=1, le=2, description="1=Urban, 2=Rural")
    region:      int = Field(..., ge=1,       description="Region code within country")

    # GSM / offline fields (optional)
    supervisor_phone: Optional[str] = Field(
        None,
        description="Supervisor phone for HIGH-risk SMS alert (international format, e.g. +2348012345678)"
    )
    offline_id: Optional[str] = Field(
        None,
        description="Client-generated UUID used to deduplicate offline submissions"
    )


# ── SHAP HELPERS ──────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    # Raw features
    'child_age_months':   'Child age',
    'child_sex':          'Child sex',
    'birth_weight_grams': 'Birth weight',
    'mother_age':         'Mother age',
    'mother_education':   'Mother education',
    'mother_bmi':         'Mother BMI',
    'wealth_index':       'Household wealth',
    'water_source':       'Water source',
    'sanitation_type':    'Sanitation access',
    'household_size':     'Household size',
    'urban_rural':        'Location (urban/rural)',
    'region':             'Geographic region',
    'country':            'Country context',
    # Engineered features (v2)
    'age_sq':             'Child age (non-linear)',
    'age_x_poverty':      'Age × poverty interaction',
    'age_x_low_edu':      'Age × low education interaction',
    'mother_risk':        'Maternal vulnerability score',
    'env_risk':           'Environmental risk score',
}

RISK_MESSAGES = {
    'birth_weight_grams': {
        'high': 'Low birth weight is significantly increasing risk',
        'low':  'Normal birth weight is reducing risk',
    },
    'mother_education': {
        'high': 'Low maternal education is increasing risk',
        'low':  'Higher maternal education is reducing risk',
    },
    'wealth_index': {
        'high': 'Low household wealth is increasing risk',
        'low':  'Higher household wealth is reducing risk',
    },
    'water_source': {
        'high': 'Unsafe water source is increasing risk',
        'low':  'Safe water source is reducing risk',
    },
    'sanitation_type': {
        'high': 'Poor sanitation is increasing risk',
        'low':  'Good sanitation is reducing risk',
    },
    'urban_rural': {
        'high': 'Rural location is increasing risk',
        'low':  'Urban location is reducing risk',
    },
    'country': {
        'high': 'Country-level health context is increasing risk',
        'low':  'Country-level health context is reducing risk',
    },
    'child_age_months': {
        'high': 'Child age is a risk factor at this stage',
        'low':  'Child age is not a concern at this stage',
    },
    'mother_bmi': {
        'high': 'Low maternal BMI is increasing risk',
        'low':  'Maternal BMI is not a concern',
    },
    'household_size': {
        'high': 'Large household size is increasing risk',
        'low':  'Household size is not a concern',
    },
}

def get_risk_factors(input_df, explainer, feature_names):
    try:
        shap_values = explainer.shap_values(input_df)
        shap_vals   = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        factors = []
        for feat, val in zip(feature_names, shap_vals):
            direction = 'high' if val > 0 else 'low'
            label     = FEATURE_LABELS.get(feat, feat)
            message   = RISK_MESSAGES.get(feat, {}).get(
                direction,
                f"{label} is {'increasing' if val > 0 else 'reducing'} risk"
            )
            factors.append({
                'feature':    feat,
                'label':      label,
                'shap_value': float(val),
                'direction':  direction,
                'message':    message,
            })

        increasing = sorted(
            [f for f in factors if f['direction'] == 'high'],
            key=lambda x: abs(x['shap_value']),
            reverse=True
        )
        return increasing[:3] if increasing else factors[:3]

    except Exception as exc:
        print(f"SHAP error: {exc}")
        return []


# ── CORE PREDICTION LOGIC (shared by /predict and /sync) ─────────────────────
RECOMMENDATIONS = {
    "HIGH": {
        "action":   "Refer immediately",
        "detail":   "This child has a high probability of stunting. Refer to the nearest nutrition clinic or health facility within 7 days. Initiate supplementary feeding if available.",
        "followup": "Follow up within 1 week to confirm referral was completed."
    },
    "MODERATE": {
        "action":   "Monitor closely",
        "detail":   "This child is at moderate risk. Schedule a follow-up home visit within 2 weeks. Provide caregiver counselling on nutrition and hygiene.",
        "followup": "Reassess in 4 weeks. Refer if condition worsens."
    },
    "LOW": {
        "action":   "Continue standard care",
        "detail":   "This child currently shows low risk of stunting. Continue routine growth monitoring and standard preventive care.",
        "followup": "Next scheduled visit as per standard protocol."
    }
}

COUNTRY_NAMES = {1: "Nigeria", 2: "Rwanda", 3: "Ethiopia"}

def _run_prediction(child: ChildInput) -> dict:
    """Run the full prediction pipeline and return a result dict."""
    if artifacts.get('model') is None:
        raise HTTPException(503, "Model not loaded. Run 2_model_training.py first.")

    model     = artifacts['model']
    explainer = artifacts['explainer']
    features  = artifacts['features']
    encoder   = artifacts.get('encoder')
    threshold = artifacts.get('threshold', 0.5)

    # Build raw input (same columns as RAW_FEATURE_COLS in training)
    input_dict = {
        'child_age_months':   child.child_age_months,
        'child_sex':          child.child_sex,
        'birth_weight_grams': child.birth_weight_grams,
        'mother_age':         child.mother_age,
        'mother_education':   child.mother_education,
        'mother_bmi':         child.mother_bmi,
        'wealth_index':       child.wealth_index,
        'water_source':       child.water_source,
        'sanitation_type':    child.sanitation_type,
        'household_size':     child.household_size,
        'urban_rural':        child.urban_rural,
        'region':             child.region,
        'country':            child.country,
    }
    input_df = pd.DataFrame([input_dict])

    # Apply feature engineering (v2 model)
    if encoder is not None:
        input_df = engineer_features(input_df)
        # One-hot encode region + country
        cat_enc   = encoder.transform(input_df[['region', 'country']])
        cat_names = encoder.get_feature_names_out(['region', 'country']).tolist()
        num_part  = input_df.drop(columns=['region', 'country']).reset_index(drop=True)
        cat_part  = pd.DataFrame(cat_enc, columns=cat_names)
        input_df  = pd.concat([num_part, cat_part], axis=1)

    # Align to training column order
    input_df = input_df.reindex(columns=features, fill_value=0)

    probability = float(model.predict_proba(input_df)[0][1])
    prediction  = int(probability >= threshold)

    if probability >= 0.60:
        risk_level, risk_color = "HIGH",     "red"
    elif probability >= 0.40:
        risk_level, risk_color = "MODERATE", "orange"
    else:
        risk_level, risk_color = "LOW",      "green"

    risk_factors = get_risk_factors(input_df, explainer, features)

    return {
        "prediction": {
            "stunted":     prediction,
            "probability": round(probability * 100, 1),
            "risk_level":  risk_level,
            "risk_color":  risk_color,
        },
        "recommendation": RECOMMENDATIONS[risk_level],
        "risk_factors":   risk_factors,
        "country":        COUNTRY_NAMES.get(child.country, "Unknown"),
    }


def _maybe_send_sms(child: ChildInput, result: dict):
    """Send SMS alert to supervisor if risk is HIGH and a phone is provided."""
    if result['prediction']['risk_level'] != 'HIGH':
        return
    if not child.supervisor_phone:
        return
    top_factor = result['risk_factors'][0]['message'] if result['risk_factors'] else ''
    sms_service.send_high_risk_alert(
        phone=child.supervisor_phone,
        age_months=child.child_age_months,
        probability=result['prediction']['probability'],
        country=result['country'],
        top_factor=top_factor,
    )


# ── PREDICT ENDPOINT ──────────────────────────────────────────────────────────
@app.post("/predict")
def predict(child: ChildInput):
    result = _run_prediction(child)
    _maybe_send_sms(child, result)
    return result


# ── SYNC ENDPOINT (offline batch) ─────────────────────────────────────────────
class SyncPayload(BaseModel):
    assessments: List[ChildInput]

@app.post("/sync")
def sync_offline(payload: SyncPayload):
    """
    Accepts a batch of offline-queued assessments, processes each one,
    and returns results in order. Called by the Service Worker when
    connectivity is restored.
    """
    results = []
    for idx, child in enumerate(payload.assessments):
        try:
            result = _run_prediction(child)
            _maybe_send_sms(child, result)
            result['index']      = idx
            result['offline_id'] = child.offline_id
            result['status']     = 'ok'
        except Exception as exc:
            result = {
                'index':      idx,
                'offline_id': child.offline_id,
                'status':     'error',
                'detail':     str(exc),
            }
        results.append(result)

    return {
        'synced':  sum(1 for r in results if r.get('status') == 'ok'),
        'failed':  sum(1 for r in results if r.get('status') != 'ok'),
        'results': results,
    }


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    model_loaded = artifacts.get('model') is not None
    return {
        "status":       "ok" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "sms_enabled":  sms_service.is_enabled(),
        "version":      "1.1.0"
    }


# ── SERVE SERVICE WORKER ──────────────────────────────────────────────────────
# Must be served from the origin root (not /static/) so its scope covers the
# whole app.  The Service-Worker-Allowed header is redundant here but explicit.
@app.get("/sw.js")
def serve_sw():
    sw_path = os.path.join(BASE_DIR, 'frontend', 'static', 'sw.js')
    if not os.path.exists(sw_path):
        raise HTTPException(404, "Service worker not found")
    return FileResponse(
        sw_path,
        media_type='application/javascript',
        headers={"Service-Worker-Allowed": "/"},
    )


# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(BASE_DIR, 'frontend', 'templates', 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>CHW Stunting Prediction API</h1><p>Frontend not found. See /docs for API.</p>"


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    uvicorn.run(
        "3_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
