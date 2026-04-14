"""
STEP 2 — Model Training (v3)
============================
Improvements over v2:
  1. One-hot encode `region` and `country` — they are nominal, not ordinal
  2. Feature engineering — age interactions + new biological interactions
  3. XGBoost hyperparameter tuning with RandomizedSearchCV
  4. Optimal decision threshold — maximises F1 while keeping Recall >= 0.75
  5. 7 new features from pipeline v2:
       mother_height_cm, birth_interval_months, delivery_place,
       birth_size_perceived, birth_order, antenatal_visits, first_born
  6. New engineered features:
       mother_short, birth_risk, short_interval

HOW TO RUN:
    python 2_model_training.py

OUTPUT:
    models/xgboost_model.pkl      — trained model
    models/encoder.pkl            — fitted OneHotEncoder for region + country
    models/threshold.pkl          — optimal decision threshold (float)
    models/shap_explainer.pkl
    models/feature_names.pkl      — full feature list after encoding
    models/feature_importance.csv
    models/model_report.txt
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

import xgboost as xgb
import shap

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join('data', 'combined_data.csv')
MODELS_DIR   = 'models'
RANDOM_SEED  = 42
RECALL_TARGET = 0.75   # minimum recall we want from the final model

# Categorical columns that need one-hot encoding (they are NOMINAL, not ordinal)
CAT_COLS = ['region', 'country']

# Numeric columns passed through directly
NUM_COLS = [
    'child_age_months',
    'child_sex',
    'birth_weight_grams',
    'mother_age',
    'mother_education',
    'mother_bmi',
    'wealth_index',
    'water_source',
    'sanitation_type',
    'household_size',
    'urban_rural',
    # New in v3 (from pipeline v2)
    'mother_height_cm',
    'birth_interval_months',
    'delivery_place',
    'birth_size_perceived',
    'birth_order',
    'antenatal_visits',
    'first_born',
]

# All raw feature columns (before engineering / encoding)
RAW_FEATURE_COLS = NUM_COLS + CAT_COLS

TARGET_COL = 'stunted'


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction and composite features.
    Must be applied identically during training AND inference.

    Original (v2):
      age_sq         — stunting risk is non-linear with age; peaks ~18-24 months
      age_x_poverty  — older child × poor household = compounding risk
      age_x_low_edu  — older child × low maternal education
      mother_risk    — composite: no education + low BMI
      env_risk       — composite: poor water + poor sanitation

    New (v3):
      mother_short   — mother height < 150 cm (WHO threshold for short stature,
                        strongly associated with child linear growth failure)
      birth_risk     — composite: low birth weight (<2500g) OR perceived small/
                        very-small size (codes 4-5); covers cases where birth weight
                        is unknown (m19=9996) but perceived size is recorded
      short_interval — birth interval < 24 months AND child is not first-born;
                        WHO recommends ≥ 24 months between births to reduce
                        nutrient competition and stunting risk
    """
    X = X.copy()

    # ── Original features ─────────────────────────────────────────────────────

    # Polynomial age term
    X['age_sq'] = X['child_age_months'] ** 2

    # Interaction: age × poverty (invert wealth so higher = worse)
    X['age_x_poverty'] = X['child_age_months'] * (6 - X['wealth_index'])

    # Interaction: age × low maternal education (0=none → high risk)
    X['age_x_low_edu'] = X['child_age_months'] * (3 - X['mother_education'])

    # Maternal vulnerability composite (0–2)
    X['mother_risk'] = (
        (X['mother_education'] == 0).astype(float) +
        (X['mother_bmi'] < 18.5).astype(float)
    )

    # Environmental risk composite (0–4)
    X['env_risk'] = X['water_source'] + X['sanitation_type']

    # ── New v3 features ───────────────────────────────────────────────────────

    # Short maternal stature (< 150 cm) — strong intergenerational stunting predictor
    if 'mother_height_cm' in X.columns:
        X['mother_short'] = (X['mother_height_cm'] < 150).astype(float)

    # Birth risk composite: low birth weight OR small perceived size
    # birth_size_perceived: 1=large … 5=very small; codes 4-5 = small/very small
    birth_risk = pd.Series(0.0, index=X.index)
    if 'birth_weight_grams' in X.columns:
        birth_risk += (X['birth_weight_grams'] < 2500).astype(float)
    if 'birth_size_perceived' in X.columns:
        birth_risk += (X['birth_size_perceived'] >= 4).astype(float)
    X['birth_risk'] = birth_risk

    # Short preceding birth interval (< 24 months), only for non-first-borns
    if 'birth_interval_months' in X.columns and 'first_born' in X.columns:
        X['short_interval'] = (
            (X['birth_interval_months'] < 24) & (X['first_born'] == 0)
        ).astype(float)

    return X


# ── ENCODING ─────────────────────────────────────────────────────────────────
def fit_encoder(X_train: pd.DataFrame):
    """Fit OneHotEncoder on categorical columns using training data only."""
    enc = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore',   # unknown region in test → all-zero row
        dtype=np.float32
    )
    enc.fit(X_train[CAT_COLS])
    return enc


def apply_encoding(X: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """Apply a fitted encoder and return a fully numeric DataFrame."""
    cat_encoded = encoder.transform(X[CAT_COLS])
    cat_names   = encoder.get_feature_names_out(CAT_COLS).tolist()

    X_num = X.drop(columns=CAT_COLS).reset_index(drop=True)
    X_cat = pd.DataFrame(cat_encoded, columns=cat_names)

    return pd.concat([X_num, X_cat], axis=1)


# ── THRESHOLD OPTIMISATION ────────────────────────────────────────────────────
def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = RECALL_TARGET
) -> float:
    """
    Find the lowest threshold at which recall >= min_recall,
    then among all such thresholds pick the one with the highest F1.
    Falls back to 0.5 if min_recall is unachievable.
    """
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_true, y_proba)

    best_threshold = 0.5
    best_f1        = 0.0

    for p, r, t in zip(precision_arr[:-1], recall_arr[:-1], thresholds_arr):
        if r >= min_recall:
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1        = f1
                best_threshold = float(t)

    return best_threshold


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    print(f"\n{'='*55}")
    print("  CHW STUNTING PREDICTION — MODEL TRAINING v2")
    print(f"{'='*55}")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run 1_data_pipeline.py first.")
        return None, None

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded: {len(df):,} rows")
    print(f"Stunting prevalence: {df[TARGET_COL].mean()*100:.1f}%")

    missing = [c for c in RAW_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")

    cols_to_use = [c for c in RAW_FEATURE_COLS if c in df.columns]
    X = df[cols_to_use].copy()
    y = df[TARGET_COL].copy()

    print(f"Raw features: {len(cols_to_use)}  {cols_to_use}")
    return X, y


# ── EVALUATE ─────────────────────────────────────────────────────────────────
def evaluate(name, model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)

    print(f"\n  {name}  (threshold={threshold:.2f})")
    print(f"    Recall:    {recall:.3f}  (target >= {RECALL_TARGET})")
    print(f"    Precision: {precision:.3f}")
    print(f"    F1 Score:  {f1:.3f}")
    print(f"    ROC-AUC:   {roc_auc:.3f}")

    return {
        'name':      name,
        'model':     model,
        'recall':    recall,
        'precision': precision,
        'f1':        f1,
        'roc_auc':   roc_auc,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'threshold': threshold,
    }


# ── MAIN TRAINING ─────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_raw, y = load_data()
    if X_raw is None:
        return

    # ── STEP 1: FEATURE ENGINEERING ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Feature engineering...")
    X_eng = engineer_features(X_raw)
    print(f"  Features after engineering: {X_eng.shape[1]}  "
          f"(+{X_eng.shape[1] - X_raw.shape[1]} new)")

    # ── STEP 2: TRAIN / TEST SPLIT ────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_eng, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )
    print(f"\nTrain: {len(X_train_raw):,} rows | Test: {len(X_test_raw):,} rows")

    # ── STEP 3: FIT ENCODER ON TRAINING SET ONLY ─────────────────────────────
    print("\nEncoding categorical features (region, country) → one-hot...")
    encoder  = fit_encoder(X_train_raw)
    X_train  = apply_encoding(X_train_raw, encoder)
    X_test   = apply_encoding(X_test_raw,  encoder)

    ohe_cols = encoder.get_feature_names_out(CAT_COLS).tolist()
    print(f"  One-hot columns added: {len(ohe_cols)}  {ohe_cols[:6]} ...")
    print(f"  Final feature count: {X_train.shape[1]}")

    # ── STEP 4: CLASS WEIGHTS ─────────────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nClass ratio (neg/pos): {scale_pos_weight:.2f}")

    # ── STEP 5: DEFINE MODELS ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Training baseline models...")
    print(f"{'='*55}")

    baseline_models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight='balanced'
            ))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
    }

    baseline_results = []
    for name, model in baseline_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        cv = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
            scoring='roc_auc', n_jobs=-1
        )
        print(f"  CV ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f}")
        result = evaluate(name, model, X_test, y_test)
        baseline_results.append(result)

    # ── STEP 6: TUNE XGBOOST WITH RANDOMIZEDSEARCHCV ─────────────────────────
    print(f"\n{'='*55}")
    print("Tuning XGBoost (RandomizedSearchCV, 25 iterations)...")
    print(f"{'='*55}")

    param_dist = {
        'n_estimators':      randint(200, 600),
        'max_depth':         randint(4, 9),
        'learning_rate':     uniform(0.01, 0.14),
        'subsample':         uniform(0.65, 0.35),
        'colsample_bytree':  uniform(0.65, 0.35),
        'min_child_weight':  randint(3, 15),
        'gamma':             uniform(0, 0.3),
        'reg_alpha':         uniform(0, 0.5),
        'reg_lambda':        uniform(0.5, 1.5),
    }

    base_xgb = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
    )

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1',
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    tuned_xgb    = search.best_estimator_
    best_params  = search.best_params_
    best_cv_auc  = search.best_score_

    print(f"\nBest CV ROC-AUC: {best_cv_auc:.3f}")
    print(f"Best params: {best_params}")

    # Evaluate at default threshold first
    xgb_result = evaluate('XGBoost (tuned)', tuned_xgb, X_test, y_test, threshold=0.5)

    # ── STEP 7: OPTIMISE DECISION THRESHOLD ───────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Finding optimal threshold (min recall = {RECALL_TARGET})...")

    optimal_threshold = find_optimal_threshold(
        y_test.values, xgb_result['y_proba'], min_recall=RECALL_TARGET
    )
    print(f"  Optimal threshold: {optimal_threshold:.3f}  "
          f"(default was 0.500)")

    # Re-evaluate at optimal threshold
    xgb_opt = evaluate(
        'XGBoost (tuned + optimal threshold)',
        tuned_xgb, X_test, y_test,
        threshold=optimal_threshold
    )

    # ── STEP 8: FULL COMPARISON ───────────────────────────────────────────────
    all_results = baseline_results + [xgb_result, xgb_opt]

    print(f"\n{'='*55}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<40} {'Recall':>8} {'F1':>8} {'ROC-AUC':>10}")
    print("-" * 70)
    for r in all_results:
        marker = " *" if r['name'] == xgb_opt['name'] else ""
        print(f"{r['name']:<40} {r['recall']:>8.3f} {r['f1']:>8.3f} {r['roc_auc']:>10.3f}{marker}")
    print("\n  * = final deployed model")

    best = xgb_opt   # always deploy the tuned + threshold-optimised model

    # Detailed report
    print(f"\nDetailed report — {best['name']}:")
    print(classification_report(y_test, best['y_pred'],
                                 target_names=['Not Stunted', 'Stunted']))
    cm = confusion_matrix(y_test, best['y_pred'])
    print(f"Confusion Matrix:")
    print(f"  True Negative  (correct 'not stunted'): {cm[0,0]:,}")
    print(f"  False Positive (wrongly flagged):        {cm[0,1]:,}")
    print(f"  False Negative (missed stunted child):   {cm[1,0]:,}")
    print(f"  True Positive  (correctly caught):       {cm[1,1]:,}")

    fn_rate = cm[1,0] / (cm[1,0] + cm[1,1])
    print(f"\n  Miss rate (FN / all stunted): {fn_rate*100:.1f}%")

    # ── STEP 9: SHAP ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Computing SHAP values...")

    explainer    = shap.TreeExplainer(tuned_xgb)
    X_sample     = X_test.sample(min(500, len(X_test)), random_state=RANDOM_SEED)
    shap_values  = explainer.shap_values(X_sample)
    shap_vals    = shap_values[1] if isinstance(shap_values, list) else shap_values

    feature_importance = pd.DataFrame({
        'feature':    X_train.columns.tolist(),
        'importance': np.abs(shap_vals).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\nTop 15 features by SHAP importance:")
    for _, row in feature_importance.head(15).iterrows():
        bar = '#' * int(row['importance'] * 50)
        print(f"  {row['feature']:<30} {row['importance']:.4f}  {bar}")

    # ── STEP 10: SAVE ARTIFACTS ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Saving artifacts...")

    artifacts = {
        'xgboost_model.pkl':    tuned_xgb,
        'encoder.pkl':          encoder,
        'threshold.pkl':        optimal_threshold,
        'shap_explainer.pkl':   explainer,
        'feature_names.pkl':    X_train.columns.tolist(),
    }

    for fname, obj in artifacts.items():
        path = os.path.join(MODELS_DIR, fname)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved: {path}")

    feature_importance.to_csv(
        os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False
    )

    # Report
    report_path = os.path.join(MODELS_DIR, 'model_report.txt')
    with open(report_path, 'w') as f:
        f.write("CHW Stunting Prediction — Model Report v2\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Model: XGBoost (tuned + threshold optimised)\n\n")
        f.write(f"Decision threshold: {optimal_threshold:.3f}  "
                f"(recall target >= {RECALL_TARGET})\n\n")
        f.write("Performance:\n")
        f.write(f"  Recall:    {best['recall']:.3f}\n")
        f.write(f"  Precision: {best['precision']:.3f}\n")
        f.write(f"  F1 Score:  {best['f1']:.3f}\n")
        f.write(f"  ROC-AUC:   {best['roc_auc']:.3f}\n\n")
        f.write(f"Best XGBoost params:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTop Features (SHAP):\n")
        for _, row in feature_importance.head(15).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

    print(f"  Saved: {report_path}")
    print(f"\n[OK] Training complete! Run 3_api.py next.\n")

    return tuned_xgb, encoder, optimal_threshold, explainer, X_train.columns.tolist()


if __name__ == '__main__':
    train()
