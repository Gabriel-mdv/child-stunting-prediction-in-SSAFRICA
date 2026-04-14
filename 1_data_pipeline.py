"""
STEP 1 — Data Pipeline (v2)
============================
Loads DHS Children's Recode files for Nigeria, Rwanda, Ethiopia.
Extracts 19 variables (up from 12 in v1), creates the stunting label,
merges into one combined CSV ready for model training.

Changes from v1:
  - Added 6 new variables: mother_height_cm, birth_interval_months,
    delivery_place, birth_size_perceived, birth_order, antenatal_visits
  - Added derived flag: first_born
  - Fixed region filter bug (>= 9 was dropping Ethiopian regions 9-11)
  - Added delivery_place grouping function

HOW TO RUN:
    python 1_data_pipeline.py

OUTPUT:
    data/combined_data.csv
    data/pipeline_report.txt
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ── FILE PATHS ──────────────────────────────────────────────────────────────
# DATA_PATHS = {
#     'Nigeria':  os.path.join('dataset', 'NGKR7BDT', 'NGKR7BFL.DTA'),
#     'Rwanda':   os.path.join('dataset', 'RWKR81DT', 'RWKR81FL.DTA'),
#     'Ethiopia': os.path.join('dataset', 'ETKR71DT', 'ETKR71FL.DTA'),
# }

# ── VARIABLES TO EXTRACT ────────────────────────────────────────────────────
TARGET = 'hw70'   # Height-for-age Z-score × 100. < -200 = stunted

FEATURES = {
    # ── Original 12 ──────────────────────────────────────────────
    'hw1':  'child_age_months',       # Child age in months (0-59)
    'b4':   'child_sex',              # 1=Male, 2=Female
    'm19':  'birth_weight_grams',     # Birth weight in grams (9996=unknown)
    'v012': 'mother_age',             # Mother's current age in years
    'v106': 'mother_education',       # 0=None 1=Primary 2=Secondary 3=Higher
    'v445': 'mother_bmi',             # BMI × 100 (e.g. 2250 = 22.5)
    'v190': 'wealth_index',           # 1=Poorest … 5=Richest
    'v113': 'water_source',           # DHS drinking water code
    'v116': 'sanitation_type',        # DHS toilet facility code
    'v136': 'household_size',         # Total household members
    'v025': 'urban_rural',            # 1=Urban 2=Rural
    'v024': 'region',                 # Administrative region within country
    # ── New variables ─────────────────────────────────────────────
    'v438': 'mother_height_cm',       # Mother height × 10 (e.g. 1580 = 158.0 cm)
    'b11':  'birth_interval_months',  # Months since preceding birth (NaN for first)
    'm15':  'delivery_place',         # Where child was born (home/public/private)
    'm18':  'birth_size_perceived',   # Perceived birth size 1=large … 5=very small
    'bord': 'birth_order',            # Birth order (1=first child)
    'm14':  'antenatal_visits',       # Number of antenatal care visits
}

# COUNTRY_CODES = {
#     'Nigeria':  1,
#     'Rwanda':   2,
#     'Ethiopia': 3,
# }


DATA_PATHS = {
    'Rwanda':   os.path.join( 'dataset', 'RWKR81DT', 'RWKR81FL.DTA'),
    'Nigeria':  os.path.join( 'dataset', 'NGKR7BDT', 'NGKR7BFL.DTA'),
    'Ethiopia': os.path.join( 'dataset', 'ETKR71DT', 'ETKR71FL.DTA'),
    'Kenya':    os.path.join( 'dataset', 'KEKR8CDT', 'KEKR8CFL.DTA'),
    'Tanzania': os.path.join( 'dataset', 'TZKR82DT', 'TZKR82FL.DTA'),
    'Uganda':   os.path.join( 'dataset', 'UGKR7BDT', 'UGKR7BFL.DTA'),
    'Ghana':    os.path.join( 'dataset', 'GHKR8CDT', 'GHKR8CFL.DTA'),
    'Zambia':   os.path.join( 'dataset', 'ZMKR81DT', 'ZMKR81FL.DTA'),
    'Burundi':  os.path.join( 'dataset', 'BUKR71DT', 'BUKR71FL.DTA'),
    'DRC':      os.path.join( 'dataset', 'CDKR81DT', 'CDKR81FL.DTA'),
}

COUNTRY_CODES = {
    'Nigeria':  1,
    'Rwanda':   2,
    'Ethiopia': 3,
    'Kenya':    4,
    'Tanzania': 5,
    'Uganda':   6,
    'Ghana':    7,
    'Zambia':   8,
    'Burundi':  9,
    'DRC':      10,
}


# ── GROUPING FUNCTIONS ──────────────────────────────────────────────────────

def group_water(code):
    """DHS water source codes → 3 categories (0=safe, 1=moderate, 2=unsafe)."""
    try:
        code = int(code)
    except:
        return np.nan
    if code in [10, 11, 12, 13, 14]:
        return 0  # piped / protected spring
    elif code in [20, 21, 30, 31, 40, 41, 51]:
        return 1  # well / borehole / unprotected spring
    elif code in [32, 42, 43, 61, 62, 71, 96]:
        return 2  # surface water / tanker / other
    else:
        return np.nan


def group_sanitation(code):
    """DHS toilet codes → 3 categories (0=good, 1=moderate, 2=poor)."""
    try:
        code = int(code)
    except:
        return np.nan
    if code in [10, 11, 12, 13, 14, 15]:
        return 0  # flush / pour-flush
    elif code in [20, 21, 22, 23, 41, 51]:
        return 1  # pit latrine
    elif code in [30, 31, 42, 43, 96]:
        return 2  # no facility / open defecation / other
    else:
        return np.nan


def group_delivery_place(code):
    """
    DHS delivery location codes → 3 categories.
      0 = home (respondent's home or other home)
      1 = public facility (government hospital / health centre / post)
      2 = private facility (private hospital / clinic / doctor)
    """
    try:
        code = int(code)
    except:
        return np.nan
    if code in [11, 12]:
        return 0   # home
    elif 20 <= code <= 29:
        return 1   # public facility
    elif 30 <= code <= 39:
        return 2   # private facility
    else:
        return np.nan  # other (96) / missing (99)


# ── LOAD ONE COUNTRY ─────────────────────────────────────────────────────────
def load_country(name, path):
    print(f"\n{'='*55}")
    print(f"Loading {name}...")

    if not os.path.exists(path):
        print(f"  ERROR: File not found at {path}")
        return None

    df = pd.read_stata(path, convert_categoricals=False)
    print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ── Check target ──────────────────────────────────────────────────────────
    if TARGET not in df.columns:
        print(f"  ERROR: Target variable '{TARGET}' not found!")
        return None

    # ── Extract available columns ─────────────────────────────────────────────
    available = [col for col in FEATURES if col in df.columns]
    missing   = [col for col in FEATURES if col not in df.columns]
    if missing:
        print(f"  WARNING: DHS columns not found: {missing}")

    df_clean = df[[TARGET] + available].copy()

    # Add any missing columns as NaN so downstream code is consistent
    for col in missing:
        df_clean[col] = np.nan

    df_clean = df_clean.rename(columns=FEATURES)

    # ── Stunting label ─────────────────────────────────────────────────────────
    df_clean['stunted'] = (df_clean[TARGET] < -200).astype(int)
    df_clean = df_clean[df_clean[TARGET].notna()]
    df_clean = df_clean[df_clean[TARGET] < 9000]   # remove DHS flagged values
    df_clean = df_clean.drop(columns=[TARGET])

    # ── Country ───────────────────────────────────────────────────────────────
    df_clean['country']      = COUNTRY_CODES[name]
    df_clean['country_name'] = name

    # ── Clean original variables ───────────────────────────────────────────────

    # Mother BMI: stored × 100 → convert to actual BMI
    if 'mother_bmi' in df_clean.columns:
        df_clean['mother_bmi'] = df_clean['mother_bmi'] / 100
        df_clean.loc[df_clean['mother_bmi'] < 10, 'mother_bmi'] = np.nan
        df_clean.loc[df_clean['mother_bmi'] > 60, 'mother_bmi'] = np.nan

    # Birth weight: remove impossible / DHS flag values
    if 'birth_weight_grams' in df_clean.columns:
        df_clean.loc[df_clean['birth_weight_grams'] > 9000, 'birth_weight_grams'] = np.nan
        df_clean.loc[df_clean['birth_weight_grams'] < 500,  'birth_weight_grams'] = np.nan

    # Child age: keep 0-59 months
    if 'child_age_months' in df_clean.columns:
        df_clean = df_clean[
            (df_clean['child_age_months'] >= 0) &
            (df_clean['child_age_months'] <= 59)
        ]

    # Water source → 3 categories
    if 'water_source' in df_clean.columns:
        df_clean['water_source'] = df_clean['water_source'].apply(group_water)

    # Sanitation → 3 categories
    if 'sanitation_type' in df_clean.columns:
        df_clean['sanitation_type'] = df_clean['sanitation_type'].apply(group_sanitation)

    # Categorical variables with small ranges — remove DHS missing codes (9, 99)
    # NOTE: region is handled separately below to avoid dropping Ethiopian regions 9-11
    for col in ['child_sex', 'mother_education', 'wealth_index', 'urban_rural']:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] >= 9, col] = np.nan

    # Region: DHS missing codes are 97-99; valid codes can go up to 11 (Ethiopia)
    if 'region' in df_clean.columns:
        df_clean.loc[df_clean['region'] >= 97, 'region'] = np.nan

    # ── Clean NEW variables ────────────────────────────────────────────────────

    # Mother height: stored × 10 → convert to cm, remove flag values
    if 'mother_height_cm' in df_clean.columns:
        df_clean['mother_height_cm'] = df_clean['mother_height_cm'] / 10
        df_clean.loc[df_clean['mother_height_cm'] < 130, 'mother_height_cm'] = np.nan
        df_clean.loc[df_clean['mother_height_cm'] > 200, 'mother_height_cm'] = np.nan

    # Birth order: remove DHS flag values (≥ 20)
    if 'birth_order' in df_clean.columns:
        df_clean.loc[df_clean['birth_order'] >= 20, 'birth_order'] = np.nan

    # First-born flag: create BEFORE imputing birth_interval_months
    # Uses birth_order rather than missing b11 so it's always accurate
    df_clean['first_born'] = 0
    if 'birth_order' in df_clean.columns:
        df_clean['first_born'] = (df_clean['birth_order'] == 1).astype(int)

    # Preceding birth interval: remove impossible / flag values
    # Valid range 6-240 months; DHS flags are ≥ 994
    if 'birth_interval_months' in df_clean.columns:
        df_clean.loc[df_clean['birth_interval_months'] > 240, 'birth_interval_months'] = np.nan
        df_clean.loc[df_clean['birth_interval_months'] < 6,   'birth_interval_months'] = np.nan

    # Antenatal visits: remove DHS flag codes (≥ 97)
    if 'antenatal_visits' in df_clean.columns:
        df_clean.loc[df_clean['antenatal_visits'] >= 97, 'antenatal_visits'] = np.nan

    # Perceived birth size: valid 1-5; codes 6-8 = don't know / missing
    if 'birth_size_perceived' in df_clean.columns:
        df_clean.loc[df_clean['birth_size_perceived'] >= 6, 'birth_size_perceived'] = np.nan

    # Delivery place: recode using grouping function
    if 'delivery_place' in df_clean.columns:
        df_clean['delivery_place'] = df_clean['delivery_place'].apply(group_delivery_place)

    # ── Report ─────────────────────────────────────────────────────────────────
    n_total   = len(df_clean)
    n_stunted = df_clean['stunted'].sum()
    pct       = n_stunted / n_total * 100 if n_total > 0 else 0
    print(f"  Clean rows : {n_total:,}")
    print(f"  Stunted    : {n_stunted:,} ({pct:.1f}%)")

    # Show missingness for new variables only
    new_vars = ['mother_height_cm', 'birth_interval_months', 'delivery_place',
                'birth_size_perceived', 'birth_order', 'antenatal_visits']
    print(f"  Missing values (new variables):")
    for col in new_vars:
        if col in df_clean.columns:
            miss = df_clean[col].isna().sum()
            print(f"    {col:<25}: {miss:,} ({miss/n_total*100:.1f}% missing)")

    return df_clean


# ── MAIN PIPELINE ────────────────────────────────────────────────────────────
def run_pipeline():
    print("\n" + "="*55)
    print("  CHW STUNTING PREDICTION — DATA PIPELINE v2")
    print("="*55)

    os.makedirs('data', exist_ok=True)

    # Load all countries
    country_dfs = {}
    for name, path in DATA_PATHS.items():
        df = load_country(name, path)
        if df is not None:
            country_dfs[name] = df

    if not country_dfs:
        print("\nERROR: No datasets loaded.")
        return

    # ── Combine ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Combining datasets...")
    combined = pd.concat(country_dfs.values(), ignore_index=True)
    print(f"  Combined: {combined.shape[0]:,} rows x {combined.shape[1]} columns")

    # ── Impute missing values ──────────────────────────────────────────────────
    print("\nHandling missing values...")

    # Numeric: fill with country median
    num_cols = [
        'birth_weight_grams', 'mother_bmi', 'household_size',
        'child_age_months', 'mother_age',
        'mother_height_cm', 'birth_interval_months', 'antenatal_visits',
    ]
    for col in num_cols:
        if col in combined.columns:
            before = combined[col].isna().sum()
            combined[col] = combined.groupby('country')[col].transform(
                lambda x: x.fillna(x.median())
            )
            after = combined[col].isna().sum()
            if before > 0:
                print(f"  {col:<26}: filled {before - after:,} NaN with country median")

    # Categorical: fill with country mode
    cat_cols = [
        'child_sex', 'mother_education', 'wealth_index',
        'water_source', 'sanitation_type', 'urban_rural', 'region',
        'delivery_place', 'birth_size_perceived', 'birth_order', 'first_born',
    ]
    for col in cat_cols:
        if col in combined.columns:
            before = combined[col].isna().sum()
            combined[col] = combined.groupby('country')[col].transform(
                lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else 0)
            )
            after = combined[col].isna().sum()
            if before > 0:
                print(f"  {col:<26}: filled {before - after:,} NaN with country mode")

    before = len(combined)
    combined = combined.dropna()
    print(f"\n  Dropped {before - len(combined):,} rows with remaining NaN")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("FINAL COMBINED DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"Total rows     : {len(combined):,}")
    print(f"Total features : {len(combined.columns) - 2}")  # exclude stunted, country_name

    print(f"\nBy country:")
    for name, code in COUNTRY_CODES.items():
        sub = combined[combined['country'] == code]
        n   = len(sub)
        s   = sub['stunted'].sum()
        p   = s / n * 100 if n > 0 else 0
        print(f"  {name:<12}: {n:,} rows | {s:,} stunted ({p:.1f}%)")

    print(f"\n  Overall stunting prevalence: {combined['stunted'].mean()*100:.1f}%")
    print(f"\nColumns in final dataset:")
    for col in combined.columns:
        print(f"  {col}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_csv = os.path.join('data', 'combined_data.csv')
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    report = os.path.join('data', 'pipeline_report.txt')
    with open(report, 'w') as f:
        f.write("CHW Stunting Prediction - Pipeline Report v2\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total rows    : {len(combined):,}\n")
        f.write(f"Total features: {len(combined.columns) - 2}\n\n")
        f.write("Columns:\n")
        for col in combined.columns:
            f.write(f"  {col}\n")
        f.write("\nBy country:\n")
        for name, code in COUNTRY_CODES.items():
            sub = combined[combined['country'] == code]
            n, s = len(sub), sub['stunted'].sum()
            f.write(f"  {name}: {n:,} rows | {s:,} stunted ({s/n*100:.1f}%)\n")

    print(f"Report: {report}")
    print(f"\n[OK] Pipeline complete. Run 2_model_training.py next.\n")
    return combined


if __name__ == '__main__':
    run_pipeline()
