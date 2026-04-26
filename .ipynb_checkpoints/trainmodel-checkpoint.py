"""
train_xgboost_lgbm.py
---------------------
FINAL VERSION (Works + Matches predict_server_xgb.py)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

CSV_FILE = "energy_data.csv"
TEST_SPLIT = 0.2

# ══════════════════════════════════════════════════════════════

def load_and_engineer_features(df):
    df = df.copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'power_lag_{lag}'] = df['power'].shift(lag)
        df[f'run_lag_{lag}'] = df['run_time_min_per_hour'].shift(lag)

    # Rolling
    for window in [3, 6, 12]:
        df[f'power_roll_mean_{window}'] = df['power'].rolling(window).mean()

    df = df.dropna().reset_index(drop=True)

    print(f"Engineered {len(df.columns)} features from {len(df)} rows")
    return df

# ══════════════════════════════════════════════════════════════

def prepare_features(df):
    exclude_cols = ['timestamp', 'power', 'run_time_min_per_hour']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    y_power = df['power'].values
    y_run = df['run_time_min_per_hour'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y_power, y_run, scaler, feature_cols

# ══════════════════════════════════════════════════════════════

def time_series_split(X, y_power, y_run):
    split = int(len(X) * (1 - TEST_SPLIT))

    X_train, X_test = X[:split], X[split:]
    yp_train, yp_test = y_power[:split], y_power[split:]
    yr_train, yr_test = y_run[:split], y_run[split:]

    print(f"Train: {len(X_train)} samples ({(1-TEST_SPLIT)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({TEST_SPLIT*100:.1f}%)")

    return X_train, X_test, yp_train, yp_test, yr_train, yr_test

# ══════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train):
    print("Training XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

# ══════════════════════════════════════════════════════════════

def train_lightgbm(X_train, y_train):
    print("Training LightGBM...")

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

# ══════════════════════════════════════════════════════════════

def train_ensemble(m1, m2, X_train, y_train):
    from sklearn.linear_model import Ridge

    p1 = m1.predict(X_train).reshape(-1, 1)
    p2 = m2.predict(X_train).reshape(-1, 1)

    X_stack = np.hstack([p1, p2])

    meta = Ridge()
    meta.fit(X_stack, y_train)

    return meta

# ══════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_true, name):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{name} → MAE: {mae:.3f}, R²: {r2:.4f}")

# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n=== Energy Prediction Training ===")

    # Load
    df = pd.read_csv(CSV_FILE)
    df = load_and_engineer_features(df)

    # Prepare
    X, y_power, y_run, scaler, feature_cols = prepare_features(df)

    # Split
    X_train, X_test, yp_train, yp_test, yr_train, yr_test = time_series_split(X, y_power, y_run)

    # Train models
    xgb_power = train_xgboost(X_train, yp_train)
    lgb_power = train_lightgbm(X_train, yp_train)

    xgb_run = train_xgboost(X_train, yr_train)
    lgb_run = train_lightgbm(X_train, yr_train)

    # Ensemble
    meta_power = train_ensemble(xgb_power, lgb_power, X_train, yp_train)
    meta_run = train_ensemble(xgb_run, lgb_run, X_train, yr_train)

    # Evaluate
    print("\n--- Power ---")
    evaluate(xgb_power, X_test, yp_test, "XGB")
    evaluate(lgb_power, X_test, yp_test, "LGB")

    print("\n--- Run Time ---")
    evaluate(xgb_run, X_test, yr_test, "XGB")
    evaluate(lgb_run, X_test, yr_test, "LGB")

    # Save ALL required files (IMPORTANT)
    joblib.dump(xgb_power, 'xgboost_power.pkl')
    joblib.dump(xgb_run, 'xgboost_run.pkl')

    joblib.dump(lgb_power, 'lightgbm_power.pkl')
    joblib.dump(lgb_run, 'lightgbm_run.pkl')

    joblib.dump(meta_power, 'meta_power.pkl')
    joblib.dump(meta_run, 'meta_run.pkl')

    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(feature_cols, 'feature_cols.pkl')

    print("\n✅ All models saved correctly!")