"""
train_lstm.py
-------------
Trains an LSTM model on energy_data.csv
Now also trains to predict appliance running time per hour/day/week/month.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    print("[OK] TensorFlow loaded")
except ImportError:
    print("[X] TensorFlow not found. Run: pip install tensorflow")
    exit(1)

# ══════════════════════════════════════════════════════════════
CSV_FILE      = "energy_data.csv"
MODEL_FILE    = "lstm_model.keras"
SCALER_FILE   = "scaler.pkl"
LOOK_BACK     = 24
EPOCHS        = 50
BATCH_SIZE    = 32
TEST_SPLIT    = 0.2
# ══════════════════════════════════════════════════════════════


def load_and_clean(path):
    print(f"\n[1/5] Loading data from '{path}'...")

    if not os.path.exists(path):
        print(f"[X] File not found: {path}")
        exit(1)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"      Total records: {len(df)}")
    print(f"      Time range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")

    if len(df) < LOOK_BACK + 10:
        print(f"[X] Not enough data! Need at least {LOOK_BACK + 10} records.")
        exit(1)

    # Time features
    df["hour"]        = df["timestamp"].dt.hour
    df["minute"]      = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Appliance on as numeric
    if "appliance_on" in df.columns:
        df["appliance_on"] = pd.to_numeric(df["appliance_on"], errors="coerce").fillna(0)
    else:
        df["appliance_on"] = 0

    # Run time per hour in minutes (0-60)
    if "run_time_min_per_hour" in df.columns:
        df["run_time_min_per_hour"] = pd.to_numeric(
            df["run_time_min_per_hour"], errors="coerce"
        ).fillna(0).clip(0, 60)
    else:
        # Estimate from appliance_on column
        df["run_time_min_per_hour"] = df["appliance_on"] * 60.0

    df = df.ffill().bfill().fillna(0)
    return df


def prepare_sequences(df):
    print("\n[2/5] Preparing sequences for LSTM...")

    features = [
        "power", "voltage", "current",
        "appliance_on", "run_time_min_per_hour",
        "hour", "day_of_week"
    ]

    # Make sure all feature columns exist
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    data = df[features].values.astype(np.float32)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    joblib.dump(scaler, SCALER_FILE)
    print(f"      Scaler saved to '{SCALER_FILE}'")
    print(f"      Features: {features}")

    X, y_power, y_run = [], [], []

    for i in range(LOOK_BACK, len(data_scaled)):
        X.append(data_scaled[i - LOOK_BACK:i])
        y_power.append(data_scaled[i, 0])           # power feature (index 0)
        y_run.append(data_scaled[i, 4])              # run_time_min_per_hour (index 4)

    X       = np.array(X)
    y_power = np.array(y_power)
    y_run   = np.array(y_run)

    print(f"      Sequences: {len(X)}  |  X shape: {X.shape}")
    return X, y_power, y_run, scaler, features


def split_data(X, y_power, y_run):
    print("\n[3/5] Splitting train/test...")
    split = int(len(X) * (1 - TEST_SPLIT))
    return (
        X[:split], X[split:],
        y_power[:split], y_power[split:],
        y_run[:split],   y_run[split:]
    )


def build_model(input_shape):
    """
    Multi-output LSTM:
    - Output 1: predicted power (watts)
    - Output 2: predicted run_time_min_per_hour
    """
    print("\n[4/5] Building multi-output LSTM model...")

    inp = Input(shape=input_shape)
    x   = LSTM(64, return_sequences=True)(inp)
    x   = Dropout(0.2)(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.2)(x)
    x   = Dense(16, activation="relu")(x)

    out_power = Dense(1, name="power_out")(x)
    out_run   = Dense(1, name="run_out")(x)

    model = Model(inputs=inp, outputs=[out_power, out_run])
    model.compile(
        optimizer="adam",
        loss={"power_out": "mse", "run_out": "mse"},
        metrics={"power_out": "mae", "run_out": "mae"}
    )
    model.summary()
    return model


def train(model, X_train, y_pow_train, y_run_train, X_test, y_pow_test, y_run_test):
    print("\n[5/5] Training...")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_FILE, monitor="val_loss", save_best_only=True),
    ]

    return model.fit(
        X_train,
        {"power_out": y_pow_train, "run_out": y_run_train},
        validation_data=(
            X_test,
            {"power_out": y_pow_test, "run_out": y_run_test}
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )


def evaluate(model, X_test, y_pow_test, y_run_test, scaler):
    print("\n-- Evaluation --")

    pred_power_s, pred_run_s = model.predict(X_test)
    n = len(pred_power_s)

    # Inverse transform power
    dummy = np.zeros((n, scaler.n_features_in_))
    dummy[:, 0] = pred_power_s.flatten()
    pred_power_w = scaler.inverse_transform(dummy)[:, 0]

    dummy2 = np.zeros((n, scaler.n_features_in_))
    dummy2[:, 0] = y_pow_test.flatten()
    true_power_w = scaler.inverse_transform(dummy2)[:, 0]

    # Inverse transform run time
    dummy3 = np.zeros((n, scaler.n_features_in_))
    dummy3[:, 4] = pred_run_s.flatten()
    pred_run_min = scaler.inverse_transform(dummy3)[:, 4]

    dummy4 = np.zeros((n, scaler.n_features_in_))
    dummy4[:, 4] = y_run_test.flatten()
    true_run_min = scaler.inverse_transform(dummy4)[:, 4]

    mae_p   = mean_absolute_error(true_power_w, pred_power_w)
    pct_p   = (mae_p / (true_power_w.mean() + 1e-6)) * 100
    mae_r   = mean_absolute_error(true_run_min, pred_run_min)

    print(f"Power   MAE: {mae_p:.2f} W  ({pct_p:.1f}% error)")
    print(f"RunTime MAE: {mae_r:.2f} min/hour")
    print(f"\n[OK] Model saved -> '{MODEL_FILE}'")
    print(f"[OK] Scaler saved -> '{SCALER_FILE}'")


if __name__ == "__main__":
    print("=" * 48)
    print("  IoT Energy Meter -- LSTM Trainer")
    print("  Now with Appliance Run Time prediction")
    print("=" * 48)

    df = load_and_clean(CSV_FILE)
    X, y_power, y_run, scaler, features = prepare_sequences(df)
    X_tr, X_te, yp_tr, yp_te, yr_tr, yr_te = split_data(X, y_power, y_run)
    model = build_model((X.shape[1], X.shape[2]))
    train(model, X_tr, yp_tr, yr_tr, X_te, yp_te, yr_te)
    evaluate(model, X_te, yp_te, yr_te, scaler)
