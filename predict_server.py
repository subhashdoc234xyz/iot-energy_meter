"""
predict_server.py
-----------------
Runs AFTER training. ESP32 calls this to get power + runtime predictions.

Endpoints:
  GET /predict          -> next-hour power + runtime prediction
  GET /predict/bill     -> predicted bill + runtime for today
  GET /dashboard        -> web page with charts + runtime
  GET /status           -> server health check
"""

from flask import Flask, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("[X] TensorFlow not found. Run: pip install tensorflow")
    exit(1)

app = Flask(__name__)

CSV_FILE    = "energy_data.csv"
MODEL_FILE  = "lstm_model.keras"
SCALER_FILE = "scaler.pkl"
LOOK_BACK   = 24
RATE_PER_UNIT = 8.0   # Rs per kWh

print("Loading model and scaler...")
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    print(f"[X] Missing {MODEL_FILE} or {SCALER_FILE}")
    print("     Run  python train_lstm.py  first!")
    exit(1)

model  = load_model(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
print("[OK] Model loaded. Ready to predict.")

N_FEATURES = scaler.n_features_in_   # should be 7


def get_last_n_readings(n=LOOK_BACK):
    if not os.path.exists(CSV_FILE):
        return None

    df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < n:
        return None

    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    if "appliance_on" not in df.columns:
        df["appliance_on"] = 0
    if "run_time_min_per_hour" not in df.columns:
        df["run_time_min_per_hour"] = df["appliance_on"] * 60.0

    features = [
        "power", "voltage", "current",
        "appliance_on", "run_time_min_per_hour",
        "hour", "day_of_week"
    ]
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    last_n = df[features].tail(n).values.astype(np.float32)
    scaled = scaler.transform(last_n)
    return scaled.reshape(1, n, N_FEATURES)


def predict_next(steps=1):
    """
    Returns list of (power_w, run_min_per_hour) tuples for next `steps` intervals.
    """
    X = get_last_n_readings(LOOK_BACK)
    if X is None:
        return None

    predictions = []
    current_seq = X.copy()

    for _ in range(steps):
        outputs = model.predict(current_seq, verbose=0)

        if isinstance(outputs, list) and len(outputs) == 2:
            pred_power_s = float(outputs[0][0, 0])
            pred_run_s   = float(outputs[1][0, 0])
        else:
            pred_power_s = float(outputs[0, 0])
            pred_run_s   = 0.5

        # Inverse transform power
        dummy_p = np.zeros((1, N_FEATURES))
        dummy_p[0, 0] = pred_power_s
        power_w = float(scaler.inverse_transform(dummy_p)[0, 0])
        power_w = max(0.0, power_w)

        # Inverse transform run time
        dummy_r = np.zeros((1, N_FEATURES))
        dummy_r[0, 4] = pred_run_s
        run_min = float(scaler.inverse_transform(dummy_r)[0, 4])
        run_min = max(0.0, min(60.0, run_min))

        predictions.append((power_w, run_min))

        # Slide window
        new_row = current_seq[0, -1, :].copy()
        new_row[0] = pred_power_s
        new_row[4] = pred_run_s
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = new_row

    return predictions


def calc_bill(power_w, hours):
    kwh = (power_w * hours) / 1000.0
    return round(kwh * RATE_PER_UNIT, 2), round(kwh, 4)


def run_min_to_times(run_min_per_hour):
    """Convert predicted run minutes per hour to per day/week/month."""
    run_hour  = run_min_per_hour / 60.0          # hours ON per hour (fraction)
    run_day   = run_hour * 24.0                  # hours ON per day
    run_week  = run_day * 7.0
    run_month = run_day * 30.0
    return round(run_hour, 3), round(run_day, 2), round(run_week, 2), round(run_month, 2)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["GET"])
def predict_endpoint():
    """ESP32 calls this every 5 minutes."""
    preds = predict_next(steps=1)
    if preds is None:
        return jsonify({"error": "Not enough data yet", "predicted_power_w": 0}), 200

    power_w, run_min = preds[0]
    bill_1h, kwh_1h  = calc_bill(power_w, 1)
    bill_1d, kwh_1d  = calc_bill(power_w, 24)
    run_hour, run_day, run_week, run_month = run_min_to_times(run_min)

    return jsonify({
        "predicted_power_w":   round(power_w, 2),
        "predicted_bill_1h":   bill_1h,
        "predicted_bill_1d":   bill_1d,
        "predicted_kwh_1d":    kwh_1d,
        "predicted_run_min_per_hour": round(run_min, 2),
        "predicted_run_hour":  run_hour,
        "predicted_run_day":   run_day,
        "predicted_run_week":  run_week,
        "predicted_run_month": run_month,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/predict/bill", methods=["GET"])
def predict_bill():
    """Predict next 24 hours power + runtime."""
    preds = predict_next(steps=24)
    if preds is None:
        return jsonify({"error": "Not enough data yet"}), 200

    powers   = [p[0] for p in preds]
    run_mins = [p[1] for p in preds]

    avg_power  = float(np.mean(powers))
    peak_power = float(np.max(powers))
    avg_run    = float(np.mean(run_mins))
    bill, kwh  = calc_bill(avg_power, 24)

    _, run_day, run_week, run_month = run_min_to_times(avg_run)

    return jsonify({
        "avg_predicted_power_w":    round(avg_power, 2),
        "peak_predicted_power_w":   round(peak_power, 2),
        "predicted_kwh_today":      kwh,
        "predicted_bill_today":     bill,
        "avg_run_min_per_hour":     round(avg_run, 2),
        "predicted_run_day_h":      run_day,
        "predicted_run_week_h":     run_week,
        "predicted_run_month_h":    run_month,
        "predictions_24h_power":    [round(p, 1) for p in powers],
        "predictions_24h_run_min":  [round(r, 1) for r in run_mins],
        "rate_per_unit":            RATE_PER_UNIT,
    })


@app.route("/dashboard", methods=["GET"])
def dashboard():
    preds = predict_next(steps=24)
    if preds is None:
        return "<h2 style='font-family:Arial;color:red;padding:20px'>Not enough data yet. Keep server.py running!</h2>"

    powers   = [p[0] for p in preds]
    run_mins = [p[1] for p in preds]

    avg_p    = round(float(np.mean(powers)), 1)
    peak_p   = round(float(np.max(powers)), 1)
    avg_r    = round(float(np.mean(run_mins)), 1)
    bill, kwh = calc_bill(avg_p, 24)
    _, run_day, run_week, run_month = run_min_to_times(avg_r)

    hours        = [(datetime.now() + timedelta(hours=i)).strftime("%H:00") for i in range(24)]
    chart_labels = str(hours)
    chart_power  = str([round(p, 1) for p in powers])
    chart_run    = str([round(r, 1) for r in run_mins])

    html = f"""<!DOCTYPE html><html>
<head>
  <title>Energy Meter - AI Predictions</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body{{font-family:Arial;background:#0f172a;color:#e2e8f0;padding:20px;margin:0}}
    h1{{color:#38bdf8;margin-bottom:4px}} h2{{color:#7dd3fc;margin:24px 0 12px}}
    .cards{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}}
    .card{{background:#1e293b;border-radius:12px;padding:18px 22px;min-width:160px;text-align:center}}
    .val{{font-size:1.8em;font-weight:bold;color:#38bdf8}} .lbl{{color:#94a3b8;font-size:0.88em;margin-top:4px}}
    .rval{{color:#fb923c}} .gval{{color:#10b981}}
    canvas{{border-radius:12px;background:#1e293b;padding:10px;max-height:260px}}
    .chart-wrap{{margin-bottom:24px}}
  </style>
</head>
<body>
  <h1>AI Energy Predictions</h1>
  <p style="color:#64748b">Next 24 hours &mdash; Updated: {datetime.now().strftime('%H:%M:%S')}</p>

  <h2>Power Predictions</h2>
  <div class="cards">
    <div class="card"><div class="val">{avg_p} W</div><div class="lbl">Avg Predicted Power</div></div>
    <div class="card"><div class="val">{peak_p} W</div><div class="lbl">Peak Predicted Power</div></div>
    <div class="card"><div class="val gval">{kwh} kWh</div><div class="lbl">Predicted Consumption</div></div>
    <div class="card"><div class="val gval">Rs.{bill}</div><div class="lbl">Predicted Bill Today</div></div>
  </div>

  <h2>Appliance Running Time Predictions</h2>
  <div class="cards">
    <div class="card"><div class="val rval">{avg_r} min</div><div class="lbl">Per Hour (avg)</div></div>
    <div class="card"><div class="val rval">{run_day} hrs</div><div class="lbl">Per Day</div></div>
    <div class="card"><div class="val rval">{run_week} hrs</div><div class="lbl">Per Week</div></div>
    <div class="card"><div class="val rval">{run_month} hrs</div><div class="lbl">Per Month</div></div>
  </div>

  <div class="chart-wrap">
    <canvas id="chartPower"></canvas>
  </div>
  <div class="chart-wrap">
    <canvas id="chartRun"></canvas>
  </div>

  <script>
    const labels = {chart_labels};
    new Chart(document.getElementById('chartPower'),{{
      type:'line',
      data:{{labels,datasets:[{{
        label:'Predicted Power (W)',data:{chart_power},
        borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,0.1)',tension:0.4,fill:true
      }}]}},
      options:{{plugins:{{legend:{{labels:{{color:'#e2e8f0'}}}}}},
        scales:{{x:{{ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}}
    }});
    new Chart(document.getElementById('chartRun'),{{
      type:'bar',
      data:{{labels,datasets:[{{
        label:'Predicted Run Time (min/hour)',data:{chart_run},
        backgroundColor:'rgba(251,146,60,0.5)',borderColor:'#fb923c',borderWidth:1
      }}]}},
      options:{{plugins:{{legend:{{labels:{{color:'#e2e8f0'}}}}}},
        scales:{{x:{{ticks:{{color:'#94a3b8'}}}},y:{{ticks:{{color:'#94a3b8'}},max:60}}}}}}
    }});
  </script>
</body></html>"""
    return html


@app.route("/status", methods=["GET"])
def status():
    count = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            count = sum(1 for _ in f) - 1
    return jsonify({
        "status":           "running",
        "model":            MODEL_FILE,
        "records":          count,
        "look_back":        LOOK_BACK,
        "rate_per_unit_rs": RATE_PER_UNIT,
    })


if __name__ == "__main__":
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print("=" * 58)
    print("  Prediction Server Running")
    print(f"  Dashboard  -> http://{ip}:5001/dashboard")
    print(f"  ESP32 API  -> http://{ip}:5001/predict")
    print(f"  Bill API   -> http://{ip}:5001/predict/bill")
    print("=" * 58)
    app.run(host="0.0.0.0", port=5001, debug=False)
