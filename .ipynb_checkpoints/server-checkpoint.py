"""
server.py
---------
Runs on your PC/laptop on the same WiFi as the ESP32.
ESP32 sends power readings here every 30s → saved to energy_data.csv
Now also logs appliance_on and run_time_sec fields.
"""

from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)

CSV_FILE = "energy_data.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "voltage", "current", "power",
            "energy", "frequency", "power_factor",
            "appliance_on", "run_time_sec", "run_time_min_per_hour"
        ])
    print(f"[OK] Created {CSV_FILE}")


@app.route("/", methods=["GET"])
def index():
    return "<h2>Server is running! Visit <a href='/data'>/data</a> or <a href='/status'>/status</a></h2>"


@app.route("/log", methods=["POST"])
def log_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [
        timestamp,
        data.get("voltage",              0),
        data.get("current",              0),
        data.get("power",                0),
        data.get("energy",               0),
        data.get("frequency",            0),
        data.get("power_factor",         0),
        1 if data.get("appliance_on", False) else 0,
        data.get("run_time_sec",         0),
        data.get("run_time_min_per_hour",0),
    ]

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    status = "ON" if data.get("appliance_on", False) else "OFF"
    print(f"[{timestamp}] Power:{data.get('power',0):.1f}W  "
          f"Current:{data.get('current',0):.2f}A  "
          f"Appliance:{status}  "
          f"RunTime:{data.get('run_time_sec',0):.0f}s")

    return jsonify({"status": "saved"}), 200


@app.route("/data", methods=["GET"])
def get_data():
    rows = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    return jsonify({"count": len(rows), "data": rows[-50:]})


@app.route("/status", methods=["GET"])
def status():
    count = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            count = sum(1 for _ in f) - 1
    return jsonify({"status": "running", "records_logged": count})


if __name__ == "__main__":
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print("=" * 52)
    print(f"  Data Collection Server")
    print(f"  PC IP: {ip}")
    print(f"  ESP32 POSTs every 30s to: http://{ip}:5000/log")
    print(f"  Saving to: {CSV_FILE}")
    print("=" * 52)
    app.run(host="0.0.0.0", port=5000, debug=False)