"""
server.py
---------
Data logging server — listens on port 5000
Receives POST /log from ESP32 and appends rows to energy_data.csv
Run this ALONGSIDE predict_server_xgb.py (port 5001)
"""

from flask import Flask, jsonify, request
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════
CSV_FILE = "energy_data.csv"

CSV_COLUMNS = [
    'timestamp',
    'voltage',
    'current',
    'power',
    'energy',
    'frequency',
    'power_factor',
    'appliance_on',
    'run_time_sec',
    'run_time_min_per_hour'
]

# ══════════════════════════════════════════════════════════════
# Ensure CSV exists with correct headers
# ══════════════════════════════════════════════════════════════
def init_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(CSV_FILE, index=False)
        print(f"✓ Created new {CSV_FILE}")
    else:
        # Check headers are correct
        existing = pd.read_csv(CSV_FILE, nrows=0)
        missing_cols = [c for c in CSV_COLUMNS if c not in existing.columns]
        if missing_cols:
            print(f"⚠️  CSV missing columns: {missing_cols} — recreating file")
            df = pd.DataFrame(columns=CSV_COLUMNS)
            df.to_csv(CSV_FILE, index=False)
        else:
            count = len(pd.read_csv(CSV_FILE))
            print(f"✓ Found existing {CSV_FILE} with {count} records")

init_csv()

# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════

@app.route('/log', methods=['POST'])
def log_data():
    """
    Receive sensor data from ESP32 and append to CSV.
    ESP32 sends JSON body with: voltage, current, power, energy,
    frequency, power_factor, appliance_on, run_time_sec,
    run_time_min_per_hour
    """
    try:
        data = request.get_json(force=True, silent=True)

        if data is None:
            return jsonify({'status': 'error', 'message': 'No JSON body received'}), 400

        # Build row — use 0 as fallback for any missing field
        row = {
            'timestamp':             datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'voltage':               float(data.get('voltage',             0)),
            'current':               float(data.get('current',             0)),
            'power':                 float(data.get('power',               0)),
            'energy':                float(data.get('energy',              0)),
            'frequency':             float(data.get('frequency',           0)),
            'power_factor':          float(data.get('power_factor',        0)),
            'appliance_on':          int(bool(data.get('appliance_on',     False))),
            'run_time_sec':          float(data.get('run_time_sec',        0)),
            'run_time_min_per_hour': float(data.get('run_time_min_per_hour', 0)),
        }

        # Append row to CSV
        df_row = pd.DataFrame([row])
        df_row.to_csv(CSV_FILE, mode='a', header=False, index=False)

        # Count total records
        total = len(pd.read_csv(CSV_FILE))

        print(
            f"[{row['timestamp']}] "
            f"V={row['voltage']:.1f}  "
            f"I={row['current']:.3f}  "
            f"P={row['power']:.1f}W  "
            f"App={'ON' if row['appliance_on'] else 'OFF'}  "
            f"Run={row['run_time_min_per_hour']:.2f}min/h  "
            f"| Total records: {total}"
        )

        return jsonify({
            'status':          'ok',
            'records_saved':   total,
            'ready_to_predict': total >= 50,
            'timestamp':       row['timestamp']
        }), 200

    except Exception as e:
        print(f"[ERROR] /log failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Quick health check — shows how many records are collected"""
    records = 0
    latest = None
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        records = len(df)
        if records > 0:
            latest = df.iloc[-1].to_dict()

    return jsonify({
        'status':           'healthy',
        'server':           'WattBot Data Logger (port 5000)',
        'csv_file':         CSV_FILE,
        'records':          records,
        'minimum_needed':   50,
        'ready_to_predict': records >= 50,
        'latest_row':       latest,
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/clear', methods=['GET'])
def clear_data():
    """Wipe CSV and start fresh — useful during testing"""
    df = pd.DataFrame(columns=CSV_COLUMNS)
    df.to_csv(CSV_FILE, index=False)
    print("[WARN] CSV cleared!")
    return jsonify({'status': 'ok', 'message': 'energy_data.csv cleared'}), 200


@app.route('/', methods=['GET'])
def home():
    records = len(pd.read_csv(CSV_FILE)) if os.path.exists(CSV_FILE) else 0
    return jsonify({
        'service':  'WattBot Data Logger',
        'port':     5000,
        'endpoints': {
            'POST /log':    'ESP32 posts sensor data here',
            'GET  /status': 'Check record count & latest row',
            'GET  /clear':  'Wipe CSV (testing only)',
        },
        'records_collected': records,
        'ready_to_predict':  records >= 50,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "=" * 60)
    print("  📡 WattBot Data Logger  (port 5000)")
    print("=" * 60)
    print(f"\n✓ CSV file  : {CSV_FILE}")
    print(f"✓ Columns   : {', '.join(CSV_COLUMNS)}")
    print(f"\n📡 Listening at:")
    print(f"   • Local  : http://localhost:5000")
    print(f"   • Network: http://{local_ip}:5000")
    print(f"\n🔗 Endpoints:")
    print(f"   • POST http://{local_ip}:5000/log     ← ESP32 sends data here")
    print(f"   • GET  http://{local_ip}:5000/status  ← check record count")
    print(f"   • GET  http://{local_ip}:5000/clear   ← wipe CSV (testing)")
    print(f"\n💡 Run predict_server_xgb.py separately on port 5001")
    print(f"💡 Need 50+ records before predictions work")
    print("\n" + "=" * 60)
    print("  Waiting for ESP32 data... Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)