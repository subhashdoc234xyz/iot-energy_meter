"""
predict_server_xgb.py
---------------------
Prediction server using XGBoost/LightGBM ensemble
Provides REST API endpoints for energy and runtime predictions
"""

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════
CSV_FILE = "energy_data.csv"
RATE_PER_UNIT = 8.0  # Rs per kWh
LOOK_BACK_HOURS = 24

# ══════════════════════════════════════════════════════════════
# Load models and preprocessors
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("  Loading XGBoost/LightGBM Models...")
print("=" * 60)

# Check if models exist
required_files = [
    'xgboost_power.pkl',
    'xgboost_run.pkl', 
    'lightgbm_power.pkl',
    'lightgbm_run.pkl',
    'meta_power.pkl',
    'meta_run.pkl',
    'feature_scaler.pkl',
    'feature_cols.pkl'
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print("\n❌ ERROR: Missing required files:")
    for f in missing_files:
        print(f"   - {f}")
    print("\n   Please run: python train_xgboost_lgbm.py first!")
    exit(1)

# Load all models
xgboost_power = joblib.load('xgboost_power.pkl')
xgboost_run = joblib.load('xgboost_run.pkl')
lightgbm_power = joblib.load('lightgbm_power.pkl')
lightgbm_run = joblib.load('lightgbm_run.pkl')
meta_power = joblib.load('meta_power.pkl')
meta_run = joblib.load('meta_run.pkl')
scaler = joblib.load('feature_scaler.pkl')
feature_cols = joblib.load('feature_cols.pkl')

print("✓ XGBoost Power model loaded")
print("✓ XGBoost Run Time model loaded")
print("✓ LightGBM Power model loaded")
print("✓ LightGBM Run Time model loaded")
print("✓ Ensemble meta-learners loaded")
print("✓ Feature scaler loaded")
print(f"✓ {len(feature_cols)} features configured")

# ══════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════

def get_features_from_csv():
    """Extract and engineer features from latest CSV data"""
    
    if not os.path.exists(CSV_FILE):
        print(f"⚠️  Warning: {CSV_FILE} not found")
        return None
    
    # Load data
    df = pd.read_csv(CSV_FILE, parse_dates=['timestamp'])
    
    if len(df) < 50:
        print(f"⚠️  Warning: Only {len(df)} records, need at least 50")
        return None
    
    # Get last 72 hours for lag features
    df = df.tail(72).copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Time segments
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    
    # Lag features (use available data)
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'power_lag_{lag}'] = df['power'].shift(lag)
        df[f'run_lag_{lag}'] = df['run_time_min_per_hour'].shift(lag)
        df[f'voltage_lag_{lag}'] = df['voltage'].shift(lag)
        df[f'current_lag_{lag}'] = df['current'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f'power_roll_mean_{window}'] = df['power'].rolling(window).mean()
        df[f'power_roll_std_{window}'] = df['power'].rolling(window).std()
        df[f'power_roll_min_{window}'] = df['power'].rolling(window).min()
        df[f'power_roll_max_{window}'] = df['power'].rolling(window).max()
        df[f'run_roll_mean_{window}'] = df['run_time_min_per_hour'].rolling(window).mean()
    
    # Exponential weighted moving averages
    for span in [6, 12, 24]:
        df[f'power_ewm_{span}'] = df['power'].ewm(span=span, adjust=False).mean()
        df[f'run_ewm_{span}'] = df['run_time_min_per_hour'].ewm(span=span, adjust=False).mean()
    
    # Rate of change
    df['power_diff_1'] = df['power'].diff(1)
    df['power_diff_12'] = df['power'].diff(12)
    df['power_diff_24'] = df['power'].diff(24)
    df['run_diff_1'] = df['run_time_min_per_hour'].diff(1)
    df['power_diff2'] = df['power_diff_1'].diff(1)
    
    # Interaction features
    df['power_voltage'] = df['power'] * df['voltage']
    df['power_current'] = df['power'] * df['current']
    df['voltage_current'] = df['voltage'] * df['current']
    
    # Appliance features
    if 'appliance_on' in df.columns:
        for lag in [1, 6, 12, 24]:
            df[f'appliance_lag_{lag}'] = df['appliance_on'].shift(lag)
        df['appliance_run_interaction'] = df['appliance_on'] * df['run_time_min_per_hour']
    else:
        df['appliance_on'] = 0
        df['appliance_run_interaction'] = 0
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Get the most recent complete row (drop NaN)
    df = df.dropna()
    
    if len(df) == 0:
        print("⚠️  Warning: No valid rows after feature engineering")
        return None
    
    # Use the latest row for prediction
    latest_row = df.iloc[-1:]
    X = latest_row[feature_cols].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

def predict_next_hours(hours=24):
    """Predict energy and runtime for next N hours"""
    
    # Get current features
    X_current = get_features_from_csv()
    if X_current is None:
        return None
    
    # Get predictions from both models
    xgb_pred_p = xgboost_power.predict(X_current)
    lgb_pred_p = lightgbm_power.predict(X_current)
    xgb_pred_r = xgboost_run.predict(X_current)
    lgb_pred_r = lightgbm_run.predict(X_current)
    
    # Stack and use meta-learner
    stacked_p = np.hstack([xgb_pred_p.reshape(-1, 1), lgb_pred_p.reshape(-1, 1)])
    stacked_r = np.hstack([xgb_pred_r.reshape(-1, 1), lgb_pred_r.reshape(-1, 1)])
    
    final_power = meta_power.predict(stacked_p)[0]
    final_run = meta_run.predict(stacked_r)[0]
    
    # Ensure valid ranges
    final_power = max(0, final_power)
    final_run = max(0, min(60, final_run))
    
    return final_power, final_run

def calculate_bill(power_w, hours):
    """Calculate electricity bill"""
    kwh = (power_w * hours) / 1000.0
    bill = kwh * RATE_PER_UNIT
    return round(bill, 2), round(kwh, 3)

def run_time_breakdown(run_min_per_hour):
    """Convert hourly run time to daily/weekly/monthly"""
    run_hour_per_day = (run_min_per_hour / 60) * 24
    run_hour_per_week = run_hour_per_day * 7
    run_hour_per_month = run_hour_per_day * 30
    
    return {
        'per_hour_min': round(run_min_per_hour, 2),
        'per_hour_hr': round(run_min_per_hour / 60, 2),
        'per_day_hr': round(run_hour_per_day, 2),
        'per_week_hr': round(run_hour_per_week, 2),
        'per_month_hr': round(run_hour_per_month, 2)
    }

# ══════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    return jsonify({
        'service': 'Energy Prediction API (XGBoost + LightGBM)',
        'version': '1.0',
        'endpoints': {
            '/predict': 'Get next hour prediction',
            '/predict/day': 'Get 24-hour prediction',
            '/predict/week': 'Get weekly forecast',
            '/dashboard': 'Web dashboard with charts',
            '/status': 'Server health check',
            '/features': 'List of features used',
            '/health': 'System health check'
        },
        'rate_per_unit': f'Rs. {RATE_PER_UNIT}/kWh',
        'models': ['XGBoost', 'LightGBM', 'Ensemble']
    })

@app.route('/predict', methods=['GET'])
def predict_next_hour():
    """Predict next hour's power consumption and runtime.
    Returns FLAT keys so ESP32 ArduinoJson can read them directly.
    """

    result = predict_next_hours(1)
    if result is None:
        return jsonify({
            'error': 'Insufficient data. Need at least 50 records in energy_data.csv',
            'status': 'error',
            'predicted_power_w':   0,
            'predicted_bill_1h':   0,
            'predicted_bill_1d':   0,
            'predicted_kwh_1h':    0,
            'predicted_kwh_1d':    0,
            'predicted_kwh_7d':    0,
            'predicted_kwh_30d':   0,
            'predicted_run_hour':  0,
            'predicted_run_day':   0,
            'predicted_run_week':  0,
            'predicted_run_month': 0,
        }), 200

    power_w, run_min   = result
    bill_1h, kwh_1h    = calculate_bill(power_w, 1)
    bill_1d, kwh_1d    = calculate_bill(power_w, 24)
    bill_7d, kwh_7d    = calculate_bill(power_w, 168)
    bill_30d, kwh_30d  = calculate_bill(power_w, 720)
    runtime            = run_time_breakdown(run_min)

    return jsonify({
        # ── FLAT keys — read directly by ESP32 fetchPrediction() ──
        'predicted_power_w':   round(power_w, 2),
        'predicted_bill_1h':   bill_1h,
        'predicted_bill_1d':   bill_1d,
        'predicted_kwh_1h':    kwh_1h,
        'predicted_kwh_1d':    kwh_1d,
        'predicted_kwh_7d':    kwh_7d,
        'predicted_kwh_30d':   kwh_30d,
        'predicted_run_hour':  runtime['per_hour_hr'],
        'predicted_run_day':   runtime['per_day_hr'],
        'predicted_run_week':  runtime['per_week_hr'],
        'predicted_run_month': runtime['per_month_hr'],
        'predicted_run_min_per_hour': runtime['per_hour_min'],
        # ── Nested keys kept for dashboard / browser use ──
        'status':    'success',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': {
            'power_w':              round(power_w, 2),
            'power_kw':             round(power_w / 1000, 3),
            'run_time_min_per_hour': round(run_min, 2),
            'cost_estimate': {
                'next_hour_rs':   bill_1h,
                'next_hour_kwh':  kwh_1h,
                'next_24h_rs':    bill_1d,
                'next_24h_kwh':   kwh_1d,
                'next_7d_rs':     bill_7d,
                'next_7d_kwh':    kwh_7d,
                'next_30d_rs':    bill_30d,
                'next_30d_kwh':   kwh_30d,
            },
            'runtime_breakdown': runtime,
        },
    })

@app.route('/predict/day', methods=['GET'])
def predict_day():
    """Predict full day (24 hours) of consumption"""
    
    # For simplicity, assume similar pattern throughout day
    result = predict_next_hours(1)
    if result is None:
        return jsonify({'error': 'Insufficient data', 'status': 'error'}), 200
    
    power_w, run_min = result
    
    # Create hourly predictions with some variation based on time
    current_hour = datetime.now().hour
    hourly_power = []
    hourly_runtime = []
    
    for i in range(24):
        hour_of_day = (current_hour + i) % 24
        
        # Adjust based on time of day
        if 6 <= hour_of_day < 12:  # Morning
            factor = 0.8
        elif 12 <= hour_of_day < 18:  # Afternoon
            factor = 1.0
        elif 18 <= hour_of_day < 22:  # Evening peak
            factor = 1.3
        else:  # Night
            factor = 0.5
        
        hourly_power.append(round(power_w * factor, 2))
        hourly_runtime.append(round(min(60, run_min * factor), 2))
    
    avg_power = np.mean(hourly_power)
    total_kwh = sum(hourly_power) / 1000
    total_bill = total_kwh * RATE_PER_UNIT
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'avg_power_w': round(avg_power, 2),
            'peak_power_w': max(hourly_power),
            'min_power_w': min(hourly_power),
            'total_kwh_24h': round(total_kwh, 3),
            'total_cost_rs': round(total_bill, 2),
            'avg_runtime_min': round(np.mean(hourly_runtime), 2)
        },
        'hourly_breakdown': [
            {
                'hour': i,
                'time': f"{i:02d}:00",
                'power_w': hourly_power[i],
                'runtime_min': hourly_runtime[i]
            }
            for i in range(24)
        ]
    })

@app.route('/predict/week', methods=['GET'])
def predict_week():
    """Predict weekly consumption and cost"""
    
    result = predict_next_hours(1)
    if result is None:
        return jsonify({'error': 'Insufficient data', 'status': 'error'}), 200
    
    power_w, run_min = result
    
    # Daily variation (weekday vs weekend)
    daily_power = []
    for day in range(7):
        if day >= 5:  # Weekend
            factor = 1.2  # Higher usage on weekends
        else:
            factor = 1.0
        daily_power.append(power_w * 24 * factor)
    
    weekly_kwh = sum(daily_power) / 1000
    weekly_bill = weekly_kwh * RATE_PER_UNIT
    monthly_bill = weekly_bill * 4.33
    
    runtime = run_time_breakdown(run_min)
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'weekly_forecast': {
            'total_kwh': round(weekly_kwh, 3),
            'total_cost_rs': round(weekly_bill, 2),
            'avg_daily_kwh': round(weekly_kwh / 7, 3),
            'avg_daily_cost_rs': round(weekly_bill / 7, 2)
        },
        'monthly_projection': {
            'total_kwh': round(weekly_kwh * 4.33, 3),
            'total_cost_rs': round(monthly_bill, 2),
            'runtime_breakdown': runtime
        },
        'daily_breakdown': [
            {
                'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day],
                'kwh': round(daily_power[day] / 1000, 3),
                'cost_rs': round(daily_power[day] / 1000 * RATE_PER_UNIT, 2)
            }
            for day in range(7)
        ]
    })

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Web dashboard with charts"""
    
    result = predict_next_hours(1)
    if result is None:
        return "<h2 style='font-family:Arial;color:red;padding:20px'>⚠️ Insufficient Data</h2><p>Need at least 50 records in energy_data.csv</p>"
    
    power_w, run_min = result
    
    # Generate 24-hour forecast
    current_hour = datetime.now().hour
    hourly_power = []
    hourly_labels = []
    
    for i in range(24):
        hour_of_day = (current_hour + i) % 24
        hourly_labels.append(f"{hour_of_day:02d}:00")
        
        if 6 <= hour_of_day < 12:
            factor = 0.8
        elif 12 <= hour_of_day < 18:
            factor = 1.0
        elif 18 <= hour_of_day < 22:
            factor = 1.3
        else:
            factor = 0.5
        
        hourly_power.append(round(power_w * factor, 2))
    
    total_kwh = sum(hourly_power) / 1000
    total_bill = total_kwh * RATE_PER_UNIT
    avg_runtime = run_min
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Energy Predictor - XGBoost/LightGBM</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #667eea; margin-bottom: 5px; }}
        .subtitle {{ color: #666; margin-bottom: 15px; }}
        .model-badge {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            display: inline-block;
            font-size: 12px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-label {{ color: #666; font-size: 14px; margin-bottom: 10px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #667eea; }}
        .stat-unit {{ font-size: 14px; color: #999; }}
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .info-text {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
            color: #666;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        .live-badge {{
            animation: pulse 2s infinite;
            background: #10b981;
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            display: inline-block;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ AI Energy Predictor</h1>
            <p class="subtitle">
                Powered by XGBoost + LightGBM Ensemble
                <span class="live-badge">LIVE</span>
            </p>
            <div>
                <span class="model-badge">🤖 XGBoost</span>
                <span class="model-badge" style="margin-left: 10px;">⚡ LightGBM</span>
                <span class="model-badge" style="margin-left: 10px;">🎯 Ensemble</span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">⚡ Predicted Power</div>
                <div class="stat-value">{power_w:.1f} <span class="stat-unit">Watts</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">⏱️ Runtime (per hour)</div>
                <div class="stat-value">{avg_runtime:.1f} <span class="stat-unit">minutes</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">💰 Today's Bill</div>
                <div class="stat-value">₹{total_bill:.2f} <span class="stat-unit">for {total_kwh:.2f} kWh</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">📊 Monthly Projection</div>
                <div class="stat-value">₹{(total_bill * 30):.0f} <span class="stat-unit">per month</span></div>
            </div>
        </div>

        <div class="chart-container">
            <h3>📈 24-Hour Power Forecast</h3>
            <canvas id="powerChart"></canvas>
        </div>

        <div class="info-text">
            💡 Predictions based on historical patterns and ensemble ML models<br>
            Updated in real-time using latest sensor data
        </div>
    </div>

    <script>
        const labels = {hourly_labels};
        const powerData = {hourly_power};
        
        new Chart(document.getElementById('powerChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Predicted Power (Watts)',
                    data: powerData,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointBackgroundColor: '#764ba2'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.parsed.y + ' Watts';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Power (Watts)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Time of Day'
                        }}
                    }}
                }}
            }}
        }});
        
        // Auto-refresh every 5 minutes
        setTimeout(() => {{
            location.reload();
        }}, 300000);
    </script>
</body>
</html>"""
    
    return html

@app.route('/status', methods=['GET'])
def status():
    """Server health check"""
    
    # Check data availability
    records = 0
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        records = len(df)
    
    return jsonify({
        'status': 'healthy',
        'server': 'Energy Prediction API',
        'models_loaded': {
            'xgboost_power': True,
            'xgboost_run': True,
            'lightgbm_power': True,
            'lightgbm_run': True,
            'meta_learners': True
        },
        'data_status': {
            'records_available': records,
            'minimum_required': 50,
            'ready_for_prediction': records >= 50,
            'csv_file': CSV_FILE
        },
        'config': {
            'rate_per_unit_rs': RATE_PER_UNIT,
            'lookback_hours': LOOK_BACK_HOURS,
            'features_count': len(feature_cols)
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/health', methods=['GET'])
def health():
    """Simple health check for monitoring"""
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/features', methods=['GET'])
def list_features():
    """List all features used by the model"""
    return jsonify({
        'total_features': len(feature_cols),
        'features': feature_cols[:50],  # Show first 50
        'note': f'Total {len(feature_cols)} features engineered from time series'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'status': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 500}), 500

# ══════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import socket
    
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "=" * 60)
    print("  🚀 XGBoost + LightGBM Prediction Server")
    print("=" * 60)
    print(f"\n✓ Models loaded successfully!")
    print(f"✓ Feature engineering ready!")
    print(f"✓ {len(feature_cols)} features available")
    print(f"✓ Ready to serve predictions")
    
    print("\n📡 Access the server at:")
    print(f"   • Local:    http://localhost:5001")
    print(f"   • Network:  http://{local_ip}:5001")
    
    print("\n🔗 Available endpoints:")
    print(f"   • Home:       http://{local_ip}:5001/")
    print(f"   • Predict:    http://{local_ip}:5001/predict  ← ESP32 calls this")
    print(f"   • Day View:   http://{local_ip}:5001/predict/day")
    print(f"   • Week View:  http://{local_ip}:5001/predict/week")
    print(f"   • Dashboard:  http://{local_ip}:5001/dashboard")
    print(f"   • Status:     http://{local_ip}:5001/status")
    
    print("\n💡 Quick test:")
    print(f"   curl http://localhost:5001/predict")
    
    print("\n" + "=" * 60)
    print("  Server is running... Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)