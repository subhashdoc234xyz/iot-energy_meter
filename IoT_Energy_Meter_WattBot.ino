/*
 * ============================================================
 * WattBot — IoT Energy Meter with AI Prediction
 * + Appliance Running Time Tracker
 * Board  : ESP32 Dev Module
 * ============================================================
 * HARDWARE CONNECTIONS:
 * PZEM004T v3.0  →  ESP32
 * TX  →  GPIO16 (RX2)   RX  →  GPIO17 (TX2)
 * VCC →  5V              GND →  GND
 *
 * OLED SSD1306 0.96"  →  ESP32
 * SDA → GPIO21   SCL → GPIO22
 * VCC → 3.3V     GND → GND
 * ============================================================
 */

#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <HTTPClient.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <PZEM004Tv30.h>
#include <ArduinoJson.h>

// ════════════════════════════════════════════════════════════
//   CONFIG — CHANGE THESE
// ════════════════════════════════════════════════════════════
const char* WIFI_SSID     = "ssid_here";
const char* WIFI_PASSWORD = "password_here";

const char* PC_IP         = "ip.address.here"; // Your PC's IP (hotspot/LAN)
const int   LOG_PORT      = 5000;
const int   PRED_PORT     = 5001;

const unsigned long SEND_INTERVAL    = 30000UL;  // 30 seconds
const unsigned long PREDICT_INTERVAL = 300000UL;  // 5 minutes

const float DEFAULT_RATE = 8.0;
// Appliance detection threshold (watts) — if power > this, appliance is ON
const float APPLIANCE_ON_THRESHOLD = 0.1;

// ════════════════════════════════════════════════════════════
//   HARDWARE PINS & OBJECTS
// ════════════════════════════════════════════════════════════
#define OLED_WIDTH   128
#define OLED_HEIGHT  64
#define OLED_RESET   -1
#define EEPROM_SIZE  16
#define EEPROM_RATE_ADDR 0

PZEM004Tv30      pzem(Serial2, 16, 17);
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, OLED_RESET);
AsyncWebServer   server(80);

// ════════════════════════════════════════════════════════════
//   GLOBAL STATE — SENSOR
// ════════════════════════════════════════════════════════════
float voltage     = 0.0;
float current     = 0.0;
float power       = 0.0;
float energy      = 0.0;
float frequency   = 0.0;
float powerFactor = 0.0;
bool  sensorOk    = false;

float ratePerUnit = DEFAULT_RATE;

float predictedPower  = 0.0;
float predictedBill1h = 0.0;
float predictedBill1d = 0.0;
float predictedKwh1d  = 0.0;
bool  predictionReady = false;
String predictionTime = "--:--";

// Predicted running times from AI
float predictedRunHour  = 0.0;
float predictedRunDay   = 0.0;
float predictedRunWeek  = 0.0;
float predictedRunMonth = 0.0;

// ════════════════════════════════════════════════════════════
//   APPLIANCE RUNNING TIME TRACKER
// ════════════════════════════════════════════════════════════
bool  applianceOn    = false;
unsigned long appOnStart = 0;       // millis when appliance turned ON
unsigned long totalOnMs  = 0;       // cumulative ms ON this session

// Computed running time (seconds)
float runTimeTotalSec  = 0.0;
float runTimePerHour   = 0.0;   // fraction of last hour it was ON
float runTimePerDay    = 0.0;   // hours ON in last 24h (estimated)
float runTimePerWeek   = 0.0;   // hours ON in last 7 days (estimated)
float runTimePerMonth  = 0.0;   // hours ON in last 30 days (estimated)

// Track session uptime for estimation
unsigned long bootMillis = 0;

// ════════════════════════════════════════════════════════════
//   OLED PAGE
// ════════════════════════════════════════════════════════════
int oledPage = 0;
unsigned long lastPageSwitch = 0;
const unsigned long PAGE_INTERVAL = 3000;

unsigned long lastSendTime    = 0;
unsigned long lastPredictTime = 0;

// ════════════════════════════════════════════════════════════
//   EEPROM HELPERS
// ════════════════════════════════════════════════════════════
void saveRate(float rate) {
  EEPROM.put(EEPROM_RATE_ADDR, rate);
  EEPROM.commit();
}

float loadRate() {
  float r;
  EEPROM.get(EEPROM_RATE_ADDR, r);
  if (isnan(r) || r <= 0 || r > 50) return DEFAULT_RATE;
  return r;
}

// ════════════════════════════════════════════════════════════
//   BILL CALCULATION
// ════════════════════════════════════════════════════════════
float calcBill(float watts, float hours) {
  return ((watts * hours) / 1000.0) * ratePerUnit;
}

// ════════════════════════════════════════════════════════════
//   APPLIANCE RUNTIME CALCULATION
// ════════════════════════════════════════════════════════════
void updateRunTime() {
  bool isOn = (sensorOk && power > APPLIANCE_ON_THRESHOLD);
  if (isOn && !applianceOn) {
    applianceOn = true;
    appOnStart  = millis();
  } else if (!isOn && applianceOn) {
    applianceOn = false;
    totalOnMs  += (millis() - appOnStart);
  }

  // Total seconds ON since boot
  unsigned long currentOnMs = totalOnMs;
  if (applianceOn) currentOnMs += (millis() - appOnStart);
  runTimeTotalSec = currentOnMs / 1000.0;

  // Uptime since boot in seconds
  float uptimeSec = (millis() - bootMillis) / 1000.0;
  if (uptimeSec < 1) uptimeSec = 1;

  // ON fraction (0.0 to 1.0)
  float onFraction   = runTimeTotalSec / uptimeSec;

  // Extrapolate to periods
  runTimePerHour   = onFraction * 3600.0;                    // seconds ON per hour
  runTimePerDay    = (onFraction * 86400.0) / 3600.0;        // hours ON per day
  runTimePerWeek   = runTimePerDay * 7.0;
  runTimePerMonth  = runTimePerDay * 30.0;
}

String formatRunTime(float seconds) {
  if (seconds < 60) {
    return String((int)seconds) + "s";
  } else if (seconds < 3600) {
    return String((int)(seconds / 60)) + "m " + String((int)((int)seconds % 60)) + "s";
  } else {
    int h = (int)(seconds / 3600);
    int m = (int)((seconds - h * 3600) / 60);
    return String(h) + "h " + String(m) + "m";
  }
}

// ════════════════════════════════════════════════════════════
//   PZEM SENSOR READ
// ════════════════════════════════════════════════════════════
void readSensor() {
  float v  = pzem.voltage();
  float c  = pzem.current();
  float p  = pzem.power();
  float e  = pzem.energy();
  float f  = pzem.frequency();
  float pf = pzem.pf();

  if (!isnan(v) && !isnan(c) && !isnan(p)) {
    voltage     = v  * 1.03607;
    current     = c;
    power       = p  * 1.03607;
    energy      = e  * 1.03607;
    frequency   = f;
    powerFactor = pf;
    sensorOk    = true;
  } else {
    sensorOk = false;
  }
  updateRunTime();
}

// ════════════════════════════════════════════════════════════
//   OLED PAGES
// ════════════════════════════════════════════════════════════
void oledPage0_LiveReadings() {
  display.setRotation(2);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);  display.println(F("-- Live Readings --"));
  display.print(F("V: "));  display.print(voltage, 1);  display.println(F(" V"));
  display.print(F("I: "));  display.print(current, 3);  display.println(F(" A"));
  display.print(F("P: "));  display.print(power, 1);    display.println(F(" W"));
  display.print(F("E: "));
  display.print(energy, 3);   display.println(F(" kWh"));
  if (!sensorOk) { display.setCursor(0, 56); display.println(F("! SENSOR ERROR")); }
  display.display();
}

void oledPage1_FreqPF() {
  display.setRotation(2);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);  display.println(F("-- Power Quality --"));
  display.print(F("Freq: ")); display.print(frequency, 1);   display.println(F(" Hz"));
  display.print(F("PF  : ")); display.println(powerFactor, 3);
  display.println();
  display.print(F("Rate: Rs.")); display.print(ratePerUnit, 1); display.println(F("/kWh"));
  display.display();
}

void oledPage2_Bills() {
  display.setRotation(2);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);  display.println(F("-- Bill Estimate --"));
  display.print(F("1 hr : Rs.")); display.println(calcBill(power, 1),   2);
  display.print(F("1 day: Rs.")); display.println(calcBill(power, 24),  2);
  display.print(F("7 day: Rs."));
  display.println(calcBill(power, 168), 2);
  display.print(F("30day: Rs.")); display.println(calcBill(power, 720), 2);
  display.display();
}

void oledPage3_RunTime() {
  display.setRotation(2);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("-- Appliance Time --"));
  display.print(F("Now: "));
  display.println(applianceOn ? F("ON") : F("OFF"));
  display.print(F("Total: "));
  display.println(formatRunTime(runTimeTotalSec));
  display.print(F("/hr : "));
  display.print((int)(runTimePerHour / 60));
  display.println(F(" min"));
  display.print(F("/day: "));
  display.print(runTimePerDay, 1);
  display.println(F(" hrs"));
  display.display();
}

void oledPage4_AIPrediction() {
  display.setRotation(2);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("-- AI Prediction --"));
  if (predictionReady) {
    display.print(F("Power: "));   display.print(predictedPower, 1);   display.println(F(" W"));
    display.print(F("Bill/h: Rs.")); display.println(predictedBill1h, 2);
    display.print(F("Bill/d: Rs.")); display.println(predictedBill1d, 2);
    display.print(F("Run/d: "));   display.print(predictedRunDay, 1);  display.println(F(" h"));
  } else {
    display.println(F("Waiting for"));
    display.println(F("AI server..."));
    display.println(F("Run predict"));
    display.println(F("_server.py on PC"));
  }
  display.display();
}

void oledPage5_WiFiStatus() {
  display.setRotation(2);
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);  display.println(F("-- Network --"));
  if (WiFi.status() == WL_CONNECTED) {
    display.println(F("WiFi: Connected"));
    display.print(F("IP: "));    display.println(WiFi.localIP());
    display.print(F("RSSI: ")); display.print(WiFi.RSSI()); display.println(F(" dBm"));
    display.println(F("Open IP in browser"));
  } else {
    display.println(F("WiFi: DISCONNECTED"));
  }
  display.display();
}

void updateOLED() {
  if (millis() - lastPageSwitch >= PAGE_INTERVAL) {
    lastPageSwitch = millis();
    oledPage = (oledPage + 1) % 6;
  }
  switch (oledPage) {
    case 0: oledPage0_LiveReadings(); break;
    case 1: oledPage1_FreqPF();       break;
    case 2: oledPage2_Bills();        break;
    case 3: oledPage3_RunTime();      break;
    case 4: oledPage4_AIPrediction(); break;
    case 5: oledPage5_WiFiStatus();   break;
  }
}

// ════════════════════════════════════════════════════════════
//   ML — SEND DATA TO server.py  (every 30s)
// ════════════════════════════════════════════════════════════
void sendDataToServer() {
  if (WiFi.status() != WL_CONNECTED || !sensorOk) return;
  HTTPClient http;
  String url = String("http://") + PC_IP + ":" + LOG_PORT + "/log";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);

  String body = "{";
  body += "\"voltage\":"         + String(voltage,      2) + ",";
  body += "\"current\":"         + String(current,      3) + ",";
  body += "\"power\":"           + String(power,        2) + ",";
  body += "\"energy\":"          + String(energy,       4) + ",";
  body += "\"frequency\":"       + String(frequency,    1) + ",";
  body += "\"power_factor\":"    + String(powerFactor,  3) + ",";
  body += "\"appliance_on\":"    + String(applianceOn ? "true" : "false") + ",";
  body += "\"run_time_sec\":"    + String(runTimeTotalSec, 1) + ",";
  body += "\"run_time_min_per_hour\":" + String(runTimePerHour / 60.0, 2);
  body += "}";

  int code = http.POST(body);
  if (code == 200) {
    Serial.println("[ML] Data logged to server OK (30s interval)");
  } else {
    Serial.print("[ML] Log failed HTTP "); Serial.println(code);
  }
  http.end();
}

// ════════════════════════════════════════════════════════════
//   ML — FETCH PREDICTION FROM predict_server.py
// ════════════════════════════════════════════════════════════
void fetchPrediction() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String("http://") + PC_IP + ":" + PRED_PORT + "/predict";
  http.begin(url);
  http.setTimeout(8000);

  int code = http.GET();
  if (code == 200) {
    String payload = http.getString();
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, payload);
    if (!err) {
      predictedPower      = doc["predicted_power_w"]     | 0.0f;
      predictedBill1h     = doc["predicted_bill_1h"]     | 0.0f;
      predictedBill1d     = doc["predicted_bill_1d"]     | 0.0f;
      predictedKwh1d      = doc["predicted_kwh_1d"]      | 0.0f;
      predictedRunHour    = doc["predicted_run_hour"]    | 0.0f;
      predictedRunDay     = doc["predicted_run_day"]     | 0.0f;
      predictedRunWeek    = doc["predicted_run_week"]    | 0.0f;
      predictedRunMonth   = doc["predicted_run_month"]   | 0.0f;
      predictionReady     = true;

      unsigned long t = millis() / 1000;
      int h = (t / 3600) % 24;
      int m = (t / 60) % 60;
      predictionTime = (h < 10 ? "0" : "") + String(h) + ":" + (m < 10 ? "0" : "") + String(m);
      Serial.print("[AI] Predicted: "); Serial.print(predictedPower, 1);
      Serial.print(" W  Run/day: ");    Serial.print(predictedRunDay, 1); Serial.println(" h");
    }
  } else {
    Serial.print("[AI] Prediction fetch failed HTTP "); Serial.println(code);
  }
  http.end();
}

// ════════════════════════════════════════════════════════════
//   WEB DASHBOARD HTML
// ════════════════════════════════════════════════════════════
String buildDashboardHTML() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="5">
<title>WattBot Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #050505;
    --surface: rgba(20, 20, 25, 0.6);
    --border: rgba(255, 255, 255, 0.08);
    --accent-1: #00f0ff;
    --accent-2: #7000ff;
    --success: #00ff9d;
    --warning: #ffb800;
    --danger: #ff003c;
    --text-main: #ffffff;
    --text-muted: #8892b0;
  }
  
  * { box-sizing: border-box; margin: 0; padding: 0; }
  
  body {
    background-color: var(--bg);
    color: var(--text-main);
    font-family: 'Outfit', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
    position: relative;
    overflow-x: hidden;
  }

  /* Animated Background Elements */
  .bg-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    z-index: -1;
    opacity: 0.5;
    animation: drift 20s infinite alternate ease-in-out;
  }
  .orb-1 { width: 400px; height: 400px; background: rgba(0, 240, 255, 0.15); top: -100px; left: -100px; }
  .orb-2 { width: 500px; height: 500px; background: rgba(112, 0, 255, 0.15); bottom: -150px; right: -100px; animation-delay: -5s; }
  .orb-3 { width: 300px; height: 300px; background: rgba(0, 255, 157, 0.1); top: 40%; left: 50%; transform: translate(-50%, -50%); animation-delay: -10s; }

  @keyframes drift {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(50px, 30px) scale(1.1); }
  }

  .container {
    max-width: 1200px;
    width: 100%;
    z-index: 1;
  }

  /* Header */
  .header {
    text-align: center;
    margin-bottom: 3rem;
    position: relative;
  }
  
  .logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: -1px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 20px rgba(165, 180, 252, 0.3));
    animation: pulse 4s infinite alternate;
  }
  
  .logo span {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .tagline {
    color: var(--text-muted);
    font-size: 1.1rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  /* Layout */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }
  
  .section-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }

  /* Glassmorphism Cards */
  .card, .section {
    background: var(--surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.5rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
  }
  
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  }

  .card:hover {
    transform: translateY(-8px);
    border-color: rgba(255, 255, 255, 0.2);
    box-shadow: 0 20px 40px -10px rgba(0,240,255,0.15);
  }

  /* Main Metrics */
  .metric-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 2rem 1.5rem;
  }
  
  .metric-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    opacity: 0.8;
  }

  .metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1.1;
    margin: 0.5rem 0;
    text-shadow: 0 0 20px rgba(255,255,255,0.1);
  }

  .metric-val span {
    font-size: 1.2rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-left: 4px;
  }

  .metric-lbl {
    font-size: 0.9rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
  }

  /* Specific Metric Colors */
  .metric-v .metric-val { color: #a78bfa; }
  .metric-a .metric-val { color: #38bdf8; }
  .metric-w .metric-val { color: #fbbf24; }
  .metric-k .metric-val { color: #34d399; }

  /* Sections */
  .section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .section-title::before {
    content: '';
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent-1);
    box-shadow: 0 0 10px var(--accent-1);
  }

  .section-title.ai-title::before {
    background: var(--accent-2);
    box-shadow: 0 0 10px var(--accent-2);
  }

  /* Data Grids */
  .data-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }

  .data-item {
    background: rgba(0,0,0,0.2);
    border-radius: 16px;
    padding: 1.2rem;
    border: 1px solid rgba(255,255,255,0.03);
    transition: all 0.3s ease;
  }

  .data-item:hover {
    background: rgba(255,255,255,0.03);
    transform: translateY(-2px);
  }

  .data-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
  }

  .data-lbl {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
  }

  /* Bill Colors */
  .bill-item .data-val { color: var(--success); }
  .ai-item .data-val { background: linear-gradient(90deg, var(--accent-1), var(--accent-2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

  /* Rate Form */
  .rate-form {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
  }

  .rate-form input {
    flex: 1;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: white;
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    transition: all 0.3s ease;
    outline: none;
  }

  .rate-form input:focus {
    border-color: var(--accent-1);
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
  }

  .rate-form button {
    background: linear-gradient(135deg, var(--accent-1), #0077ff);
    color: #000;
    border: none;
    border-radius: 12px;
    padding: 0 1.5rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 240, 255, 0.3);
  }

  .rate-form button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 240, 255, 0.5);
  }

  /* Appliance Status */
  .status-badge {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
  }

  .status-on {
    background: rgba(0, 255, 157, 0.1);
    border: 1px solid rgba(0, 255, 157, 0.2);
    color: var(--success);
  }

  .status-off {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: var(--text-muted);
  }

  .pulse-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }

  .status-on .pulse-dot {
    background: var(--success);
    box-shadow: 0 0 12px var(--success);
    animation: blink 1.5s infinite;
  }

  .status-off .pulse-dot {
    background: var(--text-muted);
  }

  @keyframes blink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }

  .footer-status {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 3rem;
    padding: 1.5rem;
    background: var(--surface);
    border-radius: 100px;
    border: 1px solid var(--border);
    font-size: 0.9rem;
    font-weight: 500;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-ok { color: var(--success); }
  .status-err { color: var(--danger); }
  .status-ip { color: var(--accent-1); }

  @media (max-width: 768px) {
    .section-grid { grid-template-columns: 1fr; }
    .footer-status { flex-direction: column; gap: 1rem; border-radius: 24px; text-align: center; }
  }
</style>
</head>
<body>

<div class="bg-orb orb-1"></div>
<div class="bg-orb orb-2"></div>
<div class="bg-orb orb-3"></div>

<div class="container">
  <div class="header">
    <div class="logo">Watt<span>Bot</span> ⚡</div>
    <div class="tagline">Next-Gen Energy Intelligence</div>
  </div>

)rawliteral";

  // Live readings
  html += "<div class='grid'>";
  html += "<div class='card metric-card metric-v'><div class='metric-icon'>⚡</div><div class='metric-val'>" + String(voltage,1)     + "<span>V</span></div><div class='metric-lbl'>Voltage</div></div>";
  html += "<div class='card metric-card metric-a'><div class='metric-icon'>🌊</div><div class='metric-val'>" + String(current,2)     + "<span>A</span></div><div class='metric-lbl'>Current</div></div>";
  html += "<div class='card metric-card metric-w'><div class='metric-icon'>🔥</div><div class='metric-val'>" + String(power,1)       + "<span>W</span></div><div class='metric-lbl'>Power</div></div>";
  html += "<div class='card metric-card metric-k'><div class='metric-icon'>📈</div><div class='metric-val'>" + String(energy,3)      + "<span>kWh</span></div><div class='metric-lbl'>Energy</div></div>";
  html += "<div class='card metric-card'><div class='metric-icon'>〰️</div><div class='metric-val'>" + String(frequency,1)   + "<span>Hz</span></div><div class='metric-lbl'>Frequency</div></div>";
  html += "<div class='card metric-card'><div class='metric-icon'>📐</div><div class='metric-val'>" + String(powerFactor,2) + "</div><div class='metric-lbl'>Power Factor</div></div>";
  html += "</div>";

  html += "<div class='section-grid'>";
  
  // Bill estimates
  html += "<div class='section'><div class='section-title'>Bill Estimate (Rs." + String(ratePerUnit,1) + "/kWh)</div>";
  html += "<div class='data-grid'>";
  html += "<div class='data-item bill-item'><div class='data-val'>Rs." + String(calcBill(power,1),2)   + "</div><div class='data-lbl'>1 Hour</div></div>";
  html += "<div class='data-item bill-item'><div class='data-val'>Rs." + String(calcBill(power,24),2)  + "</div><div class='data-lbl'>1 Day</div></div>";
  html += "<div class='data-item bill-item'><div class='data-val'>Rs." + String(calcBill(power,168),2) + "</div><div class='data-lbl'>7 Days</div></div>";
  html += "<div class='data-item bill-item'><div class='data-val'>Rs." + String(calcBill(power,720),2) + "</div><div class='data-lbl'>30 Days</div></div>";
  html += "</div>";
  html += R"rawliteral(
  <div class="rate-form">
    <input type="number" id="rI" placeholder="New rate (Rs/kWh)" step="0.1" min="0.1" max="50" inputmode="decimal">
    <button onclick="var r=document.getElementById('rI').value;if(r>0&&r<=50)window.location.href='/setrate?rate='+r;else alert('Enter a valid rate between 0.1 and 50');">Update</button>
  </div>
  </div>)rawliteral";

  // Appliance Running Time
  html += "<div class='section'><div class='section-title'>Appliance Tracking</div>";
  if (applianceOn) {
    html += "<div class='status-badge status-on'><div class='pulse-dot'></div>Active — Currently Running</div>";
  } else {
    html += "<div class='status-badge status-off'><div class='pulse-dot'></div>Standby — Powered Off</div>";
  }
  html += "<div class='data-grid' style='margin-bottom: 1rem;'>";
  html += "<div class='data-item'><div class='data-val' style='color:#fb923c;'>" + formatRunTime(runTimePerHour)         + "</div><div class='data-lbl'>Per Hour</div></div>";
  html += "<div class='data-item'><div class='data-val' style='color:#fb923c;'>" + String(runTimePerDay, 1) + "h"     + "</div><div class='data-lbl'>Per Day</div></div>";
  html += "<div class='data-item'><div class='data-val' style='color:#fb923c;'>" + String(runTimePerWeek, 1) + "h"    + "</div><div class='data-lbl'>Per Week</div></div>";
  html += "<div class='data-item'><div class='data-val' style='color:#fb923c;'>" + String(runTimePerMonth, 1) + "h"   + "</div><div class='data-lbl'>Per Month</div></div>";
  html += "</div>";
  html += "<div style='color:var(--text-muted);font-size:0.85rem;text-align:center;'>Total session ON time: <span style='color:#fb923c;font-weight:700;'>" + formatRunTime(runTimeTotalSec) + "</span> | Threshold: >" + String(APPLIANCE_ON_THRESHOLD,0) + "W</div>";
  html += "</div>";

  // AI Prediction
  html += "<div class='section' style='grid-column: 1 / -1;'><div class='section-title ai-title'>AI Forecaster (XGBoost + LightGBM)</div>";
  if (predictionReady) {
    html += "<div class='data-grid' style='grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-bottom: 1.5rem;'>";
    html += "<div class='data-item ai-item'><div class='data-val'>" + String(predictedPower,1)  + " W</div><div class='data-lbl'>Predicted Power</div></div>";
    html += "<div class='data-item ai-item'><div class='data-val'>Rs." + String(predictedBill1h,2) + "</div><div class='data-lbl'>Predicted / Hour</div></div>";
    html += "<div class='data-item ai-item'><div class='data-val'>Rs." + String(predictedBill1d,2) + "</div><div class='data-lbl'>Predicted / Day</div></div>";
    html += "<div class='data-item ai-item'><div class='data-val'>" + String(predictedKwh1d,3)  + " kWh</div><div class='data-lbl'>Predicted kWh / Day</div></div>";
    html += "</div>";
    
    html += "<div class='data-lbl' style='margin-bottom: 1rem; color:var(--accent-2);'>AI Predicted Running Time</div>";
    html += "<div class='data-grid' style='grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));'>";
    html += "<div class='data-item'><div class='data-val' style='color:#a78bfa;'>" + String(predictedRunHour*60,0)  + "m</div><div class='data-lbl'>Per Hour</div></div>";
    html += "<div class='data-item'><div class='data-val' style='color:#a78bfa;'>" + String(predictedRunDay,1)      + "h</div><div class='data-lbl'>Per Day</div></div>";
    html += "<div class='data-item'><div class='data-val' style='color:#a78bfa;'>" + String(predictedRunWeek,1)     + "h</div><div class='data-lbl'>Per Week</div></div>";
    html += "<div class='data-item'><div class='data-val' style='color:#a78bfa;'>" + String(predictedRunMonth,1)    + "h</div><div class='data-lbl'>Per Month</div></div>";
    html += "</div>";
    html += "<div style='color:var(--text-muted);font-size:0.85rem;text-align:right;margin-top:1rem;'>Last updated: " + predictionTime + "</div>";
  } else {
    html += "<div style='padding: 3rem; text-align: center; background: rgba(0,0,0,0.2); border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1);'>";
    html += "<div class='pulse-dot' style='background: var(--accent-2); box-shadow: 0 0 15px var(--accent-2); margin: 0 auto 1rem;'></div>";
    html += "<div style='font-size: 1.1rem; color: var(--text-main); margin-bottom: 0.5rem;'>Awaiting Intelligence</div>";
    html += "<div style='color: var(--text-muted); font-size: 0.9rem;'>Start predict_server.py on the master node</div>";
    html += "</div>";
  }
  html += "</div>";
  
  html += "</div>"; // End section-grid

  html += "<div class='footer-status'>";
  html += "<div class='status-item'>Sensor: <span class='" + String(sensorOk ? "status-ok" : "status-err") + "'>" + String(sensorOk ? "ONLINE" : "OFFLINE") + "</span></div>";
  html += "<div class='status-item'>Network: <span class='status-ip'>" + WiFi.localIP().toString() + "</span></div>";
  html += "<div class='status-item'>Sync: <span style='color:var(--text-main);'>5s Interval</span></div>";
  html += "</div>";

  html += "<div style='text-align:center; margin-top: 2rem; color: var(--text-muted); font-size: 0.85rem; font-weight: 500; letter-spacing: 1px;'>";
  html += "Powered by <span style='color: var(--accent-1); font-weight: 700;'>WattBot</span> ⚡ Intelligent Energy";
  html += "</div>";

  html += "</div>"; // End container
  html += "</body></html>";
  return html;
}

// ════════════════════════════════════════════════════════════
//   JSON API
// ════════════════════════════════════════════════════════════
String buildDataJSON() {
  JsonDocument doc;
  doc["voltage"]       = round(voltage * 10) / 10.0;
  doc["current"]       = round(current * 1000) / 1000.0;
  doc["power"]         = round(power * 10) / 10.0;
  doc["energy"]        = energy;
  doc["frequency"]     = round(frequency * 10) / 10.0;
  doc["power_factor"]  = round(powerFactor * 100) / 100.0;
  doc["rate"]          = ratePerUnit;
  doc["bill_1h"]       = round(calcBill(power,1) * 100) / 100.0;
  doc["bill_1d"]       = round(calcBill(power,24) * 100) / 100.0;
  doc["sensor_ok"]     = sensorOk;
  doc["appliance_on"]  = applianceOn;
  doc["run_total_sec"] = runTimeTotalSec;
  doc["run_per_hour_min"] = runTimePerHour / 60.0;
  doc["run_per_day_h"]    = runTimePerDay;
  doc["run_per_week_h"]   = runTimePerWeek;
  doc["run_per_month_h"]  = runTimePerMonth;
  JsonObject ai = doc.createNestedObject("ai_prediction");
  ai["power_w"]        = predictedPower;
  ai["bill_1h"]        = predictedBill1h;
  ai["bill_1d"]        = predictedBill1d;
  ai["kwh_1d"]         = predictedKwh1d;
  ai["run_hour"]       = predictedRunHour;
  ai["run_day"]        = predictedRunDay;
  ai["run_week"]       = predictedRunWeek;
  ai["run_month"]      = predictedRunMonth;
  ai["ready"]          = predictionReady;
  ai["updated"]        = predictionTime;
  String out;
  serializeJson(doc, out);
  return out;
}

// ════════════════════════════════════════════════════════════
//   BOOT ANIMATIONS
// ════════════════════════════════════════════════════════════

// Phase 1 — pure glitch noise (same as original)
void bootAnimationGlitch() {
  unsigned long start = millis();
  while (millis() - start < 500) {
    display.clearDisplay();
    // Random noise pixels
    for (int i = 0; i < 400; i++) {
      display.drawPixel(random(OLED_WIDTH), random(OLED_HEIGHT), SSD1306_WHITE);
    }
    // Corrupted horizontal blocks and lines
    for (int i = 0; i < 5; i++) {
      display.fillRect(random(OLED_WIDTH), random(OLED_HEIGHT), random(10, 30), random(2, 10), SSD1306_WHITE);
      display.drawLine(0, random(OLED_HEIGHT), OLED_WIDTH, random(OLED_HEIGHT), SSD1306_WHITE);
    }
    display.display();
    // Rapid flicker
    display.invertDisplay(random(2));
    delay(random(10, 50));
  }
  display.invertDisplay(false);
  display.clearDisplay();
  display.display();
  delay(200);
}

// Phase 2 — glitchy "BOOTING..." text with scanline corruption
void bootAnimationBooting() {
  const char* label = "BOOTING...";
  for (int pass = 0; pass < 18; pass++) {
    display.clearDisplay();

    // Background glitch strips — random corrupted rows
    for (int i = 0; i < 4; i++) {
      int gy = random(OLED_HEIGHT);
      display.drawLine(0, gy, random(20, OLED_WIDTH), gy, SSD1306_WHITE);
    }
    // A few noise pixels
    for (int i = 0; i < 60; i++) {
      display.drawPixel(random(OLED_WIDTH), random(OLED_HEIGHT), SSD1306_WHITE);
    }

    // Draw "BOOTING..." with random horizontal jitter per character
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    int baseX = 4;
    int baseY = 22;
    for (int c = 0; c < (int)strlen(label); c++) {
      int jx = (pass % 3 == 0) ? random(-3, 4) : 0;
      int jy = (pass % 4 == 0) ? random(-2, 3) : 0;
      display.setCursor(baseX + c * 12 + jx, baseY + jy);
      display.print(label[c]);
    }

    // Occasional full invert flash
    if (pass % 6 == 0) {
      display.display();
      display.invertDisplay(true);
      delay(40);
      display.invertDisplay(false);
    } else {
      display.display();
      delay(random(50, 100));
    }
  }
  display.clearDisplay();
  display.display();
  delay(100);
}

// Phase 3 — "WattBot" materialises through glitch then holds clean
void bootAnimationWattBot() {
  const char* line1 = "Watt";
  const char* line2 = "Bot";

  // First: glitch-reveal — text appears through noise over 12 frames
  for (int pass = 0; pass < 12; pass++) {
    display.clearDisplay();

    // Decreasing noise as pass increases
    int noiseAmt = map(pass, 0, 11, 300, 20);
    for (int i = 0; i < noiseAmt; i++) {
      display.drawPixel(random(OLED_WIDTH), random(OLED_HEIGHT), SSD1306_WHITE);
    }
    // Scanlines fading out
    if (pass < 8) {
      for (int i = 0; i < (8 - pass); i++) {
        display.drawLine(0, random(OLED_HEIGHT), OLED_WIDTH, random(OLED_HEIGHT), SSD1306_WHITE);
      }
    }

    // "Watt" — size 2, centred on upper half
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    int16_t x1, y1; uint16_t w, h;
    display.getTextBounds(line1, 0, 0, &x1, &y1, &w, &h);
    int jx = (pass < 6) ? random(-4, 5) : 0;
    int jy = (pass < 6) ? random(-2, 3) : 0;
    display.setCursor((OLED_WIDTH - w) / 2 + jx, 6 + jy);
    display.print(line1);

    // "Bot" — size 2, centred on lower half
    display.getTextBounds(line2, 0, 0, &x1, &y1, &w, &h);
    jx = (pass < 6) ? random(-4, 5) : 0;
    jy = (pass < 6) ? random(-2, 3) : 0;
    display.setCursor((OLED_WIDTH - w) / 2 + jx, 30 + jy);
    display.print(line2);

    display.display();

    // Invert flash on early frames only
    if (pass < 4 && pass % 2 == 0) {
      display.invertDisplay(true);
      delay(30);
      display.invertDisplay(false);
    }
    delay(random(40, 90));
  }

  // Final clean hold: WattBot + underline + tagline
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(2);
  int16_t x1, y1; uint16_t w, h;
  display.getTextBounds("Watt", 0, 0, &x1, &y1, &w, &h);
  display.setCursor((OLED_WIDTH - w) / 2, 6);
  display.print(F("Watt"));

  display.getTextBounds("Bot", 0, 0, &x1, &y1, &w, &h);
  display.setCursor((OLED_WIDTH - w) / 2, 28);
  display.print(F("Bot"));

  display.drawLine(18, 50, OLED_WIDTH - 18, 50, SSD1306_WHITE);

  display.setTextSize(1);
  display.getTextBounds("Energy Monitor", 0, 0, &x1, &y1, &w, &h);
  display.setCursor((OLED_WIDTH - w) / 2, 54);
  display.print(F("Energy Monitor"));

  display.display();
  delay(2000);   // hold WattBot on screen for 2 seconds

  display.clearDisplay();
  display.display();
  delay(100);
}

// ════════════════════════════════════════════════════════════
//   SETUP
// ════════════════════════════════════════════════════════════
void setup() {
  bootMillis = millis();
  Serial.begin(115200);
  Serial.println(F("\n[BOOT] Energy Meter Starting..."));

  EEPROM.begin(EEPROM_SIZE);
  ratePerUnit = loadRate();

  Wire.begin(21, 22);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("[OLED] Init failed!"));
  } else {
    display.setRotation(2);             // Kept your original rotation config
    display.setTextColor(SSD1306_WHITE); // ← FIX: required before any print
    display.setTextSize(1);

    // ── ANIMATION SEQUENCE ──
    bootAnimationGlitch();    // Phase 1: raw glitch noise (original)
    bootAnimationBooting();   // Phase 2: glitchy "BOOTING..." text
    bootAnimationWattBot();   // Phase 3: WattBot glitch-reveal + clean hold
  }

  Serial2.begin(9600, SERIAL_8N1, 16, 17);

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("Connecting WiFi..."));
  display.display();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500); Serial.print("."); attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(F("\n[WiFi] IP: ")); Serial.println(WiFi.localIP());
    display.clearDisplay(); display.setTextColor(SSD1306_WHITE); display.setCursor(0,0);
    display.println(F("WiFi Connected!"));
    display.print(F("IP: "));
    display.println(WiFi.localIP());
    display.display(); delay(2000);
  } else {
    Serial.println(F("[WiFi] FAILED — offline mode"));
    display.clearDisplay(); display.setTextColor(SSD1306_WHITE); display.setCursor(0,0);
    display.println(F("WiFi FAILED")); display.println(F("Running offline"));
    display.display(); delay(2000);
  }

  server.on("/", HTTP_GET, [](AsyncWebServerRequest* req) {
    req->send(200, "text/html", buildDashboardHTML());
  });
  server.on("/data", HTTP_GET, [](AsyncWebServerRequest* req) {
    req->send(200, "application/json", buildDataJSON());
  });
  server.on("/reset", HTTP_GET, [](AsyncWebServerRequest* req) {
    pzem.resetEnergy();
    req->send(200, "text/plain", "Energy reset");
  });
  server.on("/setrate", HTTP_GET, [](AsyncWebServerRequest* req) {
    if (req->hasParam("rate")) {
      float r = req->getParam("rate")->value().toFloat();
      if (r > 0 && r <= 50) { ratePerUnit = r; saveRate(r); }
    }
    AsyncWebServerResponse* res = req->beginResponse(302, "text/plain", "");
    res->addHeader("Location", "/"); req->send(res);
  });
  server.on("/fetchprediction", HTTP_GET, [](AsyncWebServerRequest* req) {
    fetchPrediction();
    req->send(200, "text/plain", "Prediction fetch triggered");
  });
  server.onNotFound([](AsyncWebServerRequest* req) {
    req->send(404, "text/plain", "Not found");
  });

  server.begin();
  Serial.println(F("[WEB] Server started on port 80"));
  Serial.println(F("[BOOT] Ready! Send interval: 30s"));
}

// ════════════════════════════════════════════════════════════
//   LOOP
// ════════════════════════════════════════════════════════════
void loop() {
  readSensor();
  updateOLED();
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();
    sendDataToServer();
  }
  if (millis() - lastPredictTime >= PREDICT_INTERVAL) {
    lastPredictTime = millis();
    fetchPrediction();
  }
  delay(200);
}
