/*
 * ============================================================
 * IoT Energy Meter with LSTM AI Prediction
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

const unsigned long SEND_INTERVAL    = 30000UL; // 30 seconds
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

// Predicted running times from LSTM
float predictedRunHour  = 0.0;
float predictedRunDay   = 0.0;
float predictedRunWeek  = 0.0;
float predictedRunMonth = 0.0;

// ════════════════════════════════════════════════════════════
//   APPLIANCE RUNNING TIME TRACKER
// ════════════════════════════════════════════════════════════
bool  applianceOn       = false;
unsigned long appOnStart   = 0;       // millis when appliance turned ON
unsigned long totalOnMs    = 0;       // cumulative ms ON this session

// Computed running time (seconds)
float runTimeTotalSec   = 0.0;
float runTimePerHour    = 0.0;   // fraction of last hour it was ON
float runTimePerDay     = 0.0;   // hours ON in last 24h (estimated)
float runTimePerWeek    = 0.0;   // hours ON in last 7 days (estimated)
float runTimePerMonth   = 0.0;   // hours ON in last 30 days (estimated)

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
  float onFraction = runTimeTotalSec / uptimeSec;

  // Extrapolate to periods
  runTimePerHour  = onFraction * 3600.0;          // seconds ON per hour
  runTimePerDay   = (onFraction * 86400.0) / 3600.0; // hours ON per day
  runTimePerWeek  = runTimePerDay * 7.0;
  runTimePerMonth = runTimePerDay * 30.0;
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
<title>WATTBOT</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
  @keyframes gradientShift { 0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%} }
  @keyframes float { 0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)} }
  @keyframes pulseGlow { 0%,100%{box-shadow:0 0 20px rgba(56,189,248,0.3)}50%{box-shadow:0 0 35px rgba(56,189,248,0.6)} }
  @keyframes shimmer { 0%{background-position:-200% 0}100%{background-position:200% 0} }
  @keyframes tickPulse { 0%,100%{opacity:1}50%{opacity:0.4} }

  body {
    background: linear-gradient(-45deg,#0f172a,#1e293b,#0f172a,#172554);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    color: #f1f5f9; padding: 20px; min-height: 100vh; overflow-x: hidden;
  }
  body::before, body::after {
    content:''; position:fixed; width:300px; height:300px; border-radius:50%;
    filter:blur(80px); z-index:-1; opacity:0.4; animation:float 8s ease-in-out infinite;
  }
  body::before { background:radial-gradient(circle,rgba(56,189,248,0.4),transparent 70%); top:-100px; right:-100px; }
  body::after  { background:radial-gradient(circle,rgba(16,185,129,0.3),transparent 70%); bottom:-100px; left:-100px; animation-delay:-4s;
  }

  h1 { text-align:center; font-size:2rem; margin:10px 0 25px; font-weight:800;
    background:linear-gradient(135deg,#38bdf8,#22d3ee,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:pulseGlow 3s ease-in-out infinite; letter-spacing:0.5px;
  }

  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:18px; margin-bottom:24px; }

  .card, .section, .bill-item, .ai-card, .run-item {
    background:rgba(30,41,59,0.55);
    backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,0.12);
    border-top:1px solid rgba(255,255,255,0.25);
    border-left:1px solid rgba(255,255,255,0.18);
    border-radius:20px;
    box-shadow:0 8px 32px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.1);
    transition:all 0.3s cubic-bezier(0.4,0,0.2,1); position:relative; overflow:hidden;
  }
  .card::before,.section::before,.bill-item::before,.ai-card::before,.run-item::before {
    content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);
    transition:left 0.6s ease; pointer-events:none;
  }
  .card:hover::before,.section:hover::before,.bill-item:hover::before,.ai-card:hover::before,.run-item:hover::before { left:100%; }
  .card { padding:22px 18px; text-align:center; border-bottom:3px solid rgba(56,189,248,0.5);
  }
  .card:hover { transform:translateY(-5px) scale(1.02); border-color:rgba(56,189,248,0.7); z-index:2; }
  .card .val { font-size:2rem; font-weight:800; color:#38bdf8; text-shadow:0 0 20px rgba(56,189,248,0.5);
  }
  .card .unit { font-size:0.9rem; color:#94a3b8; margin-top:4px; font-weight:600; text-transform:uppercase; letter-spacing:1px; }
  .card .lbl  { font-size:0.95rem; color:#cbd5e1; margin-top:10px;
  font-weight:600; }

  .section { padding:24px; margin-bottom:24px; border-left:4px solid rgba(56,189,248,0.6); }
  .section h2 { font-size:1.3rem; margin-bottom:20px; font-weight:700; color:#7dd3fc; display:flex;
  align-items:center; gap:8px; }
  .section h2::before { content:''; width:8px; height:8px; background:#38bdf8; border-radius:50%; box-shadow:0 0 10px rgba(56,189,248,0.8); animation:pulseGlow 2s infinite;
  }

  .bill-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }
  .bill-item,.ai-card,.run-item { padding:18px 14px; text-align:center; background:rgba(15,23,42,0.45); border:1px solid rgba(16,185,129,0.2); border-radius:16px;
  }
  .bill-item:hover,.ai-card:hover,.run-item:hover { transform:translateY(-3px); border-color:rgba(16,185,129,0.5); }
  .bill-item .bval,.ai-card .aval { font-size:1.5rem; font-weight:800; color:#10b981; text-shadow:0 0 15px rgba(16,185,129,0.4);
  }
  .bill-item .blbl,.ai-card .albl { font-size:0.8rem; color:#94a3b8; margin-top:6px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
  }
/*
 * ============================================================
 * IoT Energy Meter with LSTM AI Prediction
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

const unsigned long SEND_INTERVAL    = 30000UL; // 30 seconds
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

// Predicted running times from LSTM
float predictedRunHour  = 0.0;
float predictedRunDay   = 0.0;
float predictedRunWeek  = 0.0;
float predictedRunMonth = 0.0;

// ════════════════════════════════════════════════════════════
//   APPLIANCE RUNNING TIME TRACKER
// ════════════════════════════════════════════════════════════
bool  applianceOn       = false;
unsigned long appOnStart   = 0;       // millis when appliance turned ON
unsigned long totalOnMs    = 0;       // cumulative ms ON this session

// Computed running time (seconds)
float runTimeTotalSec   = 0.0;
float runTimePerHour    = 0.0;   // fraction of last hour it was ON
float runTimePerDay     = 0.0;   // hours ON in last 24h (estimated)
float runTimePerWeek    = 0.0;   // hours ON in last 7 days (estimated)
float runTimePerMonth   = 0.0;   // hours ON in last 30 days (estimated)

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
  float onFraction = runTimeTotalSec / uptimeSec;

  // Extrapolate to periods
  runTimePerHour  = onFraction * 3600.0;          // seconds ON per hour
  runTimePerDay   = (onFraction * 86400.0) / 3600.0; // hours ON per day
  runTimePerWeek  = runTimePerDay * 7.0;
  runTimePerMonth = runTimePerDay * 30.0;
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
<title>WATTBOT</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
  @keyframes gradientShift { 0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%} }
  @keyframes float { 0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)} }
  @keyframes pulseGlow { 0%,100%{box-shadow:0 0 20px rgba(56,189,248,0.3)}50%{box-shadow:0 0 35px rgba(56,189,248,0.6)} }
  @keyframes shimmer { 0%{background-position:-200% 0}100%{background-position:200% 0} }
  @keyframes tickPulse { 0%,100%{opacity:1}50%{opacity:0.4} }

  body {
    background: linear-gradient(-45deg,#0f172a,#1e293b,#0f172a,#172554);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    color: #f1f5f9; padding: 20px; min-height: 100vh; overflow-x: hidden;
  }
  body::before, body::after {
    content:''; position:fixed; width:300px; height:300px; border-radius:50%;
    filter:blur(80px); z-index:-1; opacity:0.4; animation:float 8s ease-in-out infinite;
  }
  body::before { background:radial-gradient(circle,rgba(56,189,248,0.4),transparent 70%); top:-100px; right:-100px; }
  body::after  { background:radial-gradient(circle,rgba(16,185,129,0.3),transparent 70%); bottom:-100px; left:-100px; animation-delay:-4s;
  }

  h1 { text-align:center; font-size:2rem; margin:10px 0 25px; font-weight:800;
    background:linear-gradient(135deg,#38bdf8,#22d3ee,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:pulseGlow 3s ease-in-out infinite; letter-spacing:0.5px;
  }

  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:18px; margin-bottom:24px; }

  .card, .section, .bill-item, .ai-card, .run-item {
    background:rgba(30,41,59,0.55);
    backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,0.12);
    border-top:1px solid rgba(255,255,255,0.25);
    border-left:1px solid rgba(255,255,255,0.18);
    border-radius:20px;
    box-shadow:0 8px 32px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.1);
    transition:all 0.3s cubic-bezier(0.4,0,0.2,1); position:relative; overflow:hidden;
  }
  .card::before,.section::before,.bill-item::before,.ai-card::before,.run-item::before {
    content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);
    transition:left 0.6s ease; pointer-events:none;
  }
  .card:hover::before,.section:hover::before,.bill-item:hover::before,.ai-card:hover::before,.run-item:hover::before { left:100%; }
  .card { padding:22px 18px; text-align:center; border-bottom:3px solid rgba(56,189,248,0.5);
  }
  .card:hover { transform:translateY(-5px) scale(1.02); border-color:rgba(56,189,248,0.7); z-index:2; }
  .card .val { font-size:2rem; font-weight:800; color:#38bdf8; text-shadow:0 0 20px rgba(56,189,248,0.5);
  }
  .card .unit { font-size:0.9rem; color:#94a3b8; margin-top:4px; font-weight:600; text-transform:uppercase; letter-spacing:1px; }
  .card .lbl  { font-size:0.95rem; color:#cbd5e1; margin-top:10px;
  font-weight:600; }

  .section { padding:24px; margin-bottom:24px; border-left:4px solid rgba(56,189,248,0.6); }
  .section h2 { font-size:1.3rem; margin-bottom:20px; font-weight:700; color:#7dd3fc; display:flex;
  align-items:center; gap:8px; }
  .section h2::before { content:''; width:8px; height:8px; background:#38bdf8; border-radius:50%; box-shadow:0 0 10px rgba(56,189,248,0.8); animation:pulseGlow 2s infinite;
  }

  .bill-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }
  .bill-item,.ai-card,.run-item { padding:18px 14px; text-align:center; background:rgba(15,23,42,0.45); border:1px solid rgba(16,185,129,0.2); border-radius:16px;
  }
  .bill-item:hover,.ai-card:hover,.run-item:hover { transform:translateY(-3px); border-color:rgba(16,185,129,0.5); }
  .bill-item .bval,.ai-card .aval { font-size:1.5rem; font-weight:800; color:#10b981; text-shadow:0 0 15px rgba(16,185,129,0.4);
  }
  .bill-item .blbl,.ai-card .albl { font-size:0.8rem; color:#94a3b8; margin-top:6px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
  }

  /* Runtime card specific */
  .run-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-bottom:16px;
  }
  .run-item { border:1px solid rgba(251,146,60,0.25); }
  .run-item:hover { border-color:rgba(251,146,60,0.5); }
  .run-item .rval { font-size:1.4rem; font-weight:800; color:#fb923c;
  text-shadow:0 0 12px rgba(251,146,60,0.4); }
  .run-item .rlbl { font-size:0.78rem; color:#94a3b8; margin-top:6px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
  .appliance-status { display:flex;
  align-items:center; gap:10px; padding:14px 18px; border-radius:14px; margin-bottom:16px; font-weight:700; font-size:1.05rem; }
  .appliance-on  { background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#10b981;
  }
  .appliance-off { background:rgba(148,163,184,0.1); border:1px solid rgba(148,163,184,0.25); color:#94a3b8; }
  .status-dot { width:10px; height:10px; border-radius:50%;
  }
  .dot-on  { background:#10b981; box-shadow:0 0 10px #10b981; animation:tickPulse 1s infinite; }
  .dot-off { background:#64748b;
  }

  /* AI run time items */
  .ai-run-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:14px; }
  .ai-run-item { padding:14px;
  text-align:center; background:rgba(15,23,42,0.45); border:1px solid rgba(167,139,250,0.2); border-radius:14px; }
  .ai-run-item:hover { border-color:rgba(167,139,250,0.5); transform:translateY(-2px); }
  .ai-run-item .arval { font-size:1.3rem; font-weight:800; color:#a78bfa;
  }
  .ai-run-item .arlbl { font-size:0.75rem; color:#94a3b8; margin-top:4px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }

  .rate-form { display:flex; gap:12px; align-items:center; margin-top:20px; flex-wrap:wrap;
  }
  .rate-form input { flex:1; min-width:120px; padding:14px 18px; border-radius:14px; border:1px solid rgba(56,189,248,0.35); background:rgba(15,23,42,0.65); color:#38bdf8; font-size:1rem; font-weight:600; outline:none;
  transition:all 0.25s ease; }
  .rate-form input::placeholder { color:#475569; }
  .rate-form input:focus { border-color:#38bdf8; box-shadow:0 0 0 3px rgba(56,189,248,0.2);
  }
  .rate-form button { padding:14px 24px; background:linear-gradient(135deg,rgba(14,165,233,0.2),rgba(56,189,248,0.15)); backdrop-filter:blur(10px); color:#f0f9ff; border:1px solid rgba(56,189,248,0.65); border-radius:14px; font-size:1rem; font-weight:700; cursor:pointer; transition:all 0.2s ease;
  }
  .rate-form button:hover { background:linear-gradient(135deg,rgba(14,165,233,0.35),rgba(56,189,248,0.25)); transform:translateY(-2px); }

  .status { font-size:0.9rem; color:#64748b; text-align:center; margin-top:24px; font-weight:600; padding:12px; background:rgba(30,41,59,0.3); border-radius:12px;
  border:1px solid rgba(255,255,255,0.08); }
  .badge-ok  { color:#10b981; font-weight:700; }
  .badge-err { color:#ef4444; font-weight:700;
  }

  @media (max-width:480px) {
    .grid { grid-template-columns:repeat(2,1fr); }
    .bill-grid,.run-grid,.ai-run-grid { grid-template-columns:1fr;
    }
    h1 { font-size:1.6rem; }
    .card .val { font-size:1.6rem;
    }
  }
</style>
</head>
<body>
<h1>WATTBOT</h1>
)rawliteral";

  // Live readings
  html += "<div class='grid'>";
  html += "<div class='card'><div class='val'>" + String(voltage,1)     + "</div><div class='unit'>V</div><div class='lbl'>Voltage</div></div>";
  html += "<div class='card'><div class='val'>" + String(current,2)     + "</div><div class='unit'>A</div><div class='lbl'>Current</div></div>";
  html += "<div class='card'><div class='val'>" + String(power,1)       + "</div><div class='unit'>W</div><div class='lbl'>Power</div></div>";
  html += "<div class='card'><div class='val'>" + String(energy,3)      + "</div><div class='unit'>kWh</div><div class='lbl'>Energy</div></div>";
  html += "<div class='card'><div class='val'>" + String(frequency,1)   + "</div><div class='unit'>Hz</div><div class='lbl'>Frequency</div></div>";
  html += "<div class='card'><div class='val'>" + String(powerFactor,2) + "</div><div class='unit'>PF</div><div class='lbl'>Power Factor</div></div>";
  html += "</div>";
  // Bill estimates
  html += "<div class='section'><h2>Bill Estimate (Rs." + String(ratePerUnit,1) + "/kWh)</h2>";
  html += "<div class='bill-grid'>";
  html += "<div class='bill-item'><div class='bval'>Rs." + String(calcBill(power,1),2)   + "</div><div class='blbl'>1 Hour</div></div>";
  html += "<div class='bill-item'><div class='bval'>Rs." + String(calcBill(power,24),2)  + "</div><div class='blbl'>1 Day</div></div>";
  html += "<div class='bill-item'><div class='bval'>Rs." + String(calcBill(power,168),2) + "</div><div class='blbl'>7 Days</div></div>";
  html += "<div class='bill-item'><div class='bval'>Rs." + String(calcBill(power,720),2) + "</div><div class='blbl'>30 Days</div></div>";
  html += "</div>";
  html += R"rawliteral(
  <div class="rate-form">
    <input type="number" id="rI" placeholder="New rate (Rs/kWh)" step="0.1" min="0.1" max="50" inputmode="decimal">
    <button onclick="var r=document.getElementById('rI').value;if(r>0&&r<=50)window.location.href='/setrate?rate='+r;else alert('Enter 0.1-50');">Set Rate</button>
  </div>
  </div>)rawliteral";
  // ── APPLIANCE RUNNING TIME ──────────────────────────────────
  html += "<div class='section'><h2>Appliance Running Time</h2>";
  if (applianceOn) {
    html += "<div class='appliance-status appliance-on'><div class='status-dot dot-on'></div>Appliance is ON — Currently Running</div>";
  } else {
    html += "<div class='appliance-status appliance-off'><div class='status-dot dot-off'></div>Appliance is OFF</div>";
  }
  html += "<div class='run-grid'>";
  html += "<div class='run-item'><div class='rval'>" + formatRunTime(runTimePerHour)         + "</div><div class='rlbl'>Per Hour</div></div>";
  html += "<div class='run-item'><div class='rval'>" + String(runTimePerDay, 1) + " hrs"     + "</div><div class='rlbl'>Per Day</div></div>";
  html += "<div class='run-item'><div class='rval'>" + String(runTimePerWeek, 1) + " hrs"    + "</div><div class='rlbl'>Per Week</div></div>";
  html += "<div class='run-item'><div class='rval'>" + String(runTimePerMonth, 1) + " hrs"   + "</div><div class='rlbl'>Per Month</div></div>";
  html += "</div>";
  html += "<div style='color:#64748b;font-size:0.82rem;margin-top:8px;'>Total session ON time: <b style='color:#fb923c'>" + formatRunTime(runTimeTotalSec) + "</b> &nbsp;|&nbsp; Threshold: >" + String(APPLIANCE_ON_THRESHOLD,0) + "W</div>";
  html += "</div>";

  // AI Prediction
  html += "<div class='section'><h2>AI Prediction (LSTM)</h2>";
  if (predictionReady) {
    html += "<div class='bill-grid'>";
    html += "<div class='ai-card'><div class='aval'>" + String(predictedPower,1)  + " W</div><div class='albl'>Predicted Power</div></div>";
    html += "<div class='ai-card'><div class='aval'>Rs." + String(predictedBill1h,2) + "</div><div class='albl'>Predicted/Hour</div></div>";
    html += "<div class='ai-card'><div class='aval'>Rs." + String(predictedBill1d,2) + "</div><div class='albl'>Predicted/Day</div></div>";
    html += "<div class='ai-card'><div class='aval'>" + String(predictedKwh1d,3)  + " kWh</div><div class='albl'>Predicted kWh/Day</div></div>";
    html += "</div>";
    html += "<div style='color:#7dd3fc;font-size:1rem;font-weight:700;margin:16px 0 8px;'>AI Predicted Running Time</div>";
    html += "<div class='ai-run-grid'>";
    html += "<div class='ai-run-item'><div class='arval'>" + String(predictedRunHour*60,0)  + " min</div><div class='arlbl'>Per Hour</div></div>";
    html += "<div class='ai-run-item'><div class='arval'>" + String(predictedRunDay,1)      + " hrs</div><div class='arlbl'>Per Day</div></div>";
    html += "<div class='ai-run-item'><div class='arval'>" + String(predictedRunWeek,1)     + " hrs</div><div class='arlbl'>Per Week</div></div>";
    html += "<div class='ai-run-item'><div class='arval'>" + String(predictedRunMonth,1)    + " hrs</div><div class='arlbl'>Per Month</div></div>";
    html += "</div>";
    html += "<div class='status' style='color:#94a3b8;margin-top:12px;'>Last updated: " + predictionTime + "</div>";
  } else {
    html += "<div style='color:#94a3b8;text-align:center;padding:16px;'>Waiting for predictions...<br><small style='color:#64748b;margin-top:8px;display:block;'>Run predict_server.py on your PC</small></div>";
  }
  html += "</div>";

  html += "<div class='status'>Sensor: <span class='" + String(sensorOk ? "badge-ok" : "badge-err") + "'>" + String(sensorOk ? "OK" : "ERROR") + "</span>";
  html += " &nbsp;|&nbsp; WiFi: <span class='badge-ok'>" + WiFi.localIP().toString() + "</span>";
  html += " &nbsp;|&nbsp; Auto-refresh: 5s</div>";
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
    display.setRotation(2);
    display.setTextColor(SSD1306_WHITE);

    // ── PHASE 1: Random noise / corrupted pixels (600ms) ──
    unsigned long noiseEnd = millis() + 600;
    while (millis() < noiseEnd) {
      display.clearDisplay();
      for (int i = 0; i < 200; i++) {
        int rx = random(0, OLED_WIDTH);
        int ry = random(0, OLED_HEIGHT);
        int rw = random(1, 6);
        int rh = random(1, 4);
        display.fillRect(rx, ry, rw, rh, SSD1306_WHITE);
      }
      display.display();
      delay(60);
    }

    // ── PHASE 2: Rapid glitch flicker (8 flashes) ──
    for (int g = 0; g < 8; g++) {
      display.clearDisplay();
      // horizontal glitch bars
      for (int i = 0; i < 4; i++) {
        int gy = random(0, OLED_HEIGHT - 6);
        int gw = random(20, OLED_WIDTH);
        int gx = random(0, OLED_WIDTH - gw);
        display.fillRect(gx, gy, gw, random(2, 6), SSD1306_WHITE);
      }
      // scattered pixels
      for (int i = 0; i < 40; i++) {
        display.drawPixel(random(0, OLED_WIDTH), random(0, OLED_HEIGHT), SSD1306_WHITE);
      }
      // ghost text — offset and corrupted
      display.setTextSize(2);
      display.setCursor(random(-6, 6), random(-4, 4) + 20);
      display.print(F("ENERGY"));
      display.display();
      delay(50);

      // blank flash between glitches
      display.clearDisplay();
      display.display();
      delay(30);
    }

    // ── PHASE 3: Snap to clean title screen ──
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(43, 0);
    display.println(F("WATTBOT"));
    display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
    display.drawLine(0, 11, 127, 11, SSD1306_WHITE);
    display.setTextSize(2);
    display.setCursor(18, 22);
    display.println(F("BOOTING"));
    display.setTextSize(1);
    display.setCursor(30, 50);
    display.println(F("Initializing..."));
    display.display();
    delay(1500);
  }

  Serial2.begin(9600, SERIAL_8N1, 16, 17);

  display.clearDisplay();
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
    display.clearDisplay(); display.setCursor(0,0);
    display.println(F("WiFi Connected!"));
    display.print(F("IP: "));
    display.println(WiFi.localIP());
    display.display(); delay(2000);
  } else {
    Serial.println(F("[WiFi] FAILED — offline mode"));
    display.clearDisplay(); display.setCursor(0,0);
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
  /* Runtime card specific */
  .run-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-bottom:16px;
  }
  .run-item { border:1px solid rgba(251,146,60,0.25); }
  .run-item:hover { border-color:rgba(251,146,60,0.5); }
  .run-item .rval { font-size:1.4rem; font-weight:800; color:#fb923c;
  text-shadow:0 0 12px rgba(251,146,60,0.4); }
  .run-item .rlbl { font-size:0.78rem; color:#94a3b8; margin-top:6px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
  .appliance-status { display:flex;
  align-items:center; gap:10px; padding:14px 18px; border-radius:14px; margin-bottom:16px; font-weight:700; font-size:1.05rem; }
  .appliance-on  { background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#10b981;
  }
  .appliance-off { background:rgba(148,163,184,0.1); border:1px solid rgba(148,163,184,0.25); color:#94a3b8; }
  .status-dot { width:10px; height:10px; border-radius:50%;
  }
  .dot-on  { background:#10b981; box-shadow:0 0 10px #10b981; animation:tickPulse 1s infinite; }
  .dot-off { background:#64748b;
  }

  /* AI run time items */
  .ai-run-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:14px; }
  .ai-run-item { padding:14px;
  text-align:center; background:rgba(15,23,42,0.45); border:1px solid rgba(167,139,250,0.2); border-radius:14px; }
  .ai-run-item:hover { border-color:rgba(167,139,250,0.5); transform:translateY(-2px); }
  .ai-run-item .arval { font-size:1.3rem; font-weight:800; color:#a78bfa;
  }
  .ai-run-item .arlbl { font-size:0.75rem; color:#94a3b8; margin-top:4px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }

  .rate-form { display:flex; gap:12px; align-items:center; margin-top:20px; flex-wrap:wrap;
  }
  .rate-form input { flex:1; min-width:120px; padding:14px 18px; border-radius:14px; border:1px solid rgba(56,189,248,0.35); background:rgba(15,23,42,0.65); color:#38bdf8; font-size:1rem; font-weight:600; outline:none;
  transition:all 0.25s ease; }
  .rate-form input::placeholder { color:#475569; }
  .rate-form input:focus { border-color:#38bdf8; box-shadow:0 0 0 3px rgba(56,189,248,0.2);
  }
  .rate-form button { padding:14px 24px; background:linear-gradient(135deg,rgba(14,165,233,0.2),rgba(56,189,248,0.15)); backdrop-filter:blur(10px); color:#f0f9ff; border:1px solid rgba(56,189,248,0.65); border-radius:14px; font-size:1rem; font-weight:700; cursor:pointer; transition:all 0.2s ease;
  }
  .rate-form button:hover { background:linear-gradient(135deg,rgba(14,165,233,0.35),rgba(56,189,248,0.25)); transform:translateY(-2px); }

  .status { font-size:0.9rem; color:#64748b; text-align:center; margin-top:24px; font-weight:600; padding:12px; background:rgba(30,41,59,0.3); border-radius:12px;
  border:1px solid rgba(255,255,255,0.08); }
  .badge-ok  { color:#10b981; font-weight:700; }
  .badge-err { color:#ef4444; font-weight:700;
  }

  @media (max-width:480px) {
    .grid { grid-template-columns:repeat(2,1fr); }
    .bill-grid,.run-grid,.ai-run-grid { grid-template-columns:1fr;
    }
    h1 { font-size:1.6rem; }
    .card .val { font-size:1.6rem;
    }
  }
</style>
</head>
<body>
<h1>WATTBOT</h1>
)rawliteral";

  // Live readings
  html += "<div class='grid'>";
  html += "<div class='card'><div class='val'>" + String(voltage,1)     + "</div><div class='unit'>V</div><div class='lbl'>Voltage</div></div>";
  html += "<div class='card'><div class='val'>" + String(current,2)     + "</div><div class='unit'>A</div><div class='lbl'>Current</div></div>";
  html += "<div class='card'><div class='val'>" + String(power,1)       + "</div><div class='unit'>W</div><div class='lbl'>Power</div></div>";
  html += "<div class='card'><div class='val'>" + String(energy,3)      + "</div><div class='unit'>kWh</div><div class='lbl'>Energy</div></div>";
  html += "<div class='card'><div class='val'>" + String(frequency,1)   + "</div><div class='unit'>Hz</div><div class='lbl'>Frequency</div></div>";
  html += "<div class='card'><div class='val'>" + String(powerFactor,2) + "</div><div class='unit'>PF</div><div class='lbl'>Power Factor</div></div>";
  html += "</div>";
  // Bill estimates
  html += "<div class='section'><h2>Bill Estimate (Rs." + String(ratePerUnit,1) + "/kWh)</h2>";
  html += "<div class='bill-grid'>";
  html += "<div class='bill-item'><div class='bval'>Rs." + String(calcBill(power,1),2)   + "</div><div class='blbl'>1 Hour</div></div>";
  html += "<div class='bill-item'><div class='bval'>Rs." + String(calcBill(power,24),2)  + "</div><div class='blbl'>1 Day</div></div>";
  html += "<div class='bill-item'><div class='bval'>Rs." + String(calcBill(power,168),2) + "</div><div class='blbl'>7 Days</div></div>";
  html += "<div class='bill-item'><div class='bval'>Rs." + String(calcBill(power,720),2) + "</div><div class='blbl'>30 Days</div></div>";
  html += "</div>";
  html += R"rawliteral(
  <div class="rate-form">
    <input type="number" id="rI" placeholder="New rate (Rs/kWh)" step="0.1" min="0.1" max="50" inputmode="decimal">
    <button onclick="var r=document.getElementById('rI').value;if(r>0&&r<=50)window.location.href='/setrate?rate='+r;else alert('Enter 0.1-50');">Set Rate</button>
  </div>
  </div>)rawliteral";
  // ── APPLIANCE RUNNING TIME ──────────────────────────────────
  html += "<div class='section'><h2>Appliance Running Time</h2>";
  if (applianceOn) {
    html += "<div class='appliance-status appliance-on'><div class='status-dot dot-on'></div>Appliance is ON — Currently Running</div>";
  } else {
    html += "<div class='appliance-status appliance-off'><div class='status-dot dot-off'></div>Appliance is OFF</div>";
  }
  html += "<div class='run-grid'>";
  html += "<div class='run-item'><div class='rval'>" + formatRunTime(runTimePerHour)         + "</div><div class='rlbl'>Per Hour</div></div>";
  html += "<div class='run-item'><div class='rval'>" + String(runTimePerDay, 1) + " hrs"     + "</div><div class='rlbl'>Per Day</div></div>";
  html += "<div class='run-item'><div class='rval'>" + String(runTimePerWeek, 1) + " hrs"    + "</div><div class='rlbl'>Per Week</div></div>";
  html += "<div class='run-item'><div class='rval'>" + String(runTimePerMonth, 1) + " hrs"   + "</div><div class='rlbl'>Per Month</div></div>";
  html += "</div>";
  html += "<div style='color:#64748b;font-size:0.82rem;margin-top:8px;'>Total session ON time: <b style='color:#fb923c'>" + formatRunTime(runTimeTotalSec) + "</b> &nbsp;|&nbsp; Threshold: >" + String(APPLIANCE_ON_THRESHOLD,0) + "W</div>";
  html += "</div>";

  // AI Prediction
  html += "<div class='section'><h2>AI Prediction (LSTM)</h2>";
  if (predictionReady) {
    html += "<div class='bill-grid'>";
    html += "<div class='ai-card'><div class='aval'>" + String(predictedPower,1)  + " W</div><div class='albl'>Predicted Power</div></div>";
    html += "<div class='ai-card'><div class='aval'>Rs." + String(predictedBill1h,2) + "</div><div class='albl'>Predicted/Hour</div></div>";
    html += "<div class='ai-card'><div class='aval'>Rs." + String(predictedBill1d,2) + "</div><div class='albl'>Predicted/Day</div></div>";
    html += "<div class='ai-card'><div class='aval'>" + String(predictedKwh1d,3)  + " kWh</div><div class='albl'>Predicted kWh/Day</div></div>";
    html += "</div>";
    html += "<div style='color:#7dd3fc;font-size:1rem;font-weight:700;margin:16px 0 8px;'>AI Predicted Running Time</div>";
    html += "<div class='ai-run-grid'>";
    html += "<div class='ai-run-item'><div class='arval'>" + String(predictedRunHour*60,0)  + " min</div><div class='arlbl'>Per Hour</div></div>";
    html += "<div class='ai-run-item'><div class='arval'>" + String(predictedRunDay,1)      + " hrs</div><div class='arlbl'>Per Day</div></div>";
    html += "<div class='ai-run-item'><div class='arval'>" + String(predictedRunWeek,1)     + " hrs</div><div class='arlbl'>Per Week</div></div>";
    html += "<div class='ai-run-item'><div class='arval'>" + String(predictedRunMonth,1)    + " hrs</div><div class='arlbl'>Per Month</div></div>";
    html += "</div>";
    html += "<div class='status' style='color:#94a3b8;margin-top:12px;'>Last updated: " + predictionTime + "</div>";
  } else {
    html += "<div style='color:#94a3b8;text-align:center;padding:16px;'>Waiting for predictions...<br><small style='color:#64748b;margin-top:8px;display:block;'>Run predict_server.py on your PC</small></div>";
  }
  html += "</div>";

  html += "<div class='status'>Sensor: <span class='" + String(sensorOk ? "badge-ok" : "badge-err") + "'>" + String(sensorOk ? "OK" : "ERROR") + "</span>";
  html += " &nbsp;|&nbsp; WiFi: <span class='badge-ok'>" + WiFi.localIP().toString() + "</span>";
  html += " &nbsp;|&nbsp; Auto-refresh: 5s</div>";
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
    display.setRotation(2);
    display.setTextColor(SSD1306_WHITE);

    // ── PHASE 1: Random noise / corrupted pixels (600ms) ──
    unsigned long noiseEnd = millis() + 600;
    while (millis() < noiseEnd) {
      display.clearDisplay();
      for (int i = 0; i < 200; i++) {
        int rx = random(0, OLED_WIDTH);
        int ry = random(0, OLED_HEIGHT);
        int rw = random(1, 6);
        int rh = random(1, 4);
        display.fillRect(rx, ry, rw, rh, SSD1306_WHITE);
      }
      display.display();
      delay(60);
    }

    // ── PHASE 2: Rapid glitch flicker (8 flashes) ──
    for (int g = 0; g < 8; g++) {
      display.clearDisplay();
      // horizontal glitch bars
      for (int i = 0; i < 4; i++) {
        int gy = random(0, OLED_HEIGHT - 6);
        int gw = random(20, OLED_WIDTH);
        int gx = random(0, OLED_WIDTH - gw);
        display.fillRect(gx, gy, gw, random(2, 6), SSD1306_WHITE);
      }
      // scattered pixels
      for (int i = 0; i < 40; i++) {
        display.drawPixel(random(0, OLED_WIDTH), random(0, OLED_HEIGHT), SSD1306_WHITE);
      }
      // ghost text — offset and corrupted
      display.setTextSize(2);
      display.setCursor(random(-6, 6), random(-4, 4) + 20);
      display.print(F("ENERGY"));
      display.display();
      delay(50);

      // blank flash between glitches
      display.clearDisplay();
      display.display();
      delay(30);
    }

    // ── PHASE 3: Snap to clean title screen ──
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(43, 0);
    display.println(F("WATTBOT"));
    display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
    display.drawLine(0, 11, 127, 11, SSD1306_WHITE);
    display.setTextSize(2);
    display.setCursor(18, 22);
    display.println(F("BOOTING"));
    display.setTextSize(1);
    display.setCursor(30, 50);
    display.println(F("Initializing..."));
    display.display();
    delay(1500);
  }

  Serial2.begin(9600, SERIAL_8N1, 16, 17);

  display.clearDisplay();
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
    display.clearDisplay(); display.setCursor(0,0);
    display.println(F("WiFi Connected!"));
    display.print(F("IP: "));
    display.println(WiFi.localIP());
    display.display(); delay(2000);
  } else {
    Serial.println(F("[WiFi] FAILED — offline mode"));
    display.clearDisplay(); display.setCursor(0,0);
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