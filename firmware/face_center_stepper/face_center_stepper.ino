/*
 * ESP8266 + ULN2003 stepper subscriber for the face-centering app.
 * Draft wiring is for a common 28BYJ-48 geared stepper and ULN2003 board.
 * Install the ESP8266 board package, PubSubClient, ArduinoJson, and AccelStepper.
 *
 * Connect ULN2003 IN1..IN4 to four ESP8266 GPIOs. Do not wire the stepper
 * coils to ESP8266 pins. Set the Wi-Fi/broker values and calibrate steps/rev.
 */
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <AccelStepper.h>
#include <math.h>
#include "secrets.h"

const char *WIFI_SSID = "<YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "<YOUR_WIFI_PASSWORD>";
const char *MQTT_HOST = "YOUR_PC_IP";  // Computer running Mosquitto
const uint16_t MQTT_PORT = 1883;
const char *COMMAND_TOPIC = "face-centering/motor/command";
const char *STATUS_TOPIC = "face-centering/motor/status";
const char *OFFLINE_STATUS = "{\"status\":\"offline\"}";

// ESP8266 board-label wiring reported for the ULN2003 inputs.
const uint8_t IN1_PIN = 5;   // D1 / GPIO5
const uint8_t IN2_PIN = 4;   // D2 / GPIO4
const uint8_t IN3_PIN = 14;  // D5 / GPIO14
const uint8_t IN4_PIN = 12;  // D6 / GPIO12

// Calibrate for the actual motor/gearbox. 4096 is a common half-step estimate
// for a 28BYJ-48, but variants differ and should be measured on the hardware.
const float STEPS_PER_REVOLUTION = 4096.0f;
const float CENTER_ANGLE = 90.0f;
const float MIN_ANGLE = 20.0f;
const float MAX_ANGLE = 160.0f;
const float MAX_SPEED = 500.0f;       // half-steps per second
const float ACCELERATION = 250.0f;    // half-steps per second squared
const int MOTOR_DIRECTION = 1;        // set -1 if positive angle turns the wrong way
const unsigned long WIFI_RETRY_MS = 10000;
const unsigned long MQTT_RETRY_MS = 5000;
const uint16_t MQTT_SOCKET_TIMEOUT_SECONDS = 1;

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
// AccelStepper's HALF4WIRE sequence interleaves the ULN2003 input order.
AccelStepper motor(AccelStepper::HALF4WIRE, IN1_PIN, IN3_PIN, IN2_PIN, IN4_PIN);
long targetSteps = 0;
bool wasMoving = false;
bool wifiAttempted = false;
unsigned long lastWifiAttempt = 0;
unsigned long lastMqttAttempt = 0;

float currentAngle() {
  return CENTER_ANGLE + (float)motor.currentPosition() * 360.0f /
         (STEPS_PER_REVOLUTION * MOTOR_DIRECTION);
}

void publishStatus(const char *status) {
  StaticJsonDocument<192> message;
  message["status"] = status;
  message["angle"] = currentAngle();
  message["position_steps"] = motor.currentPosition();
  message["target_steps"] = targetSteps;
  char payload[192];
  const size_t length = serializeJson(message, payload, sizeof(payload));
  // Retained status lets the camera app see the latest ESP8266 state even if
  // it connects after the device. The MQTT last-will replaces it on disconnect.
  mqtt.publish(STATUS_TOPIC, reinterpret_cast<const uint8_t *>(payload), length, true);
}

void onMessage(char *topic, byte *payload, unsigned int length) {
  if (String(topic) != COMMAND_TOPIC) return;

  StaticJsonDocument<256> message;
  if (deserializeJson(message, payload, length)) {
    publishStatus("invalid_json");
    return;
  }
  if (!message["angle"].is<float>() && !message["angle"].is<int>()) {
    publishStatus("missing_angle");
    return;
  }

  const float requestedAngle = message["angle"].as<float>();
  const float boundedAngle = constrain(requestedAngle, MIN_ANGLE, MAX_ANGLE);
  targetSteps = lroundf((boundedAngle - CENTER_ANGLE) * STEPS_PER_REVOLUTION / 360.0f)
                * MOTOR_DIRECTION;
  motor.enableOutputs();
  motor.moveTo(targetSteps);
  wasMoving = motor.distanceToGo() != 0;
  publishStatus(wasMoving ? "moving" : "at_target");
}

void maintainWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  const unsigned long now = millis();
  if (!wifiAttempted || now - lastWifiAttempt >= WIFI_RETRY_MS) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    lastWifiAttempt = now;
    wifiAttempted = true;
  }
}

void maintainMqtt() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;
  const unsigned long now = millis();
  if (now - lastMqttAttempt < MQTT_RETRY_MS) return;
  lastMqttAttempt = now;

  String clientId = "face-centering-esp8266-" + String(ESP.getChipId(), HEX);
  if (mqtt.connect(clientId.c_str(), STATUS_TOPIC, 0, true, OFFLINE_STATUS)) {
    mqtt.subscribe(COMMAND_TOPIC);
    publishStatus("ready");
  }
}

void setup() {
  motor.setMaxSpeed(MAX_SPEED);
  motor.setAcceleration(ACCELERATION);
  // Open-loop position has no home sensor. Manually put the motor at the
  // calibrated center before powering/resetting the ESP8266.
  motor.setCurrentPosition(0);
  motor.disableOutputs();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  wifiAttempted = true;
  lastWifiAttempt = millis();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
  mqtt.setBufferSize(512);
  // PubSubClient connects synchronously. Keep failed reconnects short so the
  // stepper loop is not starved while the broker is unavailable.
  mqtt.setSocketTimeout(MQTT_SOCKET_TIMEOUT_SECONDS);
}

void loop() {
  maintainWifi();
  if (mqtt.connected()) mqtt.loop();
  maintainMqtt();

  if (motor.distanceToGo() != 0) {
    motor.run();
    wasMoving = true;
  } else if (wasMoving) {
    wasMoving = false;
    motor.disableOutputs();
    publishStatus("at_target");
  }
  yield();
}
