# Face centering add-on

This copy keeps the original camera and identity-lock workflow and adds an
MQTT-controlled motor output. The live app sends motor angles only for the
recognized identity after **Lock** is active. It uses the locked face box's
horizontal center relative to the camera image center. A small deadband avoids jitter;
the output is smoothed and limited to the configured motor range. Unlocking
requests a return to the calibrated center angle. When a locked face is lost,
the motor holds its last commanded position. If MQTT is disconnected when
unlocking, the center command stays queued and is sent after reconnection.

The existing external camera remains the tracking input and stays fixed. The
tachometer stepper is an added output; it does not replace or pan the camera.
The motor turns its pointer to represent the locked face's horizontal offset: a
centered face box maps to the calibrated center angle, and faces to the left or
right map to lower or higher angles.

## Hardware and angle direction

The current firmware draft targets a **28BYJ-48 stepper with a ULN2003
driver**. It needs four ESP8266 output pins connected to ULN2003 `IN1` through
`IN4`. The connections are `RX/GPIO3 -> IN1`, `D1/GPIO5 -> IN2`, and,
interpreting the noted “6” as the board label D6, `D6/GPIO12 -> IN3`. On a
NodeMCU-style board, D6 means GPIO12; it does **not** mean raw GPIO6. Avoid raw
GPIO6–GPIO11 because those pins connect to flash on common ESP8266 modules
([ESP8266 Arduino pin reference](https://arduino-esp8266.readthedocs.io/en/latest/reference.html),
[NodeMCU pin map](https://arduino-esp8266.readthedocs.io/en/2.4.0/boards.html#nodemcu-0-9)). The draft uses
`D5/GPIO14 -> IN4` as its fourth control signal. A ULN2003 needs four control
signals; RX and D1 alone are not enough. The motor's coils connect to the ULN2003
board, not to ESP8266 GPIO pins. Use an appropriate 5V motor supply and
connect its ground to ESP8266 ground. Confirm actual pin labels and driver
wiring before powering the motor.

| ESP8266 / supply | ULN2003 | Status |
|---|---|---|
| RX / GPIO3 | IN1 | Confirmed |
| D1 / GPIO5 | IN2 | Confirmed |
| D6 / GPIO12 | IN3 | Interpreted from the supplied wiring note |
| D5 / GPIO14 | IN4 | Draft; confirm |
| GND | `-` | Common ground |
| 5V motor supply positive | `+` | Use the motor board's rated supply |

Because GPIO3/RX is one of the motor signals, disconnect ULN2003 `IN1` from RX
while flashing firmware over the ESP8266 serial adapter, then reconnect it
after the upload completes.

The stepper is open-loop: this firmware has no home switch or angle sensor.
Before each ESP8266 power-up/reset, manually place the shaft at the calibrated
center angle. The default half-step scale (`4096` steps/revolution) is a common
28BYJ-48 estimate and must be calibrated for the actual motor. Set
`STEPS_PER_REVOLUTION`, `MIN_ANGLE`, `MAX_ANGLE`, and `CENTER_ANGLE` in the
firmware to match the app's `--motor-min-angle`, `--motor-max-angle`, and
`--motor-center-angle`. If the motor turns in the wrong direction, change
`MOTOR_DIRECTION` in the firmware or `--motor-direction` in the app.

The firmware's `MQTT_HOST` must be the computer's LAN IP address, not
`127.0.0.1`; both devices need to reach the same broker on the network.

## Mosquitto and Python app

Install Mosquitto on the computer that runs the camera app and ESP8266 broker
connection. The plain `mosquitto -v` command starts local-only mode, which
rejects the ESP8266's network connection. Stop that broker with Ctrl+C, then
start the project's listener config. From the workspace root on this Mac:

```bash
cd "/Users/divin-d1/learning/Embedded Systems/Term 1/term 1/FaceLocking"
mosquitto -c mosquitto.conf.example -v
```

The anonymous listener is suitable only for a trusted development LAN. Get the
Mac's Wi-Fi address with `ipconfig getifaddr en0`; use that address for
`--mqtt-host` below and for `MQTT_HOST` in the ESP8266 firmware. If Mosquitto
cannot bind port 1883, stop any other Mosquitto instance first. If the ESP8266
still cannot connect, allow incoming connections for Mosquitto in the macOS
firewall. Install the project's Python dependencies and launch:

```bash
python -m pip install -r requirements.txt
CAMERA_ID=1  # replace with the external camera's existing index or device path
python -m src.camera --camera "$CAMERA_ID"  # confirm this opens the external camera
python -m src.app --camera "$CAMERA_ID" --mqtt-host 192.168.1.10
```

Replace `192.168.1.10` with the broker computer's LAN address. The default
broker is `127.0.0.1` for a broker on the same computer. The app shows MQTT
connection and ESP8266 status, and accepts `--mqtt-port`, `--mqtt-topic`,
`--mqtt-status-topic`, and angle tuning options:

```bash
python -m src.app --camera "$CAMERA_ID" --mqtt-host 192.168.1.10 \
  --motor-center-angle 90 --motor-min-angle 20 --motor-max-angle 160 \
  --motor-direction 1 --center-deadband 0.04
```

Select and lock the recognized face. Commands are JSON on
`face-centering/motor/command`, for example:

```json
{"angle": 110.0, "track_id": 2, "name": "Ada", "error_ratio": 0.25}
```

The ESP8266 publishes retained readiness, movement, and input status on
`face-centering/motor/status`. Its MQTT last-will marks the status `offline` if
the device disconnects unexpectedly, and a late-starting camera app receives
the latest retained state after subscribing.

## Check the MQTT angle mapping

Before connecting the motor, subscribe to the broker from another terminal:

```bash
mosquitto_sub -h 192.168.1.10 -t 'face-centering/motor/#' -v
```

Start the camera app, lock a recognized face, then move that face left, to the
center, and right in the image. The command payload's `angle` should move below
the center angle, return near it, and move above it, respectively. If the
direction is reversed, set `--motor-direction -1`. This checks the camera to
MQTT mapping before the stepper is connected.

## ESP8266 firmware

Open `firmware/face_center_stepper/face_center_stepper.ino` in Arduino IDE.
Install the ESP8266 board support package plus **PubSubClient**, **ArduinoJson**,
and **AccelStepper**. Copy `secrets.h.example` to `secrets.h` and put the Wi-Fi
and broker values in `secrets.h`; that local file is ignored by Git. Set the
motor-pin and steps-per-revolution values, manually center the shaft, then flash
the board. Connect four GPIOs to the ULN2003 `IN1` through `IN4` inputs.
