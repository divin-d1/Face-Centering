"""MQTT output and face-offset to motor-angle mapping for face centering."""
from __future__ import annotations

import json
import threading
import time
from typing import Optional


class MQTTMotorClient:
    """Publish absolute motor angles without blocking the camera/UI loop.

    The matching ESP8266 subscriber expects JSON containing an ``angle`` field
    in degrees. Paho's background network loop handles reconnects and delivery.
    """

    def __init__(self, host: str, port: int, topic: str, status_topic: str, client_id: str,
                 keepalive: int = 30):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("MQTT support needs paho-mqtt; install it from requirements.txt") from exc

        self.topic = topic
        self.status_topic = status_topic
        self.connected = threading.Event()
        self.connection_count = 0
        self.last_status: Optional[str] = None
        self.last_error: Optional[str] = None
        self._mqtt = mqtt
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_connect_fail = self._on_connect_fail
        self.client.on_message = self._on_message
        self.client.connect_async(host, port, keepalive)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self.last_error = f"MQTT connect rejected: {reason_code}"
            self.connected.clear()
        else:
            self.last_error = None
            self.connection_count += 1
            self.connected.set()
            client.subscribe(self.status_topic, qos=0)

    def _on_message(self, client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            status = payload.get("status")
            angle = payload.get("angle")
            self.last_status = f"ESP8266 {status}" + (f" at {float(angle):.0f}°" if angle is not None else "")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            self.last_status = "ESP8266 sent an invalid status message"

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.connected.clear()
        if reason_code.is_failure:
            self.last_error = f"MQTT disconnected: {reason_code}"

    def _on_connect_fail(self, client, userdata):
        self.connected.clear()
        self.last_error = "MQTT broker unavailable; reconnecting"

    def publish_angle(self, angle: float, **metadata) -> bool:
        if not self.connected.is_set():
            return False
        payload = {"angle": round(float(angle), 1), **metadata}
        info = self.client.publish(self.topic, json.dumps(payload), qos=0, retain=False)
        return info.rc == self._mqtt.MQTT_ERR_SUCCESS

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()


class FaceCenterController:
    """Map a locked face's horizontal image offset to a bounded motor angle."""

    def __init__(self, motor: MQTTMotorClient, center_angle: float = 90.0,
                 min_angle: float = 0.0, max_angle: float = 180.0,
                 direction: int = 1, deadband: float = 0.04,
                 publish_interval: float = 0.12, smoothing: float = 0.35):
        if not 0 <= min_angle < max_angle <= 180:
            raise ValueError("Motor limits must satisfy 0 <= min < max <= 180 degrees")
        if not min_angle <= center_angle <= max_angle:
            raise ValueError("Center angle must be within the configured motor limits")
        if direction not in (-1, 1):
            raise ValueError("Direction must be -1 or 1")
        if not 0 <= deadband < 0.5:
            raise ValueError("Deadband must be between 0 and 0.5 of frame width")
        if not 0 < smoothing <= 1:
            raise ValueError("Smoothing must be greater than 0 and at most 1")
        self.motor = motor
        self.center_angle = center_angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.direction = direction
        self.deadband = deadband
        self.publish_interval = publish_interval
        self.smoothing = smoothing
        self._filtered_error: Optional[float] = None
        self._last_publish = 0.0
        self._last_angle: Optional[float] = None
        self._connection_count = motor.connection_count
        self._pending_angle: Optional[float] = None

    def update(self, dx: float, frame_width: int, track_id: int,
               face_name: str) -> Optional[float]:
        """Publish an angle for the face's position; return angle if sent."""
        if frame_width <= 0:
            return None
        if self._connection_count != self.motor.connection_count:
            # The ESP may have rebooted or lost its last command while offline.
            self._connection_count = self.motor.connection_count
            self._last_angle = None
            self._last_publish = 0.0
        # A live face target supersedes a queued recenter request.
        self._pending_angle = None
        error = max(-1.0, min(1.0, float(dx) / (frame_width / 2.0)))
        self._filtered_error = (error if self._filtered_error is None else
                                self.smoothing * error + (1.0 - self.smoothing) * self._filtered_error)
        filtered_error = self._filtered_error
        if abs(filtered_error) <= self.deadband:
            target_angle = self.center_angle
        else:
            outside_deadband = (abs(filtered_error) - self.deadband) / (1.0 - self.deadband)
            side = 1 if filtered_error > 0 else -1
            motor_side = side * self.direction
            usable = (self.max_angle - self.center_angle if motor_side > 0 else
                      self.center_angle - self.min_angle)
            target_angle = self.center_angle + motor_side * usable * outside_deadband
        target_angle = max(self.min_angle, min(self.max_angle, target_angle))

        now = time.monotonic()
        if (now - self._last_publish < self.publish_interval or
                self._last_angle is not None and abs(target_angle - self._last_angle) < 1.0):
            return None
        sent = self.motor.publish_angle(
            target_angle, track_id=int(track_id), name=face_name,
            error_ratio=round(filtered_error, 3), error_px=round(float(dx), 1),
            frame_width=int(frame_width), timestamp=int(time.time() * 1000),
        )
        if sent:
            self._last_publish = now
            self._last_angle = target_angle
            return target_angle
        return None

    def recenter(self) -> Optional[float]:
        """Return the motor to its calibrated center angle after unlocking."""
        self._filtered_error = None
        self._pending_angle = self.center_angle
        return self.flush_pending()

    def flush_pending(self) -> Optional[float]:
        """Publish a queued center command when MQTT is available."""
        if self._pending_angle is None:
            return None
        target_angle = self._pending_angle
        if self.motor.publish_angle(target_angle, mode="recenter"):
            self._last_angle = target_angle
            self._last_publish = time.monotonic()
            self._pending_angle = None
            return target_angle
        return None
