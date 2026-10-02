"""Interactive live face detection, recognition, tracking and identity-lock demo."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog

from .camera_utils import add_camera_arg, open_camera
from .embed import ArcFaceEmbedderONNX
from .expressions import ExpressionTracker
from .face_centering import FaceCenterController, MQTTMotorClient
from .haar_5pt import Haar5ptDetector, align_face_5pt
from .tracking import IdentityLock, IoUTracker

DB_NPZ = Path("data/db/face_db.npz")
DB_JSON = Path("data/db/face_db.json")
CROPS_DIR = Path("data/enroll")
WIN = "Face Recognition Studio"
BUTTONS = ["Enroll", "Capture", "Auto Capture: off", "Save", "Cancel", "Lock", "Reload", "Quit"]


def load_db():
    db = {}
    if DB_NPZ.exists():
        with np.load(DB_NPZ, allow_pickle=True) as data:
            db = {k: data[k].astype(np.float32) for k in data.files}
    # New enrollment entries keep several pose/lighting samples. Accept legacy
    # 512-D vectors as one-template identities so existing databases still load.
    names, templates = [], []
    for name in sorted(db):
        vectors = np.asarray(db[name], dtype=np.float32).reshape(-1, 512)
        for vector in vectors:
            vector = vector / (np.linalg.norm(vector) + 1e-12)
            names.append(name)
            templates.append(vector.astype(np.float32))
    return db, names, np.stack(templates) if templates else None


def save_db(db):
    DB_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DB_NPZ, **{name: np.asarray(value, dtype=np.float32) for name, value in db.items()})
    DB_JSON.write_text(json.dumps({"names": sorted(db), "embedding_dim": 512,
                                   "templates_per_identity": {name: int(np.asarray(value).reshape(-1, 512).shape[0])
                                                               for name, value in db.items()}}, indent=2))


class FrameProcessor:
    """Run face detection, alignment and ArcFace away from the UI loop."""
    def __init__(self, detector, embedder):
        self.detector = detector
        self.embedder = embedder
        self.tracker = IoUTracker()
        self.cache = {}
        self.db_version = -1

    def process(self, frame, matrix, names, db_version, recognize_interval):
        if db_version != self.db_version:
            self.cache.clear()
            self.db_version = db_version
        faces = self.detector.detect(frame, max_faces=5)
        boxes = [(face.x1, face.y1, face.x2, face.y2) for face in faces]
        track_ids = self.tracker.update(boxes)
        now = time.time()
        records = []
        for face, track_id in zip(faces, track_ids):
            aligned, _ = align_face_5pt(frame, face.kps)
            cached = self.cache.get(track_id)
            candidate_name, distance = None, None
            if cached is None or now - cached[2] >= max(0.0, recognize_interval):
                vector = self.embedder.embed(aligned)
                if matrix is not None:
                    similarities = matrix @ vector
                    best = int(np.argmax(similarities))
                    candidate_name, distance = names[best], 1.0 - float(similarities[best])
                self.cache[track_id] = (candidate_name, distance, now, vector)
            else:
                candidate_name, distance, _, vector = cached
            records.append({"face": face, "track_id": track_id, "aligned": aligned, "vector": vector,
                            "candidate_name": candidate_name, "distance": distance})
        for stale_id in set(self.cache) - set(self.tracker.tracks):
            del self.cache[stale_id]
        return records, set(self.tracker.tracks)


def button_rects(width, height):
    margin, gap = 20, 12
    bw = (width - 2 * margin - gap * (len(BUTTONS) - 1)) // len(BUTTONS)
    button_height = min(124, max(84, int(height * .115)))
    y1, y2 = height - button_height - 14, height - 14
    return [(margin + i * (bw + gap), y1, margin + i * (bw + gap) + bw, y2) for i in range(len(BUTTONS))]


def main():
    parser = argparse.ArgumentParser(description="Interactive face recognition and identity tracking.")
    add_camera_arg(parser)
    parser.add_argument("--model", default="models/embedder_arcface.onnx")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--threshold", type=float, default=.50,
                        help="Cosine-distance match threshold; calibrate with genuine and impostor samples.")
    parser.add_argument("--detect-every", type=int, default=3,
                        help="Run face detector every N camera frames (3 gives smoother CPU preview).")
    parser.add_argument("--recognize-interval", type=float, default=.30,
                        help="Seconds between ArcFace comparisons per tracked face.")
    parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_BROKER_HOST", "127.0.0.1"),
                        help="Mosquitto broker host (default: 127.0.0.1; set MQTT_BROKER_HOST to override).")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", default="face-centering/motor/command")
    parser.add_argument("--mqtt-status-topic", default="face-centering/motor/status")
    parser.add_argument("--mqtt-client-id", default="face-centering-camera")
    parser.add_argument("--motor-center-angle", type=float, default=90.0,
                        help="Motor angle when the locked face is centered.")
    parser.add_argument("--motor-min-angle", type=float, default=20.0)
    parser.add_argument("--motor-max-angle", type=float, default=160.0)
    parser.add_argument("--motor-direction", type=int, choices=(-1, 1), default=1,
                        help="Use -1 if the motor turns away from the face with the default direction.")
    parser.add_argument("--center-deadband", type=float, default=.04,
                        help="Ignore horizontal face offsets inside this fraction of frame half-width.")
    parser.add_argument("--no-fullscreen", action="store_true")
    args = parser.parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        parser.error(f"ArcFace model not found: {model_path}. See FACE_CENTERING.md for model setup.")
    cv2.setNumThreads(2)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera!r}. Select the external HD camera with --camera.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Tk supplies the desktop dimensions and a reliable native name-entry
    # dialog. OpenCV's waitKey keyboard focus is inconsistent in fullscreen.
    ui_root = tk.Tk()
    ui_root.withdraw()
    ui_root.update_idletasks()
    screen_width, screen_height = ui_root.winfo_screenwidth(), ui_root.winfo_screenheight()
    display_width, display_height = (screen_width, screen_height) if not args.no_fullscreen else (
        min(screen_width, args.width), min(screen_height, args.height))
    detector = Haar5ptDetector(debug=False)
    embedder = ArcFaceEmbedderONNX(model_path=str(model_path))
    processor = FrameProcessor(detector, embedder)
    mqtt_motor = MQTTMotorClient(args.mqtt_host, args.mqtt_port, args.mqtt_topic,
                                 args.mqtt_status_topic,
                                 args.mqtt_client_id)
    centering = FaceCenterController(
        mqtt_motor, center_angle=args.motor_center_angle,
        min_angle=args.motor_min_angle, max_angle=args.motor_max_angle,
        direction=args.motor_direction, deadband=args.center_deadband,
    )
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-inference")
    pending = None
    database_version = 0
    current_records = []
    active_track_ids = set()
    identity_lock = IdentityLock()
    expression_trackers = {}
    expression_cache = {}
    db, names, matrix = load_db()
    identity_count = len(db)
    threshold = args.threshold
    state = {"enrolling": False, "name": "", "samples": [], "crops": [],
             "auto": False, "last_capture": 0.0, "selected": None, "message": "Ready",
             "missing_since": None}
    latest = {"faces": [], "records": [], "raw_size": (args.width, args.height),
              "display_size": (display_width, display_height)}
    frame_number = 0
    fps_clock = time.time()
    fps_frames = 0
    shown_fps = 0.0

    def on_mouse(event, x, y, flags, userdata):
        h, w = latest["display_size"][1], latest["display_size"][0]
        if event == cv2.EVENT_MOUSEMOVE:
            state["hovered"] = next((i for i, rect in enumerate(button_rects(w, h))
                                     if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]), None)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for i, (x1, y1, x2, y2) in enumerate(button_rects(w, h)):
            if x1 <= x <= x2 and y1 <= y <= y2:
                action = BUTTONS[i]
                if action == "Enroll":
                    if not args.no_fullscreen:
                        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    ui_root.attributes("-topmost", True)
                    name = simpledialog.askstring("Enroll a face", "Enter the person's name:", parent=ui_root)
                    ui_root.attributes("-topmost", False)
                    if not args.no_fullscreen:
                        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    if name and name.strip():
                        state.update(enrolling=True, name=name.strip(), samples=[], crops=[], auto=False,
                                     selected=None,
                                     message=f"Enrolling {name.strip()}: capture at least 5 samples")
                    else:
                        state["message"] = "Enrollment cancelled"
                elif action == "Capture":
                    state["capture_requested"] = True
                elif action.startswith("Auto") and state["enrolling"]:
                    state["auto"] = not state["auto"]
                    state["message"] = ("Auto capture on: collecting a sample every 0.5s; press Save when ready"
                                         if state["auto"] else "Auto capture paused; press Save to store samples")
                elif action == "Save":
                    state["save_requested"] = True
                elif action == "Cancel":
                    state.update(enrolling=False, name="", samples=[], crops=[], auto=False, message="Enrollment cancelled")
                elif action == "Lock":
                    if identity_lock.active:
                        identity_lock.unlock()
                        centered_angle = centering.recenter()
                        if centered_angle is not None:
                            state["motor_angle"] = centered_angle
                        state["message"] = "Identity unlocked"
                    else:
                        selected = next((r for r in latest["records"]
                                         if r["track_id"] == state["selected"]), None)
                        recognized = [r for r in latest["records"] if r["name"]]
                        if (selected is None or not selected["name"]) and len(recognized) == 1:
                            selected = recognized[0]
                            state["selected"] = selected["track_id"]
                        if selected and selected["name"]:
                            identity_lock.lock(selected["track_id"], selected["name"])
                            state["message"] = f"LOCKED: {selected['name']} (track {selected['track_id']})"
                        elif len(recognized) > 1:
                            state["message"] = "Click the recognized face you want, then press Lock"
                        else:
                            state["message"] = "Cannot lock: wait for an enrolled name, not Unknown"
                elif action == "Reload":
                    state["reload_requested"] = True
                elif action == "Quit":
                    state["quit"] = True
                return
        # Clicking a detected face selects its stable track ID.
        raw_w, raw_h = latest["raw_size"]
        raw_x, raw_y = x * raw_w / max(1, w), y * raw_h / max(1, h)
        for record in latest["records"]:
            x1, y1, x2, y2 = record["box"]
            if x1 <= raw_x <= x2 and y1 <= raw_y <= y2:
                state["selected"] = record["track_id"]
                state["message"] = f"Selected {record['name'] or 'Unknown'} (track {record['track_id']})"
                return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    if not args.no_fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while not state.get("quit"):
            # Pump window events before any camera or frame work so a slow
            # inference result cannot delay button clicks or keyboard actions.
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
                threshold = min(1.2, threshold + .01)
            elif key == ord("-"):
                threshold = max(.05, threshold - .01)

            ok, frame = cap.read()
            if not ok:
                state["message"] = "Camera frame unavailable"
                break
            frame_height, frame_width = frame.shape[:2]
            height, width = frame_height, frame_width
            vis = frame.copy()
            detect_every = max(1, args.detect_every)
            fresh_result = False
            if pending is not None and pending.done():
                try:
                    current_records, active_track_ids = pending.result()
                    fresh_result = True
                except Exception as exc:
                    state["message"] = f"Vision pipeline error: {exc}"
                pending = None
            if pending is None and frame_number % detect_every == 0:
                pending = worker.submit(processor.process, frame.copy(), matrix, tuple(names),
                                        database_version, args.recognize_interval)
            frame_number += 1
            records = []
            now = time.time()
            fps_frames += 1
            if now - fps_clock >= 1.0:
                shown_fps = fps_frames / (now - fps_clock)
                fps_clock, fps_frames = now, 0

            for detected in current_records:
                face = detected["face"]
                track_id = detected["track_id"]
                aligned = detected["aligned"]
                vector = detected["vector"]
                candidate_name = detected["candidate_name"]
                distance = detected["distance"]
                best_name = candidate_name if distance is not None and distance <= threshold else None
                lock_status = identity_lock.status(track_id, best_name)
                selected = track_id == state["selected"]
                color = (0, 220, 0) if best_name else (0, 70, 255)
                if selected:
                    color = (255, 190, 0)
                if lock_status == "LOCKED":
                    color = (0, 255, 255)
                label = f"{best_name or 'Unknown'}  T{track_id}"
                if distance is not None:
                    label += f"  d={distance:.3f}"
                if lock_status:
                    label += f"  {lock_status}"
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
                cv2.putText(vis, label, (face.x1, max(24, face.y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                for px, py in face.kps.astype(int):
                    cv2.circle(vis, (px, py), 3, (0, 255, 255), -1)
                expression = expression_cache.get(track_id)
                if fresh_result and face.landmarks is not None:
                    analyzer = expression_trackers.setdefault(track_id, ExpressionTracker())
                    expression = analyzer.update(face.landmarks)
                    expression_cache[track_id] = expression
                nose_x, nose_y = map(float, face.kps[2])
                dx, dy = nose_x - frame_width / 2, nose_y - frame_height / 2
                face_center_dx = ((face.x1 + face.x2) / 2.0) - frame_width / 2
                distance_from_center = float(np.hypot(dx, dy))
                horizontal = "center" if abs(dx) < frame_width * .04 else ("right" if dx > 0 else "left")
                vertical = "center" if abs(dy) < frame_height * .04 else ("down" if dy > 0 else "up")
                direction = "center" if horizontal == vertical == "center" else "-".join(
                    part for part in (vertical, horizontal) if part != "center")
                face_color = (0, 220, 0) if best_name else (0, 70, 255)
                details_y = min(height - 64, face.y2 + 18)
                if expression:
                    expr_color = (0, 220, 220) if expression["expression"] == "Smiling" else face_color
                    cv2.putText(vis, f"{expression['expression']} | eyes {expression['eyes']} | blinks {expression['blink_count']}",
                                (max(4, face.x1), max(18, details_y)), cv2.FONT_HERSHEY_SIMPLEX, .47, expr_color, 1)
                cv2.putText(vis, f"Nose {direction} | dx={dx:+.0f} dy={dy:+.0f} d={distance_from_center:.0f}px",
                            (max(4, face.x1), max(34, details_y + 19)), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 220, 80), 1)
                records.append({"track_id": track_id, "box": (face.x1, face.y1, face.x2, face.y2),
                                "name": best_name, "aligned": aligned, "expression": expression,
                                # Enrollment capture consumes this normalized ArcFace embedding.
                                # Keep it in the UI-facing record as well as the worker cache.
                                "vector": vector.copy(),
                                "nose_direction": direction, "nose_dx": dx, "nose_dy": dy,
                                "face_center_dx": face_center_dx,
                                "face_center_x": (face.x1 + face.x2) / 2.0,
                                "face_center_y": (face.y1 + face.y2) / 2.0,
                                "nose_distance": distance_from_center})

            for stale_id in set(expression_trackers) - active_track_ids:
                del expression_trackers[stale_id]
                expression_cache.pop(stale_id, None)

            latest["records"] = records
            latest["raw_size"] = (frame_width, frame_height)
            latest["display_size"] = (display_width, display_height)
            if state.pop("reload_requested", False):
                db, names, matrix = load_db()
                identity_count = len(db)
                database_version += 1
                state["message"] = f"Loaded {identity_count} identities"

            selected_record = next((r for r in records if r["track_id"] == state["selected"]), None)
            # A prior selection may refer to a face/track from before this
            # enrollment session. Use a currently visible face by default, and
            # keep Capture usable if tracking assigns a fresh ID mid-enrollment.
            if selected_record is None and records and (state["selected"] is None or state["enrolling"]):
                selected_record = records[0]
                state["selected"] = selected_record["track_id"]
            if state.pop("capture_requested", False):
                if state["enrolling"] and selected_record:
                    state["samples"].append(selected_record["vector"].copy())
                    state["crops"].append(selected_record["aligned"].copy())
                    state["message"] = f"Captured sample {len(state['samples'])} (need at least 5)"
                elif state["enrolling"]:
                    state["message"] = "No face to capture"
            if state["auto"] and state["enrolling"] and selected_record and now - state["last_capture"] >= .5:
                state["samples"].append(selected_record["vector"].copy())
                state["crops"].append(selected_record["aligned"].copy())
                state["last_capture"] = now
                state["message"] = f"Auto captured sample {len(state['samples'])} (5 minimum; press Save when ready)"
            elif state["auto"] and state["enrolling"] and selected_record is None:
                state["message"] = f"Auto Capture waiting for a face | samples: {len(state['samples'])}"
            if state.pop("save_requested", False):
                name = re.sub(r"[^\w .'-]", "", state["name"]).strip()
                if not state["enrolling"] or not name:
                    state["message"] = "Enter a valid name before saving"
                elif len(state["samples"]) < 5:
                    state["message"] = "Capture at least 5 samples before saving"
                else:
                    # Keep the captured poses as separate templates. Matching
                    # against the nearest one tolerates normal pose and camera
                    # landmark variation better than one averaged vector.
                    templates = np.stack(state["samples"]).astype(np.float32)
                    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)
                    db[name] = templates
                    save_db(db)
                    person_dir = CROPS_DIR / name
                    person_dir.mkdir(parents=True, exist_ok=True)
                    for index, crop in enumerate(state["crops"], start=1):
                        cv2.imwrite(str(person_dir / f"{int(now * 1000)}_{index:03d}.jpg"), crop)
                    db, names, matrix = load_db()
                    identity_count = len(db)
                    state.update(enrolling=False, name="", samples=[], crops=[], auto=False,
                                 message=f"Saved {name}; database has {len(names)} identities")

            locked_record = next((r for r in records if r["track_id"] == identity_lock.track_id), None)
            if identity_lock.active and (locked_record is None or locked_record["name"] != identity_lock.name):
                # IoU tracking can assign a new ID after a brief detection loss
                # or fast movement. Reattach only when face recognition again
                # identifies the locked person; never transfer to another name.
                replacement = next((r for r in records if r["name"] == identity_lock.name), None)
                if replacement is not None and identity_lock.rebind(
                        replacement["track_id"], replacement["name"]):
                    locked_record = replacement
            locked_face_visible = locked_record is not None and locked_record["name"] == identity_lock.name
            motor_angle = None
            if identity_lock.active and locked_face_visible and mqtt_motor.connected.is_set():
                motor_angle = centering.update(
                    locked_record["face_center_dx"], frame_width, locked_record["track_id"],
                    locked_record["name"],
                )
            elif not identity_lock.active:
                motor_angle = centering.flush_pending()
            else:
                motor_angle = None
            state["motor_angle"] = motor_angle if motor_angle is not None else state.get("motor_angle")
            if locked_face_visible:
                center_x, center_y = frame_width // 2, frame_height // 2
                center_band = max(1, int(frame_width * args.center_deadband / 2.0))
                cv2.line(vis, (center_x - center_band, 72), (center_x - center_band, frame_height - 1),
                         (70, 90, 70), 1)
                cv2.line(vis, (center_x + center_band, 72), (center_x + center_band, frame_height - 1),
                         (70, 90, 70), 1)
                face_x = int(locked_record["face_center_x"])
                face_y = int(locked_record["face_center_y"])
                cv2.line(vis, (center_x, center_y), (face_x, face_y), (0, 220, 255), 2)
                cv2.circle(vis, (face_x, face_y), 7, (0, 220, 255), -1)
                cv2.putText(vis, f"LOCKED FACE OFFSET {locked_record['face_center_dx']:+.0f}px",
                            (max(12, center_x - 170), 98), cv2.FONT_HERSHEY_SIMPLEX,
                            .58, (0, 220, 255), 2)
            missing = not records or (identity_lock.active and not locked_face_visible)
            if missing:
                if state["missing_since"] is None:
                    state["missing_since"] = now
            else:
                state["missing_since"] = None

            # Fill the fullscreen window even when the camera cannot negotiate
            # the requested HD capture mode (many USB cameras return 640x480).
            if (frame_width, frame_height) != (display_width, display_height):
                vis = cv2.resize(vis, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            width, height = display_width, display_height
            cv2.rectangle(vis, (0, 0), (width, 70), (22, 27, 38), -1)
            cv2.putText(vis, f"FACE RECOGNITION  |  {identity_count} identities  |  threshold {threshold:.2f}  |  {shown_fps:.1f} FPS",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 244, 250), 2)
            cv2.putText(vis, f"External camera: {args.camera}  |  Faces: {len(records)}  |  {state['message']}",
                        (16, 54), cv2.FONT_HERSHEY_SIMPLEX, .48, (190, 205, 220), 1)
            broker_label = "MQTT connected" if mqtt_motor.connected.is_set() else (
                mqtt_motor.last_error or "MQTT connecting")
            if mqtt_motor.last_status:
                broker_label += f" | {mqtt_motor.last_status}"
            motor_label = (f"Motor: {state['motor_angle']:.0f}°" if state.get("motor_angle") is not None
                           else "Motor: waiting for a locked face")
            cv2.putText(vis, f"{broker_label}  |  {motor_label}",
                        (width - 470, 28), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (100, 230, 130) if mqtt_motor.connected.is_set() else (80, 170, 255), 1)
            if state["missing_since"] is not None and now - state["missing_since"] >= .5:
                warning = (f"SEARCHING FOR LOCKED PERSON: {identity_lock.name}" if locked_record is None else
                           f"LOCKED PERSON NOT CONFIRMED: {identity_lock.name}") if identity_lock.active else "WARNING: NO FACE DETECTED"
                banner_color = (0, 90, 230) if identity_lock.active else (0, 125, 255)
                banner_w = min(width - 40, max(360, len(warning) * 18))
                left = max(20, (width - banner_w) // 2)
                cv2.rectangle(vis, (left, 125), (left + banner_w, 175), banner_color, -1)
                cv2.putText(vis, warning, (left + 14, 158), cv2.FONT_HERSHEY_SIMPLEX, .78,
                            (255, 255, 255), 2, cv2.LINE_AA)
            if state["enrolling"]:
                cv2.putText(vis, f"Enrolling: {state['name']}   |   Samples: {len(state['samples'])} / 5 minimum",
                            (16, 94), cv2.FONT_HERSHEY_SIMPLEX, .60, (255, 255, 255), 1)

            labels = BUTTONS.copy()
            labels[2] = (f"Auto: ON ({len(state['samples'])})" if state["auto"] else "Auto Capture: off")
            labels[5] = "Unlock" if identity_lock.active else "Lock"
            for label, rect in zip(labels, button_rects(width, height)):
                x1, y1, x2, y2 = rect
                enabled = (label in ("Enroll", "Reload", "Quit", "Lock", "Unlock") or
                           state["enrolling"] and label in ("Capture", "Save", "Cancel",
                                                             "Auto Capture: off") or
                           state["enrolling"] and label.startswith("Auto: ON"))
                if state.get("hovered") == labels.index(label):
                    color = (78, 148, 204)
                else:
                    color = (52, 104, 147) if enabled else (54, 58, 65)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, -1)
                baseline = y1 + (y2 - y1) // 2 + 10
                font_scale = min(.86, max(.58, width / 2300))
                cv2.putText(vis, label, (x1 + 12, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), 2)

            cv2.imshow(WIN, vis)
    finally:
        worker.shutdown(wait=True, cancel_futures=True)
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        ui_root.destroy()
        mqtt_motor.close()


if __name__ == "__main__":
    main()
