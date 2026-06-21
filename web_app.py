"""
web_app.py - Smart Vision Analysis System (Web Version)
Arsitektur: Dedicated capture thread + processing thread + MJPEG output queue
Fix: camera lag, proper close, single stream, anomaly model fix
"""

import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import json
import time
import threading
import queue
import base64
from datetime import datetime
from flask import (Flask, render_template, Response, jsonify,
                   request, send_from_directory, stream_with_context)

import database as db
import config as cfg
import utils

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from motion_detection import MotionDetector as _MotionDetector
    MD_AVAILABLE = True
except Exception:
    MD_AVAILABLE = False

try:
    import notifier as tg_notifier
except Exception:
    tg_notifier = None

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder="web/templates",
            static_folder="web/static")
app.secret_key = "svs-2024"
utils.ensure_dirs()
db.init_db()


# ══════════════════════════════════════════════════════════════════════════════
# CameraEngine — single capture + process thread, one shared MJPEG output
# ══════════════════════════════════════════════════════════════════════════════

class CameraEngine:
    """
    Mesin kamera terpusat.
    - _capture_thread : baca frame dari cv2.VideoCapture
    - _process_thread : jalankan inference (YOLO / motion)
    - _output_queue   : buffer JPEG siap stream (max 2 frame — anti-lag)
    - _sse_queues     : server-sent events ke semua browser
    """

    JPEG_QUALITY = 72        # kualitas JPEG stream (turunkan jika masih lag)
    MAX_OUTPUT   = 2         # max frame di output queue → buang yang lama
    TARGET_FPS   = 25        # target FPS capture

    def __init__(self):
        self._lock          = threading.Lock()
        self.cap            = None
        self.active         = False
        self.module         = "idle"   # idle|object|motion|anomaly|manipulation
        self.cam_index      = 0

        # Frame queues
        self._raw_q    = queue.Queue(maxsize=2)   # raw frame dari kamera
        self._out_q    = queue.Queue(maxsize=self.MAX_OUTPUT)  # JPEG terproses

        # Threads
        self._cap_thread  = None
        self._proc_thread = None
        self._stop_evt    = threading.Event()

        # Module params
        self.obj_conf   = 0.45
        self.obj_iou    = 0.45
        self.obj_model  = "yolov8n.pt"
        self.mot_sens   = 800
        self.mot_method = "MOG2 (Adaptive)"
        self.ano_conf   = 0.45
        self.ano_iou    = 0.45
        self.ano_detect_person  = True
        self.ano_detect_vehicle = False
        self.manip_filter = "none"
        self.zones: list  = []

        # Counters
        self.motion_count  = 0
        self.alert_count   = 0
        self.total_objects = 0
        self.session_labels: list = []
        self._last_motion_save = 0.0
        self._last_object_save = 0
        self._last_alert_save  = 0.0

        # YOLO model cache {filename: model}
        self._yolo: dict = {}
        # Motion detector
        self._mot_det = None

        # SSE
        self._sse: list[queue.Queue] = []
        self._sse_lock = threading.Lock()

        # FPS tracking
        self._fps_cnt  = 0
        self._fps_time = time.time()
        self.current_fps = 0.0
        self.latest_frame = None
        # Placeholder JPEG (kamera mati)
        self._placeholder = self._make_placeholder()

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _make_placeholder(self) -> bytes:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (18, 18, 30)
        cv2.putText(img, "[ Kamera Tidak Aktif ]",
                    (120, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 90), 2)
        cv2.putText(img, "Klik START untuk mengaktifkan kamera",
                    (80, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 75), 1)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    # ── Camera open/close ─────────────────────────────────────────────────────

    def start(self, index: int = 0, module: str = "idle") -> tuple[bool, str]:
        """Buka kamera dan mulai thread. Return (ok, msg)."""
        with self._lock:
            # Jika sudah aktif dengan kamera yang sama, cukup ganti module
            if self.active and self.cam_index == index:
                self.module = module
                if module == "motion":
                    self._mot_det = None  # reset bg model
                return True, f"Module berganti ke: {module}"

            # Tutup kamera lama jika ada
            self._do_stop()

            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                return False, f"Gagal membuka kamera index {index}"

            # Turunkan resolusi untuk performa lebih baik
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # buffer minimal = anti-lag

            self.cap       = cap
            self.active    = True
            self.module    = module
            self.cam_index = index
            self._stop_evt.clear()
            self._mot_det  = None  # reset motion detector

            # Bersihkan queue lama
            self._flush_queue(self._raw_q)
            self._flush_queue(self._out_q)

            # Start threads
            self._cap_thread  = threading.Thread(
                target=self._capture_loop, daemon=True, name="CaptureThread")
            self._proc_thread = threading.Thread(
                target=self._process_loop, daemon=True, name="ProcessThread")
            self._cap_thread.start()
            self._proc_thread.start()

            return True, f"Kamera {index} aktif — modul: {module}"

    def stop(self):
        """Tutup kamera dan hentikan semua thread."""
        with self._lock:
            self._do_stop()

    def set_module(self, module: str):
        """Ganti modul tanpa restart kamera."""
        with self._lock:
            self.module = module
            if module == "motion":
                self._mot_det = None

    def _do_stop(self):
        """Internal stop — harus dipanggil dengan self._lock dipegang."""
        self._stop_evt.set()
        self.active = False
        self.module = "idle"
        if self.cap:
            self.cap.release()
            self.cap = None
        self._flush_queue(self._raw_q)
        self._flush_queue(self._out_q)

    @staticmethod
    def _flush_queue(q: queue.Queue):
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break

    # ── Capture Thread ────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Dedicated thread: hanya baca frame dari kamera, masuk ke raw_q."""
        interval = 1.0 / self.TARGET_FPS
        while not self._stop_evt.is_set():
            t0 = time.time()
            if not self.cap or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Resize ke 640×480 jika perlu
            h, w = frame.shape[:2]
            if w != 640 or h != 480:
                frame = cv2.resize(frame, (640, 480))
            self.latest_frame = frame.copy()

            # Masukkan ke queue — jika penuh, buang frame lama (anti-lag)
            if self._raw_q.full():
                try: self._raw_q.get_nowait()
                except queue.Empty: pass
            try:
                self._raw_q.put_nowait(frame)
            except queue.Full:
                pass

            # Throttle agar tidak terlalu cepat
            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self.active = False

    # ── Process Thread ────────────────────────────────────────────────────────

    def _process_loop(self):
        """Dedicated thread: ambil frame dari raw_q, proses, encode JPEG."""
        while not self._stop_evt.is_set():
            try:
                frame = self._raw_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                processed = self._process_frame(frame)
                self.latest_frame = processed.copy()
            except Exception as e:
                print(f"[ProcessThread] Error: {e}")
                processed = frame

            # FPS counter
            self._fps_cnt += 1
            now = time.time()
            if now - self._fps_time >= 1.0:
                self.current_fps = self._fps_cnt / (now - self._fps_time)
                self._fps_cnt  = 0
                self._fps_time = now

            # Encode JPEG
            _, buf = cv2.imencode(".jpg", processed,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
            jpeg = buf.tobytes()

            # Output ke stream queue
            if self._out_q.full():
                try: self._out_q.get_nowait()
                except queue.Empty: pass
            try:
                self._out_q.put_nowait(jpeg)
            except queue.Full:
                pass

    # ── Frame Processing ──────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        module = self.module

        if module == "object":
            frame = self._proc_object(frame)
        elif module == "motion":
            frame = self._proc_motion(frame)
        elif module == "anomaly":
            frame = self._proc_anomaly(frame)
        elif module == "manipulation":
            frame = self._proc_manip(frame)

        utils.draw_timestamp(frame)
        return frame

    def _get_yolo(self, model_file: str = "yolov8n.pt"):
        if not YOLO_AVAILABLE: return None
        if model_file not in self._yolo:
            self._yolo[model_file] = YOLO(model_file)
        return self._yolo[model_file]

    def _proc_object(self, frame: np.ndarray) -> np.ndarray:
        model = self._get_yolo(self.obj_model)
        if not model: return frame
        results = model(frame, conf=self.obj_conf, iou=self.obj_iou, verbose=False)
        detections = []; summary = {}
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                c = float(box.conf[0]); cls = int(box.cls[0])
                label = model.names[cls]
                color = utils.class_color(cls)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                text = f"{label} {c:.0%}"
                (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame,(x1,y1-th-7),(x1+tw+4,y1),color,-1)
                cv2.putText(frame,text,(x1+2,y1-3),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
                detections.append({"label":label,"conf":round(c,2)})
                summary[label] = summary.get(label,0)+1

        count = len(detections)

        if count > 0:
            self.total_objects += count

            for d in detections:
                if d["label"] not in self.session_labels:
                    self.session_labels.append(d["label"])

            now = time.time()

            if now - self._last_object_save >= 3:
                self._last_object_save = now

                path = utils.save_frame(
                    frame,
                    "output/object",
                    "webcam_detect"
                )

                db.log_object_detection(
                    "webcam",
                    count,
                    list(summary.keys()),
                    path
                )

                self.broadcast(
        "detection",
        {
            "count": count,
            "summary": summary,
            "total": self.total_objects
        }
    )
    def _proc_motion(self, frame: np.ndarray) -> np.ndarray:
        if not MD_AVAILABLE: return frame
        if self._mot_det is None:
            self._mot_det = _MotionDetector()
        self._mot_det.sensitivity = self.mot_sens
        self._mot_det.method      = self.mot_method
        motion, frame, _ = self._mot_det.detect(frame.copy())
        if motion:
            now = time.time()
            if now - self._last_motion_save >= 2.0:   # cooldown 2 detik
                self._last_motion_save = now
                self.motion_count += 1
                path = utils.save_frame(frame, "output/motion", "motion")
                db.log_motion_detection("webcam", self.motion_count, path)
                if tg_notifier:
                    tg_notifier.get_notifier().notify_motion(path, self.motion_count)
                self.broadcast("motion", {
                    "count": self.motion_count,
                    "time": datetime.now().strftime("%H:%M:%S"),
                })
        utils.draw_status_badge(frame, "⚠ MOTION" if motion else "● NORMAL", motion)
        utils.draw_info_overlay(frame, [
            f"Events: {self.motion_count}",
            f"Method: {self.mot_method.split(' ')[0]}",
            f"FPS: {self.current_fps:.0f}",
        ])
        return frame

    def _proc_anomaly(self, frame: np.ndarray) -> np.ndarray:
        # Gunakan model nano yang cepat untuk anomaly, atau sesuai pengaturan
        model = self._get_yolo("yolov8n.pt")
        if not model: return frame

        # Tentukan classes
        classes = []
        if self.ano_detect_person:  classes.append(0)
        if self.ano_detect_vehicle: classes.extend([1,2,3,5,7])
        kwargs = dict(conf=self.ano_conf, iou=self.ano_iou, verbose=False)
        if classes: kwargs["classes"] = classes

        h, w = frame.shape[:2]

        # Gambar zona terlarang
        zone_colors = [
            (230, 80, 0), (0, 80, 230), (0, 200, 80),
            (200, 0, 200), (0, 200, 200), (200, 200, 0)
        ]
        for zi, zone in enumerate(self.zones):
            if len(zone) < 3: continue
            # Clamp koordinat ke dalam frame
            pts_raw = [[max(0,min(p[0],w-1)), max(0,min(p[1],h-1))] for p in zone]
            pts = np.array(pts_raw, dtype=np.int32)
            color = zone_colors[zi % len(zone_colors)]
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], color)
            cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], True, color, 2)
            c_x = int(np.mean([p[0] for p in pts_raw]))
            c_y = int(np.mean([p[1] for p in pts_raw]))
            # Label tengah zona
            label = f"ZONE {zi+1}"
            (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame,(c_x-lw//2-4,c_y-lh-4),(c_x+lw//2+4,c_y+4),color,-1)
            cv2.putText(frame, label, (c_x-lw//2, c_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        results = model(frame, **kwargs)
        intrusions = []; alert_zones = set()

        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                c = float(box.conf[0]); cls = int(box.cls[0])
                label = model.names[cls]
                color = utils.class_color(cls)
                cx = (x1+x2)//2; fy = y2

                hit = []
                for zi, zone in enumerate(self.zones):
                    if len(zone) < 3: continue
                    try:
                        pts = np.array(zone, dtype=np.int32)
                        mid_y = (y1 + y2) // 2
                        if (cv2.pointPolygonTest(pts,(float(cx),float(fy)),False)>=0 or
                            cv2.pointPolygonTest(pts,(float(cx),float(mid_y)),False)>=0):
                            hit.append(zi+1)
                    except Exception:
                        continue
                if hit:
                    color = utils.COLOR_RED
                    ov = frame.copy()
                    cv2.rectangle(ov,(x1,y1),(x2,y2),(0,0,255),-1)
                    cv2.addWeighted(ov,0.2,frame,0.8,0,frame)
                    intrusions.append({"label":label,"zones":hit})
                    alert_zones.update(hit)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                text = f"{label} {c:.0%}"
                (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
                cv2.rectangle(frame,(x1,y1-th-7),(x1+tw+4,y1),color,-1)
                cv2.putText(frame,text,(x1+2,y1-3),
                            cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1)

        is_anomaly = len(intrusions) > 0
        if is_anomaly:
            cv2.rectangle(frame,(0,0),(w-1,h-1),(0,0,255),5)
            txt = "!! ANOMALY DETECTED !!"
            (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_TRIPLEX,0.85,2)
            tx = (w-tw)//2; ty = h//2
            cv2.rectangle(frame,(tx-8,ty-th-8),(tx+tw+8,ty+8),(0,0,180),-1)
            cv2.putText(frame,txt,(tx,ty),cv2.FONT_HERSHEY_TRIPLEX,0.85,(255,70,70),2)

            now = time.time()
            if now - self._last_alert_save >= 3.0:
                self._last_alert_save = now
                self.alert_count += 1
                path = utils.save_frame(frame,"output/anomaly","alert")
                labels = list(set(i["label"] for i in intrusions))
                db.log_anomaly_detection("webcam",labels,str(list(alert_zones)),path)
                if tg_notifier:
                    tg_notifier.get_notifier().notify_anomaly(
                        path, str(list(alert_zones)), labels)
                self.broadcast("anomaly",{
                    "count": self.alert_count,
                    "labels": labels,
                    "zones": list(alert_zones),
                    "time": datetime.now().strftime("%H:%M:%S"),
                })

        utils.draw_status_badge(frame,"🚨 ANOMALY!" if is_anomaly else "✅ NORMAL",is_anomaly)
        utils.draw_info_overlay(frame,[
            f"Zones: {len(self.zones)}",
            f"Alerts: {self.alert_count}",
            f"FPS: {self.current_fps:.0f}",
        ])
        return frame
    
    def _proc_manip(self, frame: np.ndarray) -> np.ndarray:
        try:
            key = self.manip_filter
            print(f"[FILTER] {key}")
            
            if key in ("none", "normal", "original", ""):
                return frame.copy()

            from image_manipulation import FILTERS

            if key in FILTERS:
                return FILTERS[key][0](frame.copy())
            else:
                print(f"[FILTER NOT FOUND] {key}")

        except Exception as e:
            print(f"[Manipulation Error] {e}")

        return frame.copy() # ── MJPEG Stream ──────────────────────────────────────────────────────────

    def generate_mjpeg(self):
        """Generator MJPEG untuk Flask Response."""
        BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            if not self.active:
                jpeg = self._placeholder
                yield BOUNDARY + jpeg + b"\r\n"
                time.sleep(0.15)
                continue
            try:
                jpeg = self._out_q.get(timeout=0.5)
            except queue.Empty:
                continue
            yield BOUNDARY + jpeg + b"\r\n"

    # ── SSE ───────────────────────────────────────────────────────────────────

    def sse_subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=30)
        with self._sse_lock:
            self._sse.append(q)
        return q

    def sse_unsubscribe(self, q: queue.Queue):
        with self._sse_lock:
            try: self._sse.remove(q)
            except ValueError: pass

    def broadcast(self, event: str, data: dict):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        with self._sse_lock:
            dead = []
            for q in self._sse:
                try: q.put_nowait(msg)
                except queue.Full: dead.append(q)
            for q in dead:
                try: self._sse.remove(q)
                except ValueError: pass


# Singleton engine
engine = CameraEngine()


# ══════════════════════════════════════════════════════════════════════════════
# Flask Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    try:
        stats = db.get_summary_stats()
        return render_template("index.html", stats=stats)
    except Exception as e:
        import traceback
        return f"<pre>{traceback.format_exc()}</pre>"    

# ── Video Feed (satu endpoint, dipakai semua halaman) ─────────────────────────

@app.route("/video_feed")
def video_feed():
    return Response(
        engine.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── SSE ───────────────────────────────────────────────────────────────────────

@app.route("/events")
def sse_stream():
    q = engine.sse_subscribe()
    def _gen():
        try:
            yield f"data: {json.dumps({'type':'connected'})}\n\n"
            while True:
                try:
                    yield q.get(timeout=25)
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            engine.sse_unsubscribe(q)
    return Response(
        stream_with_context(_gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
    )


# ── Camera API ────────────────────────────────────────────────────────────────

@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    d   = request.json or {}
    idx = int(d.get("index", 0))
    mod = d.get("module", "idle")
    ok, msg = engine.start(idx, mod)
    return jsonify({"ok": ok, "msg": msg,
                    "module": mod, "active": engine.active})

@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    engine.stop()
    return jsonify({"ok": True, "msg": "Kamera dimatikan", "active": False})

@app.route("/api/camera/module", methods=["POST"])
def api_camera_module():
    mod = (request.json or {}).get("module", "idle")
    engine.set_module(mod)
    return jsonify({"ok": True, "module": mod})

@app.route("/api/camera/status")
def api_camera_status():
    return jsonify({
        "active": engine.active,
        "module": engine.module,
        "index":  engine.cam_index,
        "fps":    round(engine.current_fps, 1),
    })
@app.route("/api/camera/screenshot", methods=["POST"])
def api_screenshot():

    if engine.latest_frame is None:
        return jsonify({
            "ok": False,
            "msg": "Tidak ada frame"
        }), 400

    path = utils.screenshot(engine.latest_frame)

    return jsonify({
        "ok": True,
        "path": path
    })


# ── Module Params ─────────────────────────────────────────────────────────────

@app.route("/api/params/object", methods=["POST"])
def api_params_object():
    d = request.json or {}
    engine.obj_conf  = float(d.get("conf", 0.45))
    engine.obj_iou   = float(d.get("iou", 0.45))
    engine.obj_model = d.get("model", "yolov8n.pt")
    return jsonify({"ok": True})

@app.route("/api/params/motion", methods=["POST"])
def api_params_motion():
    d = request.json or {}
    engine.mot_sens   = int(d.get("sensitivity", 800))
    engine.mot_method = d.get("method", "MOG2 (Adaptive)")
    engine._mot_det   = None   # reset bg model
    return jsonify({"ok": True})

@app.route("/api/params/anomaly", methods=["POST"])
def api_params_anomaly():
    d = request.json or {}
    engine.ano_conf          = float(d.get("conf", 0.45))
    engine.ano_iou           = float(d.get("iou", 0.45))
    engine.ano_detect_person  = bool(d.get("detect_person", True))
    engine.ano_detect_vehicle = bool(d.get("detect_vehicle", False))
    return jsonify({"ok": True})

@app.route("/api/params/manipulation", methods=["POST"])
def api_params_manip():
    engine.manip_filter = (request.json or {}).get("filter", "grayscale")
    return jsonify({"ok": True, "filter": engine.manip_filter})

@app.route("/api/zones", methods=["GET"])
def api_zones_get():
    return jsonify({"zones": engine.zones})

@app.route("/api/zones", methods=["POST"])
def api_zones_set():
    engine.zones = (request.json or {}).get("zones", [])
    return jsonify({"ok": True, "count": len(engine.zones)})

@app.route("/api/zones/clear", methods=["POST"])
def api_zones_clear():
    engine.zones = []
    return jsonify({"ok": True})

@app.route("/api/motion/reset", methods=["POST"])
def api_motion_reset():
    engine.motion_count  = 0
    engine._mot_det      = None
    return jsonify({"ok": True})


# ── Upload — Object Detection ─────────────────────────────────────────────────

@app.route("/api/upload/detect", methods=["POST"])
def api_upload_detect():
    if "file" not in request.files:
        return jsonify({"ok": False, "msg": "No file"}), 400
    f   = request.files["file"]
    arr = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "msg": "Gambar tidak valid"}), 400

    model = engine._get_yolo(engine.obj_model)
    if not model:
        return jsonify({"ok": False, "msg": "YOLO tidak tersedia"}), 500

    results = model(img, conf=engine.obj_conf, iou=engine.obj_iou, verbose=False)
    dets = []; summary = {}
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            c=float(box.conf[0]); cls=int(box.cls[0])
            label=model.names[cls]; color=utils.class_color(cls)
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            txt=f"{label} {c:.0%}"
            (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(img,(x1,y1-th-7),(x1+tw+4,y1),color,-1)
            cv2.putText(img,txt,(x1+2,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            dets.append({"label":label,"conf":round(c,2)})
            summary[label]=summary.get(label,0)+1
    utils.draw_timestamp(img)
    out = utils.save_frame(img, "output/object", "web_detect")
    db.log_object_detection("web_upload", len(dets), list(summary.keys()), out)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({
        "ok": True, "count": len(dets), "summary": summary,
        "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
        "saved": out,
    })


# ── Upload — Manipulation ─────────────────────────────────────────────────────

@app.route("/api/upload/manipulate", methods=["POST"])
def api_upload_manip():
    if "file" not in request.files:
        return jsonify({"ok": False, "msg": "No file"}), 400
    fkey = request.form.get("filter", "grayscale")
    f    = request.files["file"]
    arr  = np.frombuffer(f.read(), dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "msg": "Gambar tidak valid"}), 400

    try:
        from image_manipulation import FILTERS
        if fkey not in FILTERS:
            return jsonify({"ok": False, "msg": "Filter tidak ditemukan"}), 400
        result = FILTERS[fkey][0](img.copy())
        filter_name = FILTERS[fkey][1]
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

    out = utils.save_frame(result, "output/image_processing", f"web_{fkey}")
    _, bo = cv2.imencode(".jpg", img,    [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, br = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jsonify({
        "ok": True, "filter": fkey, "filter_name": filter_name,
        "original": "data:image/jpeg;base64," + base64.b64encode(bo).decode(),
        "result":   "data:image/jpeg;base64," + base64.b64encode(br).decode(),
        "saved": out,
        "size_orig":   f"{img.shape[1]}×{img.shape[0]}",
        "size_result": f"{result.shape[1]}×{result.shape[0]}",
    })


# ── Data API ──────────────────────────────────────────────────────────────────

@app.route("/api/data/<module>")
def api_data_get(module):
    if module not in ("object","motion","anomaly"):
        return jsonify({"error":"Invalid"}), 400
    limit  = int(request.args.get("limit", 300))
    search = request.args.get("search","").lower()
    date   = request.args.get("date","")
    conn   = db.get_connection()
    rows   = conn.execute(
        f"SELECT * FROM {module}_detection_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    data = [dict(r) for r in rows]
    if search:
        data = [r for r in data if any(search in str(v).lower() for v in r.values())]
    if date:
        data = [r for r in data if r.get("date","").startswith(date)]
    return jsonify({"data": data, "total": len(data)})

@app.route("/api/data/<module>", methods=["POST"])
def api_data_add(module):
    d = request.json or {}
    try:
        if module == "object":
            db.log_object_detection(
                d.get("source","web"), int(d.get("total_objects",0)),
                [n.strip() for n in d.get("object_names","").split(",") if n.strip()],
                d.get("image_path",""))
        elif module == "motion":
            db.log_motion_detection(
                d.get("source","web"), int(d.get("motion_count",0)),
                d.get("image_path",""))
        elif module == "anomaly":
            db.log_anomaly_detection(
                d.get("source","web"),
                [n.strip() for n in d.get("object_names","").split(",") if n.strip()],
                d.get("area_info",""), d.get("image_path",""))
        else:
            return jsonify({"ok":False,"msg":"Invalid module"}), 400
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/api/data/<module>/<int:rid>", methods=["DELETE"])
def api_data_delete(module, rid):
    if module not in ("object","motion","anomaly"):
        return jsonify({"error":"Invalid"}), 400
    conn = db.get_connection()
    conn.execute(f"DELETE FROM {module}_detection_log WHERE id=?", (rid,))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/api/data/<module>/clear", methods=["DELETE"])
def api_data_clear(module):
    if module not in ("object","motion","anomaly"):
        return jsonify({"error":"Invalid"}), 400
    conn = db.get_connection()
    conn.execute(f"DELETE FROM {module}_detection_log")
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/api/data/export/<module>")
def api_data_export(module):
    import csv, io
    if module not in ("object","motion","anomaly"):
        return "Invalid", 400
    conn  = db.get_connection()
    rows  = conn.execute(
        f"SELECT * FROM {module}_detection_log ORDER BY id DESC").fetchall()
    conn.close()
    if not rows: return "No data", 404
    out = io.StringIO()
    w   = csv.DictWriter(out, fieldnames=dict(rows[0]).keys())
    w.writeheader(); w.writerows([dict(r) for r in rows])
    from flask import make_response
    resp = make_response(out.getvalue())
    resp.headers["Content-Type"]        = "text/csv"
    resp.headers["Content-Disposition"] = f"attachment; filename={module}_export.csv"
    return resp

@app.route("/api/stats/summary")
def api_stats():
    s = db.get_summary_stats()
    return jsonify({
        "total_objects":  s["object"]["total_objects"],
        "object_sessions":s["object"]["sessions"],
        "total_motions":  s["motion"]["total_motions"],
        "motion_events":  s["motion"]["events"],
        "anomaly_alerts": s["anomaly"]["alerts"],
        "camera_active":  engine.active,
        "active_module":  engine.module,
        "fps":            round(engine.current_fps, 1),
    })

@app.route("/api/stats/chart/<module>")
def api_chart(module):
    if module == "object":
        data = db.get_object_detection_chart_data()
    elif module == "motion":
        data = db.get_motion_chart_data()
    else:
        from collections import Counter
        logs = db.get_anomaly_logs(500)
        cnt  = Counter(l["date"] for l in logs)
        data = [(d, cnt[d]) for d in sorted(cnt)]
    return jsonify({"labels":[r[0] for r in data], "values":[r[1] for r in data]})


# ── Settings API ──────────────────────────────────────────────────────────────

@app.route("/api/settings/telegram", methods=["GET","POST"])
def api_telegram():
    if request.method == "GET":
        return jsonify(cfg.load().get("telegram",{}))
    c = cfg.load()
    c["telegram"].update(request.json or {})
    cfg.save(c)
    if tg_notifier: tg_notifier._notifier = None
    return jsonify({"ok": True})

@app.route("/api/settings/telegram/test", methods=["POST"])
def api_telegram_test():
    if not tg_notifier:
        return jsonify({"ok":False,"msg":"Notifier tidak tersedia"})
    d = request.json or {}
    token   = d.get("bot_token","")
    chat_id = d.get("chat_id","")
    if not token or not chat_id:
        return jsonify({"ok":False,"msg":"Token dan Chat ID wajib diisi"})
    orig_t = cfg.get("telegram.bot_token","")
    orig_c = cfg.get("telegram.chat_id","")
    cfg.set_value("telegram.bot_token", token)
    cfg.set_value("telegram.chat_id",   chat_id)
    n = tg_notifier.TelegramNotifier()
    ok, msg = n.test_connection()
    n.shutdown()
    cfg.set_value("telegram.bot_token", orig_t)
    cfg.set_value("telegram.chat_id",   orig_c)
    return jsonify({"ok": ok, "msg": msg})


# ── Static output files ─────────────────────────────────────1t──────────────────

@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(os.path.join(BASE_DIR, "output"), filename)

import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",  default="0.0.0.0")
    ap.add_argument("--port",  type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"\n🌐 Smart Vision Web — http://localhost:{args.port}\n")
    Timer(2, open_browser).start()
    app.run(host=args.host, port=args.port,
            debug=args.debug, threaded=True, use_reloader=False)
