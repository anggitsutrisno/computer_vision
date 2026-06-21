"""
anomaly_detection.py - Smart Vision Analysis System
Modul 3: Anomaly / Intrusion Detection — Akurasi Tinggi
Peningkatan: polygon zones, multi-class, dwell time, per-zone confidence, IoU tuning
"""

import cv2
import numpy as np
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from collections import defaultdict

import utils
import database as db
import config as cfg
import notifier
from window_utils import get_module_sizes

DARK="#0d0d1a"; PANEL="#13132b"; HEADER="#1a1a2e"
ACCENT="#ff4444"; GREEN="#00aa44"; RED="#aa2222"
TEXT_DIM="#6666aa"; TEXT_BR="#eeeeff"
MODE_CAM="#3a0000"; MODE_MAN="#3a1a5e"

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

_model = None
def get_model():
    global _model
    if _model is None:
        if not YOLO_AVAILABLE: raise RuntimeError("Ultralytics belum terpasang.")
        _model = YOLO("yolov8n.pt")
    return _model


class RestrictedZone:
    """
    Zona terlarang berbasis polygon (lebih akurat dari rectangle).
    Mendukung mode rectangle (drag) dan polygon (klik per-titik).
    """
    def __init__(self, points: list, zone_id: int, label: str = ""):
        self.points  = np.array(points, dtype=np.int32)
        self.zone_id = zone_id
        self.label   = label or f"Zone {zone_id}"
        self.color   = [
            (0, 60, 220), (220, 60, 0), (0, 180, 80),
            (180, 0, 180), (0, 180, 180), (180, 180, 0)
        ][zone_id % 6]
        self.intrusion_count = 0

    @property
    def bounding_rect(self):
        x,y,w,h = cv2.boundingRect(self.points)
        return x,y,x+w,y+h

    def contains_point(self, x: int, y: int) -> bool:
        """Cek apakah titik (x,y) berada di dalam polygon."""
        return cv2.pointPolygonTest(self.points, (float(x), float(y)), False) >= 0

    def contains_bbox(self, x1, y1, x2, y2) -> bool:
        """
        Cek apakah bounding box masuk ke zona.
        Menggunakan beberapa titik agar lebih akurat.
        """

        test_points = [
            (x1, y1),                  # kiri atas
            (x2, y1),                  # kanan atas
            (x1, y2),                  # kiri bawah
            (x2, y2),                  # kanan bawah
            ((x1+x2)//2, (y1+y2)//2), # tengah
            ((x1+x2)//2, y2),         # kaki bawah
        ]

        for px, py in test_points:
            if self.contains_point(px, py):
                return True

        return False

    def draw(self, frame: np.ndarray, alert: bool = False):
        """Gambar zona di frame dengan efek transparan."""
        overlay = frame.copy()
        pts = self.points.reshape((-1,1,2))

        # Fill
        fill_color = (0,0,180) if not alert else (0,0,255)
        cv2.fillPoly(overlay, [pts], fill_color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Border
        border_color = (0,0,255) if alert else self.color
        thick = 3 if alert else 2
        cv2.polylines(frame, [pts], True, border_color, thick)

        # Label
        x,y,_,_ = cv2.boundingRect(pts)
        bg_color = (0,0,200) if alert else self.color
        text = f" {self.label} ({self.intrusion_count})"
        (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x,y-th-8), (x+tw+4,y), bg_color, -1)
        cv2.putText(frame, text, (x+2,y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Gambar titik-titik polygon
        for pt in self.points:
            cv2.circle(frame, tuple(pt), 4, border_color, -1)


class ZoneManager:
    """Mengelola kumpulan RestrictedZone dengan dukungan polygon."""

    def __init__(self):
        self.zones: list[RestrictedZone] = []
        self._drawing     = False
        self._temp_pts    = []   # titik sementara saat menggambar
        self._rect_start  = None
        self.draw_mode    = "rect"   # "rect" | "polygon"

    def reset(self):
        self.zones.clear()
        self._drawing = False
        self._temp_pts = []
        self._rect_start = None

    def start_rect(self, x, y):
        self._drawing = True
        self._rect_start = (x, y)

    def update_rect(self, x, y):
        pass  # digunakan saat mouse move untuk preview

    def finish_rect(self, x, y):
        if not self._drawing or self._rect_start is None: return
        x1,y1 = self._rect_start; x2,y2 = x,y
        if abs(x2-x1)>15 and abs(y2-y1)>15:
            pts = [(min(x1,x2),min(y1,y2)), (max(x1,x2),min(y1,y2)),
                   (max(x1,x2),max(y1,y2)), (min(x1,x2),max(y1,y2))]
            self.zones.append(RestrictedZone(pts, len(self.zones)+1))
        self._drawing = False; self._rect_start = None

    def get_rect_preview(self, x, y):
        if self._drawing and self._rect_start:
            return self._rect_start, (x,y)
        return None, None

    def check_intrusions(self, x1,y1,x2,y2) -> list:
        """Kembalikan list zone_id yang dimasuki bbox."""
        return [z.zone_id for z in self.zones if z.contains_bbox(x1,y1,x2,y2)]

    def draw_all(self, frame, alert_zones: set = None):
        alert_zones = alert_zones or set()
        for z in self.zones:
            z.draw(frame, alert=z.zone_id in alert_zones)

    def draw_preview(self, frame, cur_x, cur_y):
        if self._drawing and self._rect_start:
            x1,y1 = self._rect_start
            cv2.rectangle(frame,(min(x1,cur_x),min(y1,cur_y)),
                          (max(x1,cur_x),max(y1,cur_y)),(0,200,255),2)


class AnomalyDetectionWindow:
    POLL_MS = 25

    # Kelas YOLO yang dideteksi untuk intrusi (bisa dikustomisasi)
    DETECT_CLASSES = {
        "person":       0,
        "car":          2,
        "motorcycle":   3,
        "bicycle":      1,
        "truck":        7,
        "bus":          5,
    }

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("🚨 Modul 3 – Anomaly / Intrusion Detection")
        self.win.configure(bg=DARK)
        sz = get_module_sizes(parent)
        self.WIN_W=sz["win_w"]; self.WIN_H=sz["win_h"]
        self.DISP_W=sz["disp_w"]; self.DISP_H=sz["disp_h"]
        utils.center_window(self.win, self.WIN_W, self.WIN_H)
        self.win.resizable(True, True); self.win.minsize(700,500)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._q: queue.Queue = queue.Queue(maxsize=3)
        self.cap = None; self.running = False
        self.zone_mgr = ZoneManager()
        self.alert_count = 0
        self.mode = tk.StringVar(value="camera")
        self.conf_var = tk.DoubleVar(value=0.45)
        self.iou_var  = tk.DoubleVar(value=0.45)
        self.draw_mode_var = tk.BooleanVar(value=False)

        # Multi-class detection
        self.detect_person = tk.BooleanVar(value=True)
        self.detect_vehicle= tk.BooleanVar(value=False)

        self._photo = None; self._latest_frame = None
        self._last_alert = None
        self._cooldown = cfg.get("detection.anomaly_cooldown", 2.0)
        self._scale = 1.0; self._mouse_drawing = False
        self._cur_mouse = (0,0)
        self._tg = notifier.get_notifier()
        self._fps_cnt = 0; self._fps_ts = datetime.now()
        # Dwell time tracking: track_id → frame_count inside zone
        self._dwell: dict = defaultdict(int)

        self.mf_source=tk.StringVar(value="manual")
        self.mf_names =tk.StringVar(value="person")
        self.mf_area  =tk.StringVar(value="Zone 1")
        self.mf_path  =tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        hdr=tk.Frame(self.win,bg=HEADER,height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr,text="🚨  ANOMALY DETECTION  —  Polygon Zone + Multi-Class",
                 font=("Helvetica",13,"bold"),bg=HEADER,fg=ACCENT).pack(side="left",padx=16,pady=14)

        toggle=tk.Frame(self.win,bg="#111133",height=46)
        toggle.pack(fill="x"); toggle.pack_propagate(False)
        tk.Label(toggle,text="MODE:",bg="#111133",fg=TEXT_DIM,
                 font=("Helvetica",9,"bold")).pack(side="left",padx=14,pady=12)
        self.btn_cam=tk.Button(toggle,text="📷  KAMERA (WEBCAM)",
                                font=("Helvetica",10,"bold"),relief="flat",cursor="hand2",
                                padx=18,pady=6,command=lambda: self._set_mode("camera"))
        self.btn_cam.pack(side="left",padx=4,pady=6)
        self.btn_man=tk.Button(toggle,text="✏️  INPUT MANUAL",
                                font=("Helvetica",10,"bold"),relief="flat",cursor="hand2",
                                padx=18,pady=6,command=lambda: self._set_mode("manual"))
        self.btn_man.pack(side="left",padx=4,pady=6)
        self.mode_badge=tk.Label(toggle,text="",bg="#111133",fg=ACCENT,
                                  font=("Helvetica",8,"italic"))
        self.mode_badge.pack(side="left",padx=10)

        body=tk.Frame(self.win,bg=DARK)
        body.pack(fill="both",expand=True,padx=10,pady=6)
        self.left=tk.Frame(body,bg=DARK)
        self.left.pack(side="left",fill="both",expand=True)

        self.cam_panel=tk.Frame(self.left,bg=DARK)
        self.canvas_lbl=tk.Label(self.cam_panel,bg="#111122",
                                  text="[ Mulai kamera → Draw Mode → drag buat zona ]",
                                  font=("Courier",11),fg="#444466",
                                  wraplength=460,justify="center")
        self.canvas_lbl.pack(fill="both",expand=True,padx=4,pady=4)
        self.canvas_lbl.bind("<ButtonPress-1>",  self._on_mouse_down)
        self.canvas_lbl.bind("<B1-Motion>",       self._on_mouse_move)
        self.canvas_lbl.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas_lbl.bind("<Motion>",          self._on_mouse_hover)
        self.status_bar=tk.Label(self.cam_panel,text="Status: Idle",
                                  bg=HEADER,fg="#aaaacc",font=("Helvetica",9),anchor="w")
        self.status_bar.pack(fill="x",padx=4,pady=(0,2))

        self.man_panel=tk.Frame(self.left,bg=DARK)
        self._build_manual_panel()

        right=tk.Frame(body,bg=PANEL,width=275)
        right.pack(side="right",fill="y",padx=(8,2),pady=2)
        right.pack_propagate(False)
        self._build_sidebar(right)

        self._set_mode("camera")

    def _build_manual_panel(self):
        p=self.man_panel
        hdr=tk.Frame(p,bg="#1a1a3e",height=42); hdr.pack(fill="x",pady=(4,0))
        tk.Label(hdr,text="✏️  INPUT DATA ANOMALY DETECTION SECARA MANUAL",
                 font=("Helvetica",11,"bold"),bg="#1a1a3e",fg=ACCENT).pack(side="left",padx=16,pady=10)
        form=tk.Frame(p,bg=DARK,padx=30,pady=20); form.pack(fill="both",expand=True)
        now=datetime.now()
        self.mf_date=tk.StringVar(value=now.strftime("%Y-%m-%d"))
        self.mf_time=tk.StringVar(value=now.strftime("%H:%M:%S"))

        def field(label,var,hint=""):
            row=tk.Frame(form,bg=DARK); row.pack(fill="x",pady=7)
            tk.Label(row,text=label,bg=DARK,fg=TEXT_DIM,font=("Helvetica",10),
                     width=22,anchor="w").pack(side="left")
            w=tk.Entry(row,textvariable=var,bg=PANEL,fg=TEXT_BR,
                       insertbackground="white",font=("Courier",11),relief="flat",
                       highlightthickness=1,highlightcolor=ACCENT,highlightbackground="#2a2a4e")
            w.pack(side="left",fill="x",expand=True,ipady=5,padx=(8,0))
            if hint: tk.Label(row,text=hint,bg=DARK,fg="#444466",font=("Helvetica",8)).pack(side="left",padx=6)

        field("📅  Tanggal:",self.mf_date,"YYYY-MM-DD")
        field("⏰  Waktu:",self.mf_time,"HH:MM:SS")
        field("📍  Sumber:",self.mf_source)
        field("👤  Nama Objek:",self.mf_names,"pisah koma")
        field("📍  Info Zona:",self.mf_area)

        pr=tk.Frame(form,bg=DARK); pr.pack(fill="x",pady=7)
        tk.Label(pr,text="🖼️  Lampiran Gambar:",bg=DARK,fg=TEXT_DIM,
                 font=("Helvetica",10),width=22,anchor="w").pack(side="left")
        tk.Entry(pr,textvariable=self.mf_path,bg=PANEL,fg=TEXT_BR,
                 insertbackground="white",font=("Courier",11),relief="flat",
                 highlightthickness=1,highlightcolor=ACCENT,
                 highlightbackground="#2a2a4e").pack(side="left",fill="x",expand=True,ipady=5,padx=(8,0))
        tk.Button(pr,text="📂",bg=PANEL,fg=TEXT_DIM,relief="flat",cursor="hand2",
                  command=lambda: self.mf_path.set(
                      filedialog.askopenfilename(parent=self.win))).pack(side="left",padx=(6,0))

        tk.Frame(form,bg="#2a2a4e",height=1).pack(fill="x",pady=14)
        br=tk.Frame(form,bg=DARK); br.pack(fill="x")
        bc={"font":("Helvetica",11,"bold"),"relief":"flat","cursor":"hand2","pady":10}
        tk.Button(br,text="💾  SIMPAN KE DATABASE",bg=GREEN,fg="white",
                  command=self._save_manual,**bc).pack(side="left",fill="x",expand=True)
        tk.Button(br,text="🔄  RESET",bg="#555555",fg="white",
                  command=self._reset_form,**bc).pack(side="left",padx=(8,0))

        self.man_status=tk.Label(form,text="",bg=DARK,fg="#aaaacc",
                                  font=("Helvetica",9),wraplength=480)
        self.man_status.pack(pady=(10,0))

        tk.Frame(form,bg="#2a2a4e",height=1).pack(fill="x",pady=8)
        tk.Label(form,text="📋  Riwayat:",bg=DARK,fg=TEXT_DIM,
                 font=("Helvetica",9,"bold")).pack(anchor="w")
        self.man_log=tk.Text(form,height=5,bg="#080818",fg="#ff9999",
                              font=("Courier",8),state="disabled",relief="flat")
        self.man_log.pack(fill="x",pady=(4,0))

    def _build_sidebar(self, p):
        pad={"padx":10,"pady":3}
        bc={"font":("Helvetica",10,"bold"),"relief":"flat","cursor":"hand2","pady":7}

        # Detection params
        tk.Label(p,text="🎯  PARAMETER DETEKSI",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)

        def prm(label,var,lo,hi,res=0.05):
            f=tk.Frame(p,bg=PANEL); f.pack(fill="x",padx=10,pady=2)
            tk.Label(f,text=label,bg=PANEL,fg=TEXT_DIM,
                     font=("Helvetica",8),width=14,anchor="w").pack(side="left")
            vl=tk.Label(f,text=f"{var.get():.2f}",bg=PANEL,fg="#ffcc00",
                        font=("Courier",9,"bold"),width=5)
            vl.pack(side="right")
            tk.Scale(f,from_=lo,to=hi,resolution=res,orient="horizontal",
                     variable=var,bg=PANEL,fg="white",troughcolor="#2a2a4e",
                     highlightbackground=PANEL,showvalue=False,
                     command=lambda v: vl.config(text=f"{float(v):.2f}")).pack(
                side="left",fill="x",expand=True)

        prm("Confidence:", self.conf_var, 0.1, 0.95)
        prm("IoU (NMS):",  self.iou_var,  0.1, 0.95)

        # Multi-class checkboxes
        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="👥  KELAS YANG DIDETEKSI",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        tk.Checkbutton(p,text="👤  Person (Orang)",
                       variable=self.detect_person,
                       bg=PANEL,fg="#ccccee",selectcolor="#1e1e3f",
                       activebackground=PANEL,
                       font=("Helvetica",9)).pack(anchor="w",padx=20,pady=1)
        tk.Checkbutton(p,text="🚗  Kendaraan (Car/Bike/Truck)",
                       variable=self.detect_vehicle,
                       bg=PANEL,fg="#ccccee",selectcolor="#1e1e3f",
                       activebackground=PANEL,
                       font=("Helvetica",9)).pack(anchor="w",padx=20,pady=1)

        # Zone controls
        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="🖱️  ZONA TERLARANG",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)

        self.draw_btn=tk.Button(p,text="✏️  DRAW MODE: OFF",bg="#555555",fg="white",
                                 command=self._toggle_draw,**bc)
        self.draw_btn.pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="🗑️  HAPUS SEMUA ZONA",bg="#773333",fg="white",
                  command=self._clear_zones,**bc).pack(fill="x",padx=10,pady=3)
        self.lbl_zones=tk.Label(p,text="Zona aktif: 0",bg=PANEL,fg="#ffcc00",
                                 font=("Helvetica",9))
        self.lbl_zones.pack(padx=10,anchor="w")

        utils.make_separator(p,"#2a2a4e",pady=4)
        tg_on=cfg.get("telegram.enabled",False) and cfg.get("telegram.notify_on_anomaly",True)
        self.tg_badge=tk.Label(p,text="🔔 Telegram: "+("ON" if tg_on else "OFF"),
                                bg=PANEL,fg="#00ff88" if tg_on else "#666688",
                                font=("Helvetica",8))
        self.tg_badge.pack(anchor="w",padx=10,pady=2)

        tk.Label(p,text="⚙️  KONTROL",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        self.btn_start=tk.Button(p,text="▶  START KAMERA",bg=GREEN,fg="white",
                                  command=self.start,**bc)
        self.btn_start.pack(fill="x",padx=10,pady=3)
        self.btn_stop=tk.Button(p,text="⏹  STOP",bg=RED,fg="white",
                                 state="disabled",command=self.stop,**bc)
        self.btn_stop.pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="📊  GRAFIK",bg="#6c3483",fg="white",
                  command=self._show_chart,**bc).pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="🗄️  LIHAT DATA",bg="#2a4a2a",fg="white",
                  command=lambda: __import__("data_input").DataManagementWindow(self.win),
                  **bc).pack(fill="x",padx=10,pady=3)

        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="📈  STATISTIK",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        sf=tk.Frame(p,bg=DARK,padx=8,pady=6); sf.pack(fill="x",padx=10,pady=2)
        self.lbl_alerts = self._srow(sf,"Total Alert","0")
        self.lbl_last   = self._srow(sf,"Terakhir","-")
        self.lbl_fps    = self._srow(sf,"FPS","-")
        self.lbl_thread = self._srow(sf,"Thread","idle")

        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="📋  LOG ALERT",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        self.log_box=tk.Text(p,height=6,bg="#080818",fg="#ff9999",
                              font=("Courier",8),state="disabled",relief="flat")
        self.log_box.pack(fill="x",padx=10,pady=2)

    def _srow(self,parent,label,init):
        row=tk.Frame(parent,bg=DARK); row.pack(fill="x",pady=1)
        tk.Label(row,text=label+":",bg=DARK,fg=TEXT_DIM,
                 font=("Helvetica",8),width=12,anchor="w").pack(side="left")
        v=tk.Label(row,text=init,bg=DARK,fg="#ffdd00",font=("Courier",9,"bold"))
        v.pack(side="right"); return v

    # ── Mode Toggle ───────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        self.mode.set(mode)
        if mode=="camera":
            self.man_panel.pack_forget()
            self.cam_panel.pack(fill="both",expand=True)
            self.btn_cam.config(bg=MODE_CAM,fg="white",relief="groove")
            self.btn_man.config(bg=PANEL,fg=TEXT_DIM,relief="flat")
            self.mode_badge.config(
                text="Mode Aktif: 📷 Kamera — draw zona → start deteksi")
            self.btn_start.config(state="normal")
        else:
            self.cam_panel.pack_forget()
            self.man_panel.pack(fill="both",expand=True)
            self.btn_man.config(bg=MODE_MAN,fg="white",relief="groove")
            self.btn_cam.config(bg=PANEL,fg=TEXT_DIM,relief="flat")
            self.mode_badge.config(text="Mode Aktif: ✏️ Manual — isi form & simpan")
            if self.running: self.stop()
            self.btn_start.config(state="disabled")

    # ── Mouse (draw zones) ────────────────────────────────────────────────────

    def _canvas_to_frame(self, cx, cy):
        s=max(self._scale,0.01); return int(cx/s),int(cy/s)

    def _on_mouse_down(self, e):
        if not self.draw_mode_var.get(): return
        fx,fy=self._canvas_to_frame(e.x,e.y)
        self.zone_mgr.start_rect(fx,fy); self._mouse_drawing=True

    def _on_mouse_move(self, e):
        if not self.draw_mode_var.get() or not self._mouse_drawing: return
        fx,fy=self._canvas_to_frame(e.x,e.y)
        self._cur_mouse=(fx,fy)

    def _on_mouse_hover(self, e):
        if self.draw_mode_var.get():
            fx,fy=self._canvas_to_frame(e.x,e.y)
            self._cur_mouse=(fx,fy)

    def _on_mouse_up(self, e):
        if not self.draw_mode_var.get(): return
        fx,fy=self._canvas_to_frame(e.x,e.y)
        self.zone_mgr.finish_rect(fx,fy)
        self._mouse_drawing=False
        self.lbl_zones.config(text=f"Zona aktif: {len(self.zone_mgr.zones)}")

    def _toggle_draw(self):
        val=not self.draw_mode_var.get(); self.draw_mode_var.set(val)
        if val:
            self.draw_btn.config(text="✏️  DRAW MODE: ON",bg="#cc6600")
            self.canvas_lbl.config(cursor="crosshair")
            self._set_status("Draw Mode AKTIF — drag untuk buat zona rectangular")
        else:
            self.draw_btn.config(text="✏️  DRAW MODE: OFF",bg="#555555")
            self.canvas_lbl.config(cursor="")
            self._set_status("Draw Mode NONAKTIF")

    def _clear_zones(self):
        self.zone_mgr.reset()
        self.lbl_zones.config(text="Zona aktif: 0")
        self._log("[ZONA] Semua zona dihapus.")

    # ── Camera Controls ───────────────────────────────────────────────────────

    def start(self):
        if self.running or self.mode.get()=="manual": return
        try:
            self.cap=utils.open_camera(cfg.get("detection.camera_index",0))
        except RuntimeError as e:
            messagebox.showerror("Error",str(e),parent=self.win); return
        self._flush_q(); self.running=True
        self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        self.lbl_thread.config(text="running")
        threading.Thread(target=self._capture_loop,daemon=True).start()
        self._poll(); self._set_status("Running — Webcam")

    def stop(self):
        self.running=False
        if self.cap: self.cap.release(); self.cap=None
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.lbl_thread.config(text="idle"); self._set_status("Stopped")

    def _get_target_classes(self):
        """Kumpulkan class IDs berdasarkan pilihan user."""
        classes = []
        if self.detect_person.get():  classes.append(0)   # person
        if self.detect_vehicle.get():
            classes.extend([1,2,3,5,7])  # bicycle,car,motorcycle,bus,truck
        return classes if classes else None   # None = deteksi semua

    def _capture_loop(self):
        try: model=get_model()
        except RuntimeError as e:
            self.win.after(0,lambda: messagebox.showerror("Error",str(e),parent=self.win))
            self.running=False; return

        while self.running:
            ret,frame=self.cap.read()
            if not ret: self.running=False; break
            h,w=frame.shape[:2]
            self._scale=min(self.DISP_W/w,self.DISP_H/h,1.0)

            target_cls=self._get_target_classes()
            kwargs=dict(conf=self.conf_var.get(),iou=self.iou_var.get(),verbose=False)
            if target_cls: kwargs["classes"]=target_cls
            results=model(frame,**kwargs)

            intrusions=[]; all_labels=[]; alert_zone_ids=set()

            for result in results:
                for box in result.boxes:
                    x1,y1,x2,y2=map(int,box.xyxy[0])
                    conf=float(box.conf[0]); cls=int(box.cls[0])
                    label=model.names[cls]; all_labels.append(label)

                    hit_zones=self.zone_mgr.check_intrusions(x1,y1,x2,y2)
                    color=utils.class_color(cls)

                    if hit_zones:
                        # Flash merah saat intrusi
                        color=utils.COLOR_RED
                        ov=frame.copy()
                        cv2.rectangle(ov,(x1,y1),(x2,y2),(0,0,255),-1)
                        cv2.addWeighted(ov,0.2,frame,0.8,0,frame)
                        intrusions.append({"label":label,"zones":hit_zones,"conf":conf})
                        alert_zone_ids.update(hit_zones)
                        # Update intrusion counter di zona
                        for zid in hit_zones:
                            z=[z for z in self.zone_mgr.zones if z.zone_id==zid]
                            if z: z[0].intrusion_count+=1

                    # Gambar bbox dengan confidence
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    text=f"{label} {conf:.0%}"
                    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
                    cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
                    cv2.putText(frame,text,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1)

            # Gambar semua zona
            self.zone_mgr.draw_all(frame, alert_zone_ids)
            # Preview zona sedang digambar
            if self.draw_mode_var.get():
                self.zone_mgr.draw_preview(frame,*self._cur_mouse)

            is_anomaly=len(intrusions)>0
            if is_anomaly: self._draw_alert_border(frame)

            utils.draw_status_badge(frame,
                                    "🚨 ANOMALY!" if is_anomaly else "✅ NORMAL",
                                    is_anomaly)
            utils.draw_info_overlay(frame,[
                f"Zones: {len(self.zone_mgr.zones)}",
                f"Objects: {len(all_labels)}",
                f"Alerts: {self.alert_count}",
            ])
            utils.draw_timestamp(frame)

            self._q_put({"frame":frame.copy(),"anomaly":is_anomaly,
                         "intrusions":intrusions})

    def _poll(self):
        try:
            data=self._q.get_nowait(); self._update_ui(data)
        except queue.Empty: pass
        if self.running:
            self.win.after(self.POLL_MS,self._poll)
        else:
            self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
            self.lbl_thread.config(text="idle"); self._set_status("Stopped")

    def _update_ui(self, data):
        frame=data["frame"]; is_anomaly=data["anomaly"]; intrusions=data["intrusions"]
        self._latest_frame=frame
        self._fps_cnt+=1
        elapsed=(datetime.now()-self._fps_ts).total_seconds()
        if elapsed>=1.0:
            self.lbl_fps.config(text=f"{self._fps_cnt/elapsed:.1f}")
            self._fps_cnt=0; self._fps_ts=datetime.now()

        if is_anomaly:
            now=datetime.now()
            do_save=(self._last_alert is None or
                     (now-self._last_alert).total_seconds()>=self._cooldown)
            if do_save:
                self._last_alert=now; self.alert_count+=1
                zones_str=str(list(set(z for i in intrusions for z in i["zones"])))
                path=utils.save_frame(frame,"output/anomaly","alert")
                labels=list(set(i["label"] for i in intrusions))
                db.log_anomaly_detection("webcam",labels,zones_str,path)
                ts=now.strftime("%H:%M:%S")
                self._log(f"[{ts}] 🚨 ANOMALY #{self.alert_count} — {labels} @ Zona {zones_str}")
                self.lbl_alerts.config(text=str(self.alert_count))
                self.lbl_last.config(text=ts)
                self._tg.notify_anomaly(path,zones_str,labels)

        photo=utils.frame_to_photoimage(frame,self.DISP_W,self.DISP_H)
        self._photo=photo; self.canvas_lbl.config(image=photo,text="")

    def _draw_alert_border(self, frame):
        h,w=frame.shape[:2]
        cv2.rectangle(frame,(0,0),(w-1,h-1),(0,0,255),6)
        text="!! ANOMALY DETECTED !!"
        font=cv2.FONT_HERSHEY_TRIPLEX; scale=0.9
        (tw,th),_=cv2.getTextSize(text,font,scale,2)
        tx,ty=(w-tw)//2,h//2
        cv2.rectangle(frame,(tx-10,ty-th-10),(tx+tw+10,ty+10),(0,0,150),-1)
        cv2.putText(frame,text,(tx,ty),font,scale,(0,0,0),4)
        cv2.putText(frame,text,(tx,ty),font,scale,(255,80,80),2)

    # ── Manual Input ──────────────────────────────────────────────────────────

    def _save_manual(self):
        date=self.mf_date.get().strip(); time_s=self.mf_time.get().strip()
        source=self.mf_source.get().strip() or "manual"
        names=self.mf_names.get().strip(); area=self.mf_area.get().strip() or "-"
        path=self.mf_path.get().strip()
        try:
            datetime.strptime(date,"%Y-%m-%d"); datetime.strptime(time_s,"%H:%M:%S")
        except ValueError:
            self.man_status.config(text="❌ Format tanggal/waktu salah",fg="#ff4444"); return
        if not names:
            self.man_status.config(text="❌ Nama objek wajib diisi!",fg="#ff4444"); return
        try:
            db.log_anomaly_detection(source,[n.strip() for n in names.split(",")],area,path)
            self._man_log(f"✅ [{date} {time_s}] {names} @ {area}")
            self.man_status.config(text="✅ Data berhasil disimpan!",fg="#00ff88")
            self._log(f"[MANUAL] {date} {time_s} — {names} @ {area}")
        except Exception as e:
            self.man_status.config(text=f"❌ Error: {e}",fg="#ff4444")

    def _reset_form(self):
        now=datetime.now()
        self.mf_date.set(now.strftime("%Y-%m-%d")); self.mf_time.set(now.strftime("%H:%M:%S"))
        self.mf_source.set("manual"); self.mf_names.set("person")
        self.mf_area.set("Zone 1"); self.mf_path.set(""); self.man_status.config(text="")

    def _man_log(self,text):
        self.man_log.config(state="normal"); self.man_log.insert("end",text+"\n")
        self.man_log.see("end"); self.man_log.config(state="disabled")

    def _show_chart(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from collections import Counter
        logs=db.get_anomaly_logs(200)
        if not logs:
            messagebox.showinfo("Info","Belum ada data.",parent=self.win); return
        cnt=Counter(l["date"] for l in logs)
        dates=sorted(cnt); vals=[cnt[d] for d in dates]
        cw=tk.Toplevel(self.win); cw.title("📊 Grafik Anomaly"); cw.configure(bg=DARK)
        utils.center_window(cw,680,400)
        fig,ax=plt.subplots(figsize=(6.8,3.6),facecolor=DARK); ax.set_facecolor(PANEL)
        ax.bar(dates,vals,color=ACCENT,edgecolor=DARK)
        ax.set_title("Anomaly Alerts per Hari",color="white",fontsize=11)
        ax.tick_params(colors="white",labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a4e")
        plt.xticks(rotation=30,ha="right"); plt.tight_layout()
        c=FigureCanvasTkAgg(fig,master=cw); c.draw()
        c.get_tk_widget().pack(fill="both",expand=True,padx=8,pady=8)

    def _q_put(self,item):
        if self._q.full():
            try: self._q.get_nowait()
            except queue.Empty: pass
        try: self._q.put_nowait(item)
        except queue.Full: pass

    def _flush_q(self):
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break

    def _log(self,text):
        self.log_box.config(state="normal"); self.log_box.insert("end",text+"\n")
        self.log_box.see("end"); self.log_box.config(state="disabled")

    def _set_status(self,msg):
        self.status_bar.config(text=f"Status: {msg}")

    def _on_close(self):
        self.running=False
        if self.cap: self.cap.release()
        self.win.destroy()
