"""
image_detection.py - Smart Vision Analysis System
Modul 1: Image Detection (YOLO v8) — Akurasi Tinggi
Peningkatan: model selector, object tracking, NMS tuning, class filter, confidence per frame
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
from window_utils import get_module_sizes
import custom_objects as co

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

DARK = "#0d0d1a"; PANEL = "#13132b"; HEADER = "#1a1a2e"
ACCENT = "#00d4ff"; GREEN = "#00aa44"; RED = "#aa2222"
TEXT_DIM = "#6666aa"; TEXT_BR = "#eeeeff"
MODE_CAM = "#1a3a5e"; MODE_MAN = "#3a1a5e"

# YOLO model options
MODELS = {
    "YOLOv8 Nano (Cepat)": "yolov8n.pt",
    "YOLOv8 Small (Seimbang)": "yolov8s.pt",
    "YOLOv8 Medium (Akurat)": "yolov8m.pt",

    "YOLO26 Nano": "yolo26n.pt",
    "YOLO26 Small": "yolo26s.pt",
    "YOLO26 Medium": "yolo26m.pt",
}

_models_cache = {}

def get_model(model_file: str):
    global _models_cache
    if model_file not in _models_cache:
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics belum terpasang. pip install ultralytics")
        _models_cache[model_file] = YOLO(model_file)
    return _models_cache[model_file]


def detect_objects(frame: np.ndarray, model,
                   conf: float = 0.45,
                   iou: float = 0.45,
                   target_classes: list = None) -> tuple:
    """
    Deteksi objek dengan YOLO.
    - iou: threshold NMS (lebih kecil = lebih sedikit duplikat)
    - target_classes: None = semua, list int = filter kelas tertentu
    """
    kwargs = dict(conf=conf, iou=iou, verbose=False)
    if target_classes:
        kwargs["classes"] = target_classes

    results = model(frame, **kwargs)
    dets = []
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            c = float(box.conf[0]); cls = int(box.cls[0])
            label = model.names[cls]
            color = utils.class_color(cls)
            # Gambar bbox lebih tebal & rapi
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            # Label background
            text = f"{label} {c:.0%}"
            (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, text, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1)
            dets.append({"label":label,"confidence":c,"bbox":(x1,y1,x2,y2),"cls":cls})
    return frame, dets


def summarize(dets):
    s = {}
    for d in dets: s[d["label"]] = s.get(d["label"],0)+1
    return s


class ImageDetectionWindow:
    POLL_MS = 25   # ~40fps display

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("📦 Modul 1 – Image Detection (YOLOv8 / YOLO26)")
        self.win.configure(bg=DARK)
        sz = get_module_sizes(parent)
        self.WIN_W=sz["win_w"]; self.WIN_H=sz["win_h"]
        self.DISP_W=sz["disp_w"]; self.DISP_H=sz["disp_h"]
        utils.center_window(self.win, self.WIN_W, self.WIN_H)
        self.win.resizable(True, True); self.win.minsize(700,500)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._q: queue.Queue = queue.Queue(maxsize=3)
        self.cap = None; self.running = False
        self.mode = tk.StringVar(value="camera")

        # Detection params
        self.model_var  = tk.StringVar(value=list(MODELS.keys())[0])
        self.conf_var   = tk.DoubleVar(value=0.45)
        self.iou_var    = tk.DoubleVar(value=0.45)
        self.source_cam = tk.StringVar(value="webcam")

        # Stats
        self.total_detected = 0
        self.session_objects = []
        self._photo = None
        self._latest_frame = None
        self._fps_cnt = 0; self._fps_ts = datetime.now()
        self._frame_count = 0
        # Per-frame object history (untuk tracking sederhana)
        self._obj_history = defaultdict(int)

        # Manual form
        self.mf_source = tk.StringVar(value="manual")
        self.mf_total  = tk.IntVar(value=1)
        self.mf_names  = tk.StringVar(value="")
        self.mf_path   = tk.StringVar(value="")

        # Custom objects
        self._alias_map: dict = co.build_alias_map()
        self._tmpl_detector: co.TemplateDetector = co.TemplateDetector()

        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self.win, bg=HEADER, height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="📦  IMAGE DETECTION  —  YOLOv8 / YOLO26",
                 font=("Helvetica",13,"bold"), bg=HEADER, fg=ACCENT).pack(
            side="left", padx=16, pady=14)

        toggle = tk.Frame(self.win, bg="#111133", height=46)
        toggle.pack(fill="x"); toggle.pack_propagate(False)
        tk.Label(toggle, text="MODE:", bg="#111133", fg=TEXT_DIM,
                 font=("Helvetica",9,"bold")).pack(side="left",padx=14,pady=12)
        self.btn_cam = tk.Button(toggle, text="📷  KAMERA / FILE",
                                  font=("Helvetica",10,"bold"), relief="flat",
                                  cursor="hand2", padx=18, pady=6,
                                  command=lambda: self._set_mode("camera"))
        self.btn_cam.pack(side="left", padx=4, pady=6)
        self.btn_man = tk.Button(toggle, text="✏️  INPUT MANUAL",
                                  font=("Helvetica",10,"bold"), relief="flat",
                                  cursor="hand2", padx=18, pady=6,
                                  command=lambda: self._set_mode("manual"))
        self.btn_man.pack(side="left", padx=4, pady=6)
        self.mode_badge = tk.Label(toggle, text="", bg="#111133", fg=ACCENT,
                                    font=("Helvetica",8,"italic"))
        self.mode_badge.pack(side="left", padx=10)

        body = tk.Frame(self.win, bg=DARK)
        body.pack(fill="both", expand=True, padx=10, pady=6)
        self.left = tk.Frame(body, bg=DARK)
        self.left.pack(side="left", fill="both", expand=True)

        # Camera panel
        self.cam_panel = tk.Frame(self.left, bg=DARK)
        self.canvas = tk.Label(self.cam_panel, bg="#111122",
                                text="[ No Input ]", font=("Courier",12), fg="#444466")
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.status_bar = tk.Label(self.cam_panel, text="Status: Idle",
                                    bg=HEADER, fg="#aaaacc",
                                    font=("Helvetica",9), anchor="w")
        self.status_bar.pack(fill="x", padx=4, pady=(0,2))

        # Manual panel
        self.man_panel = tk.Frame(self.left, bg=DARK)
        self._build_manual_panel()

        # Sidebar
        right = tk.Frame(body, bg=PANEL, width=275)
        right.pack(side="right", fill="y", padx=(8,2), pady=2)
        right.pack_propagate(False)
        self._build_sidebar(right)

        self._set_mode("camera")

    def _build_manual_panel(self):
        p = self.man_panel
        hdr = tk.Frame(p, bg="#1a1a3e", height=42); hdr.pack(fill="x",pady=(4,0))
        tk.Label(hdr, text="✏️  INPUT DATA DETEKSI OBJEK SECARA MANUAL",
                 font=("Helvetica",11,"bold"), bg="#1a1a3e", fg=ACCENT).pack(side="left",padx=16,pady=10)
        form = tk.Frame(p, bg=DARK, padx=30, pady=20)
        form.pack(fill="both", expand=True)
        now = datetime.now()
        self.mf_date = tk.StringVar(value=now.strftime("%Y-%m-%d"))
        self.mf_time = tk.StringVar(value=now.strftime("%H:%M:%S"))

        def field(label, var, hint="", spin=False):
            row=tk.Frame(form,bg=DARK); row.pack(fill="x",pady=7)
            tk.Label(row,text=label,bg=DARK,fg=TEXT_DIM,font=("Helvetica",10),
                     width=22,anchor="w").pack(side="left")
            if spin:
                w=tk.Spinbox(row,textvariable=var,from_=0,to=9999,width=10,
                             bg=PANEL,fg=TEXT_BR,insertbackground="white",
                             buttonbackground=PANEL,font=("Courier",11),relief="flat",
                             highlightthickness=1,highlightcolor=ACCENT,highlightbackground="#2a2a4e")
            else:
                w=tk.Entry(row,textvariable=var,bg=PANEL,fg=TEXT_BR,
                           insertbackground="white",font=("Courier",11),relief="flat",
                           highlightthickness=1,highlightcolor=ACCENT,highlightbackground="#2a2a4e")
            w.pack(side="left",fill="x",expand=True,ipady=5,padx=(8,0))
            if hint: tk.Label(row,text=hint,bg=DARK,fg="#444466",font=("Helvetica",8)).pack(side="left",padx=6)

        field("📅  Tanggal:", self.mf_date, "YYYY-MM-DD")
        field("⏰  Waktu:",   self.mf_time, "HH:MM:SS")
        field("📍  Sumber:",  self.mf_source)
        field("🔢  Jml Objek:", self.mf_total, spin=True)
        field("🏷️  Nama Objek:", self.mf_names, "pisah koma")

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
                  command=self._reset_manual_form,**bc).pack(side="left",padx=(8,0))

        self.man_status=tk.Label(form,text="",bg=DARK,fg="#aaaacc",
                                  font=("Helvetica",9),wraplength=480)
        self.man_status.pack(pady=(10,0))

        tk.Frame(form,bg="#2a2a4e",height=1).pack(fill="x",pady=8)
        tk.Label(form,text="📋  Riwayat:",bg=DARK,fg=TEXT_DIM,
                 font=("Helvetica",9,"bold")).pack(anchor="w")
        self.man_log=tk.Text(form,height=5,bg="#080818",fg="#aaffaa",
                              font=("Courier",8),state="disabled",relief="flat")
        self.man_log.pack(fill="x",pady=(4,0))

    def _build_sidebar(self, p):
        pad={"padx":10,"pady":3}
        bc={"font":("Helvetica",10,"bold"),"relief":"flat","cursor":"hand2","pady":7}

        # Model selector
        tk.Label(p,text="🤖  MODEL YOLO",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        ttk.Combobox(p,textvariable=self.model_var,
                     values=list(MODELS.keys()),
                     state="readonly",width=28).pack(padx=10,pady=2)

        utils.make_separator(p,"#2a2a4e",pady=4)

        # Source
        tk.Label(p,text="🎥  SUMBER INPUT",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        for val,txt in [("webcam","🎥  Webcam"),("image","🖼️  File Gambar")]:
            tk.Radiobutton(p,text=txt,variable=self.source_cam,value=val,
                           bg=PANEL,fg="#ccccee",selectcolor="#1e1e3f",
                           activebackground=PANEL,
                           font=("Helvetica",9)).pack(anchor="w",padx=20,pady=1)

        utils.make_separator(p,"#2a2a4e",pady=4)

        # Confidence & IOU
        tk.Label(p,text="🎯  PARAMETER DETEKSI",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)

        def param_row(label, var, lo, hi, res):
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

        param_row("Confidence:", self.conf_var, 0.1, 0.95, 0.05)
        param_row("IoU (NMS):",  self.iou_var,  0.1, 0.95, 0.05)

        # Hint
        tk.Label(p,text="↑ Conf lebih tinggi = lebih selektif\n↓ IoU lebih kecil = kurangi duplikat",
                 bg=PANEL,fg="#446688",font=("Helvetica",7),justify="left").pack(padx=10,anchor="w")

        utils.make_separator(p,"#2a2a4e",pady=4)

        # Controls
        tk.Label(p,text="⚙️  KONTROL",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        self.btn_start=tk.Button(p,text="▶  START",bg=GREEN,fg="white",
                                  command=self.start,**bc)
        self.btn_start.pack(fill="x",padx=10,pady=3)
        self.btn_stop=tk.Button(p,text="⏹  STOP",bg=RED,fg="white",
                                 state="disabled",command=self.stop,**bc)
        self.btn_stop.pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="📷  SCREENSHOT",bg="#1a5276",fg="white",
                  command=self._screenshot,**bc).pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="📊  GRAFIK",bg="#6c3483",fg="white",
                  command=self._show_chart,**bc).pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="🗄️  LIHAT DATA",bg="#2a4a2a",fg="white",
                  command=lambda: __import__("data_input").DataManagementWindow(self.win),
                  **bc).pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="🎯  OBJEK KUSTOM",bg="#3a2a5e",fg="white",
                  command=self._open_custom_manager,**bc).pack(fill="x",padx=10,pady=3)

        utils.make_separator(p,"#2a2a4e",pady=4)

        # Stats
        tk.Label(p,text="📈  STATISTIK",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        sf=tk.Frame(p,bg=DARK,padx=8,pady=6); sf.pack(fill="x",padx=10,pady=2)
        self.lbl_total  = self._srow(sf,"Total Objek","0")
        self.lbl_fps    = self._srow(sf,"FPS","-")
        self.lbl_conf_avg = self._srow(sf,"Avg Conf","-")
        self.lbl_thread = self._srow(sf,"Thread","idle")

        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="📋  LOG",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        self.log_box=tk.Text(p,height=7,bg="#080818",fg="#aaffaa",
                              font=("Courier",8),state="disabled",relief="flat")
        self.log_box.pack(fill="x",padx=10,pady=2)

    def _srow(self,parent,label,init):
        row=tk.Frame(parent,bg=DARK); row.pack(fill="x",pady=1)
        tk.Label(row,text=label+":",bg=DARK,fg=TEXT_DIM,
                 font=("Helvetica",8),width=12,anchor="w").pack(side="left")
        v=tk.Label(row,text=init,bg=DARK,fg="#ffdd00",font=("Courier",9,"bold"))
        v.pack(side="right"); return v

    def _set_mode(self, mode: str):
        self.mode.set(mode)
        if mode=="camera":
            self.man_panel.pack_forget()
            self.cam_panel.pack(fill="both",expand=True)
            self.btn_cam.config(bg=MODE_CAM,fg="white",relief="groove")
            self.btn_man.config(bg=PANEL,fg=TEXT_DIM,relief="flat")
            self.mode_badge.config(text="Mode Aktif: 📷 Kamera — deteksi objek real-time")
            self.btn_start.config(state="normal")
        else:
            self.cam_panel.pack_forget()
            self.man_panel.pack(fill="both",expand=True)
            self.btn_man.config(bg=MODE_MAN,fg="white",relief="groove")
            self.btn_cam.config(bg=PANEL,fg=TEXT_DIM,relief="flat")
            self.mode_badge.config(text="Mode Aktif: ✏️ Manual — isi form & simpan")
            if self.running: self.stop()
            self.btn_start.config(state="disabled")

    def start(self):
        if self.running or self.mode.get()=="manual": return
        src = self.source_cam.get()
        if src=="image":
            self._process_image_file(); return
        try:
            self.cap = utils.open_camera(cfg.get("detection.camera_index",0))
        except RuntimeError as e:
            messagebox.showerror("Error",str(e),parent=self.win); return
        self.running=True; self.total_detected=0; self.session_objects=[]
        self._frame_count=0; self._obj_history.clear()
        self._flush_q()
        self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        self.lbl_thread.config(text="running")
        threading.Thread(target=self._capture_loop,daemon=True).start()
        self._poll()
        self._set_status("Running — Webcam")

    def stop(self):
        self.running=False
        if self.cap: self.cap.release(); self.cap=None
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.lbl_thread.config(text="idle"); self._set_status("Stopped")
        if self.session_objects and self._latest_frame is not None:
            path=utils.save_frame(self._latest_frame,"output/object","session_end")
            db.log_object_detection("webcam",self.total_detected,self.session_objects,path)

    def _capture_loop(self):
        model_file = MODELS[self.model_var.get()]
        try:
            model = get_model(model_file)
        except RuntimeError as e:
            self.win.after(0,lambda: messagebox.showerror("Error",str(e),parent=self.win))
            self.running=False; return

        while self.running:
            ret,frame = self.cap.read()
            if not ret: self.running=False; break
            self. self._latest_frame = frame.copy() 
            self._frame_count += 1
            

            # Proses setiap frame (tidak skip) untuk akurasi maksimal
            try:
                proc, dets = detect_objects(
                    frame, model,
                    conf=self.conf_var.get(),
                    iou=self.iou_var.get()
                )
            except Exception as e:
                self.win.after(0,lambda: self._log(f"[ERROR] {e}"))
                self.running=False; break

            # Terapkan alias map ke label YOLO
            alias = self._alias_map
            for d in dets:
                if d["label"] in alias:
                    d["label"] = alias[d["label"]]

            # Jalankan template detector untuk objek kustom
            tmpl_dets = self._tmpl_detector.detect(frame)
            if tmpl_dets:
                co.draw_custom_detections(proc, tmpl_dets)
                dets = dets + tmpl_dets

            sm = summarize(dets)
            # Avg confidence
            avg_conf = (sum(d["confidence"] for d in dets)/len(dets)) if dets else 0

            utils.draw_info_overlay(proc, [
                f"Model: {model_file}",
                f"Objects: {len(dets)}",
                f"Conf avg: {avg_conf:.0%}",
                f"Frame: {self._frame_count}",
            ] + [f"  {k}: {v}" for k,v in list(sm.items())[:4]])
            utils.draw_timestamp(proc)
            utils.draw_status_badge(proc, f"DETECTED: {len(dets)}", len(dets)>0)

            self._q_put({"frame":proc,"dets":dets,"sm":sm,"avg_conf":avg_conf})

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
        frame=data["frame"]; dets=data["dets"]; sm=data["sm"]
        avg_conf=data.get("avg_conf",0)
        self._latest_frame=frame
        self._fps_cnt+=1
        elapsed=(datetime.now()-self._fps_ts).total_seconds()
        if elapsed>=1.0:
            self.lbl_fps.config(text=f"{self._fps_cnt/elapsed:.1f}")
            self._fps_cnt=0; self._fps_ts=datetime.now()
        self.total_detected+=len(dets)
        for d in dets:
            if d["label"] not in self.session_objects:
                self.session_objects.append(d["label"])
        self.lbl_total.config(text=str(self.total_detected))
        if avg_conf>0:
            self.lbl_conf_avg.config(text=f"{avg_conf:.0%}")
        if dets:
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] {len(dets)} — {', '.join(sm.keys())}")
        photo=utils.frame_to_photoimage(frame,self.DISP_W,self.DISP_H)
        self._photo=photo; self.canvas.config(image=photo,text="")

    def _process_image_file(self):
        path=filedialog.askopenfilename(
            parent=self.win,title="Pilih Gambar",
            filetypes=[("Image","*.jpg *.jpeg *.png *.bmp *.webp"),("All","*.*")])
        if not path: return
        try: frame=utils.read_image(path)
        except ValueError as e:
            messagebox.showerror("Error",str(e),parent=self.win); return
        self._set_status("Processing…"); self.win.update()
        def _run():
            try:
                model=get_model(MODELS[self.model_var.get()])
                rf,dets=detect_objects(frame,model,self.conf_var.get(),self.iou_var.get())
                # Alias
                alias = self._alias_map
                for d in dets:
                    if d["label"] in alias:
                        d["label"] = alias[d["label"]]
                # Template
                tmpl_dets = self._tmpl_detector.detect(frame)
                if tmpl_dets:
                    co.draw_custom_detections(rf, tmpl_dets)
                    dets = dets + tmpl_dets
                sm=summarize(dets)
                utils.draw_info_overlay(rf,[f"File: {os.path.basename(path)}",f"Objects: {len(dets)}"])
                utils.draw_timestamp(rf); utils.draw_status_badge(rf,f"DETECTED: {len(dets)}",len(dets)>0)
                out=utils.save_frame(rf,"output/object","img_detect")
                db.log_object_detection(path,len(dets),list(sm.keys()),out)
                self.win.after(0,lambda: self._show_img_result(rf,len(dets),sm,out,path))
            except Exception as e:
                self.win.after(0,lambda: messagebox.showerror("Error",str(e),parent=self.win))
        threading.Thread(target=_run,daemon=True).start()

    def _show_img_result(self,frame,count,sm,out,orig):
        self._latest_frame=frame
        photo=utils.frame_to_photoimage(frame,self.DISP_W,self.DISP_H)
        self._photo=photo; self.canvas.config(image=photo,text="")
        self.lbl_total.config(text=str(count))
        summary_str="\n".join(f"  {k}: {v}" for k,v in sm.items())
        self._log(f"[IMAGE] {os.path.basename(orig)} → {count} objek")
        self._set_status(f"Selesai — {count} objek. Disimpan: {out}")
        messagebox.showinfo("Hasil",f"{count} objek:\n{summary_str}\n\n→ {out}",parent=self.win)

    def _open_custom_manager(self):
        def _on_reload():
            self._alias_map = co.build_alias_map()
            self._tmpl_detector.reload()
            self._log("[KUSTOM] Data objek kustom diperbarui")
        co.CustomObjectManagerWindow(self.win, on_reload_callback=_on_reload)

    def _save_manual(self):
        date=self.mf_date.get().strip(); time_s=self.mf_time.get().strip()
        source=self.mf_source.get().strip() or "manual"
        total=int(self.mf_total.get()); names=self.mf_names.get().strip()
        path=self.mf_path.get().strip()
        try:
            datetime.strptime(date,"%Y-%m-%d"); datetime.strptime(time_s,"%H:%M:%S")
        except ValueError:
            self.man_status.config(text="❌ Format tanggal/waktu salah",fg="#ff4444"); return
        if not names:
            self.man_status.config(text="❌ Nama objek wajib diisi!",fg="#ff4444"); return
        try:
            db.log_object_detection(source,total,[n.strip() for n in names.split(",")],path)
            self._man_log(f"✅ [{date} {time_s}] {total} obj ({names})")
            self.man_status.config(text="✅ Data berhasil disimpan!",fg="#00ff88")
            self._log(f"[MANUAL] {date} {time_s} — {total} obj: {names}")
        except Exception as e:
            self.man_status.config(text=f"❌ Error: {e}",fg="#ff4444")

    def _reset_manual_form(self):
        now=datetime.now()
        self.mf_date.set(now.strftime("%Y-%m-%d")); self.mf_time.set(now.strftime("%H:%M:%S"))
        self.mf_source.set("manual"); self.mf_total.set(1)
        self.mf_names.set(""); self.mf_path.set(""); self.man_status.config(text="")

    def _man_log(self,text):
        self.man_log.config(state="normal"); self.man_log.insert("end",text+"\n")
        self.man_log.see("end"); self.man_log.config(state="disabled")

    def _screenshot(self):
        if self._latest_frame is None:
            messagebox.showwarning("Info","Tidak ada frame.",parent=self.win); return
        path=utils.screenshot(self._latest_frame); self._log(f"[SHOT] {path}")

    def _show_chart(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        data=db.get_object_detection_chart_data()
        if not data:
            messagebox.showinfo("Info","Belum ada data.",parent=self.win); return
        dates=[r[0] for r in data]; totals=[r[1] for r in data]
        cw=tk.Toplevel(self.win); cw.title("📊 Grafik"); cw.configure(bg=DARK)
        utils.center_window(cw,680,400)
        fig,ax=plt.subplots(figsize=(6.8,3.6),facecolor=DARK); ax.set_facecolor(PANEL)
        ax.bar(dates,totals,color=ACCENT,edgecolor=DARK)
        ax.set_title("Total Objek per Hari",color="white",fontsize=11)
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
