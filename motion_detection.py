"""
motion_detection.py - Smart Vision Analysis System
Modul 2: Motion Detection — Akurasi Tinggi
Peningkatan: adaptive threshold, shadow removal, morphological filtering,
             minimum object size, cooldown adaptif, dual-mode comparison
"""

import cv2
import numpy as np
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import utils
import database as db
import config as cfg
import notifier
from window_utils import get_module_sizes

DARK="#0d0d1a"; PANEL="#13132b"; HEADER="#1a1a2e"
ACCENT="#ff9900"; GREEN="#00aa44"; RED="#aa2222"
TEXT_DIM="#6666aa"; TEXT_BR="#eeeeff"
MODE_CAM="#3a2200"; MODE_MAN="#3a1a5e"


class MotionDetector:
    """
    Detektor gerakan akurasi tinggi.
    - MOG2: adaptif terhadap perubahan cahaya, hapus bayangan
    - KNN:  lebih akurat untuk latar belakang kompleks
    - Frame Diff: ringan, cocok untuk kondisi statis
    """
    METHODS = ["MOG2 (Adaptive)", "KNN Background", "Frame Difference"]

    def __init__(self):
        self.method      = self.METHODS[0]
        self.sensitivity = 800    # area minimum piksel
        self.blur_k      = 21
        self.min_w       = 30     # lebar minimum objek
        self.min_h       = 30     # tinggi minimum objek
        self._bg_mog2    = self._make_mog2()
        self._bg_knn     = self._make_knn()
        self._prev_gray  = None
        self._bg_learned = 0      # frame counter untuk warmup MOG2

    def _make_mog2(self):
        sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=True)
        sub.setShadowThreshold(0.5)   # deteksi bayangan lebih ketat
        return sub

    def _make_knn(self):
        return cv2.createBackgroundSubtractorKNN(
            history=300, dist2Threshold=400, detectShadows=True)

    def reset(self):
        self._bg_mog2   = self._make_mog2()
        self._bg_knn    = self._make_knn()
        self._prev_gray = None
        self._bg_learned = 0

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray, list]:
        """
        Return: (motion_detected, frame_with_overlay, contour_list)
        """
        bk = self.blur_k | 1
        blur = cv2.GaussianBlur(frame, (bk, bk), 0)

        if self.method == "MOG2 (Adaptive)":
            mask = self._mog2_mask(blur)
        elif self.method == "KNN Background":
            mask = self._knn_mask(blur)
        else:
            mask = self._frame_diff_mask(blur)

        # Post-processing mask untuk kurangi noise
        mask = self._clean_mask(mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion = False
        valid  = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.sensitivity:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter berdasarkan ukuran minimum
            if w < self.min_w or h < self.min_h:
                continue

            motion = True
            valid.append(cnt)

            # Warna berdasarkan ukuran (merah = besar, kuning = kecil)
            intensity = min(int(area / 5000 * 255), 255)
            color = (0, 255-intensity, intensity)

            cv2.rectangle(frame, (x,y), (x+w,y+h), utils.COLOR_YELLOW, 2)
            # Info area
            cv2.putText(frame, f"Motion {w}x{h}px",
                        (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        utils.COLOR_YELLOW, 1)

        # Gambar motion heatmap mini di pojok
        if len(contours) > 0:
            self._draw_motion_map(frame, mask, contours)

        self._bg_learned = min(self._bg_learned + 1, 9999)
        return motion, frame, valid

    def _mog2_mask(self, frame):
        mask = self._bg_mog2.apply(frame)
        mask[mask == 127] = 0   # hapus bayangan (value 127)
        return mask

    def _knn_mask(self, frame):
        mask = self._bg_knn.apply(frame)
        mask[mask == 127] = 0
        return mask

    def _frame_diff_mask(self, gray_bgr):
        gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros_like(gray)
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        # Adaptive threshold berdasarkan rata-rata gambar
        mean_val = np.mean(diff)
        thresh_val = max(15, int(mean_val * 2.5))
        _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        return mask

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Morphological cleaning untuk hilangkan noise kecil."""
        # Erosi kecil dulu untuk hilangkan noise
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_small)
        # Dilasi untuk hubungkan bagian yang terputus
        k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_large, iterations=2)
        return mask

    def _draw_motion_map(self, frame, mask, contours):
        """Gambar mini-map motion di pojok kiri bawah."""
        h, w = frame.shape[:2]
        mh, mw = 80, 120
        mini = cv2.resize(mask, (mw, mh))
        mini_color = cv2.applyColorMap(mini, cv2.COLORMAP_HOT)
        x1, y1 = 8, h-mh-8
        overlay = frame.copy()
        overlay[y1:y1+mh, x1:x1+mw] = mini_color
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x1-1,y1-1), (x1+mw+1,y1+mh+1), utils.COLOR_YELLOW, 1)
        cv2.putText(frame, "Motion Map", (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, utils.COLOR_YELLOW, 1)

    @property
    def is_warming_up(self):
        """MOG2 butuh beberapa frame untuk belajar background."""
        return self._bg_learned < 30


class MotionDetectionWindow:
    POLL_MS = 25

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("🏃 Modul 2 – Motion Detection")
        self.win.configure(bg=DARK)
        sz = get_module_sizes(parent)
        self.WIN_W=sz["win_w"]; self.WIN_H=sz["win_h"]
        self.DISP_W=sz["disp_w"]; self.DISP_H=sz["disp_h"]
        utils.center_window(self.win, self.WIN_W, self.WIN_H)
        self.win.resizable(True, True); self.win.minsize(700,500)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._q: queue.Queue = queue.Queue(maxsize=3)
        self.cap = None; self.running = False
        self.detector = MotionDetector()
        self.motion_count = 0
        self.mode = tk.StringVar(value="camera")
        self.method_var = tk.StringVar(value=MotionDetector.METHODS[0])
        self.sens_var   = tk.IntVar(value=800)
        self.source_cam = tk.StringVar(value="webcam")
        self._photo = None; self._latest_frame = None
        self._last_motion_time = None
        self._cooldown = cfg.get("detection.motion_cooldown", 2.0)
        self._source_label = "webcam"
        self._fps_cnt = 0; self._fps_ts = datetime.now()
        self._tg = notifier.get_notifier()
        self._warmup_warned = False

        self.mf_source = tk.StringVar(value="manual")
        self.mf_count  = tk.IntVar(value=1)
        self.mf_path   = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        hdr=tk.Frame(self.win,bg=HEADER,height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr,text="🏃  MOTION DETECTION  —  MOG2 / KNN / Frame Diff",
                 font=("Helvetica",13,"bold"),bg=HEADER,fg=ACCENT).pack(side="left",padx=16,pady=14)

        toggle=tk.Frame(self.win,bg="#111133",height=46)
        toggle.pack(fill="x"); toggle.pack_propagate(False)
        tk.Label(toggle,text="MODE:",bg="#111133",fg=TEXT_DIM,
                 font=("Helvetica",9,"bold")).pack(side="left",padx=14,pady=12)
        self.btn_cam=tk.Button(toggle,text="📷  KAMERA / VIDEO",
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
        self.canvas=tk.Label(self.cam_panel,bg="#111122",
                              text="[ No Input ]",font=("Courier",12),fg="#444466")
        self.canvas.pack(fill="both",expand=True,padx=4,pady=4)
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
        tk.Label(hdr,text="✏️  INPUT DATA MOTION DETECTION SECARA MANUAL",
                 font=("Helvetica",11,"bold"),bg="#1a1a3e",fg=ACCENT).pack(side="left",padx=16,pady=10)
        form=tk.Frame(p,bg=DARK,padx=30,pady=20); form.pack(fill="both",expand=True)
        now=datetime.now()
        self.mf_date=tk.StringVar(value=now.strftime("%Y-%m-%d"))
        self.mf_time=tk.StringVar(value=now.strftime("%H:%M:%S"))

        def field(label,var,hint="",spin=False):
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

        field("📅  Tanggal:",self.mf_date,"YYYY-MM-DD")
        field("⏰  Waktu:",self.mf_time,"HH:MM:SS")
        field("📍  Sumber:",self.mf_source)
        field("🔢  Jml Gerakan:",self.mf_count,spin=True)

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
        self.man_log=tk.Text(form,height=5,bg="#080818",fg="#ffccaa",
                              font=("Courier",8),state="disabled",relief="flat")
        self.man_log.pack(fill="x",pady=(4,0))

    def _build_sidebar(self, p):
        pad={"padx":10,"pady":3}
        bc={"font":("Helvetica",10,"bold"),"relief":"flat","cursor":"hand2","pady":7}

        tk.Label(p,text="🎥  SUMBER INPUT",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        for val,txt in [("webcam","🎥  Webcam"),("video","🎬  File Video")]:
            tk.Radiobutton(p,text=txt,variable=self.source_cam,value=val,
                           bg=PANEL,fg="#ccccee",selectcolor="#1e1e3f",
                           activebackground=PANEL,
                           font=("Helvetica",9)).pack(anchor="w",padx=20,pady=1)

        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="⚙️  METODE DETEKSI",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        method_cb=ttk.Combobox(p,textvariable=self.method_var,
                               values=MotionDetector.METHODS,
                               state="readonly",width=26)
        method_cb.pack(padx=10,pady=2)
        tk.Label(p,text="MOG2: terbaik untuk kamera statis\n"
                        "KNN: terbaik untuk latar kompleks\n"
                        "Frame Diff: paling ringan",
                 bg=PANEL,fg="#446688",font=("Helvetica",7),justify="left").pack(padx=10,anchor="w")

        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="🎚️  SENSITIVITAS (px²)",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        sr=tk.Frame(p,bg=PANEL); sr.pack(fill="x",padx=10,pady=2)
        self.sens_lbl=tk.Label(sr,text="800",bg=PANEL,fg="#ffcc00",
                                font=("Courier",11,"bold"),width=6)
        self.sens_lbl.pack(side="right")
        tk.Scale(sr,from_=100,to=8000,resolution=100,orient="horizontal",
                 variable=self.sens_var,bg=PANEL,fg="white",troughcolor="#2a2a4e",
                 highlightbackground=PANEL,
                 command=lambda v: self.sens_lbl.config(
                     text=str(int(float(v))))).pack(side="left",fill="x",expand=True)
        tk.Label(p,text="↑ lebih tinggi = hanya deteksi gerakan besar",
                 bg=PANEL,fg="#446688",font=("Helvetica",7)).pack(padx=10,anchor="w")

        utils.make_separator(p,"#2a2a4e",pady=4)
        tg_on=cfg.get("telegram.enabled",False) and cfg.get("telegram.notify_on_motion",False)
        self.tg_badge=tk.Label(p,text="🔔 Telegram: "+("ON" if tg_on else "OFF"),
                                bg=PANEL,fg="#00ff88" if tg_on else "#666688",
                                font=("Helvetica",8))
        self.tg_badge.pack(anchor="w",padx=10,pady=2)

        tk.Label(p,text="⚙️  KONTROL",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        self.btn_start=tk.Button(p,text="▶  START",bg=GREEN,fg="white",
                                  command=self.start,**bc)
        self.btn_start.pack(fill="x",padx=10,pady=3)
        self.btn_stop=tk.Button(p,text="⏹  STOP",bg=RED,fg="white",
                                 state="disabled",command=self.stop,**bc)
        self.btn_stop.pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="🔄  RESET BG MODEL",bg="#555555",fg="white",
                  command=self._reset_bg,**bc).pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="📊  GRAFIK",bg="#6c3483",fg="white",
                  command=self._show_chart,**bc).pack(fill="x",padx=10,pady=3)
        tk.Button(p,text="🗄️  LIHAT DATA",bg="#2a4a2a",fg="white",
                  command=lambda: __import__("data_input").DataManagementWindow(self.win),
                  **bc).pack(fill="x",padx=10,pady=3)

        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="📈  STATISTIK",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        sf=tk.Frame(p,bg=DARK,padx=8,pady=6); sf.pack(fill="x",padx=10,pady=2)
        self.lbl_count  = self._srow(sf,"Motion Events","0")
        self.lbl_last   = self._srow(sf,"Terakhir","-")
        self.lbl_fps    = self._srow(sf,"FPS","-")
        self.lbl_warmup = self._srow(sf,"BG Status","warmup")
        self.lbl_thread = self._srow(sf,"Thread","idle")

        utils.make_separator(p,"#2a2a4e",pady=4)
        tk.Label(p,text="📋  LOG",bg=PANEL,fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(anchor="w",**pad)
        self.log_box=tk.Text(p,height=6,bg="#080818",fg="#ffccaa",
                              font=("Courier",8),state="disabled",relief="flat")
        self.log_box.pack(fill="x",padx=10,pady=2)

    def _srow(self,parent,label,init):
        row=tk.Frame(parent,bg=DARK); row.pack(fill="x",pady=1)
        tk.Label(row,text=label+":",bg=DARK,fg=TEXT_DIM,
                 font=("Helvetica",8),width=12,anchor="w").pack(side="left")
        v=tk.Label(row,text=init,bg=DARK,fg="#ffdd00",font=("Courier",9,"bold"))
        v.pack(side="right"); return v

    def _set_mode(self, mode):
        self.mode.set(mode)
        if mode=="camera":
            self.man_panel.pack_forget()
            self.cam_panel.pack(fill="both",expand=True)
            self.btn_cam.config(bg=MODE_CAM,fg="white",relief="groove")
            self.btn_man.config(bg=PANEL,fg=TEXT_DIM,relief="flat")
            self.mode_badge.config(text="Mode Aktif: 📷 Kamera — deteksi gerakan real-time")
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
        src=self.source_cam.get()
        try:
            if src=="webcam":
                self.cap=utils.open_camera(cfg.get("detection.camera_index",0))
                self._source_label="webcam"
            else:
                path=filedialog.askopenfilename(
                    parent=self.win,title="Pilih Video",
                    filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
                if not path: return
                self.cap=utils.open_video(path)
                self._source_label=path
        except RuntimeError as e:
            messagebox.showerror("Error",str(e),parent=self.win); return

        self.detector.method=self.method_var.get()
        self.detector.sensitivity=self.sens_var.get()
        self.detector.reset()
        self._warmup_warned=False
        self._flush_q(); self.running=True
        self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        self.lbl_thread.config(text="running"); self.lbl_warmup.config(text="warmup…")
        threading.Thread(target=self._capture_loop,daemon=True).start()
        self._poll()
        self._set_status(f"Running — {os.path.basename(self._source_label)}")

    def stop(self):
        self.running=False
        if self.cap: self.cap.release(); self.cap=None
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.lbl_thread.config(text="idle"); self._set_status("Stopped")
        db.log_motion_detection(self._source_label,self.motion_count)

    def _reset_bg(self):
        self.detector.reset()
        self._log("[RESET] Background model direset — warmup ulang…")
        self.lbl_warmup.config(text="warmup…")

    def _capture_loop(self):
        while self.running:
            ret,frame=self.cap.read()
            if not ret: self.running=False; break
            self.detector.sensitivity=self.sens_var.get()
            self.detector.method=self.method_var.get()
            motion,vis,contours=self.detector.detect(frame.copy())

            is_warming=self.detector.is_warming_up
            status_text="⏳ WARMUP…" if is_warming else ("⚠ MOTION" if motion else "● NORMAL")
            utils.draw_info_overlay(vis,[
                f"Method: {self.method_var.get().split(' ')[0]}",
                f"Events: {self.motion_count}",
                f"Contours: {len(contours)}",
            ])
            utils.draw_timestamp(vis)
            utils.draw_status_badge(vis, status_text, motion and not is_warming)

            self._q_put({"frame":vis.copy(),"motion":motion and not is_warming,
                         "warming":is_warming})

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
        frame=data["frame"]; motion=data["motion"]; warming=data["warming"]
        self._latest_frame=frame
        self._fps_cnt+=1
        elapsed=(datetime.now()-self._fps_ts).total_seconds()
        if elapsed>=1.0:
            self.lbl_fps.config(text=f"{self._fps_cnt/elapsed:.1f}")
            self._fps_cnt=0; self._fps_ts=datetime.now()

        self.lbl_warmup.config(text="warmup…" if warming else "ready ✓",
                                fg="#ffcc00" if warming else "#00ff88")

        if motion:
            now=datetime.now()
            do_save=(self._last_motion_time is None or
                     (now-self._last_motion_time).total_seconds()>=self._cooldown)
            if do_save:
                self._last_motion_time=now; self.motion_count+=1
                ts=now.strftime("%H:%M:%S")
                path=utils.save_frame(frame,"output/motion","motion")
                db.log_motion_detection(self._source_label,self.motion_count,path)
                self._log(f"[{ts}] Motion #{self.motion_count} → {path}")
                self.lbl_count.config(text=str(self.motion_count))
                self.lbl_last.config(text=ts)
                self._tg.notify_motion(path,self.motion_count)

        photo=utils.frame_to_photoimage(frame,self.DISP_W,self.DISP_H)
        self._photo=photo; self.canvas.config(image=photo,text="")

    def _save_manual(self):
        date=self.mf_date.get().strip(); time_s=self.mf_time.get().strip()
        source=self.mf_source.get().strip() or "manual"
        count=int(self.mf_count.get()); path=self.mf_path.get().strip()
        try:
            datetime.strptime(date,"%Y-%m-%d"); datetime.strptime(time_s,"%H:%M:%S")
        except ValueError:
            self.man_status.config(text="❌ Format tanggal/waktu salah",fg="#ff4444"); return
        try:
            db.log_motion_detection(source,count,path)
            self._man_log(f"✅ [{date} {time_s}] {count} gerakan disimpan")
            self.man_status.config(text="✅ Data berhasil disimpan!",fg="#00ff88")
            self._log(f"[MANUAL] {date} {time_s} — {count} events")
        except Exception as e:
            self.man_status.config(text=f"❌ Error: {e}",fg="#ff4444")

    def _reset_form(self):
        now=datetime.now()
        self.mf_date.set(now.strftime("%Y-%m-%d")); self.mf_time.set(now.strftime("%H:%M:%S"))
        self.mf_source.set("manual"); self.mf_count.set(1)
        self.mf_path.set(""); self.man_status.config(text="")

    def _man_log(self,text):
        self.man_log.config(state="normal"); self.man_log.insert("end",text+"\n")
        self.man_log.see("end"); self.man_log.config(state="disabled")

    def _show_chart(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        data=db.get_motion_chart_data()
        if not data:
            messagebox.showinfo("Info","Belum ada data.",parent=self.win); return
        dates=[r[0] for r in data]; totals=[r[1] for r in data]
        cw=tk.Toplevel(self.win); cw.title("📊 Grafik Motion"); cw.configure(bg=DARK)
        utils.center_window(cw,680,400)
        fig,ax=plt.subplots(figsize=(6.8,3.6),facecolor=DARK); ax.set_facecolor(PANEL)
        ax.plot(dates,totals,color=ACCENT,linewidth=2,marker="o")
        ax.fill_between(range(len(dates)),totals,alpha=0.2,color=ACCENT)
        ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates,rotation=30,ha="right",fontsize=7,color="white")
        ax.set_title("Motion Events per Hari",color="white",fontsize=11)
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a4e")
        plt.tight_layout()
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
    