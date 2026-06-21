"""
gui.py - Smart Vision Analysis System
Dashboard utama + Settings Telegram Bot.
"""

import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import os
import threading

import utils
import database as db
import config as cfg


DARK_BG       = "#0d0d1a"
PANEL_BG      = "#13132b"
HEADER_BG     = "#1a1a2e"
ACCENT_BLUE   = "#00d4ff"
ACCENT_ORANGE = "#ff9900"
ACCENT_RED    = "#ff4444"
ACCENT_PURPLE = "#cc44ff"
ACCENT_GREEN  = "#00ff88"
TEXT_DIM      = "#6666aa"
TEXT_BRIGHT   = "#eeeeff"


# ─── Module Card ──────────────────────────────────────────────────────────────

class ModuleCard(tk.Frame):
    def __init__(self, parent, icon, title, subtitle, accent, on_click, **kw):
        super().__init__(parent, bg=PANEL_BG, cursor="hand2",
                         relief="flat", padx=18, pady=18, **kw)
        self._normal_bg = PANEL_BG
        self._hover_bg  = "#1c1c3c"

        tk.Frame(self, bg=accent, width=4).pack(side="left", fill="y", padx=(0, 14))
        tk.Label(self, text=icon, font=("Helvetica", 26),
                 bg=PANEL_BG, fg=accent).pack(side="left", padx=(0, 14))

        tf = tk.Frame(self, bg=PANEL_BG)
        tf.pack(side="left", fill="x", expand=True)
        self.title_lbl = tk.Label(tf, text=title, font=("Helvetica", 12, "bold"),
                                   bg=PANEL_BG, fg=TEXT_BRIGHT, anchor="w")
        self.title_lbl.pack(anchor="w")
        self.sub_lbl = tk.Label(tf, text=subtitle, font=("Helvetica", 9),
                                 bg=PANEL_BG, fg=TEXT_DIM, anchor="w")
        self.sub_lbl.pack(anchor="w")

        self.arrow = tk.Label(self, text="→", font=("Helvetica", 18),
                               bg=PANEL_BG, fg=accent)
        self.arrow.pack(side="right")

        for w in [self, tf, self.title_lbl, self.sub_lbl, self.arrow]:
            w.bind("<Button-1>", lambda e: on_click())
            w.bind("<Enter>",    self._enter)
            w.bind("<Leave>",    self._leave)

    def _enter(self, event=None):
        for w in [self, self.title_lbl, self.sub_lbl, self.arrow]:
            w.config(bg=self._hover_bg)

    def _leave(self, event=None):
        for w in [self, self.title_lbl, self.sub_lbl, self.arrow]:
            w.config(bg=self._normal_bg)


# ─── Telegram Settings Window ─────────────────────────────────────────────────

class TelegramSettingsWindow:
    """Dialog konfigurasi Telegram Bot."""

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("⚙️ Pengaturan Telegram Bot")
        self.win.configure(bg=DARK_BG)
        self.win.resizable(False, False)
        utils.center_window(self.win, 560, 580)
        self.win.grab_set()   # modal

        # Load current config
        tcfg = cfg.load().get("telegram", {})
        self.enabled_var       = tk.BooleanVar(value=tcfg.get("enabled", False))
        self.token_var         = tk.StringVar(value=tcfg.get("bot_token", ""))
        self.chat_id_var       = tk.StringVar(value=tcfg.get("chat_id", ""))
        self.anomaly_var       = tk.BooleanVar(value=tcfg.get("notify_on_anomaly", True))
        self.motion_var        = tk.BooleanVar(value=tcfg.get("notify_on_motion", False))
        self.send_photo_var    = tk.BooleanVar(value=tcfg.get("send_photo", True))

        self._build()

    def _build(self):
        # ── Header
        hdr = tk.Frame(self.win, bg=HEADER_BG, height=60)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📱  Pengaturan Telegram Bot",
                 font=("Helvetica", 14, "bold"),
                 bg=HEADER_BG, fg=ACCENT_BLUE).pack(side="left", padx=20, pady=16)

        body = tk.Frame(self.win, bg=DARK_BG, padx=24, pady=16)
        body.pack(fill="both", expand=True)

        # ── Cara mendapatkan bot
        info_frame = tk.Frame(body, bg=PANEL_BG, padx=12, pady=10)
        info_frame.pack(fill="x", pady=(0, 16))
        tk.Label(info_frame, text="💡  Cara membuat Telegram Bot:",
                 font=("Helvetica", 9, "bold"),
                 bg=PANEL_BG, fg=ACCENT_BLUE).pack(anchor="w")
        steps = [
            "1. Buka Telegram → cari @BotFather",
            "2. Ketik /newbot → ikuti instruksi",
            "3. Salin Bot Token yang diberikan",
            "4. Mulai chat dengan bot kamu",
            "5. Buka: api.telegram.org/bot<TOKEN>/getUpdates",
            "   untuk mendapatkan Chat ID",
        ]
        for s in steps:
            tk.Label(info_frame, text=s, font=("Courier", 8),
                     bg=PANEL_BG, fg="#aaaacc").pack(anchor="w")

        # ── Enable toggle
        toggle_row = tk.Frame(body, bg=DARK_BG)
        toggle_row.pack(fill="x", pady=(0, 12))
        tk.Checkbutton(toggle_row, text="  Aktifkan Notifikasi Telegram",
                       variable=self.enabled_var,
                       bg=DARK_BG, fg=TEXT_BRIGHT,
                       selectcolor=PANEL_BG,
                       activebackground=DARK_BG,
                       font=("Helvetica", 11, "bold"),
                       command=self._toggle_state).pack(side="left")

        # ── Token
        self._field(body, "🔑  Bot Token:", self.token_var, show="")

        # ── Chat ID
        self._field(body, "💬  Chat ID:", self.chat_id_var, show="")

        # ── Notify on
        opts_frame = tk.LabelFrame(body, text=" Kirim Notifikasi Saat: ",
                                    bg=DARK_BG, fg=TEXT_DIM,
                                    font=("Helvetica", 9),
                                    padx=12, pady=8)
        opts_frame.pack(fill="x", pady=10)
        tk.Checkbutton(opts_frame, text="🚨  Anomaly / Intrusion terdeteksi",
                       variable=self.anomaly_var,
                       bg=DARK_BG, fg=TEXT_BRIGHT,
                       selectcolor=PANEL_BG,
                       activebackground=DARK_BG,
                       font=("Helvetica", 9)).pack(anchor="w", pady=2)
        tk.Checkbutton(opts_frame, text="🏃  Motion terdeteksi",
                       variable=self.motion_var,
                       bg=DARK_BG, fg=TEXT_BRIGHT,
                       selectcolor=PANEL_BG,
                       activebackground=DARK_BG,
                       font=("Helvetica", 9)).pack(anchor="w", pady=2)
        tk.Checkbutton(opts_frame, text="📷  Sertakan foto pada notifikasi",
                       variable=self.send_photo_var,
                       bg=DARK_BG, fg=TEXT_BRIGHT,
                       selectcolor=PANEL_BG,
                       activebackground=DARK_BG,
                       font=("Helvetica", 9)).pack(anchor="w", pady=2)

        # ── Status label
        self.status_lbl = tk.Label(body, text="",
                                    bg=DARK_BG, fg="#aaaacc",
                                    font=("Courier", 9), wraplength=480)
        self.status_lbl.pack(pady=(4, 0))

        # ── Buttons
        btn_row = tk.Frame(body, bg=DARK_BG)
        btn_row.pack(fill="x", pady=12)

        bc = {"font": ("Helvetica", 10, "bold"), "relief": "flat",
              "cursor": "hand2", "pady": 8, "padx": 16}

        tk.Button(btn_row, text="🧪  TEST KONEKSI",
                  bg="#1a5276", fg="white",
                  command=self._test_connection, **bc).pack(side="left")
        tk.Button(btn_row, text="💾  SIMPAN",
                  bg="#00aa44", fg="white",
                  command=self._save, **bc).pack(side="right")
        tk.Button(btn_row, text="✖  BATAL",
                  bg="#555555", fg="white",
                  command=self.win.destroy, **bc).pack(side="right", padx=(0, 8))

        self._toggle_state()

    def _field(self, parent, label_text, var, show=""):
        row = tk.Frame(parent, bg=DARK_BG)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label_text, bg=DARK_BG, fg=TEXT_DIM,
                 font=("Helvetica", 9), width=14, anchor="w").pack(side="left")
        entry = tk.Entry(row, textvariable=var, show=show,
                         bg=PANEL_BG, fg=TEXT_BRIGHT,
                         insertbackground="white",
                         font=("Courier", 10), relief="flat",
                         highlightthickness=1,
                         highlightcolor=ACCENT_BLUE,
                         highlightbackground="#2a2a4e")
        entry.pack(side="left", fill="x", expand=True, ipady=4)
        self._entries = getattr(self, "_entries", [])
        self._entries.append(entry)

    def _toggle_state(self):
        state = "normal" if self.enabled_var.get() else "disabled"
        for e in getattr(self, "_entries", []):
            e.config(state=state)

    def _test_connection(self):
        self._set_status("⏳ Menguji koneksi…", "#ffcc00")
        self.win.update()

        # Update config sementara untuk test
        token   = self.token_var.get().strip()
        chat_id = self.chat_id_var.get().strip()

        if not token or not chat_id:
            self._set_status("❌ Token dan Chat ID harus diisi!", ACCENT_RED)
            return

        def _run():
            import notifier as nt
            # Buat notifier sementara dengan token test
            original_token   = cfg.get("telegram.bot_token", "")
            original_chat_id = cfg.get("telegram.chat_id", "")
            cfg.set_value("telegram.bot_token", token)
            cfg.set_value("telegram.chat_id", chat_id)

            n = nt.TelegramNotifier()
            ok, msg = n.test_connection()
            n.shutdown()

            # Restore
            cfg.set_value("telegram.bot_token", original_token)
            cfg.set_value("telegram.chat_id", original_chat_id)

            color  = ACCENT_GREEN if ok else ACCENT_RED
            prefix = "✅" if ok else "❌"
            self.win.after(0, lambda: self._set_status(f"{prefix} {msg}", color))

        threading.Thread(target=_run, daemon=True).start()

    def _save(self):
        c = cfg.load()
        c["telegram"]["enabled"]          = self.enabled_var.get()
        c["telegram"]["bot_token"]        = self.token_var.get().strip()
        c["telegram"]["chat_id"]          = self.chat_id_var.get().strip()
        c["telegram"]["notify_on_anomaly"] = self.anomaly_var.get()
        c["telegram"]["notify_on_motion"] = self.motion_var.get()
        c["telegram"]["send_photo"]       = self.send_photo_var.get()
        cfg.save(c)

        # Reset singleton notifier supaya pakai config baru
        import notifier as nt
        nt._notifier = None

        self._set_status("✅ Konfigurasi disimpan!", ACCENT_GREEN)
        self.win.after(1200, self.win.destroy)

    def _set_status(self, msg: str, color: str = "#aaaacc"):
        self.status_lbl.config(text=msg, fg=color)


# ─── Main Dashboard ───────────────────────────────────────────────────────────

class MainDashboard:
    WIN_W, WIN_H = 820, 660  # akan di-override di __init__

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Vision Analysis System")
        self.root.configure(bg=DARK_BG)
        from window_utils import get_module_sizes
        _sz = get_module_sizes(self.root)
        self.WIN_W = _sz["dash_w"]
        self.WIN_H = _sz["dash_h"]
        self.root.resizable(True, True)
        self.root.minsize(680, 500)
        utils.center_window(self.root, self.WIN_W, self.WIN_H)
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self._win_refs = {}
        self._build_ui()
        self._refresh_stats()

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=HEADER_BG, height=80)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="👁‍🗨", font=("Helvetica", 28),
                 bg=HEADER_BG, fg=ACCENT_BLUE).pack(side="left", padx=24, pady=14)
        tf = tk.Frame(hdr, bg=HEADER_BG)
        tf.pack(side="left")
        tk.Label(tf, text="Smart Vision Analysis System",
                 font=("Helvetica", 17, "bold"),
                 bg=HEADER_BG, fg=TEXT_BRIGHT).pack(anchor="w")
        tk.Label(tf, text="Computer Vision  ·  OpenCV  ·  YOLO v8  ·  Python",
                 font=("Helvetica", 9),
                 bg=HEADER_BG, fg=TEXT_DIM).pack(anchor="w")
        self.clock_lbl = tk.Label(hdr, text="",
                                   font=("Courier", 11),
                                   bg=HEADER_BG, fg=ACCENT_BLUE)
        self.clock_lbl.pack(side="right", padx=20)
        self._tick()

    def _tick(self):
        self.clock_lbl.config(
            text=datetime.now().strftime("🕐  %H:%M:%S\n📅  %Y-%m-%d"))
        self.root.after(1000, self._tick)

    def _build_body(self):
        body = tk.Frame(self.root, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=12)

        left = tk.Frame(body, bg=DARK_BG)
        left.pack(side="left", fill="both", expand=True)

        # Telegram status badge
        self.tg_status_lbl = tk.Label(left, text="",
                                       bg=DARK_BG, fg=TEXT_DIM,
                                       font=("Helvetica", 8))
        self.tg_status_lbl.pack(anchor="w", pady=(0, 4))
        self._update_tg_status()

        tk.Label(left, text="MODUL", font=("Helvetica", 9, "bold"),
                 bg=DARK_BG, fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))

        modules = [
            ("📦", "Image Detection",
             "YOLO · Webcam / Gambar / Manual Input · Bounding Box · Log",
             ACCENT_BLUE,   self._open_image_detection),
            ("🏃", "Motion Detection",
             "MOG2 / Frame Diff · Webcam / Video / Manual Input · Telegram",
             ACCENT_ORANGE, self._open_motion_detection),
            ("🚨", "Anomaly Detection",
             "Restricted Area · Intrusion · YOLO · Mouse Zone · Telegram Alert",
             ACCENT_RED,    self._open_anomaly_detection),
            ("🎨", "Image Manipulation",
             "Grayscale · Blur · Edge · Threshold · Rotate · Histogram",
             ACCENT_PURPLE, self._open_image_manipulation),
            ("📊", "Data Management",
             "Input Manual · Tabel Data · Edit · Hapus · Import / Export CSV",
             ACCENT_GREEN,  self._open_data_management),
        ]
        for icon, title, sub, accent, cmd in modules:
            ModuleCard(left, icon, title, sub, accent, cmd).pack(fill="x", pady=5)

        # Right panel
        right = tk.Frame(body, bg=DARK_BG, width=200)
        right.pack(side="right", fill="y", padx=(16, 0))
        right.pack_propagate(False)

        tk.Label(right, text="STATISTIK", font=("Helvetica", 9, "bold"),
                 bg=DARK_BG, fg=TEXT_DIM).pack(anchor="w", pady=(0, 6))

        sg = tk.Frame(right, bg=DARK_BG)
        sg.pack(fill="x")
        self.stat_objects  = self._stat_badge(sg, "Total Objek\nTerdeteksi", "–", ACCENT_BLUE)
        self.stat_motions  = self._stat_badge(sg, "Motion\nEvents",          "–", ACCENT_ORANGE)
        self.stat_alerts   = self._stat_badge(sg, "Anomaly\nAlerts",         "–", ACCENT_RED)
        self.stat_sessions = self._stat_badge(sg, "Sesi\nDeteksi",           "–", ACCENT_PURPLE)

        utils.make_separator(right, "#2a2a4e", pady=8)

        tk.Label(right, text="DIREKTORI OUTPUT", font=("Helvetica", 9, "bold"),
                 bg=DARK_BG, fg=TEXT_DIM).pack(anchor="w")
        for ico, folder in [("📦", "output/object"), ("🏃", "output/motion"),
                             ("🚨", "output/anomaly"), ("🎨", "output/image_processing"),
                             ("📸", "output/screenshots"), ("📊", "output/charts")]:
            row = tk.Frame(right, bg=DARK_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{ico} {folder}", font=("Courier", 8),
                     bg=DARK_BG, fg="#555588").pack(side="left")
            tk.Button(row, text="📂", bg=DARK_BG, fg=TEXT_DIM,
                      relief="flat", cursor="hand2", font=("Helvetica", 9),
                      command=lambda f=folder: self._open_folder(f)).pack(side="right")

    def _stat_badge(self, parent, label, value, accent) -> tk.Label:
        f = tk.Frame(parent, bg=PANEL_BG, padx=12, pady=8)
        f.pack(fill="x", pady=3)
        v = tk.Label(f, text=value, font=("Helvetica", 18, "bold"),
                     bg=PANEL_BG, fg=accent)
        v.pack()
        tk.Label(f, text=label, font=("Helvetica", 7),
                 bg=PANEL_BG, fg=TEXT_DIM).pack()
        return v

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=HEADER_BG, height=44)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        tk.Label(footer,
                 text="⚡ Smart Vision Analysis System  ·  Nerazurra Dev Studio",
                 font=("Helvetica", 8),
                 bg=HEADER_BG, fg=TEXT_DIM).pack(side="left", padx=16, pady=11)

        bc = {"font": ("Helvetica", 9, "bold"), "relief": "flat",
              "cursor": "hand2", "padx": 12, "pady": 4}

        tk.Button(footer, text="🚪  EXIT", bg="#aa2222", fg="white",
                  command=self._on_exit, **bc).pack(side="right", padx=16, pady=8)
        tk.Button(footer, text="📱  TELEGRAM", bg="#1a3a6e", fg="white",
                  command=self._open_telegram_settings, **bc).pack(side="right", padx=4, pady=8)
        tk.Button(footer, text="📊  GRAFIK", bg="#333355", fg="white",
                  command=self._show_summary_chart, **bc).pack(side="right", padx=4, pady=8)
        tk.Button(footer, text="🔄  REFRESH", bg="#1a3a5a", fg="white",
                  command=self._refresh_stats, **bc).pack(side="right", padx=4, pady=8)

    # ── Module Openers ────────────────────────────────────────────────────────

    def _open_image_detection(self):
        from image_detection import ImageDetectionWindow
        self._win_refs["img"] = ImageDetectionWindow(self.root)

    def _open_motion_detection(self):
        from motion_detection import MotionDetectionWindow
        self._win_refs["motion"] = MotionDetectionWindow(self.root)

    def _open_anomaly_detection(self):
        from anomaly_detection import AnomalyDetectionWindow
        self._win_refs["anomaly"] = AnomalyDetectionWindow(self.root)

    def _open_image_manipulation(self):
        from image_manipulation import ImageManipulationWindow
        self._win_refs["manip"] = ImageManipulationWindow(self.root)

    def _open_data_management(self):
        from data_input import DataManagementWindow
        self._win_refs["data"] = DataManagementWindow(self.root)

    def _open_telegram_settings(self):
        TelegramSettingsWindow(self.root)
        # Refresh status setelah settings ditutup
        self.root.after(1500, self._update_tg_status)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _refresh_stats(self):
        try:
            s = db.get_summary_stats()
            self.stat_objects.config(text=str(s["object"]["total_objects"] or 0))
            self.stat_motions.config(text=str(s["motion"]["total_motions"] or 0))
            self.stat_alerts.config(text=str(s["anomaly"]["alerts"] or 0))
            self.stat_sessions.config(text=str(s["object"]["sessions"] or 0))
        except Exception:
            pass
        self._update_tg_status()

    def _update_tg_status(self):
        enabled = cfg.get("telegram.enabled", False)
        token   = cfg.get("telegram.bot_token", "")
        if enabled and token:
            self.tg_status_lbl.config(
                text="📱 Telegram: AKTIF ✓",
                fg=ACCENT_GREEN)
        elif enabled:
            self.tg_status_lbl.config(
                text="📱 Telegram: Aktif tapi belum dikonfigurasi",
                fg=ACCENT_ORANGE)
        else:
            self.tg_status_lbl.config(
                text="📱 Telegram: Nonaktif  (klik TELEGRAM di footer untuk mengaktifkan)",
                fg=TEXT_DIM)

    def _show_summary_chart(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        try:
            s = db.get_summary_stats()
        except Exception:
            return
        cw = tk.Toplevel(self.root)
        cw.title("📊 Ringkasan Statistik")
        cw.configure(bg=DARK_BG)
        utils.center_window(cw, 640, 400)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.8), facecolor=DARK_BG)
        labels = ["Object Det", "Motion", "Anomaly"]
        values = [s["object"]["total_objects"] or 1,
                  s["motion"]["total_motions"] or 1,
                  s["anomaly"]["alerts"] or 1]
        colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_RED]

        ax1.set_facecolor(PANEL_BG)
        _, _, autotexts = ax1.pie(values, labels=labels, colors=colors,
                                   autopct="%1.0f%%", startangle=140,
                                   textprops={"color": "white", "fontsize": 8})
        for at in autotexts:
            at.set_color("black")
            at.set_fontsize(7)
        ax1.set_title("Distribusi Events", color="white", fontsize=10)

        ax2.set_facecolor(PANEL_BG)
        cats = ["Objek\nTerdeteksi", "Motion\nEvents", "Anomaly\nAlerts", "Sesi\nDeteksi"]
        vals = [s["object"]["total_objects"] or 0, s["motion"]["total_motions"] or 0,
                s["anomaly"]["alerts"] or 0, s["object"]["sessions"] or 0]
        bars = ax2.bar(cats, vals, color=colors + [ACCENT_PURPLE], edgecolor=DARK_BG)
        ax2.set_title("Total Keseluruhan", color="white", fontsize=10)
        ax2.tick_params(colors="white", labelsize=7)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#2a2a4e")
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.1,
                     str(val), ha="center", va="bottom",
                     color="white", fontsize=8, fontweight="bold")
        plt.tight_layout(pad=1.5)

        cv_c = FigureCanvasTkAgg(fig, master=cw)
        cv_c.draw()
        cv_c.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _open_folder(path: str):
        os.makedirs(path, exist_ok=True)
        import subprocess, sys
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    def _on_exit(self):
        if messagebox.askyesno("Exit", "Keluar dari Smart Vision Analysis System?",
                                parent=self.root):
            self.root.destroy()

    def run(self):
        self.root.mainloop()
