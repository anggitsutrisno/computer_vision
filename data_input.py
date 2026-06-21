"""
data_input.py - Smart Vision Analysis System
Modul 5: Data Management — Input Manual, Tabel Data, Edit, Hapus, Import CSV.
"""

import os
import csv
import sqlite3
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import utils
import database as db
import config as cfg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers warna
DARK_BG    = "#0d0d1a"
PANEL_BG   = "#13132b"
HEADER_BG  = "#1a1a2e"
ACCENT     = "#00d4ff"
GREEN      = "#00ff88"
RED        = "#ff4444"
ORANGE     = "#ff9900"
PURPLE     = "#cc44ff"
TEXT_DIM   = "#6666aa"
TEXT_BRIGHT= "#eeeeff"

TABLE_HEADINGS = {
    "object": ["ID", "Tanggal", "Waktu", "Sumber", "Jml Objek", "Nama Objek", "Gambar"],
    "motion": ["ID", "Tanggal", "Waktu", "Sumber", "Jml Gerakan", "Gambar"],
    "anomaly":["ID", "Tanggal", "Waktu", "Sumber", "Objek", "Info Area", "Gambar"],
}

MODULE_COLORS = {
    "object":  "#00d4ff",
    "motion":  "#ff9900",
    "anomaly": "#ff4444",
}

MODULE_LABELS = {
    "object":  "📦 Object Detection",
    "motion":  "🏃 Motion Detection",
    "anomaly": "🚨 Anomaly Detection",
}


# ──────────────────────────────────────────────────────────────────────────────
# Database helpers (CRUD tambahan yang belum ada di database.py)

def _get_all(table: str, limit: int = 500) -> list[dict]:
    conn = db.get_connection()
    rows = conn.execute(
        f"SELECT * FROM {table}_detection_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _delete_record(table: str, record_id: int):
    conn = db.get_connection()
    conn.execute(
        f"DELETE FROM {table}_detection_log WHERE id = ?", (record_id,)
    )
    conn.commit()
    conn.close()


def _update_object(record_id: int, date: str, time: str, source: str,
                   total: int, names: str, path: str):
    conn = db.get_connection()
    ts = f"{date}T{time}"
    conn.execute("""
        UPDATE object_detection_log
        SET timestamp=?, date=?, time=?, source=?,
            total_objects=?, object_names=?, image_path=?
        WHERE id=?
    """, (ts, date, time, source, total, names, path, record_id))
    conn.commit()
    conn.close()


def _update_motion(record_id: int, date: str, time: str, source: str,
                   count: int, path: str):
    conn = db.get_connection()
    ts = f"{date}T{time}"
    conn.execute("""
        UPDATE motion_detection_log
        SET timestamp=?, date=?, time=?, source=?, motion_count=?, image_path=?
        WHERE id=?
    """, (ts, date, time, source, count, path, record_id))
    conn.commit()
    conn.close()


def _update_anomaly(record_id: int, date: str, time: str, source: str,
                    names: str, area: str, path: str):
    conn = db.get_connection()
    ts = f"{date}T{time}"
    conn.execute("""
        UPDATE anomaly_detection_log
        SET timestamp=?, date=?, time=?, source=?,
            object_names=?, area_info=?, image_path=?
        WHERE id=?
    """, (ts, date, time, source, names, area, path, record_id))
    conn.commit()
    conn.close()


def _insert_object(date, time, source, total, names, path):
    db.log_object_detection(source, total,
                             [n.strip() for n in names.split(",")], path)


def _insert_motion(date, time, source, count, path):
    db.log_motion_detection(source, count, path)


def _insert_anomaly(date, time, source, names, area, path):
    db.log_anomaly_detection(source,
                              [n.strip() for n in names.split(",")], area, path)


# ──────────────────────────────────────────────────────────────────────────────
# Form Dialog (Add / Edit)

class RecordFormDialog:
    """Dialog form untuk tambah/edit satu record."""

    def __init__(self, parent, module: str, record: dict | None = None):
        self.module  = module
        self.record  = record   # None = add new
        self.result  = None     # dict jika berhasil save
        self.is_edit = record is not None

        self.win = tk.Toplevel(parent)
        self.win.title(
            ("✏️ Edit" if self.is_edit else "➕ Tambah") +
            f" — {MODULE_LABELS[module]}"
        )
        self.win.configure(bg=DARK_BG)
        self.win.resizable(False, False)
        utils.center_window(self.win, 520, 460 if module != "object" else 490)
        self.win.grab_set()

        # Inisialisasi Vars
        now = datetime.now()
        r   = record or {}

        self.date_var   = tk.StringVar(value=r.get("date",   now.strftime("%Y-%m-%d")))
        self.time_var   = tk.StringVar(value=r.get("time",   now.strftime("%H:%M:%S")))
        self.source_var = tk.StringVar(value=r.get("source", "manual"))
        self.path_var   = tk.StringVar(value=r.get("image_path", ""))

        # Module-specific vars
        if module == "object":
            self.total_var = tk.IntVar(
                value=r.get("total_objects", 1))
            self.names_var = tk.StringVar(
                value=r.get("object_names", ""))
        elif module == "motion":
            self.count_var = tk.IntVar(
                value=r.get("motion_count", 1))
        elif module == "anomaly":
            self.names_var = tk.StringVar(
                value=r.get("object_names", "person"))
            self.area_var  = tk.StringVar(
                value=r.get("area_info", "Zone 1"))

        self._build()

    def _build(self):
        accent = MODULE_COLORS[self.module]

        # Header
        hdr = tk.Frame(self.win, bg=HEADER_BG, height=52)
        hdr.pack(fill="x")
        tk.Label(hdr,
                 text=MODULE_LABELS[self.module] + "  —  " +
                      ("Edit Record" if self.is_edit else "Tambah Record Baru"),
                 font=("Helvetica", 11, "bold"),
                 bg=HEADER_BG, fg=accent).pack(side="left", padx=16, pady=14)

        body = tk.Frame(self.win, bg=DARK_BG, padx=24, pady=16)
        body.pack(fill="both", expand=True)

        def row(label, var, width=28, entry_type="text"):
            f = tk.Frame(body, bg=DARK_BG)
            f.pack(fill="x", pady=5)
            tk.Label(f, text=label, bg=DARK_BG, fg=TEXT_DIM,
                     font=("Helvetica", 9), width=18, anchor="w").pack(side="left")
            if entry_type == "spin":
                w = tk.Spinbox(f, textvariable=var, from_=0, to=9999,
                               width=10, bg=PANEL_BG, fg=TEXT_BRIGHT,
                               insertbackground="white",
                               buttonbackground=PANEL_BG,
                               font=("Courier", 10), relief="flat",
                               highlightthickness=1,
                               highlightcolor=accent,
                               highlightbackground="#2a2a4e")
            else:
                w = tk.Entry(f, textvariable=var, width=width,
                             bg=PANEL_BG, fg=TEXT_BRIGHT,
                             insertbackground="white",
                             font=("Courier", 10), relief="flat",
                             highlightthickness=1,
                             highlightcolor=accent,
                             highlightbackground="#2a2a4e")
            w.pack(side="left", fill="x", expand=True, ipady=4, padx=(6, 0))
            return w

        # Common fields
        row("📅  Tanggal (YYYY-MM-DD):", self.date_var)
        row("⏰  Waktu (HH:MM:SS):",     self.time_var)
        row("📍  Sumber:",               self.source_var)

        # Module-specific
        utils.make_separator(body, "#2a2a4e", pady=4)
        if self.module == "object":
            row("🔢  Jumlah Objek:",      self.total_var, entry_type="spin")
            row("🏷️  Nama Objek\n     (pisah koma):", self.names_var, width=26)
        elif self.module == "motion":
            row("🔢  Jumlah Gerakan:",    self.count_var, entry_type="spin")
        elif self.module == "anomaly":
            row("👤  Nama Objek:",        self.names_var)
            row("📍  Info Area/Zona:",    self.area_var)

        utils.make_separator(body, "#2a2a4e", pady=4)

        # Path gambar
        path_row = tk.Frame(body, bg=DARK_BG)
        path_row.pack(fill="x", pady=5)
        tk.Label(path_row, text="🖼️  Path Gambar:",
                 bg=DARK_BG, fg=TEXT_DIM,
                 font=("Helvetica", 9), width=18, anchor="w").pack(side="left")
        tk.Entry(path_row, textvariable=self.path_var, width=22,
                 bg=PANEL_BG, fg=TEXT_BRIGHT,
                 insertbackground="white",
                 font=("Courier", 9), relief="flat",
                 highlightthickness=1,
                 highlightcolor=accent,
                 highlightbackground="#2a2a4e").pack(side="left", fill="x",
                                                      expand=True, ipady=4, padx=(6, 0))
        tk.Button(path_row, text="📂",
                  bg=PANEL_BG, fg=TEXT_DIM,
                  relief="flat", cursor="hand2",
                  command=self._browse_image).pack(side="left", padx=(4, 0))

        # Status
        self.status_lbl = tk.Label(body, text="",
                                    bg=DARK_BG, fg=RED,
                                    font=("Helvetica", 8))
        self.status_lbl.pack(pady=4)

        # Buttons
        btn_row = tk.Frame(body, bg=DARK_BG)
        btn_row.pack(fill="x", pady=(8, 0))
        bc = {"font": ("Helvetica", 10, "bold"), "relief": "flat",
              "cursor": "hand2", "pady": 8, "padx": 18}
        tk.Button(btn_row, text="💾  SIMPAN",
                  bg="#00aa44", fg="white",
                  command=self._save, **bc).pack(side="right")
        tk.Button(btn_row, text="✖  BATAL",
                  bg="#555555", fg="white",
                  command=self.win.destroy, **bc).pack(side="right", padx=(0, 8))

    def _browse_image(self):
        path = filedialog.askopenfilename(
            parent=self.win, title="Pilih Gambar",
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if path:
            self.path_var.set(path)

    def _save(self):
        date   = self.date_var.get().strip()
        time   = self.time_var.get().strip()
        source = self.source_var.get().strip() or "manual"
        path   = self.path_var.get().strip()

        # Validasi tanggal & waktu
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            self.status_lbl.config(text="❌ Format tanggal salah (gunakan YYYY-MM-DD)")
            return
        try:
            datetime.strptime(time, "%H:%M:%S")
        except ValueError:
            self.status_lbl.config(text="❌ Format waktu salah (gunakan HH:MM:SS)")
            return

        try:
            if self.module == "object":
                total = int(self.total_var.get())
                names = self.names_var.get().strip()
                if self.is_edit:
                    _update_object(self.record["id"], date, time, source,
                                   total, names, path)
                else:
                    _insert_object(date, time, source, total, names, path)

            elif self.module == "motion":
                count = int(self.count_var.get())
                if self.is_edit:
                    _update_motion(self.record["id"], date, time, source,
                                   count, path)
                else:
                    _insert_motion(date, time, source, count, path)

            elif self.module == "anomaly":
                names = self.names_var.get().strip() or "person"
                area  = self.area_var.get().strip() or "-"
                if self.is_edit:
                    _update_anomaly(self.record["id"], date, time, source,
                                    names, area, path)
                else:
                    _insert_anomaly(date, time, source, names, area, path)

            self.result = True
            self.win.destroy()

        except Exception as e:
            self.status_lbl.config(text=f"❌ Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Source Selector Widget (reusable di modul lain)

class SourceSelectorWidget(tk.Frame):
    """
    Widget pemilih sumber input: Webcam / File Gambar / File Video / Input Manual.
    Dipakai oleh modul-modul deteksi sebagai komponen reusable.
    """

    SOURCES = {
        "webcam": ("🎥", "Webcam",      "#00aa44"),
        "image":  ("🖼️", "File Gambar", "#1a5276"),
        "video":  ("🎬", "File Video",  "#6c3483"),
        "manual": ("✏️", "Input Manual","#805500"),
    }

    def __init__(self, parent, available=("webcam", "image", "video", "manual"),
                 on_change=None, **kw):
        super().__init__(parent, bg=PANEL_BG, **kw)
        self.available  = available
        self.on_change  = on_change
        self.source_var = tk.StringVar(value=available[0])

        self._build()

    def _build(self):
        tk.Label(self, text="▶  PILIH INPUT SUMBER",
                 bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w",
                                                       padx=12, pady=(8, 4))

        btn_frame = tk.Frame(self, bg=PANEL_BG)
        btn_frame.pack(fill="x", padx=12, pady=(0, 8))

        self._btns = {}
        for src in self.available:
            ico, label, color = self.SOURCES[src]
            b = tk.Button(btn_frame,
                          text=f"{ico}  {label}",
                          bg=PANEL_BG, fg=TEXT_DIM,
                          font=("Helvetica", 9, "bold"),
                          relief="flat", cursor="hand2",
                          padx=8, pady=6,
                          command=lambda s=src: self._select(s))
            b.pack(side="left", fill="x", expand=True, padx=2)
            self._btns[src] = (b, color)

        self._select(self.available[0])

    def _select(self, src: str):
        self.source_var.set(src)
        for s, (btn, color) in self._btns.items():
            if s == src:
                btn.config(bg=color, fg="white",
                            relief="groove")
            else:
                btn.config(bg=PANEL_BG, fg=TEXT_DIM,
                            relief="flat")
        if self.on_change:
            self.on_change(src)

    def get(self) -> str:
        return self.source_var.get()


# ──────────────────────────────────────────────────────────────────────────────
# Data Table Widget

class DataTableWidget(tk.Frame):
    """Tabel data dengan fitur sort, filter, pagination."""

    PAGE_SIZE = 50

    def __init__(self, parent, module: str, **kw):
        super().__init__(parent, bg=DARK_BG, **kw)
        self.module   = module
        self._data    = []
        self._page    = 0
        self._filter  = ""
        self._sort_col= ""
        self._sort_rev= False
        self._build()

    def _build(self):
        accent = MODULE_COLORS[self.module]

        # ── Toolbar
        tb = tk.Frame(self, bg=PANEL_BG, pady=6)
        tb.pack(fill="x")

        tk.Label(tb, text="🔍", bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Helvetica", 12)).pack(side="left", padx=(10, 4))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._apply_filter())
        tk.Entry(tb, textvariable=self.search_var,
                 bg=DARK_BG, fg=TEXT_BRIGHT,
                 insertbackground="white",
                 font=("Courier", 9), relief="flat",
                 highlightthickness=1,
                 highlightcolor=accent,
                 highlightbackground="#2a2a4e",
                 width=24).pack(side="left", ipady=3)

        tk.Label(tb, text="  Filter tanggal:",
                 bg=PANEL_BG, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")
        self.date_filter_var = tk.StringVar()
        self.date_filter_var.trace_add("write", lambda *_: self._apply_filter())
        tk.Entry(tb, textvariable=self.date_filter_var,
                 bg=DARK_BG, fg=TEXT_BRIGHT,
                 insertbackground="white",
                 font=("Courier", 9), relief="flat",
                 highlightthickness=1,
                 highlightcolor=accent,
                 highlightbackground="#2a2a4e",
                 width=12).pack(side="left", ipady=3, padx=(2, 0))

        self.count_lbl = tk.Label(tb, text="0 records",
                                   bg=PANEL_BG, fg=TEXT_DIM,
                                   font=("Helvetica", 8))
        self.count_lbl.pack(side="right", padx=12)

        # ── Treeview
        cols = TABLE_HEADINGS[self.module]
        tree_frame = tk.Frame(self, bg=DARK_BG)
        tree_frame.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                         background=DARK_BG,
                         foreground=TEXT_BRIGHT,
                         fieldbackground=DARK_BG,
                         rowheight=26,
                         font=("Courier", 9))
        style.configure("Dark.Treeview.Heading",
                         background=PANEL_BG,
                         foreground=accent,
                         font=("Helvetica", 9, "bold"))
        style.map("Dark.Treeview",
                  background=[("selected", "#1c3a5a")])

        self.tree = ttk.Treeview(tree_frame,
                                   columns=cols,
                                   show="headings",
                                   style="Dark.Treeview",
                                   selectmode="browse")
        # Column widths
        widths = {
            "ID": 40, "Tanggal": 90, "Waktu": 72, "Sumber": 110,
            "Jml Objek": 72, "Jml Gerakan": 80,
            "Nama Objek": 160, "Objek": 120,
            "Info Area": 120, "Gambar": 180,
        }
        for col in cols:
            w = widths.get(col, 100)
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=w, minwidth=40, anchor="w")

        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal",
                             command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        # Row tags (alternating + highlight)
        self.tree.tag_configure("odd",  background="#0f0f22")
        self.tree.tag_configure("even", background=DARK_BG)
        self.tree.tag_configure("alert", background="#2a0000", foreground="#ff8888")

        # Double-click → preview gambar
        self.tree.bind("<Double-1>", self._on_double_click)

        # ── Pagination bar
        page_bar = tk.Frame(self, bg=PANEL_BG, pady=4)
        page_bar.pack(fill="x")

        bc = {"font": ("Helvetica", 9), "relief": "flat",
              "cursor": "hand2", "bg": PANEL_BG, "fg": TEXT_DIM,
              "padx": 8, "pady": 2}
        self.btn_prev = tk.Button(page_bar, text="◀ Prev",
                                   command=self._prev_page, **bc)
        self.btn_prev.pack(side="left", padx=8)
        self.page_lbl = tk.Label(page_bar, text="Hal 1",
                                  bg=PANEL_BG, fg=TEXT_DIM,
                                  font=("Helvetica", 9))
        self.page_lbl.pack(side="left")
        self.btn_next = tk.Button(page_bar, text="Next ▶",
                                   command=self._next_page, **bc)
        self.btn_next.pack(side="left", padx=8)

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self):
        """Ambil data dari database dan tampilkan."""
        self._data = _get_all(self.module)
        self._page = 0
        self._apply_filter()

    def get_selected_record(self) -> dict | None:
        sel = self.tree.selection()
        if not sel:
            return None
        iid  = sel[0]
        vals = self.tree.item(iid, "values")
        cols = TABLE_HEADINGS[self.module]
        rec  = dict(zip(cols, vals))
        # Cari record asli dari _filtered (berdasarkan ID)
        rid  = int(rec.get("ID", -1))
        for r in self._filtered:
            if r.get("id") == rid:
                return r
        return None

    # ── Filtering & Sorting ───────────────────────────────────────────────────

    def _apply_filter(self):
        kw   = self.search_var.get().lower().strip()
        date = self.date_filter_var.get().strip()

        self._filtered = [
            r for r in self._data
            if (not kw or any(kw in str(v).lower() for v in r.values()))
            and (not date or r.get("date", "").startswith(date))
        ]

        if self._sort_col:
            key_map = {
                "ID": "id", "Tanggal": "date", "Waktu": "time",
                "Sumber": "source",
                "Jml Objek": "total_objects",
                "Jml Gerakan": "motion_count",
                "Nama Objek": "object_names", "Objek": "object_names",
                "Info Area": "area_info", "Gambar": "image_path",
            }
            db_key = key_map.get(self._sort_col, "id")
            self._filtered.sort(
                key=lambda r: str(r.get(db_key, "")),
                reverse=self._sort_rev
            )

        total = len(self._filtered)
        pages = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self._page = min(self._page, pages - 1)
        self._render_page()
        self.count_lbl.config(text=f"{total} records")

    def _sort_by(self, col: str):
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        self._apply_filter()

    def _render_page(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        start = self._page * self.PAGE_SIZE
        end   = start + self.PAGE_SIZE
        page_data = getattr(self, "_filtered", self._data)[start:end]

        for i, r in enumerate(page_data):
            tag = "even" if i % 2 == 0 else "odd"
            vals = self._record_to_row(r)
            self.tree.insert("", "end", values=vals, tags=(tag,))

        total  = len(getattr(self, "_filtered", self._data))
        pages  = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page_lbl.config(
            text=f"Hal {self._page+1}/{pages}")
        self.btn_prev.config(state="normal" if self._page > 0 else "disabled")
        self.btn_next.config(state="normal" if self._page < pages-1 else "disabled")

    def _record_to_row(self, r: dict) -> tuple:
        if self.module == "object":
            return (r.get("id",""), r.get("date",""), r.get("time",""),
                    r.get("source",""), r.get("total_objects",""),
                    r.get("object_names",""), r.get("image_path",""))
        elif self.module == "motion":
            return (r.get("id",""), r.get("date",""), r.get("time",""),
                    r.get("source",""), r.get("motion_count",""),
                    r.get("image_path",""))
        elif self.module == "anomaly":
            return (r.get("id",""), r.get("date",""), r.get("time",""),
                    r.get("source",""), r.get("object_names",""),
                    r.get("area_info",""), r.get("image_path",""))

    def _prev_page(self):
        if self._page > 0:
            self._page -= 1
            self._render_page()

    def _next_page(self):
        total = len(getattr(self, "_filtered", self._data))
        pages = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        if self._page < pages - 1:
            self._page += 1
            self._render_page()

    def _on_double_click(self, event):
        rec = self.get_selected_record()
        if rec and rec.get("image_path") and os.path.exists(rec["image_path"]):
            self._preview_image(rec["image_path"])

    def _preview_image(self, path: str):
        import cv2
        img = cv2.imread(path)
        if img is None:
            return
        from PIL import Image, ImageTk
        import tkinter as tk
        pw = tk.Toplevel()
        pw.title(f"🖼️ {os.path.basename(path)}")
        pw.configure(bg=DARK_BG)
        h, w = img.shape[:2]
        scale = min(800/w, 600/h, 1.0)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        lbl = tk.Label(pw, image=photo, bg=DARK_BG)
        lbl.image = photo
        lbl.pack(padx=8, pady=8)


# ──────────────────────────────────────────────────────────────────────────────
# Main Data Management Window

class DataManagementWindow:
    WIN_W, WIN_H = 1200, 780  # akan di-override di __init__

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("📊 Modul 5 – Data Management")
        self.win.configure(bg=DARK_BG)
        from window_utils import get_module_sizes
        _sz = get_module_sizes(parent)
        self.WIN_W = _sz["data_w"]
        self.WIN_H = _sz["data_h"]
        utils.center_window(self.win, self.WIN_W, self.WIN_H)
        self.win.resizable(True, True)
        self.win.minsize(820, 520)
        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)

        self._tables: dict[str, DataTableWidget] = {}
        self._active_module = tk.StringVar(value="object")

        self._build_ui()
        self._load_current()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.win, bg=HEADER_BG, height=56)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📊  DATA MANAGEMENT  —  Input · Lihat · Edit · Hapus · Import CSV",
                 font=("Helvetica", 13, "bold"),
                 bg=HEADER_BG, fg=ACCENT).pack(side="left", padx=18, pady=15)

        # ── Module Tab Bar
        tab_bar = tk.Frame(self.win, bg=PANEL_BG, height=48)
        tab_bar.pack(fill="x")
        tab_bar.pack_propagate(False)

        self._tab_btns = {}
        for mod, label in MODULE_LABELS.items():
            color = MODULE_COLORS[mod]
            b = tk.Button(tab_bar, text=label,
                          bg=PANEL_BG, fg=TEXT_DIM,
                          font=("Helvetica", 10, "bold"),
                          relief="flat", cursor="hand2",
                          padx=16, pady=10,
                          command=lambda m=mod: self._switch_module(m))
            b.pack(side="left")
            self._tab_btns[mod] = (b, color)

        # ── Body: Left sidebar + Right table
        body = tk.Frame(self.win, bg=DARK_BG)
        body.pack(fill="both", expand=True)

        # Left sidebar (actions)
        self.sidebar = tk.Frame(body, bg=PANEL_BG, width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # Right: stacked table widgets, satu per modul
        self.table_container = tk.Frame(body, bg=DARK_BG)
        self.table_container.pack(side="left", fill="both", expand=True)

        for mod in MODULE_LABELS:
            tbl = DataTableWidget(self.table_container, mod)
            self._tables[mod] = tbl

        # ── Status bar
        self.status_bar = tk.Label(self.win,
                                    text="Siap",
                                    bg=HEADER_BG, fg=TEXT_DIM,
                                    font=("Helvetica", 8), anchor="w")
        self.status_bar.pack(fill="x", side="bottom", ipady=3, padx=8)

        # Aktifkan tab pertama
        self._switch_module("object")

    def _build_sidebar(self):
        pad = {"padx": 12, "pady": 4}
        bc  = {"font": ("Helvetica", 10, "bold"), "relief": "flat",
               "cursor": "hand2", "pady": 9}

        tk.Label(self.sidebar, text="⚙️  AKSI DATA",
                 bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", **pad)

        tk.Button(self.sidebar, text="➕  TAMBAH RECORD",
                  bg="#00aa44", fg="white",
                  command=self._add_record, **bc).pack(fill="x", padx=10, pady=3)

        tk.Button(self.sidebar, text="✏️  EDIT RECORD",
                  bg="#1a5276", fg="white",
                  command=self._edit_record, **bc).pack(fill="x", padx=10, pady=3)

        tk.Button(self.sidebar, text="🗑️  HAPUS RECORD",
                  bg="#773333", fg="white",
                  command=self._delete_record, **bc).pack(fill="x", padx=10, pady=3)

        tk.Button(self.sidebar, text="🔄  REFRESH DATA",
                  bg="#333355", fg="white",
                  command=self._load_current, **bc).pack(fill="x", padx=10, pady=3)

        utils.make_separator(self.sidebar, "#2a2a4e", pady=6)

        tk.Label(self.sidebar, text="📤  EXPORT / IMPORT",
                 bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", **pad)

        tk.Button(self.sidebar, text="📥  IMPORT CSV",
                  bg="#555555", fg="white",
                  command=self._import_csv, **bc).pack(fill="x", padx=10, pady=3)

        tk.Button(self.sidebar, text="📤  EXPORT CSV",
                  bg="#555555", fg="white",
                  command=self._export_csv, **bc).pack(fill="x", padx=10, pady=3)

        utils.make_separator(self.sidebar, "#2a2a4e", pady=6)

        tk.Label(self.sidebar, text="🧹  KELOLA DATA",
                 bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", **pad)

        tk.Button(self.sidebar, text="🗑️  HAPUS SEMUA DATA",
                  bg="#550000", fg="white",
                  command=self._delete_all, **bc).pack(fill="x", padx=10, pady=3)

        utils.make_separator(self.sidebar, "#2a2a4e", pady=6)

        # ── Summary stats
        tk.Label(self.sidebar, text="📈  RINGKASAN",
                 bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", **pad)
        self.summary_frame = tk.Frame(self.sidebar, bg=DARK_BG, padx=8, pady=8)
        self.summary_frame.pack(fill="x", padx=10, pady=2)
        self.lbl_total     = self._srow(self.summary_frame, "Total Record", "0")
        self.lbl_today     = self._srow(self.summary_frame, "Hari ini", "0")
        self.lbl_with_img  = self._srow(self.summary_frame, "Dengan Foto", "0")

    def _srow(self, parent, label, init) -> tk.Label:
        row = tk.Frame(parent, bg=DARK_BG)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label+":", bg=DARK_BG, fg=TEXT_DIM,
                 font=("Helvetica", 8), width=13, anchor="w").pack(side="left")
        v = tk.Label(row, text=init, bg=DARK_BG,
                     fg="#ffdd00", font=("Courier", 9, "bold"))
        v.pack(side="right")
        return v

    # ── Tab switching ─────────────────────────────────────────────────────────

    def _switch_module(self, mod: str):
        self._active_module.set(mod)
        color = MODULE_COLORS[mod]

        # Update tab appearance
        for m, (btn, c) in self._tab_btns.items():
            if m == mod:
                btn.config(bg=c, fg="white",
                            relief="groove")
            else:
                btn.config(bg=PANEL_BG, fg=TEXT_DIM,
                            relief="flat")

        # Show/hide tables
        for m, tbl in self._tables.items():
            if m == mod:
                tbl.pack(fill="both", expand=True)
                tbl.load()
            else:
                tbl.pack_forget()

        self._update_summary()
        self._set_status(f"Menampilkan data: {MODULE_LABELS[mod]}")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def _add_record(self):
        mod = self._active_module.get()
        dlg = RecordFormDialog(self.win, mod, record=None)
        self.win.wait_window(dlg.win)
        if dlg.result:
            self._tables[mod].load()
            self._update_summary()
            self._set_status("✅ Record baru berhasil ditambahkan.")

    def _edit_record(self):
        mod = self._active_module.get()
        rec = self._tables[mod].get_selected_record()
        if not rec:
            messagebox.showwarning("Info", "Pilih record terlebih dahulu.",
                                   parent=self.win)
            return
        dlg = RecordFormDialog(self.win, mod, record=rec)
        self.win.wait_window(dlg.win)
        if dlg.result:
            self._tables[mod].load()
            self._update_summary()
            self._set_status(f"✅ Record ID={rec['id']} berhasil diupdate.")

    def _delete_record(self):
        mod = self._active_module.get()
        rec = self._tables[mod].get_selected_record()
        if not rec:
            messagebox.showwarning("Info", "Pilih record terlebih dahulu.",
                                   parent=self.win)
            return
        if not messagebox.askyesno("Konfirmasi",
                                    f"Hapus record ID={rec['id']}?",
                                    parent=self.win):
            return
        _delete_record(mod, rec["id"])
        self._tables[mod].load()
        self._update_summary()
        self._set_status(f"🗑️ Record ID={rec['id']} dihapus.")

    def _delete_all(self):
        mod = self._active_module.get()
        if not messagebox.askyesno("Konfirmasi",
                                    f"Hapus SEMUA data {MODULE_LABELS[mod]}?\n"
                                    "Tindakan ini tidak bisa dibatalkan!",
                                    icon="warning",
                                    parent=self.win):
            return
        conn = db.get_connection()
        conn.execute(f"DELETE FROM {mod}_detection_log")
        conn.commit()
        conn.close()
        self._tables[mod].load()
        self._update_summary()
        self._set_status(f"🗑️ Semua data {MODULE_LABELS[mod]} dihapus.")

    # ── Import / Export CSV ───────────────────────────────────────────────────

    def _import_csv(self):
        mod  = self._active_module.get()
        path = filedialog.askopenfilename(
            parent=self.win, title="Import CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return

        self._set_status("⏳ Mengimport data…")
        self.win.update()

        def _run():
            try:
                count = 0
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            if mod == "object":
                                _insert_object(
                                    row.get("date", ""),
                                    row.get("time", ""),
                                    row.get("source", "import"),
                                    int(row.get("total_objects", 0)),
                                    row.get("object_names", ""),
                                    row.get("image_path", ""))
                            elif mod == "motion":
                                _insert_motion(
                                    row.get("date", ""),
                                    row.get("time", ""),
                                    row.get("source", "import"),
                                    int(row.get("motion_count", 0)),
                                    row.get("image_path", ""))
                            elif mod == "anomaly":
                                _insert_anomaly(
                                    row.get("date", ""),
                                    row.get("time", ""),
                                    row.get("source", "import"),
                                    row.get("object_names", ""),
                                    row.get("area_info", ""),
                                    row.get("image_path", ""))
                            count += 1
                        except Exception:
                            continue   # skip baris error

                self.win.after(0, lambda: self._after_import(count))
            except Exception as e:
                self.win.after(0, lambda: messagebox.showerror(
                    "Error Import", str(e), parent=self.win))

        threading.Thread(target=_run, daemon=True).start()

    def _after_import(self, count: int):
        mod = self._active_module.get()
        self._tables[mod].load()
        self._update_summary()
        self._set_status(f"✅ Import selesai — {count} record berhasil dimasukkan.")
        messagebox.showinfo("Import Selesai",
                             f"✅ {count} record berhasil diimport.",
                             parent=self.win)

    def _export_csv(self):
        mod = self._active_module.get()
        data = _get_all(mod)
        if not data:
            messagebox.showinfo("Info", "Tidak ada data untuk diekspor.", parent=self.win)
            return

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"{mod}_export_{ts}.csv"
        path = filedialog.asksaveasfilename(
            parent=self.win,
            title="Simpan CSV",
            initialfile=default,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

        self._set_status(f"✅ Ekspor selesai → {path}")
        messagebox.showinfo("Export Selesai",
                             f"✅ {len(data)} record diekspor:\n{path}",
                             parent=self.win)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_current(self):
        mod = self._active_module.get()
        self._tables[mod].load()
        self._update_summary()

    def _update_summary(self):
        mod  = self._active_module.get()
        data = _get_all(mod)
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = sum(1 for r in data if r.get("date") == today)
        with_img    = sum(1 for r in data if r.get("image_path", ""))
        self.lbl_total.config(text=str(len(data)))
        self.lbl_today.config(text=str(today_count))
        self.lbl_with_img.config(text=str(with_img))

    def _set_status(self, msg: str):
        self.status_bar.config(text=f"  {msg}")
