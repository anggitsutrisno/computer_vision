"""
custom_objects.py - Smart Vision Analysis System
Modul manajemen objek kustom untuk Image Detection.

Mendukung dua strategi pengenalan:
  1. YOLO Alias  — ganti nama label YOLO yang sudah ada dengan nama kustom
  2. Template Matching — kenali objek baru dari foto contoh (ORB feature matching)

Data disimpan di: custom_objects/
  ├── registry.json          ← daftar semua objek terdaftar
  └── templates/<nama>/      ← folder foto template per objek
"""

import cv2
import json
import os
import shutil
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR   = os.path.join(_BASE_DIR, "custom_objects")
REGISTRY_PATH = os.path.join(CUSTOM_DIR, "registry.json")
TEMPLATE_DIR = os.path.join(CUSTOM_DIR, "templates")

# ─── 80 Kelas COCO (YOLO default) ─────────────────────────────────────────────

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush",
]

# ─── Registry helpers ─────────────────────────────────────────────────────────

def _ensure_dirs():
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)


def load_registry() -> dict:
    _ensure_dirs()
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(reg: dict):
    _ensure_dirs()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def add_alias(custom_name: str, yolo_class: str) -> str:
    """
    Tambah objek tipe ALIAS: saat YOLO mendeteksi `yolo_class`,
    label ditampilkan sebagai `custom_name`.
    Return: pesan sukses/error
    """
    reg = load_registry()
    key = custom_name.strip().lower()
    if not key:
        return "❌ Nama tidak boleh kosong"
    reg[key] = {
        "name": custom_name.strip(),
        "type": "alias",
        "yolo_class": yolo_class,
        "added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "templates": [],
    }
    save_registry(reg)
    return f"✅ Alias '{custom_name}' → '{yolo_class}' berhasil ditambahkan"


def add_template_object(custom_name: str, image_paths: list) -> str:
    """
    Tambah objek tipe TEMPLATE: kenali lewat foto contoh.
    Return: pesan sukses/error
    """
    key = custom_name.strip().lower()
    if not key:
        return "❌ Nama tidak boleh kosong"
    if not image_paths:
        return "❌ Minimal 1 foto template diperlukan"

    tmpl_folder = os.path.join(TEMPLATE_DIR, key)
    os.makedirs(tmpl_folder, exist_ok=True)

    # Salin foto ke folder template
    saved = []
    for src in image_paths:
        if not os.path.exists(src):
            continue
        ext = os.path.splitext(src)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        dst = os.path.join(tmpl_folder, f"tmpl_{len(saved)+1:03d}{ext}")
        shutil.copy2(src, dst)
        saved.append(dst)

    if not saved:
        return "❌ Tidak ada file gambar yang valid"

    reg = load_registry()
    reg[key] = {
        "name": custom_name.strip(),
        "type": "template",
        "yolo_class": None,
        "added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "templates": saved,
    }
    save_registry(reg)
    return f"✅ '{custom_name}' ditambahkan dengan {len(saved)} foto template"


def delete_object(key: str) -> str:
    reg = load_registry()
    key = key.lower()
    if key not in reg:
        return "❌ Objek tidak ditemukan"
    obj = reg.pop(key)
    save_registry(reg)
    # Hapus folder template jika ada
    tmpl_folder = os.path.join(TEMPLATE_DIR, key)
    if os.path.isdir(tmpl_folder):
        shutil.rmtree(tmpl_folder)
    return f"✅ '{obj['name']}' berhasil dihapus"


# ─── Detection helpers ────────────────────────────────────────────────────────

def build_alias_map() -> dict:
    """
    Return dict {yolo_class_name: custom_name} untuk semua alias terdaftar.
    Digunakan saat render bounding box.
    """
    reg = load_registry()
    return {
        v["yolo_class"]: v["name"]
        for v in reg.values()
        if v.get("type") == "alias" and v.get("yolo_class")
    }


class TemplateDetector:
    """
    Deteksi objek kustom lewat ORB feature matching.
    Dibuat sekali, digunakan berulang kali di capture loop.
    """

    MIN_MATCH = 12       # jumlah keypoint match minimum
    RATIO_THRESH = 0.75  # Lowe's ratio test

    def __init__(self):
        self._orb = cv2.ORB_create(nfeatures=500)
        self._bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._db: dict = {}   # {key: [(des, kp, img_gray), ...]}
        self._reload()

    def _reload(self):
        """Muat ulang semua template dari disk."""
        self._db.clear()
        reg = load_registry()
        for key, obj in reg.items():
            if obj.get("type") != "template":
                continue
            templates = []
            for path in obj.get("templates", []):
                if not os.path.exists(path):
                    continue
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                kp, des = self._orb.detectAndCompute(img, None)
                if des is not None and len(des) >= 5:
                    templates.append((des, kp, img, obj["name"]))
            if templates:
                self._db[key] = templates

    def reload(self):
        self._reload()

    def detect(self, frame: np.ndarray) -> list:
        """
        Cari objek kustom di frame.
        Return: list of {"label": str, "confidence": float, "bbox": (x1,y1,x2,y2)}
        """
        if not self._db:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_q, des_q = self._orb.detectAndCompute(gray, None)
        if des_q is None or len(des_q) < 5:
            return []

        results = []
        for key, templates in self._db.items():
            for (des_t, kp_t, img_t, name) in templates:
                try:
                    matches = self._bf.knnMatch(des_t, des_q, k=2)
                except Exception:
                    continue

                # Lowe's ratio test
                good = []
                for m_pair in matches:
                    if len(m_pair) == 2:
                        m, n = m_pair
                        if m.distance < self.RATIO_THRESH * n.distance:
                            good.append(m)

                if len(good) < self.MIN_MATCH:
                    continue

                # Estimasi lokasi dengan homografi
                src_pts = np.float32(
                    [kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp_q[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is None:
                    continue

                inliers = int(mask.sum()) if mask is not None else 0
                if inliers < 8:
                    continue

                # Proyeksikan sudut template ke frame
                h, w = img_t.shape[:2]
                corners = np.float32(
                    [[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
                dst_corners = cv2.perspectiveTransform(corners, M)

                xs = dst_corners[:, 0, 0]
                ys = dst_corners[:, 0, 1]
                fh, fw = frame.shape[:2]
                x1 = max(0, int(xs.min()))
                y1 = max(0, int(ys.min()))
                x2 = min(fw, int(xs.max()))
                y2 = min(fh, int(ys.max()))

                if x2 <= x1 or y2 <= y1:
                    continue

                conf = min(1.0, inliers / 20.0)
                results.append({
                    "label": name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "cls": -1,   # marker: ini bukan kelas YOLO
                })
                break  # satu template cukup per objek per frame

        return results


def draw_custom_detections(frame: np.ndarray, dets: list) -> np.ndarray:
    """Gambar bounding box untuk hasil template detection."""
    COLOR = (0, 220, 140)   # hijau toska — beda dari YOLO
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        label = d["label"]
        conf  = d["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        text = f"[C] {label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), COLOR, -1)
        cv2.putText(frame, text, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# CustomObjectManager — Jendela manajemen objek kustom
# ══════════════════════════════════════════════════════════════════════════════

DARK   = "#0d0d1a"
PANEL  = "#13132b"
HEADER = "#1a1a2e"
ACCENT = "#00d4ff"
GREEN  = "#00aa44"
RED    = "#aa2222"
TEXT_DIM = "#6666aa"
TEXT_BR  = "#eeeeff"


class CustomObjectManagerWindow:
    """
    Jendela manajemen objek kustom.
    Dipanggil dari ImageDetectionWindow via tombol "🎯 OBJEK KUSTOM".
    """

    def __init__(self, parent, on_reload_callback=None):
        self.parent = parent
        self.on_reload = on_reload_callback   # dipanggil setelah add/delete

        self.win = tk.Toplevel(parent)
        self.win.title("🎯 Manajemen Objek Kustom")
        self.win.configure(bg=DARK)
        self.win.geometry("860x620")
        self.win.minsize(800, 560)
        self.win.resizable(True, True)
        _center(self.win, 860, 620)

        self._photo_refs = []      # cegah GC PhotoImage
        self._selected_paths = []  # path foto yang dipilih untuk template

        self._build_ui()
        self._refresh_list()

    # ── UI Builder ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.win, bg=HEADER, height=50)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🎯  MANAJEMEN OBJEK KUSTOM",
                 font=("Helvetica", 13, "bold"),
                 bg=HEADER, fg=ACCENT).pack(side="left", padx=16, pady=12)
        tk.Label(hdr,
                 text="Tambah nama alias atau objek baru lewat foto template",
                 font=("Helvetica", 9), bg=HEADER, fg=TEXT_DIM
                 ).pack(side="left", padx=4)

        # Body: kiri=list, kanan=form
        body = tk.Frame(self.win, bg=DARK)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=PANEL, width=310)
        left.pack(side="left", fill="both", padx=(0, 6))
        left.pack_propagate(False)

        tk.Label(left, text="📋  OBJEK TERDAFTAR",
                 bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 4))

        # Treeview
        cols = ("Nama", "Tipe", "Info")
        self.tree = ttk.Treeview(left, columns=cols, show="headings",
                                  selectmode="browse", height=18)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#0d0d1a", foreground="#ccccee",
                        fieldbackground="#0d0d1a", rowheight=26,
                        font=("Helvetica", 9))
        style.configure("Treeview.Heading",
                        background="#1a1a2e", foreground=ACCENT,
                        font=("Helvetica", 9, "bold"))
        style.map("Treeview", background=[("selected", "#1a3a5e")])

        self.tree.heading("Nama", text="Nama Kustom")
        self.tree.heading("Tipe", text="Tipe")
        self.tree.heading("Info", text="Detail")
        self.tree.column("Nama", width=110)
        self.tree.column("Tipe", width=70, anchor="center")
        self.tree.column("Info", width=120)

        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=4)
        sb.pack(side="left", fill="y", pady=4)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Tombol bawah
        btn_row = tk.Frame(left, bg=PANEL)
        btn_row.pack(fill="x", padx=10, pady=6)
        bk = {"font": ("Helvetica", 9, "bold"), "relief": "flat",
              "cursor": "hand2", "pady": 6}
        tk.Button(btn_row, text="🗑  HAPUS",
                  bg="#6b1010", fg="white",
                  command=self._delete_selected, **bk
                  ).pack(side="left", fill="x", expand=True)
        tk.Button(btn_row, text="🔄  REFRESH",
                  bg="#1a3a5e", fg="white",
                  command=self._refresh_list, **bk
                  ).pack(side="left", fill="x", expand=True, padx=(6, 0))

        # Preview panel
        tk.Frame(left, bg="#2a2a4e", height=1).pack(fill="x", padx=10, pady=2)
        tk.Label(left, text="🖼  Preview template:",
                 bg=PANEL, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(anchor="w", padx=10)
        self.preview_label = tk.Label(left, bg="#080818",
                                       text="Pilih objek template",
                                       fg=TEXT_DIM, font=("Helvetica", 8))
        self.preview_label.pack(fill="x", padx=10, pady=(2, 8))

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=PANEL)
        right.pack(side="right", fill="both", expand=True)

        # Tab: Alias vs Template
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        style = ttk.Style()
        style.configure("TNotebook", background=PANEL)
        style.configure("TNotebook.Tab",
                        background=DARK, foreground=TEXT_DIM,
                        padding=[12, 5], font=("Helvetica", 9, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#1a1a3e")],
                  foreground=[("selected", ACCENT)])

        self._build_alias_tab(nb)
        self._build_template_tab(nb)

        # Status bar
        self.status = tk.Label(right, text="", bg=PANEL, fg="#aaffaa",
                                font=("Helvetica", 9), wraplength=500)
        self.status.pack(pady=(0, 8), padx=8)

    def _build_alias_tab(self, nb):
        frame = tk.Frame(nb, bg=DARK, padx=20, pady=20)
        nb.add(frame, text="  🔖  Alias YOLO  ")

        tk.Label(frame,
                 text="Ganti nama label YOLO yang sudah ada dengan nama kustom kamu.",
                 bg=DARK, fg=TEXT_DIM, font=("Helvetica", 9),
                 wraplength=480, justify="left").pack(anchor="w", pady=(0, 16))

        # Nama kustom
        self._al_name = tk.StringVar()
        self._form_row(frame, "📛  Nama kustom:", self._al_name,
                        hint="Contoh: Motor Honda")

        # YOLO class combobox
        tk.Frame(frame, bg=DARK, height=6).pack()
        row = tk.Frame(frame, bg=DARK); row.pack(fill="x", pady=4)
        tk.Label(row, text="🤖  Kelas YOLO:", bg=DARK, fg=TEXT_DIM,
                 font=("Helvetica", 10), width=20, anchor="w").pack(side="left")
        self._al_yolo_var = tk.StringVar(value=COCO_CLASSES[0])
        cb = ttk.Combobox(row, textvariable=self._al_yolo_var,
                           values=COCO_CLASSES, state="readonly", width=28,
                           font=("Helvetica", 10))
        cb.pack(side="left", fill="x", expand=True, ipady=4, padx=(8, 0))

        # Info hint box
        info = tk.Frame(frame, bg="#0a1a2a", padx=12, pady=10)
        info.pack(fill="x", pady=16)
        tk.Label(info,
                 text="ℹ️  Cara kerja Alias:",
                 bg="#0a1a2a", fg=ACCENT,
                 font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(info,
                 text=(
                     "Saat YOLO mendeteksi kelas yang dipilih, nama yang\n"
                     "ditampilkan di bounding box akan diganti dengan nama\n"
                     "kustom kamu. Cocok untuk objek yang sudah dikenali\n"
                     "YOLO tapi ingin diberi label berbeda."
                 ),
                 bg="#0a1a2a", fg="#7799bb",
                 font=("Helvetica", 8), justify="left").pack(anchor="w", pady=(4, 0))

        tk.Frame(frame, bg="#2a2a4e", height=1).pack(fill="x", pady=12)
        bk = {"font": ("Helvetica", 11, "bold"), "relief": "flat",
              "cursor": "hand2", "pady": 10}
        tk.Button(frame, text="💾  TAMBAH ALIAS",
                  bg=GREEN, fg="white",
                  command=self._save_alias, **bk
                  ).pack(fill="x")

    def _build_template_tab(self, nb):
        frame = tk.Frame(nb, bg=DARK, padx=20, pady=20)
        nb.add(frame, text="  📸  Template Baru  ")

        tk.Label(frame,
                 text="Daftarkan objek baru yang tidak ada di YOLO lewat foto contoh.",
                 bg=DARK, fg=TEXT_DIM, font=("Helvetica", 9),
                 wraplength=480, justify="left").pack(anchor="w", pady=(0, 12))

        # Nama
        self._tmpl_name = tk.StringVar()
        self._form_row(frame, "📛  Nama objek:", self._tmpl_name,
                        hint="Contoh: Helm Proyek")

        # Upload foto
        tk.Frame(frame, bg=DARK, height=8).pack()
        row = tk.Frame(frame, bg=DARK); row.pack(fill="x", pady=4)
        tk.Label(row, text="📂  Foto template:", bg=DARK, fg=TEXT_DIM,
                 font=("Helvetica", 10), width=20, anchor="w").pack(side="left")
        self._tmpl_count_lbl = tk.Label(row, text="0 foto dipilih",
                                         bg=DARK, fg="#ffcc00",
                                         font=("Courier", 10, "bold"))
        self._tmpl_count_lbl.pack(side="left", padx=(8, 0))
        tk.Button(row, text="  📂 Pilih Foto  ",
                  bg="#1a3a5e", fg="white",
                  font=("Helvetica", 9, "bold"), relief="flat",
                  cursor="hand2", pady=4,
                  command=self._pick_templates
                  ).pack(side="right")

        # Thumbnail strip
        self.thumb_frame = tk.Frame(frame, bg="#080818", height=80)
        self.thumb_frame.pack(fill="x", pady=6)
        self.thumb_frame.pack_propagate(False)

        # Info hint
        info = tk.Frame(frame, bg="#0a1a2a", padx=12, pady=10)
        info.pack(fill="x", pady=12)
        tk.Label(info, text="ℹ️  Tips foto template:",
                 bg="#0a1a2a", fg=ACCENT,
                 font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(info,
                 text=(
                     "• Gunakan 3–10 foto dari sudut berbeda\n"
                     "• Foto terang & jelas, background kontras\n"
                     "• Resolusi minimal 200×200 piksel\n"
                     "• Cocok untuk objek unik/spesifik (logo, produk, alat)"
                 ),
                 bg="#0a1a2a", fg="#7799bb",
                 font=("Helvetica", 8), justify="left").pack(anchor="w", pady=(4, 0))

        tk.Frame(frame, bg="#2a2a4e", height=1).pack(fill="x", pady=10)
        bk = {"font": ("Helvetica", 11, "bold"), "relief": "flat",
              "cursor": "hand2", "pady": 10}
        tk.Button(frame, text="💾  TAMBAH OBJEK TEMPLATE",
                  bg=GREEN, fg="white",
                  command=self._save_template, **bk
                  ).pack(fill="x")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _form_row(self, parent, label, var, hint=""):
        row = tk.Frame(parent, bg=DARK); row.pack(fill="x", pady=4)
        tk.Label(row, text=label, bg=DARK, fg=TEXT_DIM,
                 font=("Helvetica", 10), width=20, anchor="w").pack(side="left")
        e = tk.Entry(row, textvariable=var, bg=PANEL, fg=TEXT_BR,
                     insertbackground="white",
                     font=("Courier", 11), relief="flat",
                     highlightthickness=1,
                     highlightcolor=ACCENT,
                     highlightbackground="#2a2a4e")
        e.pack(side="left", fill="x", expand=True, ipady=5, padx=(8, 0))
        if hint:
            tk.Label(row, text=hint, bg=DARK, fg="#334455",
                     font=("Helvetica", 8)).pack(side="left", padx=6)
        return e

    def _set_status(self, msg: str, ok: bool = True):
        color = "#00ff88" if ok else "#ff4444"
        self.status.config(text=msg, fg=color)

    def _refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        reg = load_registry()
        for key, obj in reg.items():
            tipe = "Alias" if obj["type"] == "alias" else "Template"
            info = (obj["yolo_class"] if obj["type"] == "alias"
                    else f"{len(obj['templates'])} foto")
            self.tree.insert("", "end", iid=key,
                              values=(obj["name"], tipe, info))

    def _on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        key = sel[0]
        reg = load_registry()
        obj = reg.get(key)
        if not obj or obj["type"] != "template":
            self.preview_label.config(image="", text="(alias — tidak ada foto)",
                                       compound="none")
            return

        templates = [p for p in obj["templates"] if os.path.exists(p)]
        if not templates:
            self.preview_label.config(image="", text="Foto template tidak ditemukan",
                                       compound="none")
            return

        # Tampilkan foto pertama sebagai preview
        try:
            img = Image.open(templates[0])
            img.thumbnail((270, 80))
            photo = ImageTk.PhotoImage(img)
            self._photo_refs = [photo]
            self.preview_label.config(image=photo, text="", compound="none")
        except Exception:
            self.preview_label.config(image="", text="Gagal load preview", compound="none")

    def _pick_templates(self):
        paths = filedialog.askopenfilenames(
            parent=self.win,
            title="Pilih foto template (bisa lebih dari 1)",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if not paths:
            return
        self._selected_paths = list(paths)
        self._tmpl_count_lbl.config(text=f"{len(self._selected_paths)} foto dipilih")

        # Tampilkan thumbnails
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self._photo_refs = []
        for p in self._selected_paths[:8]:
            try:
                img = Image.open(p)
                img.thumbnail((70, 70))
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)
                lbl = tk.Label(self.thumb_frame, image=photo,
                               bg="#080818", relief="flat")
                lbl.pack(side="left", padx=3, pady=4)
            except Exception:
                pass

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Info", "Pilih objek yang ingin dihapus.",
                                   parent=self.win)
            return
        key = sel[0]
        reg = load_registry()
        name = reg.get(key, {}).get("name", key)
        if not messagebox.askyesno("Konfirmasi",
                                    f"Hapus objek '{name}'?", parent=self.win):
            return
        msg = delete_object(key)
        self._set_status(msg, ok="✅" in msg)
        self._refresh_list()
        self.preview_label.config(image="", text="Pilih objek template",
                                   compound="none")
        if self.on_reload:
            self.on_reload()

    def _save_alias(self):
        name  = self._al_name.get().strip()
        yolo  = self._al_yolo_var.get()
        if not name:
            self._set_status("❌ Nama kustom tidak boleh kosong", ok=False)
            return
        msg = add_alias(name, yolo)
        self._set_status(msg, ok="✅" in msg)
        if "✅" in msg:
            self._al_name.set("")
            self._refresh_list()
            if self.on_reload:
                self.on_reload()

    def _save_template(self):
        name = self._tmpl_name.get().strip()
        if not name:
            self._set_status("❌ Nama objek tidak boleh kosong", ok=False)
            return
        if not self._selected_paths:
            self._set_status("❌ Pilih minimal 1 foto template dulu", ok=False)
            return
        msg = add_template_object(name, self._selected_paths)
        self._set_status(msg, ok="✅" in msg)
        if "✅" in msg:
            self._tmpl_name.set("")
            self._selected_paths = []
            self._tmpl_count_lbl.config(text="0 foto dipilih")
            for w in self.thumb_frame.winfo_children():
                w.destroy()
            self._refresh_list()
            if self.on_reload:
                self.on_reload()


# ── Misc ──────────────────────────────────────────────────────────────────────

def _center(win, w, h):
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
