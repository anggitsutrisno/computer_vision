"""
utils.py - Smart Vision Analysis System
Fungsi-fungsi utilitas bersama untuk seluruh modul.
"""

import cv2
import numpy as np
import os
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk


# ─── Konstanta Warna (BGR) ────────────────────────────────────────────────────

COLOR_GREEN   = (0, 255, 0)
COLOR_RED     = (0, 0, 255)
COLOR_BLUE    = (255, 0, 0)
COLOR_YELLOW  = (0, 255, 255)
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)
COLOR_ORANGE  = (0, 165, 255)
COLOR_CYAN    = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)

# Palette warna untuk YOLO classes
YOLO_COLORS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (255, 152, 0), (236, 112, 99), (255, 87, 34), (211, 47, 47),
    (233, 30, 99), (156, 39, 176), (103, 58, 183), (63, 81, 181),
]


# ─── Path Helper ─────────────────────────────────────────────────────────────

def ensure_dirs():
    """Buat semua folder output yang diperlukan."""
    dirs = [
        "output/object",
        "output/motion",
        "output/anomaly",
        "output/image_processing",
        "output/charts",
        "output/screenshots",
        "logs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def timestamped_filename(prefix: str, ext: str = "jpg") -> str:
    """Buat nama file berdasarkan timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
    return f"{prefix}_{ts}.{ext}"


def save_frame(frame: np.ndarray, folder: str, prefix: str) -> str:
    """Simpan frame ke folder output. Kembalikan path file."""
    ensure_dirs()
    filename = timestamped_filename(prefix)
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame)
    return path


def screenshot(frame: np.ndarray) -> str:
    """Simpan screenshot ke folder screenshots."""
    return save_frame(frame, "output/screenshots", "screenshot")


# ─── Frame / Image Conversion ─────────────────────────────────────────────────

def frame_to_photoimage(frame: np.ndarray,
                         max_w: int = 800,
                         max_h: int = 600) -> ImageTk.PhotoImage:
    """Konversi OpenCV frame (BGR) ke Tkinter PhotoImage."""
    frame = resize_keep_aspect(frame, max_w, max_h)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)


def resize_keep_aspect(frame: np.ndarray,
                        max_w: int,
                        max_h: int) -> np.ndarray:
    """Resize frame sambil menjaga aspek rasio."""
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


# ─── Drawing Helpers ──────────────────────────────────────────────────────────

def draw_bounding_box(frame: np.ndarray, x1: int, y1: int,
                       x2: int, y2: int, label: str,
                       confidence: float, color=COLOR_GREEN):
    """Gambar bounding box dengan label dan confidence."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    font_scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, thickness)
    # Background label
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BLACK, thickness)


def draw_info_overlay(frame: np.ndarray, lines: list,
                       pos: tuple = (10, 30), color=COLOR_GREEN):
    """Tampilkan beberapa baris teks di atas frame."""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    line_h = 28
    for i, line in enumerate(lines):
        cy = y + i * line_h
        cv2.putText(frame, line, (x, cy), font, scale, COLOR_BLACK, thick + 2)
        cv2.putText(frame, line, (x, cy), font, scale, color, thick)


def draw_status_badge(frame: np.ndarray, status: str,
                       is_alert: bool = False):
    """Tampilkan status badge di pojok kanan atas."""
    h, w = frame.shape[:2]
    color = COLOR_RED if is_alert else COLOR_GREEN
    bg_color = (20, 20, 20)
    text = f" {status} "
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x1 = w - tw - 20
    y1 = 10
    x2 = w - 10
    y2 = y1 + th + 14
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1 + 4, y2 - 8),
                font, scale, color, thick)


def draw_timestamp(frame: np.ndarray):
    """Tampilkan timestamp di pojok kiri bawah."""
    h, _ = frame.shape[:2]
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, ts, (8, h - 10),
                font, 0.5, COLOR_BLACK, 3)
    cv2.putText(frame, ts, (8, h - 10),
                font, 0.5, COLOR_WHITE, 1)


# ─── YOLO Color ───────────────────────────────────────────────────────────────

def class_color(class_id: int) -> tuple:
    return YOLO_COLORS[class_id % len(YOLO_COLORS)]


# ─── Webcam Helper ────────────────────────────────────────────────────────────

def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Buka webcam dengan index tertentu."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka kamera index {index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def open_video(path: str) -> cv2.VideoCapture:
    """Buka file video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {path}")
    return cap


def read_image(path: str) -> np.ndarray:
    """Baca file gambar."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    return img


# ─── Validasi ────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


# ─── Tkinter Helpers ─────────────────────────────────────────────────────────

def center_window(window: tk.Toplevel | tk.Tk,
                  width: int, height: int):
    """Tampilkan window di tengah layar."""
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")


def make_separator(parent, color: str = "#2a2a3e", pady: int = 4):
    """Buat garis pemisah tipis."""
    sep = tk.Frame(parent, height=1, bg=color)
    sep.pack(fill="x", pady=pady)
    return sep


# ─── Init ─────────────────────────────────────────────────────────────────────

ensure_dirs()
