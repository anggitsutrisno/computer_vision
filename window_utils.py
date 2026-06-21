"""
window_utils.py - Smart Vision Analysis System
Utilitas ukuran window adaptif berdasarkan resolusi layar.
"""

import tkinter as tk


def get_screen_size(root=None) -> tuple[int, int]:
    """Kembalikan (screen_width, screen_height) dalam pixel."""
    if root is None:
        tmp = tk.Tk()
        tmp.withdraw()
        w = tmp.winfo_screenwidth()
        h = tmp.winfo_screenheight()
        tmp.destroy()
    else:
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
    return w, h


def adaptive_window_size(
    target_w: int,
    target_h: int,
    max_ratio: float = 0.90,
    root=None,
) -> tuple[int, int]:
    """
    Hitung ukuran window yang aman berdasarkan resolusi layar.

    target_w/h : ukuran ideal
    max_ratio  : maksimum % dari layar yang boleh dipakai (default 90%)
    """
    sw, sh = get_screen_size(root)
    usable_w = int(sw * max_ratio)
    usable_h = int(sh * max_ratio)
    return min(target_w, usable_w), min(target_h, usable_h)


def adaptive_display_size(
    win_w: int,
    win_h: int,
    sidebar_w: int = 260,
    margin: int = 40,
) -> tuple[int, int]:
    """
    Hitung ukuran area display video berdasarkan ukuran window aktual.
    """
    disp_w = win_w - sidebar_w - margin
    disp_h = win_h - 120          # header + status bar
    return max(disp_w, 320), max(disp_h, 240)


def center_and_apply(win, width: int, height: int):
    """Terapkan ukuran dan pusatkan window."""
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = (sw - width) // 2
    y = (sh - height) // 2
    win.geometry(f"{width}x{height}+{x}+{y}")


# ── Preset ukuran berdasarkan resolusi layar ──────────────────────────────────

def get_module_sizes(root=None) -> dict:
    """
    Kembalikan dict berisi ukuran window dan display area
    yang sudah disesuaikan dengan resolusi layar saat ini.

    Keys:
        win_w, win_h   — ukuran total window modul deteksi
        disp_w, disp_h — ukuran area video
        manip_w, manip_h  — ukuran window image manipulation
        data_w, data_h    — ukuran window data management
        dash_w, dash_h    — ukuran dashboard utama
    """
    sw, sh = get_screen_size(root)

    # Tentukan preset berdasarkan resolusi
    if sw >= 1920 and sh >= 1080:          # Full HD / lebih besar
        preset = "fhd"
    elif sw >= 1366 and sh >= 768:          # HD / laptop standar
        preset = "hd"
    elif sw >= 1280 and sh >= 720:          # HD min
        preset = "hd_min"
    else:                                   # Kecil (< 1280)
        preset = "small"

    presets = {
        "fhd": {
            "win_w": 1100, "win_h": 740,
            "disp_w": 760, "disp_h": 520,
            "manip_w": 1100, "manip_h": 760,
            "data_w": 1100, "data_h": 720,
            "dash_w": 820,  "dash_h": 660,
        },
        "hd": {
            "win_w": 960, "win_h": 660,
            "disp_w": 640, "disp_h": 450,
            "manip_w": 960, "manip_h": 660,
            "data_w": 980, "data_h": 640,
            "dash_w": 760, "dash_h": 600,
        },
        "hd_min": {
            "win_w": 900, "win_h": 620,
            "disp_w": 600, "disp_h": 420,
            "manip_w": 900, "manip_h": 620,
            "data_w": 920, "data_h": 600,
            "dash_w": 720, "dash_h": 560,
        },
        "small": {
            "win_w": 800, "win_h": 560,
            "disp_w": 520, "disp_h": 380,
            "manip_w": 800, "manip_h": 560,
            "data_w": 820, "data_h": 540,
            "dash_w": 680, "dash_h": 520,
        },
    }

    sizes = presets[preset]
    # Safety clamp — tidak melebihi 92% layar
    sizes["win_w"]   = min(sizes["win_w"],   int(sw * 0.92))
    sizes["win_h"]   = min(sizes["win_h"],   int(sh * 0.92))
    sizes["manip_w"] = min(sizes["manip_w"], int(sw * 0.92))
    sizes["manip_h"] = min(sizes["manip_h"], int(sh * 0.92))
    sizes["data_w"]  = min(sizes["data_w"],  int(sw * 0.92))
    sizes["data_h"]  = min(sizes["data_h"],  int(sh * 0.92))
    sizes["dash_w"]  = min(sizes["dash_w"],  int(sw * 0.88))
    sizes["dash_h"]  = min(sizes["dash_h"],  int(sh * 0.88))

    # Recalculate display area
    sidebar_w = 270
    sizes["disp_w"] = max(sizes["win_w"] - sidebar_w - 30, 320)
    sizes["disp_h"] = max(sizes["win_h"] - 110, 240)

    return sizes
