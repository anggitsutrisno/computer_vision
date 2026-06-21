"""
image_manipulation.py - Smart Vision Analysis System
Modul 4: Image Manipulation — Versi Lengkap
Fitur: Undo/Redo, Real-time Preview, 35+ Filter, Crop Visual, Zoom, Batch, Watermark
"""

import cv2
import numpy as np
import os
import queue
import threading
import copy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFont

import utils
import database as db
import config as cfg
from window_utils import get_module_sizes

# ── Tema ──────────────────────────────────────────────────────────────────────
DARK    = "#0d0d1a"
PANEL   = "#13132b"
HEADER  = "#1a1a2e"
ACCENT  = "#cc44ff"
GREEN   = "#00aa44"
RED     = "#aa2222"
BLUE    = "#1a5276"
ORANGE  = "#7d4800"
DIM     = "#6666aa"
BR      = "#eeeeff"
MODE_CAM  = "#2a0a4e"
MODE_FILE = "#0a2a4e"
MODE_MAN  = "#2a1a0a"


# ══════════════════════════════════════════════════════════════════════════════
# FILTER FUNCTIONS (35+ filter)
# ══════════════════════════════════════════════════════════════════════════════

def safe(fn):
    """Decorator — handle error & pastikan output uint8."""
    def wrapper(img, **kw):
        try:
            r = fn(img, **kw)
            if r is None: return img
            return np.clip(r, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[Filter Error] {fn.__name__}: {e}")
            return img
    wrapper.__name__ = fn.__name__
    return wrapper

# ── COLOR ────────────────────────────────────────────────────────────────────
@safe
def f_grayscale(img, **_):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

@safe
def f_invert(img, **_):
    return cv2.bitwise_not(img)

@safe
def f_sepia(img, **_):
    k = np.array([[0.272,0.534,0.131],
                  [0.349,0.686,0.168],
                  [0.393,0.769,0.189]])
    return cv2.transform(img.astype(np.float32), k)

@safe
def f_warm(img, **_):
    """Tone hangat — boost merah & kurangi biru."""
    lut_r = np.clip(np.arange(256) * 1.2, 0, 255).astype(np.uint8)
    lut_b = np.clip(np.arange(256) * 0.8, 0, 255).astype(np.uint8)
    out = img.copy()
    out[:,:,2] = cv2.LUT(img[:,:,2], lut_r)
    out[:,:,0] = cv2.LUT(img[:,:,0], lut_b)
    return out

@safe
def f_cool(img, **_):
    """Tone dingin — boost biru & kurangi merah."""
    lut_r = np.clip(np.arange(256) * 0.8, 0, 255).astype(np.uint8)
    lut_b = np.clip(np.arange(256) * 1.2, 0, 255).astype(np.uint8)
    out = img.copy()
    out[:,:,2] = cv2.LUT(img[:,:,2], lut_r)
    out[:,:,0] = cv2.LUT(img[:,:,0], lut_b)
    return out

@safe
def f_histeq(img, **_):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

@safe
def f_clahe(img, clip=2.0, tile=8, **_):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile),int(tile)))
    ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

@safe
def f_brightness_contrast(img, alpha=1.0, beta=0, **_):
    return cv2.convertScaleAbs(img, alpha=float(alpha), beta=int(beta))

@safe
def f_colormap(img, cmap="JET", **_):
    maps = {"JET":cv2.COLORMAP_JET,"HOT":cv2.COLORMAP_HOT,"COOL":cv2.COLORMAP_COOL,
            "BONE":cv2.COLORMAP_BONE,"PLASMA":cv2.COLORMAP_PLASMA,
            "VIRIDIS":cv2.COLORMAP_VIRIDIS,"RAINBOW":cv2.COLORMAP_RAINBOW,
            "PINK":cv2.COLORMAP_PINK,"MAGMA":cv2.COLORMAP_MAGMA,
            "INFERNO":cv2.COLORMAP_INFERNO}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, maps.get(cmap, cv2.COLORMAP_JET))

@safe
def f_hue_shift(img, hue_shift=30, **_):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:,:,0] = (hsv[:,:,0] + int(hue_shift)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

@safe
def f_saturation(img, sat=1.5, **_):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * float(sat), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

@safe
def f_negative(img, **_):
    return 255 - img

# ── BLUR ─────────────────────────────────────────────────────────────────────
@safe
def f_blur_gaussian(img, ksize=15, **_):
    k = max(1, int(ksize)) | 1
    return cv2.GaussianBlur(img, (k,k), 0)

@safe
def f_blur_median(img, ksize=15, **_):
    k = max(1, int(ksize)) | 1
    return cv2.medianBlur(img, k)

@safe
def f_blur_bilateral(img, d=9, sigma=75, **_):
    return cv2.bilateralFilter(img, int(d), float(sigma), float(sigma))

@safe
def f_blur_motion(img, ksize=20, angle=0, **_):
    k = max(3, int(ksize))
    M = np.zeros((k, k))
    cx = k // 2
    M[cx, :] = 1.0
    M = M / k
    rad = np.radians(float(angle))
    rot = cv2.getRotationMatrix2D((cx, cx), float(angle), 1.0)
    kernel_rot = cv2.warpAffine(M, rot, (k, k))
    kernel_rot = kernel_rot / (kernel_rot.sum() + 1e-8)
    return cv2.filter2D(img, -1, kernel_rot)

@safe
def f_denoise(img, h=10, **_):
    return cv2.fastNlMeansDenoisingColored(img, None, int(h), int(h), 7, 21)

# ── EDGE & SHARPEN ────────────────────────────────────────────────────────────
@safe
def f_edge_canny(img, low=50, high=150, **_):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.Canny(g, int(low), int(high)), cv2.COLOR_GRAY2BGR)

@safe
def f_edge_sobel(img, **_):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.clip(np.sqrt(sx**2 + sy**2), 0, 255)
    return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)

@safe
def f_edge_laplacian(img, **_):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_64F))
    return cv2.cvtColor(np.clip(lap,0,255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

@safe
def f_sharpen(img, strength=1.5, **_):
    s = float(strength)
    k = np.array([[-1,-1,-1],[-1, 8+1/s,-1],[-1,-1,-1]]) * s
    k[1,1] += 1
    return cv2.filter2D(img, -1, k.astype(np.float32))

@safe
def f_unsharp_mask(img, strength=1.5, ksize=9, **_):
    blur = cv2.GaussianBlur(img, (int(ksize)|1, int(ksize)|1), 0)
    s = float(strength)
    return cv2.addWeighted(img, 1 + s, blur, -s, 0)

@safe
def f_emboss(img, **_):
    k = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32)
    return cv2.filter2D(img, -1, k) + 128

# ── THRESHOLD ─────────────────────────────────────────────────────────────────
@safe
def f_threshold(img, thresh=127, method="Binary", **_):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = {"Binary":cv2.THRESH_BINARY,"Binary Inv":cv2.THRESH_BINARY_INV,
             "Truncate":cv2.THRESH_TRUNC,"To Zero":cv2.THRESH_TOZERO,
             "To Zero Inv":cv2.THRESH_TOZERO_INV,
             "Otsu":cv2.THRESH_BINARY+cv2.THRESH_OTSU}
    flag = flags.get(method, cv2.THRESH_BINARY)
    tv = 0 if method=="Otsu" else int(thresh)
    _, r = cv2.threshold(g, tv, 255, flag)
    return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

@safe
def f_adaptive_threshold(img, block=11, C=2, **_):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, int(block)|1, C)
    return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

@safe
def f_morphology(img, op="Dilate", ksize=5, **_):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ksize)|1, int(ksize)|1))
    ops = {"Dilate":cv2.MORPH_DILATE,"Erode":cv2.MORPH_ERODE,
           "Open":cv2.MORPH_OPEN,"Close":cv2.MORPH_CLOSE,
           "Gradient":cv2.MORPH_GRADIENT,"TopHat":cv2.MORPH_TOPHAT,
           "BlackHat":cv2.MORPH_BLACKHAT}
    return cv2.morphologyEx(img, ops.get(op, cv2.MORPH_DILATE), k)

# ── TRANSFORM ─────────────────────────────────────────────────────────────────
@safe
def f_rotate(img, angle=90, **_):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), float(angle), 1.0)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REPLICATE)

@safe
def f_flip(img, mode="Horizontal", **_):
    return cv2.flip(img, {"Horizontal":1,"Vertical":0,"Both":-1}.get(mode,1))

@safe
def f_resize(img, width=640, height=480, **_):
    return cv2.resize(img, (max(1,int(width)), max(1,int(height))))

@safe
def f_crop(img, x1=0, y1=0, x2=640, y2=480, **_):
    H, W = img.shape[:2]
    x1,y1 = max(0,int(x1)), max(0,int(y1))
    x2,y2 = min(W,int(x2)), min(H,int(y2))
    if x2>x1 and y2>y1: return img[y1:y2, x1:x2]
    return img

@safe
def f_perspective(img, **_):
    """Efek perspektif — koreksi slight tilt."""
    h, w = img.shape[:2]
    margin = int(min(h,w)*0.05)
    src = np.float32([[margin,0],[w-margin,0],[0,h],[w,h]])
    dst = np.float32([[0,0],[w,0],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w,h))

# ── FX EFFECTS ────────────────────────────────────────────────────────────────
@safe
def f_cartoon(img, **_):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

@safe
def f_pencil_sketch(img, **_):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(255 - gray, (21,21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

@safe
def f_pencil_color(img, **_):
    """Pencil sketch dengan warna."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(255 - gray, (21,21), 0)
    sketch_gray = cv2.divide(gray, 255 - blur, scale=256)
    sketch3 = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.5, sketch3, 0.5, 0)

@safe
def f_hdr(img, **_):
    img_f = img.astype(np.float32) / 255.0
    tonemap = cv2.createTonemapReinhard(gamma=1.5, intensity=0,
                                         light_adapt=0.8, color_adapt=0)
    return np.clip(tonemap.process(img_f) * 255, 0, 255)

@safe
def f_vignette(img, strength=0.5, **_):
    h, w = img.shape[:2]
    s = float(strength)
    X = cv2.getGaussianKernel(w, int(w*(1-s)+1))
    Y = cv2.getGaussianKernel(h, int(h*(1-s)+1))
    mask = (Y * X.T); mask = mask / mask.max()
    result = img.astype(np.float32)
    for i in range(3): result[:,:,i] *= mask
    return result

@safe
def f_oil_painting(img, size=7, dyn=1, **_):
    """Oil painting effect menggunakan xphoto jika tersedia, fallback bilateral."""
    try:
        return cv2.xphoto.oilPainting(img, int(size)|1, int(dyn))
    except AttributeError:
        # Fallback: bilateral + saturation boost
        r = cv2.bilateralFilter(img, 9, 200, 200)
        hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1]*1.4, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

@safe
def f_watercolor(img, **_):
    """Watercolor-like effect."""
    # Multi-scale bilateral
    r = img.copy()
    for _ in range(3):
        r = cv2.bilateralFilter(r, 9, 75, 75)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    edges = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(r, 0.8, edges, 0.2, 0)

@safe
def f_pixelate(img, pixelate_block=12, **_):
    """Mosaic/pixelate effect."""
    h, w = img.shape[:2]
    b = max(2, int(pixelate_block))
    small = cv2.resize(img, (w//b, h//b), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

@safe
def f_glitch(img, intensity=10, **_):
    """RGB channel shift glitch effect."""
    h, w = img.shape[:2]
    s = int(intensity)
    result = img.copy()
    result[:,:,2] = np.roll(img[:,:,2], s, axis=1)    # R shift kanan
    result[:,:,0] = np.roll(img[:,:,0], -s, axis=1)   # B shift kiri
    # Random horizontal strip
    rng = np.random.RandomState(42)
    for _ in range(s // 2 + 1):
        y = rng.randint(0, h)
        dy = rng.randint(2, max(3, s))
        dx = rng.randint(-s*2, s*2)
        result[y:y+dy, :] = np.roll(result[y:y+dy, :], dx, axis=1)
    return result

@safe
def f_night_vision(img, **_):
    """Night vision green effect."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    out = np.zeros_like(img)
    out[:,:,1] = enhanced   # only green channel
    noise = np.random.normal(0, 8, enhanced.shape).astype(np.int16)
    out[:,:,1] = np.clip(out[:,:,1].astype(np.int16) + noise, 0, 255)
    return out.astype(np.uint8)

@safe
def f_add_noise(img, amount=25, **_):
    """Tambah noise gaussian."""
    noise = np.random.normal(0, float(amount), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255)


# ── FILTER REGISTRY ───────────────────────────────────────────────────────────
@safe
def f_normal(img, **_):
    return img.copy()
FILTERS = {
    # Color
     "normal":       (f_normal,         "Normal",              "color"),
    "grayscale":    (f_grayscale,        "Grayscale",          "Color"),
    "invert":       (f_invert,           "Invert",             "Color"),
    "sepia":        (f_sepia,            "Sepia",              "Color"),
    "warm":         (f_warm,             "Warm Tone",          "Color"),
    "cool":         (f_cool,             "Cool Tone",          "Color"),
    "histeq":       (f_histeq,           "Histogram EQ",       "Color"),
    "clahe":        (f_clahe,            "CLAHE",              "Color"),
    "brightness":   (f_brightness_contrast,"Brightness/Contrast","Color"),
    "saturation":   (f_saturation,       "Saturation",         "Color"),
    "hue_shift":    (f_hue_shift,        "Hue Shift",          "Color"),
    "colormap":     (f_colormap,         "Colormap",           "Color"),
    "negative":     (f_negative,         "Negative",           "Color"),
    # Blur
    "blur_gaussian":  (f_blur_gaussian,  "Gaussian Blur",      "Blur"),
    "blur_median":    (f_blur_median,    "Median Blur",        "Blur"),
    "blur_bilateral": (f_blur_bilateral, "Bilateral Blur",     "Blur"),
    "blur_motion":    (f_blur_motion,    "Motion Blur",        "Blur"),
    "denoise":        (f_denoise,        "Denoise (NLM)",      "Blur"),
    # Edge & Sharpen
    "edge_canny":     (f_edge_canny,     "Canny Edge",         "Edge"),
    "edge_sobel":     (f_edge_sobel,     "Sobel Edge",         "Edge"),
    "edge_laplacian": (f_edge_laplacian, "Laplacian",          "Edge"),
    "sharpen":        (f_sharpen,        "Sharpen",            "Edge"),
    "unsharp":        (f_unsharp_mask,   "Unsharp Mask",       "Edge"),
    "emboss":         (f_emboss,         "Emboss",             "Edge"),
    # Threshold
    "threshold":      (f_threshold,      "Threshold",          "Thresh"),
    "adaptive_thresh":(f_adaptive_threshold,"Adaptive Thresh", "Thresh"),
    "morph":          (f_morphology,     "Morphology",         "Thresh"),
    # Transform
    "rotate":         (f_rotate,         "Rotate",             "Transform"),
    "flip":           (f_flip,           "Flip",               "Transform"),
    "resize":         (f_resize,         "Resize",             "Transform"),
    "crop":           (f_crop,           "Crop",               "Transform"),
    "perspective":    (f_perspective,    "Perspective Fix",    "Transform"),
    # FX
    "cartoon":        (f_cartoon,        "Cartoon",            "FX"),
    "pencil":         (f_pencil_sketch,  "Pencil Sketch",      "FX"),
    "pencil_color":   (f_pencil_color,   "Pencil Color",       "FX"),
    "hdr":            (f_hdr,            "HDR Effect",         "FX"),
    "vignette":       (f_vignette,       "Vignette",           "FX"),
    "oil_painting":   (f_oil_painting,   "Oil Painting",       "FX"),
    "watercolor":     (f_watercolor,     "Watercolor",         "FX"),
    "pixelate":       (f_pixelate,       "Pixelate",           "FX"),
    "glitch":         (f_glitch,         "Glitch",             "FX"),
    "night_vision":   (f_night_vision,   "Night Vision",       "FX"),
    "add_noise":      (f_add_noise,      "Add Noise",          "FX"),
}
CATEGORIES = ["Color","Blur","Edge","Thresh","Transform","FX"]


# ══════════════════════════════════════════════════════════════════════════════
# ImageManipulationWindow — GUI Lengkap
# ══════════════════════════════════════════════════════════════════════════════

class ImageManipulationWindow:
    POLL_MS   = 35
    UNDO_LIMIT = 30   # maks langkah undo

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("🎨 Modul 4 – Image Manipulation")
        self.win.configure(bg=DARK)
        sz = get_module_sizes(parent)
        self.WIN_W  = sz["manip_w"]
        self.WIN_H  = sz["manip_h"]
        self.DISP_W = sz["disp_w"]
        self.DISP_H = sz["disp_h"]
        utils.center_window(self.win, self.WIN_W, self.WIN_H)
        self.win.resizable(True, True)
        self.win.minsize(820, 560)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Mode
        self.mode = tk.StringVar(value="file")

        # Camera
        self._q: queue.Queue = queue.Queue(maxsize=3)
        self.cap = None; self.running = False
        self._photo_live = None; self._cam_latest = None

        # Image state — undo/redo stack
        self._orig_img: np.ndarray | None = None
        self._undo_stack: list[tuple[np.ndarray, str]] = []  # (img, op_name)
        self._redo_stack: list[tuple[np.ndarray, str]] = []
        self._current_img: np.ndarray | None = None
        self._current_file: str = ""
        self._photo_orig = None; self._photo_curr = None

        # Real-time preview debounce
        self._preview_job = None

        # Crop visual state
        self._crop_rect = None  # (x1,y1,x2,y2) in image coords
        self._crop_drag_start = None
        self._crop_dragging = False

        # Zoom
        self.zoom_level = 1.0

        # Active filter
        self._active_filter = tk.StringVar(value="grayscale")
        self._live_filter   = tk.BooleanVar(value=True)

        # ── Parameter vars ────────────────────────────────────────────────────
        self.p_blur_k      = tk.IntVar(value=15)
        self.p_canny_lo    = tk.IntVar(value=50)
        self.p_canny_hi    = tk.IntVar(value=150)
        self.p_thresh      = tk.IntVar(value=127)
        self.p_thresh_m    = tk.StringVar(value="Binary")
        self.p_rotate      = tk.DoubleVar(value=90.0)
        self.p_resize_w    = tk.IntVar(value=640)
        self.p_resize_h    = tk.IntVar(value=480)
        self.p_keep_ratio  = tk.BooleanVar(value=True)
        self.p_crop_x1     = tk.IntVar(value=0)
        self.p_crop_y1     = tk.IntVar(value=0)
        self.p_crop_x2     = tk.IntVar(value=640)
        self.p_crop_y2     = tk.IntVar(value=480)
        self.p_alpha       = tk.DoubleVar(value=1.0)
        self.p_beta        = tk.IntVar(value=0)
        self.p_flip_m      = tk.StringVar(value="Horizontal")
        self.p_cmap        = tk.StringVar(value="JET")
        self.p_morph_op    = tk.StringVar(value="Dilate")
        self.p_morph_k     = tk.IntVar(value=5)
        self.p_sharpen     = tk.DoubleVar(value=1.5)
        self.p_clahe_cl    = tk.DoubleVar(value=2.0)
        self.p_denoise_h   = tk.IntVar(value=10)
        self.p_vignette    = tk.DoubleVar(value=0.5)
        self.p_bilateral   = tk.IntVar(value=75)
        self.p_hue_shift   = tk.IntVar(value=30)
        self.p_saturation  = tk.DoubleVar(value=1.5)
        self.p_pixelate    = tk.IntVar(value=12)
        self.p_glitch      = tk.IntVar(value=10)
        self.p_noise       = tk.IntVar(value=25)
        self.p_motion_k    = tk.IntVar(value=20)
        self.p_motion_angle= tk.IntVar(value=0)
        self.p_block_size  = tk.IntVar(value=11)

        # Watermark
        self.wm_text   = tk.StringVar(value="Smart Vision")
        self.wm_pos    = tk.StringVar(value="Bottom Right")
        self.wm_size   = tk.IntVar(value=24)
        self.wm_color  = "#ffffff"
        self.wm_alpha  = tk.DoubleVar(value=0.7)

        # Manual form
        self.mf_op   = tk.StringVar(value="grayscale")
        self.mf_note = tk.StringVar(value="")

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header
        hdr = tk.Frame(self.win, bg=HEADER, height=50)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🎨  IMAGE MANIPULATION  —  35+ Filter · Undo/Redo · Real-time Preview",
                 font=("Helvetica", 12, "bold"), bg=HEADER, fg=ACCENT).pack(side="left", padx=14, pady=13)
        # Image info
        self.info_lbl = tk.Label(hdr, text="—", font=("Courier", 8), bg=HEADER, fg=DIM)
        self.info_lbl.pack(side="right", padx=14)

        # ── Mode Toggle
        tbar = tk.Frame(self.win, bg="#111133", height=44)
        tbar.pack(fill="x"); tbar.pack_propagate(False)
        tk.Label(tbar, text="MODE:", bg="#111133", fg=DIM,
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=12, pady=11)

        self.btn_cam  = self._mkbtn(tbar, "📷  Kamera Live",  lambda: self._set_mode("camera"))
        self.btn_file = self._mkbtn(tbar, "🖼️  File Gambar",  lambda: self._set_mode("file"))
        self.btn_man  = self._mkbtn(tbar, "✏️  Input Manual", lambda: self._set_mode("manual"))

        self.mode_lbl = tk.Label(tbar, text="", bg="#111133", fg=ACCENT,
                                  font=("Helvetica", 8, "italic"))
        self.mode_lbl.pack(side="left", padx=10)

        # ── Body
        body = tk.Frame(self.win, bg=DARK)
        body.pack(fill="both", expand=True, padx=6, pady=5)

        # Left: content area
        self.left = tk.Frame(body, bg=DARK)
        self.left.pack(side="left", fill="both", expand=True)
        self.left.pack_propagate(False)

        self._build_cam_panel()
        self._build_file_panel()
        self._build_manual_panel()

        # Right: filter + param panel (scrollable)
        right_frame = tk.Frame(body, bg=PANEL, width=500)
        right_frame.pack(side="right", fill="y", padx=(6,2), pady=2)
        right_frame.pack_propagate(False)
        self._build_right_panel(right_frame)

        self._set_mode("file")

    def _mkbtn(self, parent, text, cmd):
        b = tk.Button(parent, text=text, font=("Helvetica", 9, "bold"),
                       relief="flat", cursor="hand2", padx=14, pady=5, command=cmd)
        b.pack(side="left", padx=3, pady=6)
        return b

    # ── Camera Panel ──────────────────────────────────────────────────────────

    def _build_cam_panel(self):
        self.cam_panel = tk.Frame(self.left, bg=DARK)
        self.cam_canvas = tk.Label(self.cam_panel, bg="#111122",
                                    text="[ Klik START untuk preview kamera ]",
                                    font=("Courier", 11), fg="#444466")
        self.cam_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        bar = tk.Frame(self.cam_panel, bg=HEADER)
        bar.pack(fill="x", padx=4, pady=(0,2))
        self.cam_stat = tk.Label(bar, text="Idle", bg=HEADER, fg=DIM,
                                  font=("Helvetica", 8), anchor="w")
        self.cam_stat.pack(side="left", padx=4)
        for text, cmd, color in [
            ("▶ START",   self._cam_start,    GREEN),
            ("⏹ STOP",    self._cam_stop,     RED),
            ("📷 CAPTURE", self._capture_cam, BLUE),
        ]:
            tk.Button(bar, text=text, bg=color, fg="white",
                       font=("Helvetica", 8, "bold"), relief="flat",
                       cursor="hand2", padx=8, pady=2,
                       command=cmd).pack(side="right", padx=3)
        tk.Checkbutton(bar, text="Live Filter", variable=self._live_filter,
                       bg=HEADER, fg=ACCENT, selectcolor=PANEL,
                       activebackground=HEADER,
                       font=("Helvetica", 8)).pack(side="right", padx=6)

    # ── File Panel ────────────────────────────────────────────────────────────

    def _build_file_panel(self):
        self.file_panel = tk.Frame(self.left, bg=DARK)

        # Label bar
        lbar = tk.Frame(self.file_panel, bg=DARK)
        lbar.pack(fill="x")
        for text, anchor, fg in [("  ORIGINAL", "w", "#888899"), ("HASIL  ", "e", ACCENT)]:
            tk.Label(lbar, text=text, bg=HEADER, fg=fg,
                     font=("Helvetica", 9, "bold"),
                     anchor=anchor).pack(side="left" if anchor=="w" else "right",
                                          fill="x", expand=True)

        # Preview area (dua panel side-by-side)
        prev = tk.Frame(self.file_panel, bg=DARK)
        prev.pack(fill="both", expand=True)

        hw = max(450, self.DISP_W // 2)
        dh = max(550, self.DISP_H)
        # Original
        orig_frame = tk.Frame(prev, bg="#111122", width=hw, height=dh)
        orig_frame.pack(side="left", padx=2, pady=4, fill="both", expand=True)
        orig_frame.pack_propagate(False)
        self.cv_orig = tk.Label(orig_frame, bg="#111122",
                                 text="Original\n(Buka file dulu)",
                                 font=("Courier", 10), fg="#444466")
        self.cv_orig.pack(fill="both", expand=True)
        # Crop drag events on original
        self.cv_orig.bind("<ButtonPress-1>",   self._crop_drag_start)
        self.cv_orig.bind("<B1-Motion>",        self._crop_drag_move)
        self.cv_orig.bind("<ButtonRelease-1>",  self._crop_drag_end)

        # Result
        res_frame = tk.Frame(prev, bg="#111122", width=hw, height=dh)
        res_frame.pack(side="right", padx=2, pady=4, fill="both", expand=True)
        res_frame.pack_propagate(False)
        self.cv_curr = tk.Label(res_frame, bg="#111122",
                                 text="Hasil Filter\n(Pilih filter)",
                                 font=("Courier", 10), fg="#444466")
        self.cv_curr.pack(fill="both", expand=True)

        # Status bar
        sbar = tk.Frame(self.file_panel, bg=HEADER)
        sbar.pack(fill="x", padx=4, pady=(0,2))
        self.file_status = tk.Label(sbar, text="Belum ada file",
                                     bg=HEADER, fg=DIM,
                                     font=("Helvetica", 8), anchor="w")
        self.file_status.pack(side="left", padx=4)
        self.crop_info = tk.Label(sbar, text="", bg=HEADER, fg="#ffcc00",
                                   font=("Courier", 8))
        self.crop_info.pack(side="right", padx=4)

    # ── Manual Panel ──────────────────────────────────────────────────────────

    def _build_manual_panel(self):
        self.man_panel = tk.Frame(self.left, bg=DARK)
        hdr2 = tk.Frame(self.man_panel, bg="#1a1a3e", height=42)
        hdr2.pack(fill="x", pady=(4,0))
        tk.Label(hdr2, text="✏️  INPUT DATA IMAGE MANIPULATION SECARA MANUAL",
                 font=("Helvetica", 10, "bold"), bg="#1a1a3e", fg=ACCENT).pack(
            side="left", padx=14, pady=10)

        form = tk.Frame(self.man_panel, bg=DARK, padx=28, pady=18)
        form.pack(fill="both", expand=True)

        now = datetime.now()
        self.mf_date = tk.StringVar(value=now.strftime("%Y-%m-%d"))
        self.mf_time = tk.StringVar(value=now.strftime("%H:%M:%S"))

        info = tk.Frame(form, bg="#0d1a0d", padx=10, pady=8)
        info.pack(fill="x", pady=(0,12))
        tk.Label(info,
                 text="💡  Gunakan mode ini untuk mencatat secara manual proses\n"
                      "    pengolahan gambar yang sudah dilakukan di luar aplikasi.",
                 bg="#0d1a0d", fg="#88aacc",
                 font=("Helvetica", 9), justify="left").pack(anchor="w")

        def fld(label, var, hint=""):
            row = tk.Frame(form, bg=DARK); row.pack(fill="x", pady=6)
            tk.Label(row, text=label, bg=DARK, fg=DIM, font=("Helvetica", 10),
                     width=22, anchor="w").pack(side="left")
            e = tk.Entry(row, textvariable=var, bg=PANEL, fg=BR,
                         insertbackground="white", font=("Courier", 10),
                         relief="flat", highlightthickness=1,
                         highlightcolor=ACCENT, highlightbackground="#2a2a4e")
            e.pack(side="left", fill="x", expand=True, ipady=4, padx=(8,0))
            if hint:
                tk.Label(row, text=hint, bg=DARK, fg="#444466",
                         font=("Helvetica", 8)).pack(side="left", padx=5)

        fld("📅  Tanggal:", self.mf_date, "YYYY-MM-DD")
        fld("⏰  Waktu:",   self.mf_time, "HH:MM:SS")

        fr = tk.Frame(form, bg=DARK); fr.pack(fill="x", pady=6)
        tk.Label(fr, text="🎨  Filter/Operasi:", bg=DARK, fg=DIM,
                 font=("Helvetica", 10), width=22, anchor="w").pack(side="left")
        filter_names = [v[1] for v in FILTERS.values()]
        self.mf_op_cb = ttk.Combobox(fr, textvariable=self.mf_op,
                                      values=filter_names,
                                      state="readonly", width=28)
        self.mf_op_cb.pack(side="left", padx=(8,0), ipady=3)

        fld("📝  Catatan:", self.mf_note, "opsional")

        tk.Frame(form, bg="#2a2a4e", height=1).pack(fill="x", pady=12)

        br = tk.Frame(form, bg=DARK); br.pack(fill="x")
        bc = {"font":("Helvetica",10,"bold"),"relief":"flat","cursor":"hand2","pady":9}
        tk.Button(br, text="💾  SIMPAN KE DATABASE", bg=GREEN, fg="white",
                  command=self._save_manual, **bc).pack(side="left", fill="x", expand=True)
        tk.Button(br, text="🔄  RESET", bg="#555555", fg="white",
                  command=self._reset_manual_form, **bc).pack(side="left", padx=(8,0))

        self.man_status = tk.Label(form, text="", bg=DARK, fg=DIM,
                                    font=("Helvetica", 9), wraplength=480)
        self.man_status.pack(pady=(10,0))

    # ── Right Panel (Filter + Params) ─────────────────────────────────────────

    def _build_right_panel(self, parent):
        sb = tk.Scrollbar(parent)
        sb.pack(side="right", fill="y")
        cv = tk.Canvas(parent, bg=PANEL, yscrollcommand=sb.set, highlightthickness=0)
        cv.pack(fill="both", expand=True)
        sb.config(command=cv.yview)
        # Enable mousewheel scroll
        cv.bind("<Enter>", lambda e: cv.bind_all("<MouseWheel>",
                lambda ev: cv.yview_scroll(int(-1*(ev.delta/120)),"units")))
        cv.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))

        p = tk.Frame(cv, bg=PANEL)
        cv.create_window((0,0), window=p, anchor="nw")
        p.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))

        pad = {"padx":10,"pady":3}
        bc  = {"font":("Helvetica",9,"bold"),"relief":"flat","cursor":"hand2","pady":6}

        # ── File Controls ──────────────────────────────────────────────────────
        sec = self._section(p, "📂  FILE & AKSI")
        btns = [
            ("📂  Buka Gambar",     "#1a5276", self._open_file),
            ("💾  Simpan Hasil",    GREEN,     self._save_result),
            ("💾  Simpan Sebagai…", "#1a3a26", self._save_as),
            ("🔄  Reset Original",  "#555555", self._reset),
            ("📊  Histogram",       "#6c3483", self._show_histogram),
            ("🏷️  Tambah Watermark","#7d4800", self._add_watermark_dialog),
        ]
        for text, color, cmd in btns:
            tk.Button(sec, text=text, bg=color, fg="white",
                       command=cmd, **bc).pack(fill="x", padx=8, pady=2)

        # Undo / Redo
        und_row = tk.Frame(sec, bg=PANEL); und_row.pack(fill="x", padx=8, pady=2)
        self.btn_undo = tk.Button(und_row, text="↩  Undo", bg="#333355", fg="white",
                                   command=self._undo, **{k:v for k,v in bc.items()})
        self.btn_undo.pack(side="left", fill="x", expand=True)
        self.btn_redo = tk.Button(und_row, text="↪  Redo", bg="#333355", fg="white",
                                   command=self._redo, **{k:v for k,v in bc.items()})
        self.btn_redo.pack(side="right", fill="x", expand=True, padx=(5,0))

        utils.make_separator(sec, "#2a2a4e", pady=4)

        # ── Filter Picker ──────────────────────────────────────────────────────
        cat_colors = {
            "Color":     "#1a2840",
            "Blur":      "#1a3020",
            "Edge":      "#2a2010",
            "Thresh":    "#1a1030",
            "Transform": "#102030",
            "FX":        "#2a1010",
        }

        for cat in CATEGORIES:
            cf = tk.LabelFrame(p, text=f" {cat} ",
                                bg=PANEL, fg=ACCENT,
                                font=("Helvetica", 8, "bold"),
                                padx=5, pady=4)
            cf.pack(fill="x", padx=8, pady=3)
            cbg = cat_colors.get(cat, PANEL)
            items = [(k,v) for k,v in FILTERS.items() if v[2]==cat]
            for i, (key, (fn, name, _)) in enumerate(items):
                b = tk.Button(cf, text=name, bg=cbg, fg=BR,
                               font=("Helvetica", 8), relief="flat",
                               cursor="hand2", padx=5, pady=4,
                               command=lambda k=key: self._select_filter(k))
                row, col = divmod(i, 2)
                b.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            cf.columnconfigure(0, weight=1); cf.columnconfigure(1, weight=1)

        # ── Parameters ────────────────────────────────────────────────────────
        self._param_sec = self._section(p, "⚙️  PARAMETER AKTIF")
        self._build_param_panel(self._param_sec)

        # ── History Log ───────────────────────────────────────────────────────
        hist_sec = self._section(p, "📋  RIWAYAT FILTER")
        self.hist_box = tk.Text(hist_sec, height=6, bg="#080818", fg="#cc99ff",
                                 font=("Courier", 8), state="disabled",
                                 relief="flat", padx=4)
        self.hist_box.pack(fill="x", padx=8, pady=2)

        clr_row = tk.Frame(hist_sec, bg=PANEL); clr_row.pack(fill="x", padx=8, pady=2)
        tk.Button(clr_row, text="🗑️ Hapus Log", bg="#333355", fg="white",
                   command=self._clear_hist, **{k:v for k,v in bc.items() if k!="pady"},
                   pady=4).pack(fill="x")

    def _section(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=PANEL)
        outer.pack(fill="x", padx=0, pady=0)
        tk.Label(outer, text=title, bg="#1a1a2e", fg=ACCENT,
                 font=("Helvetica", 9, "bold"),
                 anchor="w", padx=10, pady=6).pack(fill="x")
        inner = tk.Frame(outer, bg=PANEL)
        inner.pack(fill="x", pady=(0,4))
        utils.make_separator(parent, "#2a2a4e", pady=0)
        return inner

    def _build_param_panel(self, parent):
        """Parameter slider panel — auto-preview on change."""

        def slider(label, var, lo, hi, res=1, dec=0):
            f = tk.Frame(parent, bg=PANEL); f.pack(fill="x", padx=8, pady=2)
            tk.Label(f, text=label+":", bg=PANEL, fg=DIM,
                     font=("Helvetica", 8), width=16, anchor="w").pack(side="left")
            vl = tk.Label(f, text=str(var.get()), bg=PANEL, fg="#ffcc00",
                          font=("Courier", 8, "bold"), width=6)
            vl.pack(side="right")
            def update(v):
                vl.config(text=f"{float(v):.{dec}f}")
                self._schedule_preview()
            tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal",
                     variable=var, bg=PANEL, fg="white", troughcolor="#2a2a4e",
                     highlightbackground=PANEL, showvalue=False,
                     command=update).pack(side="left", fill="x", expand=True)

        def combo(label, var, opts):
            f = tk.Frame(parent, bg=PANEL); f.pack(fill="x", padx=8, pady=2)
            tk.Label(f, text=label+":", bg=PANEL, fg=DIM,
                     font=("Helvetica", 8), width=16, anchor="w").pack(side="left")
            cb = ttk.Combobox(f, textvariable=var, values=opts,
                              state="readonly", width=14)
            cb.pack(side="right"); cb.bind("<<ComboboxSelected>>",
                                            lambda e: self._schedule_preview())

        slider("Blur Kernel",    self.p_blur_k,    1,  51, 2)
        slider("Canny Low",      self.p_canny_lo,  0, 255)
        slider("Canny High",     self.p_canny_hi,  0, 255)
        slider("Threshold",      self.p_thresh,    0, 255)
        slider("Block Size",     self.p_block_size,3,  51, 2)
        slider("Rotate °",       self.p_rotate,  -360,360, 1, 0)
        slider("Alpha (C.)",     self.p_alpha,   0.1,4.0, 0.1, 1)
        slider("Beta (B.)",      self.p_beta,   -127, 127)
        slider("Sharpen",        self.p_sharpen, 0.1, 5.0, 0.1, 1)
        slider("CLAHE Clip",     self.p_clahe_cl,1.0,10.0, 0.5, 1)
        slider("Denoise h",      self.p_denoise_h, 1, 30)
        slider("Morph Size",     self.p_morph_k,   1, 25, 2)
        slider("Vignette",       self.p_vignette,0.1, 0.9, 0.1, 1)
        slider("Bilateral σ",    self.p_bilateral,  1, 200)
        slider("Hue Shift",      self.p_hue_shift, -90, 90)
        slider("Saturation",     self.p_saturation,0.0, 3.0, 0.1, 1)
        slider("Pixelate Blk",   self.p_pixelate,   2,  50)
        slider("Glitch",         self.p_glitch,     1,  30)
        slider("Noise Amount",   self.p_noise,      1,  60)
        slider("Motion Blur K",  self.p_motion_k,   3,  50)
        slider("Motion Angle°",  self.p_motion_angle,0,180)

        combo("Thresh Method",   self.p_thresh_m,
              ["Binary","Binary Inv","Truncate","To Zero","To Zero Inv","Otsu"])
        combo("Flip Mode",       self.p_flip_m,
              ["Horizontal","Vertical","Both"])
        combo("Colormap",        self.p_cmap,
              ["JET","HOT","COOL","BONE","PLASMA","VIRIDIS","RAINBOW","PINK","MAGMA","INFERNO"])
        combo("Morph Op",        self.p_morph_op,
              ["Dilate","Erode","Open","Close","Gradient","TopHat","BlackHat"])

        # Resize
        tk.Label(parent, text="Resize (W × H):", bg=PANEL, fg=DIM,
                 font=("Helvetica", 8)).pack(anchor="w", padx=8)
        rz = tk.Frame(parent, bg=PANEL); rz.pack(fill="x", padx=8, pady=2)
        for var, label in [(self.p_resize_w,"W"),(self.p_resize_h,"H")]:
            tk.Label(rz, text=label+":", bg=PANEL, fg=DIM, font=("Helvetica",8)).pack(side="left")
            tk.Entry(rz, textvariable=var, width=6, bg=DARK, fg=BR,
                     insertbackground="white", font=("Courier",9),
                     relief="flat").pack(side="left", padx=(2,6))
        tk.Checkbutton(rz, text="Keep ratio", variable=self.p_keep_ratio,
                       bg=PANEL, fg=DIM, selectcolor=DARK,
                       activebackground=PANEL, font=("Helvetica",8)).pack(side="left")

        # Crop
        tk.Label(parent, text="Crop Region (klik & drag di gambar):", bg=PANEL, fg=DIM,
                 font=("Helvetica", 8)).pack(anchor="w", padx=8, pady=(6,0))
        cr = tk.Frame(parent, bg=PANEL); cr.pack(fill="x", padx=8, pady=2)
        for var, lbl in [(self.p_crop_x1,"x1"),(self.p_crop_y1,"y1"),
                          (self.p_crop_x2,"x2"),(self.p_crop_y2,"y2")]:
            tk.Label(cr, text=lbl+":", bg=PANEL, fg=DIM, font=("Helvetica",8)).pack(side="left")
            tk.Entry(cr, textvariable=var, width=5, bg=DARK, fg=BR,
                     insertbackground="white", font=("Courier",9),
                     relief="flat").pack(side="left", padx=(2,4))

        # Apply button (manual trigger)
        tk.Button(parent, text="▶  TERAPKAN FILTER",
                  bg="#6c3483", fg="white",
                  font=("Helvetica", 10, "bold"), relief="flat",
                  cursor="hand2", pady=8,
                  command=lambda: self._apply_filter(self._active_filter.get())).pack(
            fill="x", padx=8, pady=8)

    # ─────────────────────────────────────────────────────────────────────────
    # Mode toggle
    # ─────────────────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self.mode.set(mode)
        for panel in [self.cam_panel, self.file_panel, self.man_panel]:
            panel.pack_forget()
        for btn in [self.btn_cam, self.btn_file, self.btn_man]:
            btn.config(bg=PANEL, fg=DIM, relief="flat")

        if mode == "camera":
            self.cam_panel.pack(fill="both", expand=True)
            self.btn_cam.config(bg=MODE_CAM, fg="white", relief="groove")
            self.mode_lbl.config(text="Mode: 📷 Kamera — filter diterapkan real-time")
        elif mode == "file":
            self.file_panel.pack(fill="both", expand=True)
            self.btn_file.config(bg=MODE_FILE, fg="white", relief="groove")
            self.mode_lbl.config(text="Mode: 🖼️ File — edit & simpan | drag di gambar untuk crop")
        else:
            self.man_panel.pack(fill="both", expand=True)
            self.btn_man.config(bg=MODE_MAN, fg="white", relief="groove")
            self.mode_lbl.config(text="Mode: ✏️ Manual — catat proses pengolahan ke database")
            if self.running: self._cam_stop()

    # ─────────────────────────────────────────────────────────────────────────
    # Camera
    # ─────────────────────────────────────────────────────────────────────────

    def _cam_start(self):
        if self.running: return
        try:
            self.cap = utils.open_camera(cfg.get("detection.camera_index", 0))
        except RuntimeError as e:
            messagebox.showerror("Error", str(e), parent=self.win); return
        self.running = True; self._flush_q()
        self.cam_stat.config(text="Running — Webcam")
        threading.Thread(target=self._cam_loop, daemon=True).start()
        self._cam_poll()

    def _cam_stop(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap = None
        self.cam_stat.config(text="Stopped")

    def _cam_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: self.running = False; break
            if self._live_filter.get():
                try:
                    frame = self._apply_params(frame, self._active_filter.get())
                except Exception:
                    pass
            self._q_put(frame)

    def _cam_poll(self):
        try:
            frame = self._q.get_nowait()
            photo = utils.frame_to_photoimage(frame, self.DISP_W, self.DISP_H)
            self._photo_live = photo
            self.cam_canvas.config(image=photo, text="")
            self._cam_latest = frame
        except queue.Empty:
            pass
        if self.running:
            self.win.after(self.POLL_MS, self._cam_poll)
        else:
            self.cam_stat.config(text="Stopped")

    def _capture_cam(self):
        if self._cam_latest is None:
            messagebox.showwarning("Info", "Belum ada frame.", parent=self.win); return
        path = utils.save_frame(self._cam_latest, "output/image_processing", "cam_capture")
        self._orig_img    = self._cam_latest.copy()
        self._current_img = self._cam_latest.copy()
        self._undo_stack.clear(); self._redo_stack.clear()
        self._update_display()
        self.file_status.config(text=f"Captured → {path}")
        self._set_mode("file")
        self._log(f"[CAPTURE] {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # File mode
    # ─────────────────────────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            parent=self.win, title="Pilih Gambar",
            filetypes=[("Image","*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),("All","*.*")])
        if not path: return
        try:
            img = utils.read_image(path)
        except ValueError as e:
            messagebox.showerror("Error", str(e), parent=self.win); return

        self._orig_img    = img.copy()
        self._current_img = img.copy()
        self._current_file = path
        self._undo_stack.clear(); self._redo_stack.clear()
        self._crop_rect = None

        h, w = img.shape[:2]
        self.p_crop_x2.set(w); self.p_crop_y2.set(h)
        self.p_resize_w.set(w); self.p_resize_h.set(h)
        self._update_display()
        self._update_info()
        self.file_status.config(text=f"{os.path.basename(path)}  ({w}×{h})")
        self._log(f"[OPEN] {os.path.basename(path)}")

    def _select_filter(self, key: str):
        """Pilih filter — update active filter label + schedule real-time preview."""
        self._active_filter.set(key)
        if self.mode.get() == "camera":
            self.cam_stat.config(text=f"Filter aktif: {FILTERS[key][1]}")
            return
        if self._current_img is None: return
        self._schedule_preview()

    def _schedule_preview(self):
        """Real-time preview: debounce 80ms agar tidak lag saat drag slider."""
        if self._preview_job:
            self.win.after_cancel(self._preview_job)
        self._preview_job = self.win.after(80, self._do_live_preview)

    def _do_live_preview(self):
        """Render preview di cv_curr tanpa modify _current_img."""
        self._preview_job = None
        if self._current_img is None: return
        key = self._active_filter.get()
        if key not in FILTERS: return
        try:
            preview = self._apply_params(self._current_img.copy(), key)
            h, w = preview.shape[:2]
            curr_w = max(100, self.cv_curr.winfo_width())
            curr_h = max(100, self.cv_curr.winfo_height())

            photo = utils.frame_to_photoimage(
                preview,
                curr_w,
                curr_h
            )
            self._photo_curr = photo
            self.cv_curr.config(image=photo, text="")
            self.file_status.config(text=f"Preview: {FILTERS[key][1]}  ({w}×{h})")
        except Exception as e:
            self.file_status.config(text=f"Error: {e}")

    def _apply_filter(self, key: str):
        """Terapkan filter ke _current_img — push ke undo stack."""
        if self.mode.get() == "camera":
            self._active_filter.set(key)
            return
        if self._current_img is None:
            messagebox.showwarning("Info", "Buka gambar terlebih dahulu.", parent=self.win)
            return
        try:
            result = self._apply_params(self._current_img.copy(), key)
        except Exception as e:
            messagebox.showerror("Filter Error", str(e), parent=self.win); return

        # Push undo
        self._push_undo(f"Apply: {FILTERS[key][1]}")
        self._current_img = result
        self._redo_stack.clear()
        self._update_display()
        h, w = result.shape[:2]
        name = FILTERS[key][1]
        self.file_status.config(text=f"✅ {name} diterapkan  ({w}×{h})")
        self._log(f"[{name}] → {w}×{h}")

    def _apply_params(self, img: np.ndarray, key: str) -> np.ndarray:
        """Kumpulkan semua parameter dan panggil filter function."""
        if key not in FILTERS: return img
        fn = FILTERS[key][0]
        return fn(img,
            ksize=self.p_blur_k.get(),
            low=self.p_canny_lo.get(), high=self.p_canny_hi.get(),
            thresh=self.p_thresh.get(), method=self.p_thresh_m.get(),
            block=self.p_block_size.get(), C=2,
            angle=self.p_rotate.get(),
            width=self.p_resize_w.get(), height=self.p_resize_h.get(),
            x1=self.p_crop_x1.get(), y1=self.p_crop_y1.get(),
            x2=self.p_crop_x2.get(), y2=self.p_crop_y2.get(),
            alpha=self.p_alpha.get(), beta=self.p_beta.get(),
            mode=self.p_flip_m.get(), cmap=self.p_cmap.get(),
            op=self.p_morph_op.get(), ksize_morph=self.p_morph_k.get(),
            d=9, sigma=self.p_bilateral.get(),
            strength=self.p_sharpen.get(),
            clip=self.p_clahe_cl.get(), tile=8,
            h=self.p_denoise_h.get(),
            strength_v=self.p_vignette.get(),
            hue_shift=self.p_hue_shift.get(),
            sat=self.p_saturation.get(),
            pixelate_block=self.p_pixelate.get(),
            intensity=self.p_glitch.get(),
            amount=self.p_noise.get(),
            ksize_motion=self.p_motion_k.get(),
            angle_motion=self.p_motion_angle.get(),
        )

    # ─── Undo / Redo ──────────────────────────────────────────────────────────

    def _push_undo(self, op_name: str):
        self._undo_stack.append((self._current_img.copy(), op_name))
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._update_undo_btns()

    def _undo(self):
        if not self._undo_stack:
            toast_info = "Tidak ada yang bisa di-undo."; return
        img, name = self._undo_stack.pop()
        self._redo_stack.append((self._current_img.copy(), name))
        self._current_img = img
        self._update_display()
        self.file_status.config(text=f"↩ Undo: {name}")
        self._log(f"[UNDO] {name}")
        self._update_undo_btns()

    def _redo(self):
        if not self._redo_stack: return
        img, name = self._redo_stack.pop()
        self._undo_stack.append((self._current_img.copy(), name))
        self._current_img = img
        self._update_display()
        self.file_status.config(text=f"↪ Redo: {name}")
        self._log(f"[REDO] {name}")
        self._update_undo_btns()

    def _update_undo_btns(self):
        self.btn_undo.config(state="normal" if self._undo_stack else "disabled")
        self.btn_redo.config(state="normal" if self._redo_stack else "disabled")

    def _reset(self):
        if self._orig_img is None: return
        self._push_undo("Reset")
        self._current_img = self._orig_img.copy()
        self._redo_stack.clear(); self._crop_rect = None
        self._update_display()
        self.file_status.config(text="Reset ke gambar asli")
        self._log("[RESET]")

    # ─── Save ─────────────────────────────────────────────────────────────────

    def _save_result(self):
        if self._current_img is None:
            messagebox.showwarning("Info", "Belum ada gambar.", parent=self.win); return
        path = utils.save_frame(self._current_img, "output/image_processing", "result")
        self.file_status.config(text=f"Disimpan: {path}")
        self._log(f"[SAVE] {path}")
        messagebox.showinfo("Tersimpan", f"✅ Gambar disimpan:\n{path}", parent=self.win)

    def _save_as(self):
        if self._current_img is None:
            messagebox.showwarning("Info", "Belum ada gambar.", parent=self.win); return
        path = filedialog.asksaveasfilename(
            parent=self.win, title="Simpan Sebagai",
            defaultextension=".jpg",
            filetypes=[("JPEG","*.jpg *.jpeg"),("PNG","*.png"),
                       ("BMP","*.bmp"),("All","*.*")])
        if not path: return
        ext = os.path.splitext(path)[1].lower()
        params = []
        if ext in (".jpg",".jpeg"):
            q = messagebox.askquestion("Kualitas",
                "Gunakan kualitas tinggi (95)?\n[No = kualitas standar 85]",
                parent=self.win)
            params = [cv2.IMWRITE_JPEG_QUALITY, 95 if q=="yes" else 85]
        cv2.imwrite(path, self._current_img, params)
        self.file_status.config(text=f"Disimpan: {path}")
        self._log(f"[SAVE AS] {path}")

    # ─── Crop drag ────────────────────────────────────────────────────────────

    def _img_coord(self, canvas_w, canvas_h, cx, cy) -> tuple[int,int]:
        """Konversi koordinat canvas → koordinat gambar asli."""
        if self._current_img is None: return cx, cy
        h, w = self._current_img.shape[:2]
        scale = min(canvas_w/w, canvas_h/h, 1.0)
        disp_w = int(w*scale); disp_h = int(h*scale)
        off_x = (canvas_w - disp_w)//2
        off_y = (canvas_h - disp_h)//2
        ix = int((cx - off_x) / scale)
        iy = int((cy - off_y) / scale)
        return max(0,min(ix,w)), max(0,min(iy,h))

    def _crop_drag_start(self, event):
        if self._active_filter.get() != "crop": return
        w = self.cv_orig.winfo_width(); h = self.cv_orig.winfo_height()
        ix, iy = self._img_coord(w, h, event.x, event.y)
        self._crop_drag_start_pt = (ix, iy)
        self._crop_dragging = True

    def _crop_drag_move(self, event):
        if not self._crop_dragging or self._active_filter.get() != "crop": return
        w = self.cv_orig.winfo_width(); h = self.cv_orig.winfo_height()
        ix, iy = self._img_coord(w, h, event.x, event.y)
        sx, sy = self._crop_drag_start_pt
        x1,y1 = min(sx,ix), min(sy,iy)
        x2,y2 = max(sx,ix), max(sy,iy)
        self._crop_rect = (x1,y1,x2,y2)
        self.p_crop_x1.set(x1); self.p_crop_y1.set(y1)
        self.p_crop_x2.set(x2); self.p_crop_y2.set(y2)
        self.crop_info.config(text=f"Crop: ({x1},{y1})→({x2},{y2})  {x2-x1}×{y2-y1}px")
        self._schedule_preview()

    def _crop_drag_end(self, event):
        if self._crop_dragging:
            self._crop_dragging = False

    # ─── Histogram ────────────────────────────────────────────────────────────

    def _show_histogram(self):
        img = self._current_img
        if img is None:
            messagebox.showwarning("Info","Buka gambar dulu.",parent=self.win); return
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        hw = tk.Toplevel(self.win)
        hw.title("📊 Histogram Gambar")
        hw.configure(bg=DARK)
        utils.center_window(hw, 760, 520)

        fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.8), facecolor=DARK)
        fig.suptitle("Image Histogram Analysis", color="white", fontsize=11, y=0.98)

        # Color histogram
        ax1 = axes[0,0]; ax1.set_facecolor(PANEL)
        for i,(name,color) in enumerate([("Blue","#4488ff"),("Green","#44ff88"),("Red","#ff4444")]):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
            ax1.plot(hist, color=color, linewidth=1.2, label=name, alpha=0.85)
        ax1.set_title("Color Histogram", color="white", fontsize=9)
        ax1.legend(fontsize=7, labelcolor="white")
        for sp in ax1.spines.values(): sp.set_edgecolor("#2a2a4e")
        ax1.tick_params(colors="white", labelsize=7)

        # Grayscale histogram
        ax2 = axes[0,1]; ax2.set_facecolor(PANEL)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hg = cv2.calcHist([gray],[0],None,[256],[0,256])
        ax2.fill_between(range(256), hg.flatten(), color="#888888", alpha=0.8)
        ax2.set_title("Grayscale", color="white", fontsize=9)
        for sp in ax2.spines.values(): sp.set_edgecolor("#2a2a4e")
        ax2.tick_params(colors="white", labelsize=7)

        # Statistics table
        ax3 = axes[1,0]; ax3.set_facecolor(PANEL); ax3.axis("off")
        h, w, c = img.shape
        stats = [
            ["Dimensi",    f"{w} × {h}"],
            ["Channels",   str(c)],
            ["Min px",     str(img.min())],
            ["Max px",     str(img.max())],
            ["Mean",       f"{img.mean():.1f}"],
            ["Std Dev",    f"{img.std():.1f}"],
        ]
        tbl = ax3.table(cellText=stats, colLabels=["Stat","Value"],
                         cellLoc="left", loc="center",
                         colWidths=[0.5,0.5])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        for (r,col), cell in tbl.get_celld().items():
            cell.set_facecolor(HEADER if r==0 else DARK)
            cell.set_text_props(color="white")
            cell.set_edgecolor("#2a2a4e")
        ax3.set_title("Image Stats", color="white", fontsize=9)

        # CDF
        ax4 = axes[1,1]; ax4.set_facecolor(PANEL)
        cdf = np.cumsum(hg.flatten())
        cdf_norm = cdf / cdf.max()
        ax4.plot(cdf_norm, color="#cc44ff", linewidth=1.5)
        ax4.set_title("CDF (Grayscale)", color="white", fontsize=9)
        for sp in ax4.spines.values(): sp.set_edgecolor("#2a2a4e")
        ax4.tick_params(colors="white", labelsize=7)

        plt.tight_layout(rect=[0,0,1,0.96])
        cv_cv = FigureCanvasTkAgg(fig, master=hw)
        cv_cv.draw()
        cv_cv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        chart_path = os.path.join("output/charts",
                                   utils.timestamped_filename("histogram","png"))
        fig.savefig(chart_path, dpi=120, facecolor=fig.get_facecolor())
        self._log(f"[HIST] {chart_path}")

    # ─── Watermark ────────────────────────────────────────────────────────────

    def _add_watermark_dialog(self):
        if self._current_img is None:
            messagebox.showwarning("Info","Buka gambar dulu.",parent=self.win); return

        dlg = tk.Toplevel(self.win)
        dlg.title("🏷️ Tambah Watermark")
        dlg.configure(bg=DARK)
        dlg.resizable(False, False)
        utils.center_window(dlg, 400, 360)
        dlg.grab_set()

        def field(label, var):
            f = tk.Frame(dlg, bg=DARK, padx=20); f.pack(fill="x", pady=5)
            tk.Label(f, text=label, bg=DARK, fg=DIM, font=("Helvetica",9),
                     width=16, anchor="w").pack(side="left")
            e = tk.Entry(f, textvariable=var, bg=PANEL, fg=BR,
                         insertbackground="white", font=("Courier",10), relief="flat")
            e.pack(side="left", fill="x", expand=True, ipady=4, padx=(8,0))

        tk.Label(dlg, text="🏷️  Pengaturan Watermark", bg=HEADER, fg=ACCENT,
                 font=("Helvetica",12,"bold")).pack(fill="x", pady=14)
        field("📝  Teks:", self.wm_text)

        f2 = tk.Frame(dlg, bg=DARK, padx=20); f2.pack(fill="x", pady=5)
        tk.Label(f2, text="📍  Posisi:", bg=DARK, fg=DIM, font=("Helvetica",9),
                 width=16, anchor="w").pack(side="left")
        ttk.Combobox(f2, textvariable=self.wm_pos,
                     values=["Top Left","Top Right","Bottom Left","Bottom Right","Center"],
                     state="readonly", width=16).pack(side="left", padx=(8,0))

        f3 = tk.Frame(dlg, bg=DARK, padx=20); f3.pack(fill="x", pady=5)
        tk.Label(f3, text="🔤  Ukuran Font:", bg=DARK, fg=DIM, font=("Helvetica",9),
                 width=16, anchor="w").pack(side="left")
        tk.Scale(f3, from_=10, to=80, orient="horizontal", variable=self.wm_size,
                 bg=DARK, fg="white", troughcolor="#2a2a4e",
                 highlightbackground=DARK).pack(side="left", fill="x", expand=True)

        f4 = tk.Frame(dlg, bg=DARK, padx=20); f4.pack(fill="x", pady=5)
        tk.Label(f4, text="🌫️  Opacity:", bg=DARK, fg=DIM, font=("Helvetica",9),
                 width=16, anchor="w").pack(side="left")
        tk.Scale(f4, from_=0.1, to=1.0, resolution=0.05, orient="horizontal",
                 variable=self.wm_alpha, bg=DARK, fg="white",
                 troughcolor="#2a2a4e", highlightbackground=DARK).pack(
            side="left", fill="x", expand=True)

        def pick_color():
            color = colorchooser.askcolor(title="Warna Teks",
                                           initialcolor=self.wm_color,
                                           parent=dlg)
            if color[1]:
                self.wm_color = color[1]
                clr_btn.config(bg=self.wm_color)

        cf = tk.Frame(dlg, bg=DARK, padx=20); cf.pack(fill="x", pady=5)
        tk.Label(cf, text="🎨  Warna Teks:", bg=DARK, fg=DIM, font=("Helvetica",9),
                 width=16, anchor="w").pack(side="left")
        clr_btn = tk.Button(cf, text="  Pilih Warna  ", bg=self.wm_color,
                             relief="flat", cursor="hand2", command=pick_color)
        clr_btn.pack(side="left", padx=(8,0))

        tk.Frame(dlg, bg="#2a2a4e", height=1).pack(fill="x", pady=10, padx=20)
        br = tk.Frame(dlg, bg=DARK, padx=20); br.pack(fill="x")
        bc = {"font":("Helvetica",10,"bold"),"relief":"flat","cursor":"hand2","pady":9}
        tk.Button(br, text="✅  Terapkan", bg=GREEN, fg="white",
                  command=lambda: (self._apply_watermark(), dlg.destroy()), **bc).pack(
            side="left", fill="x", expand=True)
        tk.Button(br, text="✖  Batal", bg="#555555", fg="white",
                  command=dlg.destroy, **bc).pack(side="left", padx=(8,0))

    def _apply_watermark(self):
        if self._current_img is None: return
        img = self._current_img.copy()
        h, w = img.shape[:2]
        text  = self.wm_text.get() or "Watermark"
        pos   = self.wm_pos.get()
        size  = self.wm_size.get()
        alpha = float(self.wm_alpha.get())
        # Hex color → BGR
        c = self.wm_color.lstrip("#")
        r2,g2,b2 = int(c[0:2],16), int(c[2:4],16), int(c[4:6],16)
        color_bgr = (b2, g2, r2)

        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = size / 28.0
        thick = max(1, int(scale * 1.2))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        margin = 16

        positions = {
            "Top Left":     (margin, margin + th),
            "Top Right":    (w - tw - margin, margin + th),
            "Bottom Left":  (margin, h - margin),
            "Bottom Right": (w - tw - margin, h - margin),
            "Center":       ((w-tw)//2, (h+th)//2),
        }
        px, py = positions.get(pos, (w-tw-margin, h-margin))

        # Draw with transparency via overlay
        overlay = img.copy()
        cv2.putText(overlay, text, (px, py), font, scale, color_bgr, thick)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

        self._push_undo("Watermark")
        self._current_img = img
        self._redo_stack.clear()
        self._update_display()
        self._log(f"[WATERMARK] '{text}' @ {pos}")

    # ─── Display update ───────────────────────────────────────────────────────
    def _update_display(self):
        ow = max(100, self.cv_orig.winfo_width())
        oh = max(100, self.cv_orig.winfo_height())

        rw = max(100, self.cv_curr.winfo_width())
        rh = max(100, self.cv_curr.winfo_height())

        if self._orig_img is not None:
            disp = self._draw_crop_overlay(self._orig_img.copy())

            ph = utils.frame_to_photoimage(
                disp,
                ow,
                oh
            )

            self._photo_orig = ph
            self.cv_orig.config(image=ph, text="")

        if self._current_img is not None:
            ph2 = utils.frame_to_photoimage(
                self._current_img,
                rw,
                rh
            )

            self._photo_curr = ph2
            self.cv_curr.config(image=ph2, text="")

        self._update_undo_btns()

    def _draw_crop_overlay(self, img: np.ndarray) -> np.ndarray:
        """Gambar rectangle crop pada preview original."""
        if self._crop_rect is None: return img
        x1,y1,x2,y2 = self._crop_rect
        h,w = img.shape[:2]
        # Overlay gelap di luar crop region
        ov = img.copy()
        ov[:y1, :] = (ov[:y1,:]*0.4).astype(np.uint8)
        ov[y2:, :] = (ov[y2:,:]*0.4).astype(np.uint8)
        ov[y1:y2, :x1] = (ov[y1:y2,:x1]*0.4).astype(np.uint8)
        ov[y1:y2, x2:]  = (ov[y1:y2,x2:]*0.4).astype(np.uint8)
        cv2.rectangle(ov, (x1,y1), (x2,y2), (0,212,255), 2)
        # Corner handles
        d = 8
        for px,py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            cv2.rectangle(ov,(px-d,py-d),(px+d,py+d),(0,212,255),-1)
        return ov

    def _update_info(self):
        if self._current_img is None: return
        h, w = self._current_img.shape[:2]
        fname = os.path.basename(self._current_file) if self._current_file else "—"
        undo_cnt = len(self._undo_stack)
        self.info_lbl.config(
            text=f"{fname}  {w}×{h}  |  Undo: {undo_cnt}")

    # ─── Manual ───────────────────────────────────────────────────────────────

    def _save_manual(self):
        date   = self.mf_date.get().strip()
        time_s = self.mf_time.get().strip()
        op     = self.mf_op.get().strip() or "manual"
        note   = self.mf_note.get().strip()
        try:
            datetime.strptime(date, "%Y-%m-%d")
            datetime.strptime(time_s, "%H:%M:%S")
        except ValueError:
            self.man_status.config(text="❌ Format tanggal/waktu salah", fg=RED); return
        try:
            db.log_object_detection(
                source=f"image_manipulation:{op}",
                total_objects=0,
                object_names=[f"op:{op}", f"note:{note}" if note else "no_note"],
                image_path=""
            )
            self.man_status.config(text="✅ Catatan berhasil disimpan!", fg="#00ff88")
        except Exception as e:
            self.man_status.config(text=f"❌ Error: {e}", fg=RED)

    def _reset_manual_form(self):
        now = datetime.now()
        self.mf_date.set(now.strftime("%Y-%m-%d"))
        self.mf_time.set(now.strftime("%H:%M:%S"))
        self.mf_note.set("")
        self.man_status.config(text="")

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _log(self, text: str):
        self.hist_box.config(state="normal")
        self.hist_box.insert("end", text+"\n")
        self.hist_box.see("end")
        self.hist_box.config(state="disabled")
        self._update_info()

    def _clear_hist(self):
        self.hist_box.config(state="normal")
        self.hist_box.delete("1.0","end")
        self.hist_box.config(state="disabled")

    def _q_put(self, item):
        if self._q.full():
            try: self._q.get_nowait()
            except queue.Empty: pass
        try: self._q.put_nowait(item)
        except queue.Full: pass

    def _flush_q(self):
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break

    def _on_close(self):
        self.running = False
        if self.cap: self.cap.release()
        self.win.destroy()
