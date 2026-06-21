"""
main.py - Smart Vision Analysis System
Entry point utama. Bisa dijalankan langsung atau sebagai modul.

Cara pakai:
  python main.py              → Buka Dashboard
  python main.py --module 1   → Langsung modul Image Detection
  python main.py --module 2   → Langsung modul Motion Detection
  python main.py --module 3   → Langsung modul Anomaly Detection
  python main.py --module 4   → Langsung modul Image Manipulation
"""

import sys
import os
import argparse
import tkinter as tk

# Tambahkan direktori ini ke path
sys.path.insert(0, os.path.dirname(__file__))

import utils
import database as db


def check_dependencies():
    """Cek ketersediaan paket utama."""
    missing = []
    warnings = []

    try:
        import cv2
        ver = cv2.__version__
        print(f"   OpenCV {ver}")
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy as np
        print(f"   NumPy {np.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        from PIL import Image
        import PIL
        print(f"   Pillow {PIL.__version__}")
    except ImportError:
        missing.append("Pillow")

    try:
        import matplotlib
        print(f"   Matplotlib {matplotlib.__version__}")
    except ImportError:
        missing.append("matplotlib")

    try:
        from ultralytics import YOLO
        print("   Ultralytics YOLO")
    except ImportError:
        warnings.append("ultralytics (Modul 1 & 3 memerlukan ini)")

    try:
        import tkinter
        print("   Tkinter")
    except ImportError:
        missing.append("tkinter")

    if missing:
        print("\n Paket wajib belum terpasang:")
        for m in missing:
            print(f"   pip install {m}")
        sys.exit(1)

    if warnings:
        print("\n  Paket opsional belum terpasang:")
        for w in warnings:
            print(f"   pip install {w}")
        print("   (Modul lain tetap bisa digunakan)\n")


def launch_dashboard():
    """Buka dashboard utama."""
    from gui import MainDashboard
    app = MainDashboard()
    app.run()


def launch_module(module_id: int):
    """Buka modul tertentu secara langsung."""
    root = tk.Tk()
    root.withdraw()   # sembunyikan root window

    if module_id == 1:
        from image_detection import ImageDetectionWindow
        win = ImageDetectionWindow(root)
        win.win.protocol("WM_DELETE_WINDOW", root.destroy)
    elif module_id == 2:
        from motion_detection import MotionDetectionWindow
        win = MotionDetectionWindow(root)
        win.win.protocol("WM_DELETE_WINDOW", root.destroy)
    elif module_id == 3:
        from anomaly_detection import AnomalyDetectionWindow
        win = AnomalyDetectionWindow(root)
        win.win.protocol("WM_DELETE_WINDOW", root.destroy)
    elif module_id == 4:
        from image_manipulation import ImageManipulationWindow
        win = ImageManipulationWindow(root)
        win.win.protocol("WM_DELETE_WINDOW", root.destroy)
    else:
        print(f"ID modul tidak valid: {module_id} (gunakan 1-4)")
        sys.exit(1)

    root.mainloop()


def main():
    print("=" * 56)
    print("  Smart Vision Analysis System")
    print("  Computer Vision · OpenCV · YOLO v8 / YOLO26 · Python")
    print("  Nerazurra Dev Studio")
    print("=" * 56)
    print("\n Memeriksa dependensi…")
    check_dependencies()

    parser = argparse.ArgumentParser(
        description="Smart Vision Analysis System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--module", "-m",
        type=int,
        choices=[1, 2, 3, 4],
        help=(
            "Jalankan modul tertentu secara langsung:\n"
            "  1 = Image Detection\n"
            "  2 = Motion Detection\n"
            "  3 = Anomaly Detection\n"
            "  4 = Image Manipulation"
        ),
    )
    args = parser.parse_args()

    print("\n✨ Menginisialisasi sistem…")
    utils.ensure_dirs()
    db.init_db()
    print("   Folder output dibuat")
    print("   Database diinisialisasi")
    print()

    if args.module:
        module_names = {
            1: "Image Detection",
            2: "Motion Detection",
            3: "Anomaly Detection",
            4: "Image Manipulation",
        }
        print(f" Membuka Modul {args.module}: {module_names[args.module]}")
        launch_module(args.module)
    else:
        print(" Membuka Dashboard Utama…")
        launch_dashboard()


if __name__ == "__main__":
    main()
