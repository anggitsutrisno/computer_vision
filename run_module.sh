#!/bin/bash
cd "$(dirname "$0")"

echo "============================================"
echo "  Smart Vision Analysis System"
echo "  Nerazurra Dev Studio"
echo "============================================"
echo ""
echo "[1] Dashboard Utama"
echo "[2] Image Detection (YOLO)"
echo "[3] Motion Detection"
echo "[4] Anomaly Detection"
echo "[5] Image Manipulation"
echo "[0] Exit"
echo ""
read -p "Pilih modul (0-5): " choice

case $choice in
  0) exit 0 ;;
  1) python3 main.py ;;
  2) python3 main.py --module 1 ;;
  3) python3 main.py --module 2 ;;
  4) python3 main.py --module 3 ;;
  5) python3 main.py --module 4 ;;
  *) echo "Pilihan tidak valid" ;;
esac
