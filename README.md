# 👁‍🗨 Smart Vision Analysis System

> **Computer Vision berbasis Python** — OpenCV · YOLO v8 · Tkinter GUI

---

## 📋 Deskripsi

Smart Vision Analysis System adalah aplikasi Computer Vision lengkap dengan GUI berbasis Tkinter yang terdiri dari **4 modul utama**:

| # | Modul | Teknologi | Input |
|---|-------|-----------|-------|
| 1 | **Image Detection** | YOLO v8 (Ultralytics) | Webcam, Gambar |
| 2 | **Motion Detection** | MOG2 / Frame Difference | Webcam, Video |
| 3 | **Anomaly Detection** | YOLO + Mouse-drawn Zone | Webcam |
| 4 | **Image Manipulation** | OpenCV Filters | Gambar |

---

## 🗂️ Struktur Proyek

```
SmartVisionSystem/
├── main.py                  # Entry point (CLI + argparse)
├── gui.py                   # Dashboard utama
├── image_detection.py       # Modul 1
├── motion_detection.py      # Modul 2
├── anomaly_detection.py     # Modul 3
├── image_manipulation.py    # Modul 4
├── database.py              # SQLite + CSV logging
├── utils.py                 # Fungsi utilitas bersama
├── requirements.txt         # Dependensi
├── smart_vision.db          # Database SQLite (auto-created)
├── logs/
│   ├── object_detection.csv
│   ├── motion_detection.csv
│   └── anomaly_detection.csv
└── output/
    ├── object/              # Hasil deteksi objek
    ├── motion/              # Foto saat ada gerakan
    ├── anomaly/             # Foto saat ada intrusi
    ├── image_processing/    # Hasil manipulasi gambar
    ├── charts/              # Grafik Matplotlib
    └── screenshots/         # Screenshot manual
```

---

## ⚙️ Instalasi

### 1. Clone / Download
```bash
cd SmartVisionSystem
```

### 2. Buat Virtual Environment (opsional tapi direkomendasikan)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependensi
```bash
pip install opencv-python opencv-contrib-python ultralytics numpy Pillow matplotlib pandas
```

---

## 🚀 Cara Menjalankan

### Buka Dashboard Utama
```bash
python main.py
```

### Buka Modul Tertentu Langsung
```bash
python main.py --module 1   # Image Detection
python main.py --module 2   # Motion Detection
python main.py --module 3   # Anomaly Detection
python main.py --module 4   # Image Manipulation

# Shorthand
python main.py -m 1
```

### Jalankan Per-Modul (standalone)
```bash
# Setiap modul bisa dijalankan sebagai script
python image_detection.py
python motion_detection.py
python anomaly_detection.py
python image_manipulation.py
```

---

## 📦 Modul 1 — Image Detection

**Teknologi:** YOLO v8n (YOLOv8 nano, download otomatis)

**Fitur:**
- Input webcam real-time atau file gambar
- Mendeteksi 80+ kelas objek COCO
- Bounding box berwarna per kelas
- Label dan confidence score
- Counter objek per frame
- Auto-save ke `output/object/`
- Log ke SQLite + CSV
- Grafik total objek per hari

**Cara pakai:**
1. Pilih source (Webcam / File Gambar)
2. Atur confidence threshold (default 0.45)
3. Klik **START DETECTION**
4. Klik **SCREENSHOT** untuk simpan manual
5. Klik **LIHAT GRAFIK** untuk chart historis

---

## 🏃 Modul 2 — Motion Detection

**Teknologi:** Background Subtractor MOG2 + Frame Difference

**Fitur:**
- Input webcam atau file video
- Dua metode deteksi:
  - **Background Subtraction (MOG2)** — lebih akurat, adaptif
  - **Frame Difference** — lebih ringan, real-time
- Sensitivitas area gerakan yang bisa diatur
- Auto-capture foto saat ada gerakan (dengan cooldown 2 detik)
- Motion counter per sesi
- Log ke SQLite + CSV
- Grafik events per hari

**Cara pakai:**
1. Pilih source dan metode
2. Atur sensitivitas (area minimum pixel)
3. Klik **START**
4. Gerakan terdeteksi → foto tersimpan otomatis

---

## 🚨 Modul 3 — Anomaly Detection (Restricted Area)

**Teknologi:** YOLO v8 + Polygon Zone Mouse Drawing

**Fitur:**
- Input webcam
- Gambar **zona terlarang** langsung di layar dengan mouse
- Bisa menambahkan **banyak zona** sekaligus
- Deteksi khusus **person** (orang) yang masuk zona
- Alert visual: border merah + teks "ANOMALY DETECTED"
- Auto-save foto saat intrusi terdeteksi
- Log lokasi zona ke SQLite + CSV

**Cara pakai:**
1. Klik **START KAMERA**
2. Aktifkan **DRAW MODE** (tombol berubah jadi oranye)
3. Drag mouse di video untuk menggambar zona (bisa banyak)
4. Nonaktifkan Draw Mode
5. Sistem akan otomatis alert saat ada orang masuk zona

---

## 🎨 Modul 4 — Image Manipulation

**Teknologi:** OpenCV Filters + PIL

**Operasi yang tersedia:**

| Kategori | Operasi |
|----------|---------|
| Color | Grayscale, Histogram Equalization, Colormap (6 pilihan) |
| Blur | Gaussian Blur, Median Blur |
| Edge | Canny Edge Detection, Sharpen, Emboss |
| Threshold | Binary, Binary Inv, Truncate, To Zero, Otsu, Adaptive |
| Transform | Rotate, Flip (H/V/Both), Resize, Crop |
| Adjust | Brightness/Contrast (alpha + beta) |
| Analyze | Histogram Warna + Grayscale |

**Cara pakai:**
1. Klik **BUKA GAMBAR**
2. Klik operasi yang diinginkan (bisa berlapis/stack)
3. Klik **RESET ORIGINAL** untuk kembali ke awal
4. Klik **SIMPAN HASIL** untuk export

---

## 🗄️ Database & Logging

### SQLite (smart_vision.db)
- `object_detection_log` — timestamp, total_objects, object_names, image_path
- `motion_detection_log` — timestamp, motion_count, image_path
- `anomaly_detection_log` — timestamp, object_names, area_info, image_path

### CSV (logs/)
- `object_detection.csv`
- `motion_detection.csv`
- `anomaly_detection.csv`

---

## 📊 Grafik

Setiap modul memiliki tombol **LIHAT GRAFIK** yang menampilkan chart Matplotlib:
- **Modul 1:** Bar chart total objek per hari
- **Modul 2:** Line chart motion events per hari
- **Modul 3:** Bar chart anomaly alerts per hari
- **Dashboard:** Pie chart distribusi + bar chart total keseluruhan

Semua grafik tersimpan otomatis ke `output/charts/`.

---

## 🛠️ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Webcam tidak bisa dibuka | Pastikan tidak ada app lain pakai kamera. Coba index `1` atau `2`. |
| YOLO error download | Pastikan ada koneksi internet. Model `yolov8n.pt` (~6MB) diunduh otomatis. |
| Tkinter tidak tersedia | `sudo apt-get install python3-tk` (Linux) |
| Import error | `pip install -r requirements.txt` |
| Performa lambat | Turunkan resolusi atau ganti ke `yolov8n.pt` (paling ringan). |

---

## 📌 Requirements

```
Python         >= 3.10
opencv-python  >= 4.8
ultralytics    >= 8.0  (YOLO v8)
numpy          >= 1.24
Pillow         >= 10.0
matplotlib     >= 3.7
```

---

*Smart Vision Analysis System — Nerazurra Dev Studio*
