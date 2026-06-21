"""
database.py - Smart Vision Analysis System
Modul pengelolaan database SQLite dan logging CSV.
"""

import sqlite3
import csv
import os
from datetime import datetime
from pathlib import Path


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_vision.db")


def get_connection():
    """Mengembalikan koneksi SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Inisialisasi tabel database."""
    conn = get_connection()
    cur = conn.cursor()

    # Tabel object detection
    cur.execute("""
        CREATE TABLE IF NOT EXISTS object_detection_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            date        TEXT NOT NULL,
            time        TEXT NOT NULL,
            source      TEXT,
            total_objects INTEGER,
            object_names  TEXT,
            image_path    TEXT
        )
    """)

    # Tabel motion detection
    cur.execute("""
        CREATE TABLE IF NOT EXISTS motion_detection_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            date          TEXT NOT NULL,
            time          TEXT NOT NULL,
            source        TEXT,
            motion_count  INTEGER,
            image_path    TEXT
        )
    """)

    # Tabel anomaly detection
    cur.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_detection_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            date          TEXT NOT NULL,
            time          TEXT NOT NULL,
            source        TEXT,
            object_names  TEXT,
            area_info     TEXT,
            image_path    TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database diinisialisasi.")


# ─── Object Detection ──────────────────────────────────────────────────────────

def log_object_detection(source: str, total_objects: int,
                          object_names: list, image_path: str = ""):
    now = datetime.now()
    conn = get_connection()
    conn.execute("""
        INSERT INTO object_detection_log
            (timestamp, date, time, source, total_objects, object_names, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        now.isoformat(),
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        source,
        total_objects,
        ", ".join(object_names),
        image_path,
    ))
    conn.commit()
    conn.close()
    _append_csv("logs/object_detection.csv",
                ["timestamp", "date", "time", "source",
                 "total_objects", "object_names", "image_path"],
                [now.isoformat(), now.strftime("%Y-%m-%d"),
                 now.strftime("%H:%M:%S"), source,
                 total_objects, ", ".join(object_names), image_path])


def get_object_logs(limit: int = 100):
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM object_detection_log
        ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Motion Detection ─────────────────────────────────────────────────────────

def log_motion_detection(source: str, motion_count: int, image_path: str = ""):
    now = datetime.now()
    conn = get_connection()
    conn.execute("""
        INSERT INTO motion_detection_log
            (timestamp, date, time, source, motion_count, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        now.isoformat(),
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        source,
        motion_count,
        image_path,
    ))
    conn.commit()
    conn.close()
    _append_csv("logs/motion_detection.csv",
                ["timestamp", "date", "time", "source", "motion_count", "image_path"],
                [now.isoformat(), now.strftime("%Y-%m-%d"),
                 now.strftime("%H:%M:%S"), source, motion_count, image_path])


def get_motion_logs(limit: int = 100):
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM motion_detection_log
        ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Anomaly Detection ────────────────────────────────────────────────────────

def log_anomaly_detection(source: str, object_names: list,
                           area_info: str, image_path: str = ""):
    now = datetime.now()
    conn = get_connection()
    conn.execute("""
        INSERT INTO anomaly_detection_log
            (timestamp, date, time, source, object_names, area_info, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        now.isoformat(),
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        source,
        ", ".join(object_names),
        area_info,
        image_path,
    ))
    conn.commit()
    conn.close()
    _append_csv("logs/anomaly_detection.csv",
                ["timestamp", "date", "time", "source",
                 "object_names", "area_info", "image_path"],
                [now.isoformat(), now.strftime("%Y-%m-%d"),
                 now.strftime("%H:%M:%S"), source,
                 ", ".join(object_names), area_info, image_path])


def get_anomaly_logs(limit: int = 100):
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM anomaly_detection_log
        ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Statistik ────────────────────────────────────────────────────────────────

def get_summary_stats():
    """Mengembalikan statistik ringkasan semua modul."""
    conn = get_connection()
    stats = {}

    # Object detection stats
    row = conn.execute("""
        SELECT COUNT(*) as sessions,
               SUM(total_objects) as total_objects
        FROM object_detection_log
    """).fetchone()
    stats["object"] = {
        "sessions": row["sessions"] or 0,
        "total_objects": row["total_objects"] or 0,
    }

    # Motion detection stats
    row = conn.execute("""
        SELECT COUNT(*) as events,
               SUM(motion_count) as total_motions
        FROM motion_detection_log
    """).fetchone()
    stats["motion"] = {
        "events": row["events"] or 0,
        "total_motions": row["total_motions"] or 0,
    }

    # Anomaly detection stats
    row = conn.execute("""
        SELECT COUNT(*) as alerts
        FROM anomaly_detection_log
    """).fetchone()
    stats["anomaly"] = {
        "alerts": row["alerts"] or 0,
    }

    conn.close()
    return stats


def get_object_detection_chart_data():
    """Data untuk grafik deteksi objek per hari."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT date, SUM(total_objects) as total
        FROM object_detection_log
        GROUP BY date
        ORDER BY date DESC
        LIMIT 14
    """).fetchall()
    conn.close()
    return [(r["date"], r["total"]) for r in reversed(rows)]


def get_motion_chart_data():
    """Data untuk grafik motion per hari."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT date, SUM(motion_count) as total
        FROM motion_detection_log
        GROUP BY date
        ORDER BY date DESC
        LIMIT 14
    """).fetchall()
    conn.close()
    return [(r["date"], r["total"]) for r in reversed(rows)]


# ─── CSV Helper ───────────────────────────────────────────────────────────────

def _append_csv(filepath: str, headers: list, row: list):
    """Append satu baris ke file CSV, buat header jika baru."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)


# Inisialisasi saat modul diimpor
os.makedirs("logs", exist_ok=True)
init_db()
