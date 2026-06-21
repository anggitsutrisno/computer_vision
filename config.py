"""
config.py - Smart Vision Analysis System
Manajemen konfigurasi terpusat via JSON.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_CONFIG = {
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
        "notify_on_anomaly": True,
        "notify_on_motion": False,
        "send_photo": True,
    },
    "detection": {
        "default_confidence": 0.45,
        "camera_index": 0,
        "motion_sensitivity": 500,
        "motion_cooldown": 2.0,
        "anomaly_cooldown": 3.0,
        "skip_frames": 1,
    },
    "output": {
        "base_dir": "output",
        "auto_screenshot": True,
    },
}


def load() -> dict:
    """Muat config dari file. Merge dengan default jika ada key yang hilang."""
    if not os.path.exists(CONFIG_PATH):
        save(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Deep merge dengan default
    _deep_merge(data, DEFAULT_CONFIG)
    return data


def save(config: dict):
    """Simpan config ke file JSON."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get(key_path: str, default=None):
    """
    Ambil nilai dengan dot-notation.
    Contoh: get('telegram.bot_token')
    """
    cfg = load()
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def set_value(key_path: str, value):
    """
    Set nilai dengan dot-notation dan simpan.
    Contoh: set_value('telegram.enabled', True)
    """
    cfg = load()
    keys = key_path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    save(cfg)


def _deep_merge(target: dict, source: dict):
    """Tambahkan key dari source ke target jika belum ada."""
    for key, val in source.items():
        if key not in target:
            target[key] = val
        elif isinstance(val, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], val)
