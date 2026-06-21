"""
notifier.py - Smart Vision Analysis System
Notifikasi Telegram Bot — non-blocking, queue-based.
Menggunakan requests langsung ke Telegram Bot API (tanpa library eksternal khusus).
"""

import os
import queue
import threading
import logging
from datetime import datetime
from typing import Optional

import requests

import config as cfg

logger = logging.getLogger(__name__)


# ─── TelegramNotifier ─────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Mengirim pesan dan foto ke Telegram Bot secara non-blocking.
    Semua pengiriman berjalan di background thread via internal queue.
    """

    API_BASE = "https://api.telegram.org/bot{token}/{method}"
    TIMEOUT  = 10   # detik timeout per request

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=50)
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = False
        self._start_worker()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return cfg.get("telegram.enabled", False)

    @property
    def token(self) -> str:
        return cfg.get("telegram.bot_token", "")

    @property
    def chat_id(self) -> str:
        return cfg.get("telegram.chat_id", "")

    @property
    def send_photo(self) -> bool:
        return cfg.get("telegram.send_photo", True)

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def notify_anomaly(self, image_path: str = "", zone_info: str = "",
                        objects: list = None):
        """Kirim notifikasi anomaly/intrusion."""
        if not self.enabled or not cfg.get("telegram.notify_on_anomaly", True):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        obj_str = ", ".join(objects or []) or "person"
        text = (
            f"🚨 *ANOMALY DETECTED*\n"
            f"📅 {ts}\n"
            f"👤 Objek: {obj_str}\n"
            f"📍 Zona: {zone_info or '-'}\n"
            f"#SmartVision #Anomaly"
        )
        self._enqueue("anomaly", text, image_path)

    def notify_motion(self, image_path: str = "", count: int = 0):
        """Kirim notifikasi motion terdeteksi."""
        if not self.enabled or not cfg.get("telegram.notify_on_motion", False):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"🏃 *MOTION DETECTED*\n"
            f"📅 {ts}\n"
            f"🔢 Event ke-{count}\n"
            f"#SmartVision #Motion"
        )
        self._enqueue("motion", text, image_path)

    def test_connection(self) -> tuple[bool, str]:
        """
        Uji koneksi secara sinkron (untuk settings panel).
        Return (success: bool, message: str).
        """
        if not self.is_configured():
            return False, "Token atau Chat ID belum diisi."
        try:
            # Cek bot info
            url = self.API_BASE.format(token=self.token, method="getMe")
            resp = requests.get(url, timeout=self.TIMEOUT)
            if not resp.ok:
                return False, f"Token tidak valid. ({resp.status_code})"

            bot_name = resp.json().get("result", {}).get("username", "?")

            # Kirim pesan test
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"✅ *Smart Vision — Test Koneksi*\n"
                f"📅 {ts}\n"
                f"Bot @{bot_name} berhasil terhubung!"
            )
            url2 = self.API_BASE.format(token=self.token, method="sendMessage")
            resp2 = requests.post(url2, data={
                "chat_id": self.chat_id,
                "text": msg,
                "parse_mode": "Markdown",
            }, timeout=self.TIMEOUT)

            if resp2.ok:
                self._connected = True
                return True, f"Berhasil! Bot: @{bot_name}"
            else:
                err = resp2.json().get("description", "Unknown error")
                return False, f"Chat ID salah atau bot belum di-start. ({err})"

        except requests.exceptions.ConnectionError:
            return False, "Tidak ada koneksi internet."
        except requests.exceptions.Timeout:
            return False, "Timeout — server Telegram tidak merespons."
        except Exception as e:
            return False, str(e)

    def shutdown(self):
        """Hentikan worker thread dengan graceful."""
        self._stop_event.set()
        self._queue.put(None)   # sentinel

    # ── Internal ──────────────────────────────────────────────────────────────

    def _enqueue(self, kind: str, text: str, image_path: str = ""):
        if not self.is_configured():
            logger.warning("[Telegram] Token/Chat ID belum dikonfigurasi.")
            return
        try:
            self._queue.put_nowait({
                "kind": kind,
                "text": text,
                "image_path": image_path,
            })
        except queue.Full:
            logger.warning("[Telegram] Queue penuh, notifikasi dibuang.")

    def _start_worker(self):
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="TelegramWorker",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self):
        """Background thread — ambil item dari queue dan kirim."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:   # sentinel → keluar
                break

            try:
                self._send(item)
            except Exception as e:
                logger.error(f"[Telegram] Gagal kirim: {e}")
            finally:
                self._queue.task_done()

    def _send(self, item: dict):
        text       = item["text"]
        image_path = item.get("image_path", "")

        if self.send_photo and image_path and os.path.exists(image_path):
            self._send_photo(image_path, caption=text)
        else:
            self._send_message(text)

    def _send_message(self, text: str):
        url = self.API_BASE.format(token=self.token, method="sendMessage")
        resp = requests.post(url, data={
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }, timeout=self.TIMEOUT)
        if not resp.ok:
            logger.error(f"[Telegram] sendMessage gagal: {resp.text}")

    def _send_photo(self, path: str, caption: str = ""):
        url = self.API_BASE.format(token=self.token, method="sendPhoto")
        with open(path, "rb") as photo_file:
            resp = requests.post(url, data={
                "chat_id":    self.chat_id,
                "caption":    caption[:1024],  # Telegram max caption
                "parse_mode": "Markdown",
            }, files={"photo": photo_file}, timeout=self.TIMEOUT)
        if not resp.ok:
            logger.error(f"[Telegram] sendPhoto gagal: {resp.text}")


# ─── Singleton ────────────────────────────────────────────────────────────────

_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Kembalikan instance singleton TelegramNotifier."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
