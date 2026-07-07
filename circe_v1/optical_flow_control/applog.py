"""
Shared logging setup for the DM002HW controller app.

Added 2026-07-05 after repeatedly needing the user to copy-paste console
output back for diagnosis — plain print() statements aren't saved anywhere,
so if the terminal scrolls or closes, that history is gone. This gives
every module a real, leveled, timestamped, persistent log file in addition
to the console, using stdlib `logging` (no new dependency).

Usage:
    from applog import get_logger
    log = get_logger(__name__)
    log.info("connected")
    log.warning("stall detected")
    log.error("send failed: %s", exc)

Log file: app_logs/app.log next to this script, rotated at 5MB x 5 backups
so it never grows unbounded. Every process run appends (doesn't overwrite),
same "never delete history automatically" policy as video_debug/sessions/.
"""

import logging
import logging.handlers
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "app_logs")
LOG_PATH = os.path.join(LOG_DIR, "app.log")

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger("dm002hw")
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-16s pid=%(process)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    console_handler.setLevel(logging.INFO)
    root.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"dm002hw.{name}")


def close_logging():
    """Flush and close every file handler so the OS releases the lock on the
    log file — on Windows an open RotatingFileHandler keeps app_logs/ from
    being deletable. Called from main.py's shutdown() on exit."""
    global _configured
    root = logging.getLogger("dm002hw")
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)
    _configured = False
