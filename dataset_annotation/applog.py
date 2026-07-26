"""
Shared logging setup for gradio_app.py (the dataset_annotation pipeline's own UI wrapper) -
adapted from circe_v1/optical_flow_control/applog.py, same pattern.

This is the UI app's OWN log (ui_logs/app.log) - a SEPARATE concern from each pipeline stage
script's own run_log.txt (01/03/04/05/06 all log to <out>/run_log.txt via their own
logging.basicConfig calls). gradio_app.py doesn't need to touch this file's logger for
stage-subprocess output at all - it captures each subprocess's stdout/stderr directly for the
live in-UI log panel (see gradio_app.py) and streams it straight into the Gradio Textbox. This
module is only for the app's OWN lifecycle messages (started/stopped, subprocess launched/exited,
errors in the UI layer itself) - the same "history survives even if the terminal scrolls or
closes" rationale as the original.

Usage:
    from applog import get_logger
    log = get_logger(__name__)
    log.info("UI started")
    log.error("subprocess launch failed: %s", exc)

Log file: ui_logs/app.log next to this script, rotated at 5MB x 5 backups so it never grows
unbounded. Every process run appends (doesn't overwrite).
"""

import logging
import logging.handlers
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "ui_logs")
LOG_PATH = os.path.join(LOG_DIR, "app.log")

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger("circe_annotate_ui")
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
    return logging.getLogger(f"circe_annotate_ui.{name}")


def close_logging():
    """Flush and close every file handler so the OS releases the lock on the log file - on
    Windows/WSL an open RotatingFileHandler can keep ui_logs/ from being deletable. Called from
    gradio_app.py's shutdown() on exit."""
    global _configured
    root = logging.getLogger("circe_annotate_ui")
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)
    _configured = False
