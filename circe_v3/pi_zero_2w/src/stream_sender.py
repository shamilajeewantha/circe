"""CSI camera capture + H.264 TCP stream. Runs ON the Raspberry Pi Zero 2 W.

Wraps `rpicam-vid` and pipes its H.264 elementary stream over a TCP socket. Two modes:

    connect  push to a known receiver          (Pi -> RTX 4050, or Pi -> Uno Q)
    listen   wait for a receiver to pull       (handy for ad-hoc debugging)

`rpicam-vid` is used rather than picamera2 because it is confirmed present on the image
(see ../README.md); picamera2 has not been verified on this card.

Destination is a CLI argument on purpose. Per circe_v1/docs/mothership-scout.md the CSI
frames are bound for the RTX 4050, while the Uno Q <-> RPi 2W link is still undecided
(USB gadget-Ethernet vs UART). Both candidate links are IP, so only the address changes.

Examples
--------
    # push to the RTX 4050 box (receiver must be listening first)
    python3 stream_sender.py connect 192.168.1.4 8888

    # push to an Uno Q reached over USB gadget-Ethernet
    python3 stream_sender.py connect 10.12.194.2 8888

    # let a viewer pull instead
    python3 stream_sender.py listen 8888
"""

import argparse
import logging
import shutil
import signal
import socket
import subprocess
import sys
import time

log = logging.getLogger("stream_sender")

RPICAM_VID = "rpicam-vid"
CHUNK = 4096
STATS_EVERY_S = 5.0


def build_rpicam_cmd(args):
    """rpicam-vid writing H.264 to stdout.

    --inline repeats SPS/PPS headers on every I-frame so a receiver that joins late can
    still start decoding; without it a mid-stream client sees nothing but garbage.
    """
    return [
        RPICAM_VID,
        "-t", "0",                    # run until killed
        "-n",                         # no preview window (there is no display over SSH)
        "--inline",
        "--width", str(args.width),
        "--height", str(args.height),
        "--framerate", str(args.fps),
        "--bitrate", str(args.bitrate),
        "--codec", "h264",
        "-o", "-",                    # stdout
    ]


def pump(camera_stdout, conn, label):
    """Copy camera bytes to the socket until either end closes.

    Returns (bytes_sent, seconds_elapsed).
    """
    total = 0
    chunks = 0
    started = time.monotonic()
    last_report = started

    while True:
        buf = camera_stdout.read(CHUNK)
        if not buf:
            log.warning("camera stream ended (rpicam-vid exited?)")
            break
        try:
            conn.sendall(buf)
        except (BrokenPipeError, ConnectionResetError) as exc:
            log.warning("receiver went away: %s", exc)
            break

        total += len(buf)
        chunks += 1

        now = time.monotonic()
        if now - last_report >= STATS_EVERY_S:
            window = now - last_report
            log.info(
                "[%s] %d chunks | %.1f MB total | %.0f s elapsed | %.2f Mbit/s",
                label, chunks, total / 1e6, now - started,
                (total * 8) / (now - started) / 1e6,
            )
            last_report = now
            _ = window

    return total, time.monotonic() - started


def serve_listen(args, cam_proc):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    log.info("listening on 0.0.0.0:%d - waiting for a receiver", args.port)

    conn, peer = srv.accept()
    log.info("receiver connected from %s:%d", *peer)
    try:
        total, secs = pump(cam_proc.stdout, conn, f"{peer[0]}:{peer[1]}")
    finally:
        conn.close()
        srv.close()
    return total, secs


def serve_connect(args, cam_proc):
    log.info("connecting to %s:%d", args.host, args.port)
    conn = socket.create_connection((args.host, args.port), timeout=args.timeout)
    conn.settimeout(None)
    log.info("connected - streaming")
    try:
        total, secs = pump(cam_proc.stdout, conn, f"{args.host}:{args.port}")
    finally:
        conn.close()
    return total, secs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--width", type=int, default=1280)
    common.add_argument("--height", type=int, default=720)
    common.add_argument("--fps", type=int, default=30)
    common.add_argument("--bitrate", type=int, default=4_000_000,
                        help="H.264 bitrate in bits/sec (default 4 Mbit/s)")

    p_conn = sub.add_parser("connect", parents=[common],
                            help="push the stream to a listening receiver")
    p_conn.add_argument("host")
    p_conn.add_argument("port", type=int)
    p_conn.add_argument("--timeout", type=float, default=10.0,
                        help="TCP connect timeout in seconds")

    p_listen = sub.add_parser("listen", parents=[common],
                              help="wait for a receiver to connect and pull")
    p_listen.add_argument("port", type=int)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if shutil.which(RPICAM_VID) is None:
        log.error("%s not found on PATH. This script must run on the Pi.", RPICAM_VID)
        return 2

    cmd = build_rpicam_cmd(args)
    log.info("starting camera: %s", " ".join(cmd))
    cam_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _stop(signum, _frame):
        log.info("signal %d - shutting down", signum)
        cam_proc.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        if args.mode == "listen":
            total, secs = serve_listen(args, cam_proc)
        else:
            total, secs = serve_connect(args, cam_proc)
    except ConnectionRefusedError:
        log.error("connection refused - is the receiver running on %s:%d?",
                  args.host, args.port)
        return 1
    except OSError as exc:
        log.error("network error: %s", exc)
        return 1
    finally:
        cam_proc.terminate()
        try:
            cam_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cam_proc.kill()

    log.info("done: %.1f MB in %.0f s (%.2f Mbit/s average)",
             total / 1e6, secs, (total * 8) / secs / 1e6 if secs else 0.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
