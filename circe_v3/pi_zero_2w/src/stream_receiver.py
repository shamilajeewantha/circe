"""Receive the Pi's H.264 stream. Runs on the RTX 4050 box (or the Uno Q).

Counterpart to stream_sender.py. Writes the raw H.264 elementary stream to a file and/or
pipes it into a player, so the link can be verified before anything is built on top of it.

Two modes, mirroring the sender:

    listen   wait for the Pi to push       (pairs with `stream_sender.py connect`)
    connect  pull from a listening Pi      (pairs with `stream_sender.py listen`)

Examples
--------
    # wait for the Pi to push, save to disk
    python3 stream_receiver.py listen 8888 --out rover.h264

    # wait for the Pi to push, watch it live (needs mpv or ffplay installed)
    python3 stream_receiver.py listen 8888 --play

    # pull from a Pi that is listening
    python3 stream_receiver.py connect raspberrypi.local 8888 --play
"""

import argparse
import logging
import shutil
import signal
import socket
import subprocess
import sys
import time

log = logging.getLogger("stream_receiver")

CHUNK = 4096
STATS_EVERY_S = 5.0
PLAYERS = (
    ("mpv", ["--profile=low-latency", "--untimed", "--no-cache",
             "--demuxer-lavf-format=h264", "-"]),
    ("ffplay", ["-fflags", "nobuffer", "-flags", "low_delay",
                "-framedrop", "-f", "h264", "-"]),
)


def pick_player():
    """First available player, or None. Returns (argv, name)."""
    for name, extra in PLAYERS:
        path = shutil.which(name)
        if path:
            return [path, *extra], name
    return None, None


def open_sinks(args):
    """Build the list of write targets: an optional file and an optional player."""
    sinks = []
    player_proc = None
    out_fh = None

    if args.out:
        out_fh = open(args.out, "wb")
        sinks.append(out_fh.write)
        log.info("writing to %s", args.out)

    if args.play:
        argv, name = pick_player()
        if argv is None:
            log.error("--play requested but no player found. "
                      "Install one:  sudo apt install mpv")
            if out_fh:
                out_fh.close()
            return None, None
        log.info("piping to %s", name)
        player_proc = subprocess.Popen(argv, stdin=subprocess.PIPE)
        sinks.append(player_proc.stdin.write)

    if not sinks:
        log.error("nothing to do: pass --out FILE and/or --play")
        return None, None

    return sinks, player_proc


def drain(conn, sinks, label):
    """Read from the socket into every sink until the sender closes."""
    total = 0
    chunks = 0
    started = time.monotonic()
    last_report = started

    while True:
        buf = conn.recv(CHUNK)
        if not buf:
            log.info("sender closed the connection")
            break
        for write in sinks:
            try:
                write(buf)
            except BrokenPipeError:
                log.warning("a sink closed (player quit?) - stopping")
                return total, time.monotonic() - started

        total += len(buf)
        chunks += 1

        now = time.monotonic()
        if now - last_report >= STATS_EVERY_S:
            log.info(
                "[%s] %d chunks | %.1f MB total | %.0f s elapsed | %.2f Mbit/s",
                label, chunks, total / 1e6, now - started,
                (total * 8) / (now - started) / 1e6,
            )
            last_report = now

    return total, time.monotonic() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", metavar="FILE",
                        help="write the raw H.264 elementary stream here")
    common.add_argument("--play", action="store_true",
                        help="pipe into mpv/ffplay for a live view")

    p_listen = sub.add_parser("listen", parents=[common],
                              help="wait for the Pi to push")
    p_listen.add_argument("port", type=int)

    p_conn = sub.add_parser("connect", parents=[common],
                            help="pull from a Pi that is listening")
    p_conn.add_argument("host")
    p_conn.add_argument("port", type=int)
    p_conn.add_argument("--timeout", type=float, default=10.0)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    sinks, player_proc = open_sinks(args)
    if sinks is None:
        return 2

    srv = None
    conn = None
    try:
        if args.mode == "listen":
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", args.port))
            srv.listen(1)
            log.info("listening on 0.0.0.0:%d - waiting for the Pi", args.port)
            conn, peer = srv.accept()
            label = f"{peer[0]}:{peer[1]}"
            log.info("sender connected from %s", label)
        else:
            log.info("connecting to %s:%d", args.host, args.port)
            conn = socket.create_connection((args.host, args.port), timeout=args.timeout)
            conn.settimeout(None)
            label = f"{args.host}:{args.port}"
            log.info("connected - receiving")

        signal.signal(signal.SIGINT, lambda *_: conn.close())
        total, secs = drain(conn, sinks, label)
    except ConnectionRefusedError:
        log.error("connection refused - is the sender running on %s:%d?",
                  getattr(args, "host", "?"), args.port)
        return 1
    except OSError as exc:
        log.error("network error: %s", exc)
        return 1
    finally:
        if conn:
            conn.close()
        if srv:
            srv.close()
        if player_proc:
            try:
                player_proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            player_proc.wait(timeout=5)

    log.info("done: %.1f MB in %.0f s (%.2f Mbit/s average)",
             total / 1e6, secs, (total * 8) / secs / 1e6 if secs else 0.0)
    if args.out:
        log.info("play it back with:  mpv %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
