"""
Experimental video receiver for the DM002HW's UDP JPEG-fragment stream.

STATUS: implemented but UNTESTED against real hardware — it was written
while offline from the drone (see PROTOCOL_NOTES.md). It reuses whatever
socket the Drone control channel is already using (video arrives back on the
same local port the control packets were sent from), reassembles JPEG
fragments per the offsets confirmed against our own capture, and synthesizes
the JPEG header bytes the drone doesn't send (ported from
https://github.com/marshallrichards/turbodrone, MIT-style RE reference —
see PROTOCOL_NOTES.md for the exact byte-offset citations).

Everything this module sees is logged to disk (video_debug/packets.jsonl +
saved .jpg frames) so a session without live access to this file can inspect
what actually happened on the wire and fix header/resolution mismatches
without needing to reconnect to the drone.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# ── JPEG header synthesis (ported from turbodrone's wifi_uav_jpeg.py) ─────

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"

# fmt: off
STD_LUMINANCE_QT = [
     16, 11, 10, 16, 24,  40,  51,  61,
     12, 12, 14, 19, 26,  58,  60,  55,
     14, 13, 16, 24, 40,  57,  69,  56,
     14, 17, 22, 29, 51,  87,  80,  62,
     18, 22, 37, 56, 68, 109, 103,  77,
     24, 35, 55, 64, 81, 104, 113,  92,
     49, 64, 78, 87,103, 121, 120, 101,
     72, 92, 95, 98,112, 100, 103,  99,
]
STD_CHROMINANCE_QT = [
    17, 18, 24, 47, 99,  99,  99,  99,
    18, 21, 26, 66, 99,  99,  99,  99,
    24, 26, 56, 99, 99,  99,  99,  99,
    47, 66, 99, 99, 99,  99,  99,  99,
    99, 99, 99, 99, 99,  99,  99,  99,
    99, 99, 99, 99, 99,  99,  99,  99,
    99, 99, 99, 99, 99,  99,  99,  99,
    99, 99, 99, 99, 99,  99,  99,  99,
]

# Standard baseline Huffman tables (ITU T.81 Annex K.3). Almost every cheap
# hardware JPEG encoder (this drone included, going by our capture) uses
# these rather than custom-optimized tables, and doesn't bother sending them
# over the wire since they're "well known" — but a generic decoder (PIL/
# libjpeg) still needs an explicit DHT segment to use them.
DC_LUMA_COUNTS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
DC_LUMA_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

DC_CHROMA_COUNTS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
DC_CHROMA_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

AC_LUMA_COUNTS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D]
AC_LUMA_VALUES = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
]

AC_CHROMA_COUNTS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
AC_CHROMA_VALUES = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
]
# fmt: on


def _dqt_segment(table_id: int, table: list[int]) -> bytes:
    payload = bytes([table_id]) + bytes(table)
    length = len(payload) + 2
    return b"\xff\xdb" + length.to_bytes(2, "big") + payload


def _dht_segment(table_class: int, table_id: int, counts: list[int], values: list[int]) -> bytes:
    payload = bytes([(table_class << 4) | table_id]) + bytes(counts) + bytes(values)
    length = len(payload) + 2
    return b"\xff\xc4" + length.to_bytes(2, "big") + payload


def _sof0_segment(width: int, height: int, num_components: int) -> bytes:
    if num_components == 1:
        comps = [(1, 1, 1, 0)]
    else:
        comps = [(1, 1, 1, 0), (2, 1, 1, 1), (3, 1, 1, 1)]  # Y, Cb, Cr
    body = bytearray()
    for cid, h, v, qt in comps:
        body += bytes([cid, (h << 4) | v, qt])
    length = 8 + 3 * num_components
    return (
        b"\xff\xc0"
        + length.to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + bytes([num_components])
        + bytes(body)
    )


def _sos_segment(num_components: int) -> bytes:
    if num_components == 1:
        sel = [(1, 0, 0)]
    else:
        sel = [(1, 0, 0), (2, 1, 1), (3, 1, 1)]
    body = bytearray()
    for cid, dc, ac in sel:
        body += bytes([cid, (dc << 4) | ac])
    length = 6 + 2 * num_components
    return (
        b"\xff\xda"
        + length.to_bytes(2, "big")
        + bytes([num_components])
        + bytes(body)
        + bytes([0, 63, 0])
    )


def generate_jpeg_header(width: int, height: int, num_components: int = 3) -> bytes:
    header = bytearray(SOI)
    header += _dqt_segment(0, STD_LUMINANCE_QT)
    if num_components == 3:
        header += _dqt_segment(1, STD_CHROMINANCE_QT)
    header += _sof0_segment(width, height, num_components)
    header += _dht_segment(0, 0, DC_LUMA_COUNTS, DC_LUMA_VALUES)
    header += _dht_segment(1, 0, AC_LUMA_COUNTS, AC_LUMA_VALUES)
    if num_components == 3:
        header += _dht_segment(0, 1, DC_CHROMA_COUNTS, DC_CHROMA_VALUES)
        header += _dht_segment(1, 1, AC_CHROMA_COUNTS, AC_CHROMA_VALUES)
    header += _sos_segment(num_components)
    return bytes(header)


# ── captured-from-the-wire video "kick" / wake triggers ────────────────────
# Exact bytes observed in wireshark_1.pcapng, sent client->8800. Timestamp
# analysis of the real capture (see PROTOCOL_NOTES.md) shows this exact
# triplet is what the original app sends whenever the video stream stalls —
# every stall in the reference capture (3.6-3.7s gaps) was immediately
# followed by a new frame right after this was resent. This is NOT a
# per-frame request; it's a stall-recovery "kick", used only when idle.
VIDEO_WAKE_PACKETS = [
    bytes.fromhex("ef2006000165"),
    bytes.fromhex("ef20190001673c693d325e62665f73736964" "3d636d643d323e"),
    bytes.fromhex("ef20190001673c693d325e62665f73736964" "3d636d643d333e"),
]

# Same handshake drone.py sends once at connect() — the reference capture
# also shows the original app re-sending bursts of this during the session,
# not just at startup. Used here as a second-tier stall-recovery escalation.
_HANDSHAKE_REPEAT = bytes([0xef, 0x00, 0x04, 0x00])


@dataclass
class _FrameBuffer:
    total: int
    fragments: dict = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)

    def complete(self) -> bool:
        return self.total > 0 and len(self.fragments) >= self.total

    def ordered_payload(self) -> bytes:
        return b"".join(self.fragments[i] for i in range(self.total) if i in self.fragments)


class PacketLogger:
    """Append-only JSONL packet log + bounded frame-capture dump for offline analysis."""

    def __init__(self, out_dir: str, max_saved_frames: int = 60):
        # Every VideoReceiver session gets its OWN uniquely-named log file and
        # frame directory (pid + start timestamp) — never overwritten, never
        # deleted by this code, so history from every past run stays on disk
        # for later offline analysis. `out_dir/latest_session.txt` always
        # points at the most recent one for convenience.
        self.pid = os.getpid()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_id = f"{stamp}_pid{self.pid}"
        self.out_dir = out_dir
        self.session_dir = os.path.join(out_dir, "sessions")
        self.frames_dir = os.path.join(self.session_dir, f"frames_{session_id}")
        os.makedirs(self.frames_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self.session_dir, f"packets_{session_id}.jsonl")
        self._lock = threading.Lock()
        self._saved = 0
        self._max_saved = max_saved_frames
        with open(os.path.join(out_dir, "latest_session.txt"), "w") as f:
            f.write(self._jsonl_path + "\n")
        self.log({"event": "logger_init", "pid": self.pid, "session_id": session_id})

    def log(self, record: dict):
        record["t"] = time.time()
        record.setdefault("pid", self.pid)
        with self._lock:
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def save_frame(self, jpeg_bytes: bytes, tag: str) -> Optional[str]:
        with self._lock:
            if self._saved >= self._max_saved:
                return None
            self._saved += 1
            n = self._saved
        path = os.path.join(self.frames_dir, f"frame_{n:03d}_{tag}.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        return path


class VideoReceiver:
    """
    Reads video fragments off an existing (already-bound) UDP socket shared
    with the control channel, reassembles them into JPEG frames, and keeps
    the most recently decoded frame available for the UI to poll.
    """

    def __init__(self, sock: socket.socket, drone_ip: str, drone_port: int,
                 width: int = 640, height: int = 360, components: int = 3,
                 log_dir: Optional[str] = None):
        self.sock = sock
        self.drone_ip = drone_ip
        self.drone_port = drone_port
        self.width = width
        self.height = height
        self.components = components
        self.logger = PacketLogger(log_dir or os.path.join(os.path.dirname(__file__), "video_debug"))

        self._buffers: dict[int, _FrameBuffer] = {}
        self._latest_frame_bytes: Optional[bytes] = None
        self._latest_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

        self._last_frame_rx_ts = time.time()
        self._first_frame_seen = False
        self._start_ts = time.time()
        self.last_decoded_ts: Optional[float] = None

        self.frames_ok = 0
        self.frames_failed = 0
        self.packets_seen = 0
        self.unknown_packets = 0
        self.kicks_sent = 0
        self.handshake_resends = 0
        self.last_error: Optional[str] = None

    # ── lifecycle ──────────────────────────────────────────────────────
    #
    # This is a FREE-RUNNING stream, not request-gated — confirmed against
    # real capture timing (77 frames arrived in ~10s with zero acks of any
    # kind sent by the original app). An earlier version of this code
    # implemented a turbodrone-style per-frame ACK/request loop; that was
    # solving the wrong problem (ported from a *different* drone in the same
    # OEM family) and very likely actively caused the stream to stop early
    # during live testing. See PROTOCOL_NOTES.md for the full analysis.
    #
    # What real captured sessions actually show recovering a stall: resending
    # the exact 3 `ef 20 ...` "wake" packets (VIDEO_WAKE_PACKETS) — every
    # stall in the reference capture was immediately followed by a new frame
    # right after this triplet was resent. This code replicates that as a
    # stall-triggered "kick", not a per-frame requirement.

    def start(self):
        if self._running:
            return
        self._running = True
        self._start_ts = time.time()
        self.sock.settimeout(0.5)
        self._send_wake(reason="initial_start")
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=1.5)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._latest_lock:
            return self._latest_frame_bytes

    # ── stall recovery: "kick" packets, not per-frame acks ──────────────

    def _send_wake(self, reason: str):
        for pkt in VIDEO_WAKE_PACKETS:
            try:
                self.sock.sendto(pkt, (self.drone_ip, self.drone_port))
            except OSError as e:
                self.logger.log({"event": "wake_send_failed", "reason": reason, "error": str(e)})
                return
        self.kicks_sent += 1
        self.logger.log({"event": "wake_sent", "reason": reason,
                          "since_start": time.time() - self._start_ts})

    def _send_handshake_resend(self, reason: str):
        # Mirrors the repeated `ef 00 04 00` bursts seen live in the capture
        # around longer stalls (in addition to the ef20 kick). Escalation
        # tier for stalls the plain wake doesn't clear.
        for _ in range(5):
            try:
                self.sock.sendto(_HANDSHAKE_REPEAT, (self.drone_ip, self.drone_port))
            except OSError as e:
                self.logger.log({"event": "handshake_resend_failed", "reason": reason, "error": str(e)})
                return
            time.sleep(0.05)
        self.handshake_resends += 1
        self.logger.log({"event": "handshake_resent", "reason": reason,
                          "since_start": time.time() - self._start_ts})

    def _watchdog_loop(self):
        # Stall detection: if no new *complete* frame for a while, escalate:
        # first re-send the ef20 wake triplet (matches what recovered every
        # stall in the reference capture), then if that doesn't clear it
        # within a few more seconds, also resend the full handshake burst.
        STALL_KICK_AFTER = 2.0
        STALL_HANDSHAKE_AFTER = 8.0
        last_kick_ts = 0.0
        last_handshake_ts = 0.0
        while self._running:
            time.sleep(0.3)
            idle = time.time() - self._last_frame_rx_ts
            if idle < STALL_KICK_AFTER:
                continue
            now = time.time()
            if idle >= STALL_HANDSHAKE_AFTER and now - last_handshake_ts > STALL_HANDSHAKE_AFTER:
                self._send_handshake_resend(reason=f"stalled {idle:.1f}s")
                last_handshake_ts = now
            elif now - last_kick_ts > STALL_KICK_AFTER:
                self._send_wake(reason=f"stalled {idle:.1f}s")
                last_kick_ts = now

    # ── background loops ──────────────────────────────────────────────

    def _rx_loop(self):
        header = generate_jpeg_header(self.width, self.height, self.components)
        while self._running:
            try:
                payload, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as e:
                self.logger.log({"event": "socket_closed", "error": str(e)})
                break

            self.packets_seen += 1
            self._handle_packet(payload, addr, header)

    def _handle_packet(self, payload: bytes, addr, header: bytes):
        if len(payload) < 8 or payload[0] != 0x93:
            self.unknown_packets += 1
            self.logger.log({
                "event": "unknown_packet", "from": f"{addr[0]}:{addr[1]}",
                "len": len(payload), "hex_head": payload[:32].hex(" "),
            })
            return

        msg_type = payload[1]

        if msg_type == 0x04:
            self.logger.log({
                "event": "info_packet", "len": len(payload),
                "ascii": payload[8:].decode("ascii", "replace"),
            })
            return

        if msg_type != 0x01 or len(payload) < 56:
            self.logger.log({
                "event": "other_type_packet", "type": msg_type, "len": len(payload),
                "hex_head": payload[:32].hex(" "),
            })
            return

        frame_id = int.from_bytes(payload[8:16], "little")
        frag_id = int.from_bytes(payload[32:36], "little")
        frag_total = int.from_bytes(payload[36:40], "little")
        body_len = int.from_bytes(payload[40:44], "little")
        quality = payload[48] if len(payload) > 48 else None
        body = payload[56:]

        self._first_frame_seen = True
        self._last_frame_rx_ts = time.time()

        self.logger.log({
            "event": "jpeg_fragment", "frame_id": frame_id, "frag_id": frag_id,
            "frag_total": frag_total, "body_len": body_len, "quality": quality,
            "payload_len": len(body),
        })

        buf = self._buffers.get(frame_id)
        if buf is None:
            buf = _FrameBuffer(total=frag_total)
            self._buffers[frame_id] = buf
        buf.fragments[frag_id] = body

        # Drop stale in-progress frames so the buffer dict doesn't grow forever.
        stale = [fid for fid, b in self._buffers.items()
                 if fid != frame_id and time.time() - b.first_seen > 5.0]
        for fid in stale:
            del self._buffers[fid]

        if buf.complete():
            del self._buffers[frame_id]
            self._assemble_and_decode(frame_id, buf, header)

    def _assemble_and_decode(self, frame_id: int, buf: _FrameBuffer, header: bytes):
        jpeg_bytes = header + buf.ordered_payload() + EOI
        try:
            from PIL import Image
            import io
            Image.open(io.BytesIO(jpeg_bytes)).load()
            self.frames_ok += 1
            self.last_decoded_ts = time.time()
            with self._latest_lock:
                self._latest_frame_bytes = jpeg_bytes
            self.logger.log({"event": "frame_decoded_ok", "frame_id": frame_id, "size": len(jpeg_bytes)})
            self.logger.save_frame(jpeg_bytes, "ok")
        except Exception as e:  # noqa: BLE001 - want to survive any decode failure
            self.frames_failed += 1
            self.last_error = str(e)
            self.logger.log({"event": "frame_decode_failed", "frame_id": frame_id,
                              "size": len(jpeg_bytes), "error": str(e)})
            self.logger.save_frame(jpeg_bytes, "err")
