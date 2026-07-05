"""
Consolidated .pcapng analyzer for DM002HW protocol work.

Built during the 2026-07-05 video reverse-engineering session (see
PROTOCOL_NOTES.md) to replace a pile of one-off scratchpad scripts with a
single reusable, documented tool. Pure stdlib (struct + socket), no external
pcap libraries — auditable and dependency-free.

USAGE (run from this directory, or pass any .pcapng path as first arg):

    python pcap_analyzer.py summary  [path.pcapng]
        Flow overview: every UDP/TCP (src,dst,ports) pair and how many
        packets on each. Good first look at a new capture.

    python pcap_analyzer.py control  [path.pcapng]
        Dumps every client->8800 control packet with decoded
        roll/pitch/throttle/yaw/cmd/checksum columns, flagging any
        non-neutral (interesting) rows. Use to find flight commands.

    python pcap_analyzer.py video    [path.pcapng]
        Full video-channel analysis: lists every distinct JPEG frame_id,
        when it completed (fragment reassembly), gap-flags any pause > 0.3s,
        and prints every non-fragment packet (info/handshake/wake) from the
        drone on port 1234 with real timestamps. THIS is the tool to run
        first when investigating "why did video stall/stop".

    python pcap_analyzer.py bytediff  <path_a.pcapng> <path_b.pcapng>
        Byte-position variance scan comparing control-packet content between
        TWO captures (e.g. a real working session vs. our own reimplementation)
        — for every packet length seen in each, reports every byte offset
        that varies at all and what values were seen, catching differences
        in fields nobody's specifically decoded yet. This is what found the
        ctr2/ctr3 keepalive-counter lead (see PROTOCOL_NOTES.md) — reach for
        this whenever "why is A different from B" instead of only checking
        already-known fields.

    python pcap_analyzer.py decode-frame N [path.pcapng]
        Reassembles video frame_id N from the capture and decodes it via
        video_stream.py's real JPEG-header-reconstruction pipeline, saving
        it as decoded_frame_N.png next to this script. Use this to visually
        confirm a capture actually contains real (not corrupted) video, or
        to eyeball what a specific frame looked like.

If no path is given, defaults to wireshark_1.pcapng in this directory.

### Why a from-scratch parser instead of scapy/dpkt
No external dependencies — avoids environment/install issues for whoever
picks this up next, and keeps the parsing logic fully explicit and auditable
line by line rather than hidden inside a library.

### pcapng format notes (see also PROTOCOL_NOTES.md)
A .pcapng file is a sequence of typed blocks: [Block Type: 4B][Block Length: 4B][Body][Block Length: 4B].
Relevant block types:
  0x0A0D0D0A — Section Header Block: starts a new section: interface IDs
               are scoped to their section, so a parser MUST reset its
               interface table (link types + timestamp resolutions) on
               every Section Header Block, or packets from a second
               section will silently get the wrong timestamp/link-type
               (this was a real bug found and fixed this session — a
               capture with multiple sections showed a video frame's
               timestamp jump backwards to ~0.000s until this was fixed).
  0x00000001 — Interface Description Block: declares one capture interface
               (link type + optional if_tsresol timestamp-resolution option,
               default 1 microsecond/tick if absent).
  0x00000006 — Enhanced Packet Block: one captured frame — interface id,
               64-bit timestamp (as tick-pair, scaled by that interface's
               resolution), captured length, then the raw frame bytes
               starting at body offset 20.
Link-layer type from the owning Interface Description Block tells us how to
strip the L2 header before reaching IP: type 1 (Ethernet) skips 14 bytes
(or 18 if a VLAN tag, ethertype 0x8100, is present); type 101 (Raw IP, seen
in some Android captures) has no L2 header to skip at all.
"""

from __future__ import annotations

import io
import os
import socket
import struct
import sys
from collections import Counter, defaultdict

DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "wireshark_1.pcapng")
DRONE_IP = "192.168.169.1"
CONTROL_PORT = 8800
VIDEO_PORT = 1234


def read_pcapng(path: str):
    """Parse a .pcapng file into a list of (link_type, raw_frame_bytes, abs_timestamp_seconds).

    Section-aware: interface tables reset on every Section Header Block, so
    timestamps and link types stay correct across multi-section captures
    (e.g. ones produced by restarting a capture mid-session).
    """
    frames = []
    link_types: dict[int, int] = {}
    tsresol: dict[int, float] = {}
    with open(path, "rb") as f:
        raw = f.read()
    pos = 0
    endian = "<"
    iface_count = 0
    while pos + 12 <= len(raw):
        btype = struct.unpack_from("<I", raw, pos)[0]
        blen = struct.unpack_from("<I", raw, pos + 4)[0]
        if blen < 12 or pos + blen > len(raw):
            break
        body = raw[pos + 8: pos + blen - 4]

        if btype == 0x0A0D0D0A:  # Section Header Block
            bom = struct.unpack_from("<I", body, 0)[0]
            endian = "<" if bom == 0x1A2B3C4D else ">"
            link_types = {}
            tsresol = {}
            iface_count = 0

        elif btype == 0x00000001:  # Interface Description Block
            lt = struct.unpack_from(endian + "H", body, 0)[0]
            idx = iface_count
            link_types[idx] = lt
            resol = 1e-6  # default: microseconds per tick
            opos = 8
            while opos + 4 <= len(body):
                ocode, olen = struct.unpack_from(endian + "HH", body, opos)
                if ocode == 0 and olen == 0:
                    break
                oval = body[opos + 4: opos + 4 + olen]
                if ocode == 9 and len(oval) >= 1:  # if_tsresol
                    b0 = oval[0]
                    resol = (2.0 ** -(b0 & 0x7F)) if (b0 & 0x80) else (10.0 ** -b0)
                opos += 4 + olen + ((4 - olen % 4) % 4)
            tsresol[idx] = resol
            iface_count += 1

        elif btype == 0x00000006:  # Enhanced Packet Block
            iface = struct.unpack_from(endian + "I", body, 0)[0]
            ts_hi = struct.unpack_from(endian + "I", body, 4)[0]
            ts_lo = struct.unpack_from(endian + "I", body, 8)[0]
            caplen = struct.unpack_from(endian + "I", body, 12)[0]
            frame = body[20: 20 + caplen]
            ts = ((ts_hi << 32) | ts_lo) * tsresol.get(iface, 1e-6)
            frames.append((link_types.get(iface, 1), frame, ts))

        pos += blen
    return frames


def decode_udp_or_tcp(link_type: int, frame: bytes):
    """Strip L2/IP headers, return (proto, src_ip, src_port, dst_ip, dst_port, payload) or None."""
    if link_type == 1:  # Ethernet
        if len(frame) < 14:
            return None
        et = struct.unpack_from(">H", frame, 12)[0]
        if et == 0x8100:
            ip_start = 18
        elif et == 0x0800:
            ip_start = 14
        else:
            return None
    else:  # Raw IP (e.g. link type 101, some Android captures)
        ip_start = 0

    ip = frame[ip_start:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        return None
    ihl = (ip[0] & 0xF) * 4
    proto = ip[9]
    src = socket.inet_ntoa(ip[12:16])
    dst = socket.inet_ntoa(ip[16:20])

    if proto == 17:  # UDP
        if len(ip) < ihl + 8:
            return None
        udp = ip[ihl:]
        sp = struct.unpack_from(">H", udp, 0)[0]
        dp = struct.unpack_from(">H", udp, 2)[0]
        ulen = struct.unpack_from(">H", udp, 4)[0]
        pay = udp[8: 8 + max(0, ulen - 8)]
        return proto, src, sp, dst, dp, pay

    if proto == 6:  # TCP
        if len(ip) < ihl + 20:
            return None
        tcp = ip[ihl:]
        sp = struct.unpack_from(">H", tcp, 0)[0]
        dp = struct.unpack_from(">H", tcp, 2)[0]
        doff = (tcp[12] >> 4) * 4
        pay = tcp[doff:]
        return (proto, src, sp, dst, dp, pay) if pay else None

    return None


# ── mode: summary ──────────────────────────────────────────────────────────

def mode_summary(path: str):
    frames = read_pcapng(path)
    print(f"Total link-layer frames: {len(frames)}")
    if frames:
        print(f"Time span: {frames[-1][2] - frames[0][2]:.2f}s")

    udp = defaultdict(int)
    tcp = defaultdict(int)
    for lt, frame, ts in frames:
        r = decode_udp_or_tcp(lt, frame)
        if not r:
            continue
        proto, src, sp, dst, dp, pay = r
        if not pay:
            continue
        (udp if proto == 17 else tcp)[(src, sp, dst, dp)] += 1

    for label, flows in [("UDP", udp), ("TCP", tcp)]:
        print(f"\n{'=' * 60}\n{label} — {len(flows)} flows")
        for (src, sp, dst, dp), count in sorted(flows.items(), key=lambda x: -x[1])[:20]:
            print(f"  {src}:{sp:<5} -> {dst}:{dp:<5}  {count} packets")


# ── mode: control ───────────────────────────────────────────────────────────

def mode_control(path: str):
    """Dump client->8800 control packets with decoded axis columns.
    Flags non-neutral rows (real flight commands) for quick scanning."""
    frames = read_pcapng(path)
    print(f"{'idx':<6}{'t(s)':>9}  {'len':<5}{'ctr':>6}  sp   roll pitch thro  yaw  cmd  chk")
    t0 = frames[0][2] if frames else 0
    idx = 0
    for lt, frame, ts in frames:
        r = decode_udp_or_tcp(lt, frame)
        if not r:
            continue
        proto, src, sp, dst, dp, pay = r
        if dst != DRONE_IP or dp != CONTROL_PORT or src == DRONE_IP:
            continue
        idx += 1
        if len(pay) < 2 or pay[0] != 0xEF:
            continue
        if pay[1] != 0x02 or len(pay) < 28:
            # handshake (ef 00), or the "ef 20" SSID/cmd-query packets, or short
            print(f"{idx:<6}{ts - t0:9.3f}  {len(pay):<5}  (non-control) hex={pay[:16].hex(' ')}")
            continue
        # Inner control block sits at offset 18: 66 roll pitch throttle yaw cmd chk 99
        # (verified structure per drone.py / PROTOCOL_NOTES.md — NOT offset 20,
        # which was a stale offset copied from the older dump_drone_packets.py
        # that used a different assumed frame layout).
        ctr = pay[12] + pay[13] * 256
        speed = pay[16] if len(pay) > 16 else 0
        roll, pitch, throttle, yaw = pay[19], pay[20], pay[21], pay[22]
        cmd, chk = pay[23], pay[24]
        marker = "  <-- non-neutral" if (roll, pitch, throttle, yaw, cmd) not in ((128, 128, 128, 128, 0x40), (128, 128, 128, 128, 0x41)) else ""
        print(f"{idx:<6}{ts - t0:9.3f}  {len(pay):<5}{ctr:>6} sp={speed:3}  "
              f"{roll:3}  {pitch:3}  {throttle:3}  {yaw:3}  {cmd:3}  {chk:3}{marker}")


# ── mode: bytediff ──────────────────────────────────────────────────────────

def _control_payloads(path: str, t_start=None, t_end=None):
    frames = read_pcapng(path)
    t0 = frames[0][2] if frames else 0
    out = []
    for lt, frame, ts in frames:
        r = decode_udp_or_tcp(lt, frame)
        if not r:
            continue
        proto, src, sp, dst, dp, pay = r
        if dst != DRONE_IP or dp != CONTROL_PORT or src == DRONE_IP:
            continue
        rel = ts - t0
        if t_start is not None and rel < t_start:
            continue
        if t_end is not None and rel > t_end:
            continue
        if len(pay) >= 2 and pay[0] == 0xEF and pay[1] == 0x02:
            out.append((rel, pay))
    return out


def mode_bytediff(path_a: str, path_b: str):
    """Byte-position variance scan across two captures' control packets —
    this is what found the ctr2/ctr3 keepalive lead (see PROTOCOL_NOTES.md).
    For every packet length seen in each file, reports which exact byte
    offsets vary at all across packets of that length, and what values were
    seen. Use this whenever comparing "why does A behave differently from
    B" instead of only checking the specific fields you already know about —
    it catches fields nobody's looked at yet.
    """
    a = _control_payloads(path_a)
    b = _control_payloads(path_b)
    print(f"{path_a}: {len(a)} control packets")
    print(f"{path_b}: {len(b)} control packets")

    lengths = sorted(set(len(p) for _, p in a) | set(len(p) for _, p in b))
    for length in lengths:
        pa = [p for _, p in a if len(p) == length]
        pb = [p for _, p in b if len(p) == length]
        if not pa or not pb:
            continue
        print(f"\n=== len={length}  (A: {len(pa)} pkts, B: {len(pb)} pkts) ===")
        for i in range(length):
            va = sorted(set(p[i] for p in pa))
            vb = sorted(set(p[i] for p in pb))
            if va != vb:
                print(f"  byte[{i:3d}]  A={va[:8]}{'...' if len(va) > 8 else ''}  "
                      f"B={vb[:8]}{'...' if len(vb) > 8 else ''}")


# ── mode: video ─────────────────────────────────────────────────────────────

def mode_video(path: str):
    """The main diagnostic tool for video streaming issues. Shows:
    - every completed frame_id and when it finished reassembling
    - gaps > 0.3s (stalls) flagged inline
    - every non-fragment (info/handshake) packet from the drone's video port
    with REAL capture timestamps (not just packet arrival order), so you can
    correlate stalls/resumes against what the client was sending at the time.
    """
    frames = read_pcapng(path)
    if not frames:
        print("No frames parsed.")
        return
    t0 = frames[0][2]

    frame_frag_seen: dict[int, set] = defaultdict(set)
    frame_frag_total: dict[int, int] = {}
    frame_first_ts: dict[int, float] = {}
    completed_order: list[int] = []
    other_events: list[tuple[float, str, str]] = []

    for lt, frame, ts in frames:
        r = decode_udp_or_tcp(lt, frame)
        if not r:
            continue
        proto, src, sp, dst, dp, pay = r
        rel = ts - t0

        if src == DRONE_IP and sp == VIDEO_PORT:
            if len(pay) < 2 or pay[0] != 0x93:
                continue
            if pay[1] == 0x01 and len(pay) >= 44:
                fid = int.from_bytes(pay[8:16], "little")
                frag = int.from_bytes(pay[32:36], "little")
                total = int.from_bytes(pay[36:40], "little")
                if fid not in frame_first_ts:
                    frame_first_ts[fid] = rel
                frame_frag_total[fid] = total
                frame_frag_seen[fid].add(frag)
                if len(frame_frag_seen[fid]) >= total and fid not in completed_order:
                    completed_order.append(fid)
            elif pay[1] == 0x04:
                other_events.append((rel, "info_packet", pay[8:].decode("ascii", "replace")))

        elif dst == DRONE_IP and dp == CONTROL_PORT and src != DRONE_IP:
            if pay[:2] == b"\xef\x00":
                other_events.append((rel, "client_handshake", pay.hex(" ")))
            elif pay[:2] == b"\xef\x20":
                other_events.append((rel, "client_ef20_wake", pay.hex(" ")))

    print(f"Distinct frame_ids with fragments: {len(frame_frag_total)}")
    print(f"Fully completed frames: {len(completed_order)}")
    if frame_frag_total:
        print(f"frame_id range: {min(frame_frag_total)}-{max(frame_frag_total)}")

    # merge frame completions + other events into one chronological timeline
    timeline = [(frame_first_ts[fid], "frame_complete", f"fid={fid}") for fid in completed_order]
    timeline += other_events
    timeline.sort(key=lambda x: x[0])

    print(f"\n{'t(s)':>9}  event")
    prev_frame_ts = None
    for ts, kind, detail in timeline:
        gap_marker = ""
        if kind == "frame_complete":
            if prev_frame_ts is not None and ts - prev_frame_ts > 0.3:
                gap_marker = f"   <-- GAP {ts - prev_frame_ts:.2f}s since last frame"
            prev_frame_ts = ts
        print(f"{ts:9.3f}  {kind:<18} {detail}{gap_marker}")


# ── mode: decode-frame ──────────────────────────────────────────────────────

def mode_decode_frame(path: str, target_fid: int):
    """Reassemble and decode one specific frame_id, saving it as a PNG next
    to this script — visual proof the capture contains real, valid video."""
    sys.path.insert(0, os.path.dirname(__file__))
    import video_stream as vs  # local import: only needed for this mode

    frames = read_pcapng(path)
    dummy_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv = vs.VideoReceiver(dummy_sock, DRONE_IP, CONTROL_PORT, width=640, height=360,
                             log_dir=os.path.join(os.path.dirname(__file__), "video_debug", "pcap_analyzer_tmp"))
    header = vs.generate_jpeg_header(recv.width, recv.height, recv.components)

    for lt, frame, ts in frames:
        r = decode_udp_or_tcp(lt, frame)
        if not r:
            continue
        proto, src, sp, dst, dp, pay = r
        if src != DRONE_IP or sp != VIDEO_PORT:
            continue
        before = recv.frames_ok
        recv._handle_packet(pay, (src, sp), header)
        if recv.frames_ok > before and recv.frames_ok == target_fid:
            from PIL import Image
            jpeg = recv.latest_jpeg()
            im = Image.open(io.BytesIO(jpeg)).convert("RGB")
            out_path = os.path.join(os.path.dirname(__file__), f"decoded_frame_{target_fid}.png")
            im.save(out_path)
            print(f"Decoded frame {target_fid} ({im.size[0]}x{im.size[1]}) -> {out_path}")
            return
    print(f"Frame {target_fid} was never fully reassembled in this capture "
          f"(frames_ok reached {recv.frames_ok}).")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "decode-frame":
        target_fid = int(sys.argv[2])
        path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_FILE
        mode_decode_frame(path, target_fid)
        return
    if mode == "bytediff":
        mode_bytediff(sys.argv[2], sys.argv[3])
        return
    path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FILE
    {"summary": mode_summary, "control": mode_control, "video": mode_video}[mode](path)


if __name__ == "__main__":
    main()
