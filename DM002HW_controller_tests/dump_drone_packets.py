"""
Extract control packets sent to 192.168.169.1:8800.
Shows full bytes AND focuses on the control region (bytes 20-27)
where roll/pitch/throttle/yaw/command/headless live.
"""

import struct
import socket

FILE = r"d:\my_github\circe\wireshark_1.pcapng"


def read_pcapng(path):
    frames = []
    link_types = {}
    with open(path, "rb") as f:
        raw = f.read()
    pos = 0
    while pos + 12 <= len(raw):
        btype = struct.unpack_from("<I", raw, pos)[0]
        blen  = struct.unpack_from("<I", raw, pos + 4)[0]
        if blen < 12 or pos + blen > len(raw):
            break
        body = raw[pos + 8 : pos + blen - 4]
        if btype == 0x00000001:
            link_types[len(link_types)] = struct.unpack_from("<H", body, 0)[0]
        elif btype == 0x00000006:
            iface  = struct.unpack_from("<I", body, 0)[0]
            caplen = struct.unpack_from("<I", body, 8)[0]
            frames.append((link_types.get(iface, 1), body[20:20+caplen]))
        pos += blen
    return frames


def decode_udp(link_type, frame):
    ip = frame[14:] if link_type == 1 else frame
    if len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 17:
        return None
    ihl = (ip[0] & 0xF) * 4
    udp = ip[ihl:]
    dp  = struct.unpack_from(">H", udp, 2)[0]
    dst = socket.inet_ntoa(ip[16:20])
    ulen = struct.unpack_from(">H", udp, 4)[0]
    return dst, dp, udp[8:8+max(0,ulen-8)]


def hexdump(data, indent="    "):
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        h = ' '.join(f'{b:02x}' for b in chunk)
        a = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"{indent}{i:04x}  {h:<47}  |{a}|")


def main():
    frames = read_pcapng(FILE)

    # ── 1. Show all distinct full payloads (group by full content) ────────────
    print("=" * 70)
    print("ALL DISTINCT PACKET CONTENTS -> 192.168.169.1:8800\n")

    seen_full = {}
    all_payloads = []

    for lt, frame in frames:
        r = decode_udp(lt, frame)
        if not r:
            continue
        dst, dp, pay = r
        if dst != "192.168.169.1" or dp != 8800 or not pay:
            continue
        all_payloads.append(pay)
        if pay not in seen_full:
            seen_full[pay] = len(seen_full) + 1

    print(f"Total control packets: {len(all_payloads)}")
    print(f"Distinct payloads:     {len(seen_full)}\n")

    for pay, idx in sorted(seen_full.items(), key=lambda x: x[1]):
        print(f"--- Distinct #{idx}  len={len(pay)} ---")
        hexdump(pay)
        print()

    # ── 2. Focus on control bytes in ef-02 packets ────────────────────────────
    print("=" * 70)
    print("CONTROL BYTE ANALYSIS (bytes 20-27 of ef-02-xx packets)\n")
    print(f"  {'#':<5} {'len':<5} {'ctr':>5}  roll pitch thro  yaw  cmd  head  chk")
    print(f"  {'-'*65}")

    for i, pay in enumerate(all_payloads):
        if len(pay) < 28 or pay[0] != 0xef or pay[1] != 0x02:
            continue
        ctr = pay[12] + pay[13] * 256
        roll, pitch, throttle, yaw = pay[20], pay[21], pay[22], pay[23]
        cmd, headless = pay[24], pay[25]
        chk = pay[26] if len(pay) > 26 else 0
        # only print lines where something interesting is non-zero
        if roll != 0 or pitch != 0 or throttle != 0 or yaw != 0 or cmd != 0:
            print(f"  {i:<5} {len(pay):<5} {ctr:>5}   "
                  f"{roll:3}  {pitch:3}  {throttle:3}  {yaw:3}  {cmd:3}  {headless:3}  {chk:3}")

    print("\n(only rows with non-zero control values shown)")


if __name__ == "__main__":
    import sys
    with open("drone_packets.txt", "w") as f:
        sys.stdout = f
        main()
    sys.stdout = sys.__stdout__
    print("Written to drone_packets.txt")
