"""
Reads wireshark_1.pcapng and prints every UDP/TCP flow with raw payloads.
"""

import struct
import socket
from collections import defaultdict

FILE = r"d:\my_github\circe\wireshark_1.pcapng"


def read_pcapng(path):
    frames = []
    link_types = {}

    with open(path, "rb") as f:
        raw = f.read()

    pos = 0
    endian = "<"

    while pos + 12 <= len(raw):
        btype = struct.unpack_from("<I", raw, pos)[0]
        blen  = struct.unpack_from("<I", raw, pos + 4)[0]
        if blen < 12 or pos + blen > len(raw):
            break
        body = raw[pos + 8 : pos + blen - 4]

        if btype == 0x0A0D0D0A:                         # Section Header
            bom = struct.unpack_from("<I", body, 0)[0]
            endian = "<" if bom == 0x1A2B3C4D else ">"

        elif btype == 0x00000001:                        # Interface Description
            lt = struct.unpack_from(endian + "H", body, 0)[0]
            link_types[len(link_types)] = lt

        elif btype == 0x00000006:                        # Enhanced Packet Block
            iface  = struct.unpack_from(endian + "I", body, 0)[0]
            caplen = struct.unpack_from(endian + "I", body, 8)[0]
            frame  = body[20 : 20 + caplen]
            frames.append((link_types.get(iface, 1), frame))

        elif btype == 0x00000002:                        # Obsolete Packet Block
            caplen = struct.unpack_from(endian + "I", body, 4)[0]
            frame  = body[16 : 16 + caplen]
            frames.append((link_types.get(0, 1), frame))

        pos += blen

    return frames


def decode(link_type, frame):
    if link_type == 1:                                   # Ethernet
        if len(frame) < 14:
            return None
        et = struct.unpack_from(">H", frame, 12)[0]
        if et == 0x8100:   ip_start = 18
        elif et == 0x0800: ip_start = 14
        else: return None
    else:
        ip_start = 0                                     # raw IP (101/228)

    ip = frame[ip_start:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        return None

    ihl   = (ip[0] & 0xF) * 4
    proto = ip[9]
    src   = socket.inet_ntoa(ip[12:16])
    dst   = socket.inet_ntoa(ip[16:20])

    if proto == 17:                                      # UDP
        if len(ip) < ihl + 8: return None
        udp  = ip[ihl:]
        sp   = struct.unpack_from(">H", udp, 0)[0]
        dp   = struct.unpack_from(">H", udp, 2)[0]
        ulen = struct.unpack_from(">H", udp, 4)[0]
        pay  = udp[8 : 8 + max(0, ulen - 8)]
        return proto, src, sp, dst, dp, pay

    if proto == 6:                                       # TCP
        if len(ip) < ihl + 20: return None
        tcp  = ip[ihl:]
        sp   = struct.unpack_from(">H", tcp, 0)[0]
        dp   = struct.unpack_from(">H", tcp, 2)[0]
        doff = (tcp[12] >> 4) * 4
        pay  = tcp[doff:]
        return (proto, src, sp, dst, dp, pay) if pay else None

    return None


def main():
    frames = read_pcapng(FILE)
    print(f"Total frames: {len(frames)}\n")

    udp = defaultdict(list)
    tcp = defaultdict(list)

    for lt, frame in frames:
        r = decode(lt, frame)
        if not r:
            continue
        proto, src, sp, dst, dp, pay = r
        if not pay:
            continue
        (udp if proto == 17 else tcp)[(src, sp, dst, dp)].append(pay)

    for label, flows in [("UDP", udp), ("TCP", tcp)]:
        print("=" * 72)
        print(f"{label}  —  {len(flows)} flows, {sum(len(v) for v in flows.values())} packets\n")
        for (src, sp, dst, dp), payloads in sorted(flows.items()):
            print(f"  {src}:{sp}  ->  {dst}:{dp}   ({len(payloads)} packets)")
            seen = []
            for p in payloads:
                if p not in seen:
                    seen.append(p)
            for i, p in enumerate(seen[:15]):
                asc = ''.join(chr(b) if 32 <= b < 127 else '.' for b in p[:48])
                print(f"    [{i+1:2}] len={len(p):4}  hex= {p[:24].hex(' ')}  |{asc}|")
            if len(seen) > 15:
                print(f"         ... {len(seen)} unique payloads")
            print()


if __name__ == "__main__":
    import sys
    with open("capture_analysis.txt", "w") as out:
        sys.stdout = out
        main()
    sys.stdout = sys.__stdout__
    print("Written to capture_analysis.txt")
