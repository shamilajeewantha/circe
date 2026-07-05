"""
Probe common WiFi drone UDP ports to find which one the DM002HW listens on.
We send a neutral control packet and a known-format test packet to each port,
then listen briefly for any response.
"""

import socket
import time

DRONE_IP = "192.168.169.1"
TIMEOUT = 0.3  # seconds to wait for a reply per port

# Ports seen across WiFi UAV / Chinese drone family
PORTS_TO_TRY = [50000, 8080, 8888, 8800, 9090, 2000, 7060, 8060, 1234, 40000, 8000, 8001, 8002]

# Standard WiFi UAV neutral packet (from blog.horner.tj protocol)
NEUTRAL_PKT = bytes([0x66, 0x80, 0x80, 0x80, 0x80, 0x00, 0x80 ^ 0x80 ^ 0x80 ^ 0x80, 0x99])

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(TIMEOUT)

print(f"Probing {DRONE_IP} on {len(PORTS_TO_TRY)} ports...\n")

for port in PORTS_TO_TRY:
    try:
        sock.sendto(NEUTRAL_PKT, (DRONE_IP, port))
        try:
            data, addr = sock.recvfrom(1024)
            print(f"[!!!] PORT {port} RESPONDED: {data.hex()} from {addr}")
        except socket.timeout:
            print(f"[ - ] port {port}: no reply (sent OK)")
    except Exception as e:
        print(f"[ERR] port {port}: {e}")
    time.sleep(0.05)

sock.close()
print("\nDone. If nothing responded, use PCAPdroid on your phone to sniff the real port.")
