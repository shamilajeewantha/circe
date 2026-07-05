"""Quick TCP port scan on the drone to find open services (video feed, control, etc.)"""

import socket

DRONE_IP = "192.168.169.1"
PORTS = list(range(1, 10001))
TIMEOUT = 0.3
OPEN = []

print(f"Scanning {DRONE_IP} TCP ports 1-10000 (this takes ~30s)...\n")

for port in PORTS:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(TIMEOUT)
        result = s.connect_ex((DRONE_IP, port))
        if result == 0:
            print(f"  [OPEN] TCP port {port}")
            OPEN.append(port)
        s.close()
    except Exception:
        pass

print(f"\nDone. Open TCP ports: {OPEN if OPEN else 'none found'}")
