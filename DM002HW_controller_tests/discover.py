"""
Send the known handshake to the drone and listen for any response.
Run this while connected to the drone's WiFi.

Also listens on all ports for any unsolicited broadcasts from the drone.
"""

import socket
import threading
import time

DRONE_IP   = "192.168.169.1"
HANDSHAKE  = bytes([0xEF, 0x00, 0x04, 0x00])
LISTEN_PORTS = [8800, 8801, 8802, 9060, 7060, 8060, 50000, 8888, 8080]


def listen_udp(port, results):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(0.5)
        s.bind(("0.0.0.0", port))
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                data, addr = s.recvfrom(4096)
                if data:
                    results.append((port, addr, data))
                    print(f"  [RECV on :{port}] from {addr}  len={len(data)}  hex={data[:32].hex(' ')}")
            except socket.timeout:
                pass
        s.close()
    except Exception as e:
        pass  # port already in use etc


def send_handshake():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1)
    for port in [8800, 8801]:
        for _ in range(5):
            s.sendto(HANDSHAKE, (DRONE_IP, port))
            print(f"  [SENT] ef 00 04 00  →  {DRONE_IP}:{port}")
            time.sleep(0.05)
    s.close()


def main():
    print(f"Discovering drone protocol at {DRONE_IP}")
    print(f"Listening on ports: {LISTEN_PORTS}\n")

    results = []
    threads = []
    for port in LISTEN_PORTS:
        t = threading.Thread(target=listen_udp, args=(port, results), daemon=True)
        t.start()
        threads.append(t)

    time.sleep(0.2)  # let listeners bind first

    print("[*] Sending handshakes...")
    send_handshake()

    print("\n[*] Listening for 8 seconds for any drone response or broadcast...\n")
    for t in threads:
        t.join()

    if results:
        print(f"\n[+] Got {len(results)} response(s)!")
        for port, addr, data in results:
            print(f"  port={port}  from={addr}  hex={data.hex(' ')}")
    else:
        print("\n[-] No response received.")
        print("    Try: make sure you're connected to drone WiFi (192.168.169.x)")
        print("    The drone may need the motors armed or props spinning first.")


if __name__ == "__main__":
    main()
