"""
DM002HW drone controller — protocol reverse-engineered from Wireshark capture.

Structure confirmed from live traffic:
  Outer wrapper: 88-byte or 124-byte UDP to 192.168.169.1:8800
  Inner control at byte offset 18: 66 [roll][pitch][throttle][yaw][cmd][chk] 99
  Checksum = roll ^ pitch ^ throttle ^ yaw ^ cmd
  Neutral = 0x80 (128) for all axes
  CMD_ARMED = 0x40 (normal flight / motors unlocked)
"""

import socket
import threading
import time

DRONE_IP   = "192.168.169.1"
DRONE_PORT = 8800
NEUTRAL    = 0x80   # 128
SPEED      = 0x08   # as observed in capture

# Inner command byte (byte 5 of inner 8-byte block)
CMD_ARMED    = 0x40  # normal flight state
CMD_TAKEOFF  = 0x01  # auto take-off / land toggle
CMD_STOP     = 0x02  # emergency stop
CMD_LAND     = 0x03  # land
CMD_CALIBRATE = 0x80  # gyro calibration

# Fixed outer packet fragments (confirmed from capture)
_HDR_88  = bytes([0xef,0x02,0x58,0x00, 0x02,0x02,0x00,0x01, 0x00,0x00,0x00,0x00])
_HDR_124 = bytes([0xef,0x02,0x7c,0x00, 0x02,0x02,0x00,0x01, 0x02,0x00,0x00,0x00])
_TAIL    = bytes([0x32,0x4b,0x14,0x2d,0x00,0x00])
_CTR2_SFX = bytes([0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,
                   0x00,0x00,0x14,0x00,0x00,0x00,0xff,0xff,0xff,0xff])
_CTR3_SFX = bytes([0x00,0x00,0x00,0x00,0x00,0x00,0x03,0x00,
                   0x00,0x00,0x10,0x00,0x00,0x00])

_HANDSHAKE = bytes([0xef, 0x00, 0x04, 0x00])


def _inner(roll, pitch, throttle, yaw, cmd):
    chk = roll ^ pitch ^ throttle ^ yaw ^ cmd
    return bytes([0x66, roll, pitch, throttle, yaw, cmd, chk, 0x99])


def _build(counter, roll, pitch, throttle, yaw, cmd, long=False):
    ctr1  = bytes([counter & 0xFF, (counter >> 8) & 0xFF])
    speed = bytes([0x00, 0x00, SPEED, 0x00])
    inner = _inner(roll, pitch, throttle, yaw, cmd)
    pad   = bytes(56)
    base  = (_HDR_124 if long else _HDR_88) + ctr1 + speed + inner + pad + _TAIL
    if long:
        ctr2 = bytes([(counter+1) & 0xFF, ((counter+1) >> 8) & 0xFF])
        ctr3 = bytes([(counter+2) & 0xFF, ((counter+2) >> 8) & 0xFF])
        base += ctr2 + _CTR2_SFX + ctr3 + _CTR3_SFX
    return base  # 88 or 124 bytes


class Drone:
    def __init__(self, ip=DRONE_IP, port=DRONE_PORT):
        self.ip        = ip
        self.port      = port
        self.sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._roll     = NEUTRAL
        self._pitch    = NEUTRAL
        self._throttle = NEUTRAL
        self._yaw      = NEUTRAL
        self._cmd      = CMD_ARMED
        self._counter  = 0
        self._running  = False
        self._thread   = None
        self._armed    = False
        self.last_error = None

    def _loop(self):
        while self._running:
            try:
                # Alternate short / long packets exactly as the app does
                self.sock.sendto(_build(self._counter, self._roll, self._pitch,
                                        self._throttle, self._yaw, self._cmd,
                                        long=False), (self.ip, self.port))
                self.sock.sendto(_build(self._counter, self._roll, self._pitch,
                                        self._throttle, self._yaw, self._cmd,
                                        long=True),  (self.ip, self.port))
            except OSError as e:
                self.last_error = str(e)
                self._armed = False
                print(f"[!] Control loop stopped — send failed: {e}")
                break
            self._counter += 1
            time.sleep(0.02)   # 50 Hz

    def connect(self):
        # A previous disconnect() closed the old socket — always start fresh
        # so reconnecting after a disconnect doesn't send/recv on a dead fd.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_error = None

        # Step 1: handshake
        print("[*] Sending handshake...")
        for _ in range(5):
            self.sock.sendto(_HANDSHAKE, (self.ip, self.port))
            time.sleep(0.05)

        # Step 2: a few zero-control init packets (as the app does)
        print("[*] Sending init packets...")
        for i in range(6):
            self.sock.sendto(_build(i, 0,0,0,0,0, long=False), (self.ip, self.port))
            self.sock.sendto(_build(i, 0,0,0,0,0, long=True),  (self.ip, self.port))
            time.sleep(0.05)

        # Step 3: start control loop in armed state
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._armed = True
        print(f"[+] Connected — sending to {self.ip}:{self.port}")

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        self.sock.close()
        self._armed = False
        print("[+] Disconnected")

    # ── flight controls ───────────────────────────────────────────────────────

    def set_controls(self, roll=NEUTRAL, pitch=NEUTRAL,
                     throttle=NEUTRAL, yaw=NEUTRAL):
        self._roll     = max(0, min(255, roll))
        self._pitch    = max(0, min(255, pitch))
        self._throttle = max(0, min(255, throttle))
        self._yaw      = max(0, min(255, yaw))

    def hover(self):
        self.set_controls()
        self._cmd = CMD_ARMED

    def takeoff(self):
        print("[*] Takeoff")
        self._cmd = CMD_TAKEOFF
        time.sleep(0.5)
        self._cmd = CMD_ARMED

    def land(self):
        print("[*] Land")
        self._cmd = CMD_LAND
        time.sleep(0.5)
        self._cmd = CMD_ARMED

    def stop(self):
        print("[!] Emergency stop")
        self._cmd = CMD_STOP

    def calibrate(self):
        print("[*] Calibrating gyro (keep flat)...")
        self._cmd = CMD_CALIBRATE
        time.sleep(1.0)
        self._cmd = CMD_ARMED

    # ── movements (duration in seconds) ──────────────────────────────────────

    def up(self, duration=1.0, power=180):
        print(f"[>] Up        duration={duration}s  throttle={power}")
        self.set_controls(throttle=power); time.sleep(duration); self.hover()

    def down(self, duration=1.0, power=80):
        print(f"[>] Down      duration={duration}s  throttle={power}")
        self.set_controls(throttle=power); time.sleep(duration); self.hover()

    def forward(self, duration=1.0, power=160):
        print(f"[>] Forward   duration={duration}s  pitch={power}")
        self.set_controls(pitch=power); time.sleep(duration); self.hover()

    def backward(self, duration=1.0, power=96):
        print(f"[>] Backward  duration={duration}s  pitch={power}")
        self.set_controls(pitch=power); time.sleep(duration); self.hover()

    def turn_left(self, duration=1.0, power=63):
        print(f"[>] Turn left  duration={duration}s  yaw={power}")
        self.set_controls(yaw=power); time.sleep(duration); self.hover()

    def turn_right(self, duration=1.0, power=191):
        print(f"[>] Turn right duration={duration}s  yaw={power}")
        self.set_controls(yaw=power); time.sleep(duration); self.hover()

    def move_left(self, duration=1.0, power=96):
        print(f"[>] Move left  duration={duration}s  roll={power}")
        self.set_controls(roll=power); time.sleep(duration); self.hover()

    def move_right(self, duration=1.0, power=160):
        print(f"[>] Move right duration={duration}s  roll={power}")
        self.set_controls(roll=power); time.sleep(duration); self.hover()


# ── demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    drone = Drone()
    drone.connect()
    try:
        print("[*] Arming (3s)...")
        time.sleep(3)

        print("[*] Calibrating gyro...")
        drone.calibrate()
        time.sleep(1)

        print("[*] Taking off...")
        drone.takeoff()
        time.sleep(3)

        print("[>] Hover 2s")
        drone.hover()
        time.sleep(2)

        drone.up(duration=1.5)

        print("[>] Hover 1s")
        drone.hover()
        time.sleep(1)

        drone.down(duration=1.5)

        print("[>] Hover 1s")
        drone.hover()
        time.sleep(1)

        drone.forward(duration=1.5)

        print("[>] Hover 1s")
        drone.hover()
        time.sleep(1)

        drone.backward(duration=1.5)

        print("[>] Hover 1s")
        drone.hover()
        time.sleep(1)

        drone.move_left(duration=1.0)

        print("[>] Hover 1s")
        drone.hover()
        time.sleep(1)

        drone.move_right(duration=1.0)

        print("[>] Hover 1s")
        drone.hover()
        time.sleep(1)

        drone.turn_left(duration=1.0)

        print("[>] Hover 1s")
        drone.hover()
        time.sleep(1)

        drone.turn_right(duration=1.0)

        print("[>] Hover 3s before landing...")
        drone.hover()
        time.sleep(3)

        print("[*] Landing...")
        drone.land()
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n[!] Interrupted — landing")
        drone.land()
        time.sleep(2)
    finally:
        drone.disconnect()
