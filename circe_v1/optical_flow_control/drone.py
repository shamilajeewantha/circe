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

from applog import get_logger

log = get_logger("drone")

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


def _build(counter, roll, pitch, throttle, yaw, cmd, long=False, long_counter=None):
    # long_counter drives the ctr2/ctr3 fields inside the "long" packet's
    # suffix (bytes 88 and 108). Normally these are just counter+1/counter+2
    # (kept as the default for backwards compatibility with existing flight
    # behavior), but a real capture showed that during a healthy sustained
    # video session, ctr1 (this function's `counter`) FREEZES while ctr2/ctr3
    # keep incrementing independently at ~6.9Hz — decoupled entirely from
    # ctr1. See PROTOCOL_NOTES.md. Pass long_counter explicitly to replicate
    # that (see Drone._loop's idle-mode handling).
    if long_counter is None:
        long_counter = counter
    ctr1  = bytes([counter & 0xFF, (counter >> 8) & 0xFF])
    speed = bytes([0x00, 0x00, SPEED, 0x00])
    inner = _inner(roll, pitch, throttle, yaw, cmd)
    pad   = bytes(56)
    base  = (_HDR_124 if long else _HDR_88) + ctr1 + speed + inner + pad + _TAIL
    if long:
        c2, c3 = long_counter + 1, long_counter + 2
        ctr2 = bytes([c2 & 0xFF, (c2 >> 8) & 0xFF])
        ctr3 = bytes([c3 & 0xFF, (c3 >> 8) & 0xFF])
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
        self._long_counter = 0.0
        self._last_long_tick = None
        self._running  = False
        self._thread   = None
        self._armed    = False
        self.last_error = None
        self._idle_mode = False
        # Command arbitration (generation counter). Every "authoritative" write
        # (set_controls/hover/takeoff/land/calibrate/stop) bumps _gen under
        # _state_lock and returns the new value as a token. A timed command
        # (moves, takeoff/land/calibrate) captures its token before sleeping and
        # only applies its terminal revert-to-neutral/ARMED if _gen is unchanged
        # — so a newer command, an e-stop, or a disconnect cancels a stale
        # revert. Fixes: e-stop being silently undone by a later move's hover(),
        # and land/takeoff/calibrate truncated by a concurrent move's hover().
        self._state_lock = threading.Lock()
        self._gen = 0
        self._disconnect_event = threading.Event()

    def set_idle_mode(self, enabled: bool):
        """Video-streaming aid: the real WiFi UAV app spends ~95% of a
        healthy, sustained video session sending cmd=0 with all axes at
        zero and a FROZEN packet counter (disarmed/idle) — it only sends
        the continuous armed/neutral pattern this class normally sends
        forever for about half a second right after connecting. Confirmed
        from a real capture's control-channel bytes, see PROTOCOL_NOTES.md.
        Movement commands have no effect while this is enabled — call
        set_idle_mode(False) (or just fly normally) to resume flight."""
        self._idle_mode = enabled

    # Rate ctr2/ctr3 (bytes 88/108 of the "long" packet) advance at in a real
    # healthy sustained-video capture — measured directly from newtest.pcapng:
    # 17 increments over 2.461s = ~6.9 Hz. Confirmed to keep advancing even
    # while ctr1 (the main counter) is frozen. See PROTOCOL_NOTES.md.
    LONG_CTR_HZ = 6.9

    def _loop(self):
        while self._running:
            try:
                if self._idle_mode:
                    roll = pitch = throttle = yaw = 0
                    cmd = 0
                else:
                    with self._state_lock:  # atomic snapshot — no torn read vs. a command write
                        roll, pitch, throttle, yaw, cmd = (self._roll, self._pitch,
                                                           self._throttle, self._yaw, self._cmd)

                now = time.time()
                if self._last_long_tick is None:
                    self._last_long_tick = now
                self._long_counter += (now - self._last_long_tick) * self.LONG_CTR_HZ
                self._last_long_tick = now
                long_ctr = int(self._long_counter)

                # Send ONLY the 124-byte "long" packet, at ~18 Hz. A fresh
                # flight-with-video capture (flight_with_video.pcapng, see
                # PROTOCOL_NOTES.md "flight-with-video") proved the real app,
                # while ARMED and actively flying, streams video continuously
                # (two clean 60-70s windows) using 124-byte packets EXCLUSIVELY
                # — zero 88-byte short packets in the healthy window — with ctr1
                # advancing at ~17.6 Hz and ctr2/ctr3 decoupled at ~6 Hz. Our
                # 124-byte build is already byte-for-byte identical to the real
                # app's flight packet (verified). This loop previously sent an
                # 88+124 pair every iteration at 50 Hz — ~5.7x the packet rate
                # the real app uses — flooding the shared control/video socket
                # and starving the video RX, which is the actual cause of the
                # "video stalls after a few commands" stall (NOT being armed;
                # the real app is armed the whole flight and video is fine).
                self.sock.sendto(_build(self._counter, roll, pitch,
                                        throttle, yaw, cmd,
                                        long=True, long_counter=long_ctr),  (self.ip, self.port))
            except OSError as e:
                self.last_error = str(e)
                self._armed = False
                self._running = False  # so disconnect()/reconnect logic sees this loop as stopped
                log.error("Control loop stopped — send failed: %s", e)
                break
            if not self._idle_mode:
                self._counter += 1
            time.sleep(0.055)   # ~18 Hz, matching the real app's flight cadence

    def connect(self):
        # A previous disconnect() closed the old socket — always start fresh
        # so reconnecting after a disconnect doesn't send/recv on a dead fd.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_error = None

        # Real state-management bug found live 2026-07-05: none of these were
        # ever reset here, only in __init__. A live capture (gradio_test3.pcapng)
        # showed every reconnect within the same process carried over stale
        # counter values from the previous session — ctr2/ctr3 in particular
        # jumped to huge, discontinuous numbers (500+) instead of starting
        # fresh at 1, which the one session that DID reset (by luck) proved
        # was the difference between a stalled-at-7-frames session and a
        # 258-frame, 37-second healthy stream. The real app always starts
        # ctr1=0, ctr2=1 fresh on every connect — do the same here.
        self._counter = 0
        self._long_counter = 0.0
        self._last_long_tick = None
        # Fresh session: re-enable interruptible sleeps and invalidate any
        # command left sleeping from a previous connection (its stale revert
        # will find _gen changed and no-op).
        self._disconnect_event.clear()
        with self._state_lock:
            self._gen += 1

        # Step 1: handshake
        log.info("Sending handshake to %s:%s...", self.ip, self.port)
        for _ in range(5):
            self.sock.sendto(_HANDSHAKE, (self.ip, self.port))
            time.sleep(0.05)

        # Step 2: a few zero-control init packets (as the app does)
        log.info("Sending init packets...")
        for i in range(6):
            self.sock.sendto(_build(i, 0,0,0,0,0, long=False), (self.ip, self.port))
            self.sock.sendto(_build(i, 0,0,0,0,0, long=True),  (self.ip, self.port))
            time.sleep(0.05)

        # Step 3: start control loop in armed state
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._armed = True
        log.info("Connected — sending to %s:%s", self.ip, self.port)

    def disconnect(self):
        # Wake any command mid-sleep and invalidate its pending revert so it
        # can't fire on this (or the next) session.
        self._disconnect_event.set()
        with self._state_lock:
            self._gen += 1
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                # Don't close the socket out from under a thread that's
                # still using it — that's exactly the kind of race that
                # can leave sockets/threads piling up across repeated
                # connect/disconnect cycles. Wait longer instead of
                # proceeding regardless.
                log.warning("Control loop still running after 1s, waiting longer before closing socket...")
                self._thread.join(timeout=3)
                if self._thread.is_alive():
                    log.error("Control loop did not stop — socket left open to avoid a use-after-close race. "
                              "This connection may be in a bad state; consider restarting the app.")
                    self._armed = False
                    return
        self.sock.close()
        self._armed = False
        log.info("Disconnected")

    # ── command arbitration primitives ───────────────────────────────────────

    def _write_axes(self, roll=NEUTRAL, pitch=NEUTRAL, throttle=NEUTRAL,
                    yaw=NEUTRAL, cmd=None) -> int:
        """Authoritative axis (and optional cmd) write. Bumps the generation
        counter and returns the new token."""
        with self._state_lock:
            self._roll     = max(0, min(255, roll))
            self._pitch    = max(0, min(255, pitch))
            self._throttle = max(0, min(255, throttle))
            self._yaw      = max(0, min(255, yaw))
            if cmd is not None:
                self._cmd = cmd
            self._gen += 1
            return self._gen

    def _write_cmd(self, cmd) -> int:
        """Authoritative cmd-only write (axes untouched). Bumps _gen."""
        with self._state_lock:
            self._cmd = cmd
            self._gen += 1
            return self._gen

    def _revert_if_current(self, token: int, *, axes: bool, cmd=CMD_ARMED) -> bool:
        """Terminal revert for a timed command — applies ONLY if no newer
        authoritative write happened since `token` and we're still armed, so a
        stale command can never clobber an e-stop, a newer command, or a fresh
        session. Returns whether it applied."""
        with self._state_lock:
            if self._gen != token or not self._armed:
                return False
            if axes:
                self._roll = self._pitch = self._throttle = self._yaw = NEUTRAL
            self._cmd = cmd
            self._gen += 1
            return True

    def _interruptible_sleep(self, duration: float) -> bool:
        """Sleep that returns early (False) if a disconnect fires meanwhile."""
        return not self._disconnect_event.wait(timeout=duration)

    # ── flight controls ───────────────────────────────────────────────────────

    def set_controls(self, roll=NEUTRAL, pitch=NEUTRAL,
                     throttle=NEUTRAL, yaw=NEUTRAL) -> int:
        return self._write_axes(roll, pitch, throttle, yaw)

    def hover(self) -> int:
        # Authoritative — always applies (centers sticks, cmd=ARMED).
        return self._write_axes(cmd=CMD_ARMED)

    def takeoff(self):
        log.info("Takeoff")
        token = self._write_cmd(CMD_TAKEOFF)
        self._interruptible_sleep(0.5)
        self._revert_if_current(token, axes=False)

    def land(self):
        log.info("Land")
        token = self._write_cmd(CMD_LAND)
        self._interruptible_sleep(0.5)
        self._revert_if_current(token, axes=False)

    def stop(self):
        log.warning("Emergency stop")
        self._write_cmd(CMD_STOP)   # unconditional, ungated — never blockable

    def calibrate(self):
        log.info("Calibrating gyro (keep flat)...")
        token = self._write_cmd(CMD_CALIBRATE)
        self._interruptible_sleep(1.0)
        self._revert_if_current(token, axes=False)

    # ── movements (duration in seconds) ──────────────────────────────────────

    # Each timed move captures its write token, sleeps (interruptibly), then
    # reverts to neutral ONLY if still current — see _revert_if_current.

    def up(self, duration=1.0, power=180):
        log.info("Up duration=%ss throttle=%s", duration, power)
        token = self.set_controls(throttle=power)
        self._interruptible_sleep(duration)
        self._revert_if_current(token, axes=True)

    def down(self, duration=1.0, power=80):
        log.info("Down duration=%ss throttle=%s", duration, power)
        token = self.set_controls(throttle=power)
        self._interruptible_sleep(duration)
        self._revert_if_current(token, axes=True)

    def forward(self, duration=1.0, power=160):
        log.info("Forward duration=%ss pitch=%s", duration, power)
        token = self.set_controls(pitch=power)
        self._interruptible_sleep(duration)
        self._revert_if_current(token, axes=True)

    def backward(self, duration=1.0, power=96):
        log.info("Backward duration=%ss pitch=%s", duration, power)
        token = self.set_controls(pitch=power)
        self._interruptible_sleep(duration)
        self._revert_if_current(token, axes=True)

    def turn_left(self, duration=1.0, power=63):
        log.info("Turn left duration=%ss yaw=%s", duration, power)
        token = self.set_controls(yaw=power)
        self._interruptible_sleep(duration)
        self._revert_if_current(token, axes=True)

    def turn_right(self, duration=1.0, power=191):
        log.info("Turn right duration=%ss yaw=%s", duration, power)
        token = self.set_controls(yaw=power)
        self._interruptible_sleep(duration)
        self._revert_if_current(token, axes=True)

    def move_left(self, duration=1.0, power=96):
        log.info("Move left duration=%ss roll=%s", duration, power)
        token = self.set_controls(roll=power)
        self._interruptible_sleep(duration)
        self._revert_if_current(token, axes=True)

    def move_right(self, duration=1.0, power=160):
        log.info("Move right duration=%ss roll=%s", duration, power)
        token = self.set_controls(roll=power)
        self._interruptible_sleep(duration)
        self._revert_if_current(token, axes=True)


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
