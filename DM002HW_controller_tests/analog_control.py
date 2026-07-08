"""
Analog "gyro-style" continuous control for the DM002HW.

Packet analysis of the phone app WITH ITS GYRO/TILT MODE ENABLED
(`flight_with_video.pcapng`, see PROTOCOL_NOTES.md) proved the phone's smooth
flight is PURELY CLIENT-SIDE: it maps phone-tilt to fine analog axis bytes
(55-61 distinct values/axis, drifting +/-1-3 every ~50ms around 0x80) and
streams the SAME protocol we already send (cmd=0x40, 124-byte, ~18-20Hz). There
is no drone-side gyro-mode byte. So we replicate the phone's smoothness by
feeding CONTINUOUS fine axis values into `Drone.set_controls()` at loop rate
instead of 1-second discrete button pulses.

This module is the input side of that: a small side-channel HTTP server (same
idiom as `video_stream.MjpegServer`, deliberately OUTSIDE Gradio's event queue)
that a browser virtual-joystick posts held stick positions to at ~20Hz. Each
`GET /set?roll=&pitch=&throttle=&yaw=` clamps and forwards straight into
`drone.set_controls()`. A watchdog forces neutral if input goes stale (browser
tab closed / JS stopped / network hiccup) so a lost joystick can't leave the
drone pinned off-neutral.

Composes with drone.py's generation-counter arbitration for free: set_controls
bumps the generation each call, so a stale timed-move's terminal revert loses to
the live analog stream, and analog mode has no terminal revert of its own.
"""

import http.server
import socketserver
import threading
import time
from urllib.parse import urlparse, parse_qs

NEUTRAL = 0x80  # 128


def _clamp_axis(raw, default=NEUTRAL):
    try:
        return max(0, min(255, int(round(float(raw)))))
    except (TypeError, ValueError):
        return default


class AnalogAxisState:
    """Thread-safe holder for the latest analog stick position + when it arrived."""

    def __init__(self):
        self._lock = threading.Lock()
        self.roll = NEUTRAL
        self.pitch = NEUTRAL
        self.throttle = NEUTRAL
        self.yaw = NEUTRAL
        self.last_update = 0.0
        self.count = 0

    def update(self, roll, pitch, throttle, yaw):
        with self._lock:
            self.roll, self.pitch = roll, pitch
            self.throttle, self.yaw = throttle, yaw
            self.last_update = time.time()
            self.count += 1

    def snapshot(self):
        with self._lock:
            return (self.roll, self.pitch, self.throttle, self.yaw,
                    self.last_update, self.count)

    def age(self):
        with self._lock:
            if self.last_update == 0.0:
                return float("inf")
            return time.time() - self.last_update

    def touch(self):
        """Seed freshness (grace window) without recording a stick position."""
        with self._lock:
            self.last_update = time.time()


class _AnalogHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep console clean

    def _no_content(self):
        self.send_response(204)
        # CORS: the joystick JS runs on the Gradio origin (a different port), so
        # allow the cross-origin GET. (The request already reaches us regardless;
        # this just keeps the browser from logging a blocked-response warning.)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/set":
            self.send_response(404)
            self.end_headers()
            return
        q = parse_qs(parsed.query)
        roll     = _clamp_axis(q.get("roll",     [NEUTRAL])[0])
        pitch    = _clamp_axis(q.get("pitch",    [NEUTRAL])[0])
        throttle = _clamp_axis(q.get("throttle", [NEUTRAL])[0])
        yaw      = _clamp_axis(q.get("yaw",      [NEUTRAL])[0])
        self.server.controller.on_input(roll, pitch, throttle, yaw)
        self._no_content()


class _ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    # SO_REUSEADDR so re-engaging analog mode on the same port right after
    # disengaging binds cleanly (Windows holds the port briefly). Same reason
    # video_stream.MjpegServer sets it.
    allow_reuse_address = True


class AnalogInputServer:
    """Threaded HTTP server accepting `GET /set?roll=&pitch=&throttle=&yaw=` and
    forwarding clamped values straight into `drone.set_controls()`. A watchdog
    forces neutral if no fresh input within `stale_after` seconds."""

    def __init__(self, drone, axis_state=None, logger=None,
                 host="127.0.0.1", port=8095, stale_after=0.3, invert=None):
        self.drone = drone
        self.axis_state = axis_state or AnalogAxisState()
        self.logger = logger  # optional: a callable logger(event, **fields)
        self.host = host
        self.port = port
        self.stale_after = stale_after
        # Per-axis sign inversion (sign convention is a live-drone unknown; the
        # UI's invert checkboxes flip these). Passed by reference so a checkbox
        # change takes effect on the running stream without a restart.
        self.invert = invert if invert is not None else {
            "roll": False, "pitch": False, "throttle": False, "yaw": False}
        self._httpd = None
        self._thread = None
        self._watchdog_thread = None
        self._running = False
        self._forced_neutral = False

    # ── input path ───────────────────────────────────────────────────────────
    def on_input(self, roll, pitch, throttle, yaw):
        # Apply per-axis inversion by mirroring around neutral, then re-clamp.
        if self.invert.get("roll"):     roll     = max(0, min(255, 2 * NEUTRAL - roll))
        if self.invert.get("pitch"):    pitch    = max(0, min(255, 2 * NEUTRAL - pitch))
        if self.invert.get("throttle"): throttle = max(0, min(255, 2 * NEUTRAL - throttle))
        if self.invert.get("yaw"):      yaw      = max(0, min(255, 2 * NEUTRAL - yaw))
        self.axis_state.update(roll, pitch, throttle, yaw)
        self._forced_neutral = False
        # Wake the drone if it's idling and drive the sticks. set_controls leaves
        # _cmd untouched (engage_hover() below sets it to ARMED once on start).
        try:
            self.drone.set_idle_mode(False)
        except AttributeError:
            pass  # duck-typed fake drone in tests may not have it
        self.drone.set_controls(roll=roll, pitch=pitch, throttle=throttle, yaw=yaw)

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self):
        if self._httpd is not None:
            return
        # Arm + center once so the analog stream drives from a known neutral
        # hover, exactly like releasing the phone's stick to center.
        try:
            self.drone.set_idle_mode(False)
            self.drone.hover()
        except AttributeError:
            pass
        handler = type("_BoundAnalogHandler", (_AnalogHandler,), {})
        self._httpd = _ReusableThreadingTCPServer((self.host, self.port), handler)
        self._httpd.daemon_threads = True
        self._httpd.controller = self
        self._running = True
        self._forced_neutral = False
        # Grace window: don't let the watchdog force neutral until stale_after
        # elapses without a real post (avoids a redundant neutral right after
        # the engage-time hover, and gives the browser time to start streaming).
        self.axis_state.touch()
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        if self.logger:
            self.logger("analog_started", host=self.host, port=self.port,
                        stale_after=self.stale_after)

    def stop(self):
        self._running = False
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None
        # Leave the drone centered (not idle) so it hovers rather than holding a
        # stale stick position; the caller decides whether to land/disconnect.
        try:
            self.drone.hover()
        except AttributeError:
            pass
        if self.logger:
            self.logger("analog_stopped")

    def _watchdog_loop(self):
        # If the joystick stops posting (tab closed, JS stalled, network drop),
        # force neutral once so the drone doesn't stay pinned off-center. Hard
        # safety invariant — mirrors VideoReceiver._watchdog_loop.
        while self._running:
            if not self._forced_neutral and self.axis_state.age() > self.stale_after:
                self.drone.set_controls(roll=NEUTRAL, pitch=NEUTRAL,
                                        throttle=NEUTRAL, yaw=NEUTRAL)
                self._forced_neutral = True
                if self.logger:
                    self.logger("analog_stale_neutral", age=self.axis_state.age())
            time.sleep(0.1)

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/set"
