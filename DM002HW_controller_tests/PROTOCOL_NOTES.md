# DM002HW protocol notes (for future agents/sessions)

Drone: DM002HW DIY WiFi+camera quad. AP at `192.168.169.1`, client gets
`192.168.169.3`. Reverse-engineered from `wireshark_1.pcapng` in this folder
(see `analyze_capture.py` / `capture_analysis.txt` / `dump_drone_packets.py`).
That capture appears to have been shot at night (most frames decode near-black).

## Current status (read this first — the rest of this file is a chronological
## record of how we got here, including dead ends; this section is the answer)

- **Control (fly the drone)**: confirmed working, has been the whole time.
  `drone.py`, no open issues.
- **Video streaming**: **confirmed working live** — a real session
  (`gradio_test3.pcapng`) streamed 258 frames continuously over 37.7
  seconds, plus several more good runs in the same session.
  - **UPDATE 2026-07-08 — read "FLIGHT-WITH-VIDEO FINDING" below; it corrects
    the emphasis here.** The real cause of our armed-mode stall was **packet
    RATE**, not arm state: `drone.py` sent an 88+124 packet pair at 50 Hz
    (~100 pkt/s), ~5.7× the real app's ~18 pkt/s (124-byte-only). The real
    app is *armed the whole flight* and video is fine. Fix: `_loop()` now
    sends 124-byte-only at ~18 Hz. `set_idle_mode` is no longer required for
    video (and must never be force-restored mid-flight — that's a disarm).
  - The earlier fix chain still holds as contributing factors: decoupling the
    `ctr2`/`ctr3` counter fields from the main counter (they keep advancing at
    ~6 Hz even while `ctr1` freezes) + resetting all counter state at the top
    of every `connect()` (not just `__init__`, which was silently breaking
    every reconnect within the same process). See "CONTROL-CHANNEL FINDING"
    and "ctr2/ctr3 decoupling" sections below for the evidence.
  - Ignore the "FREE-RUNNING stream ... recovered by a 'kick'" section
    below — that was a real hypothesis, live-tested, and **disproven** (see
    "Live-tested and disproven hypotheses"). It's kept for the record, not
    because it's the current answer.
- **Gyro / tilt "smooth control" mode is PURELY CLIENT-SIDE (2026-07-08)** —
  see "GYRO/TILT-MODE FINDING" below. The phone app's gyro button (fine, smooth
  tilt-to-fly with good hover) sends the **exact same protocol we already use**
  (`cmd=0x40`, 124-byte, ~18–20 Hz); there is **no drone-side gyro-mode byte**.
  The only difference is the phone streams a *continuous* fine spread of axis
  values (55–61 distinct values/axis, drifting ±1–3 every ~50 ms around 0x80)
  instead of our 1-second discrete button pulses (96/128/160). Replicated in
  both apps via `analog_control.py` (a browser joystick → `set_controls()` at
  ~20 Hz). No protocol change required.
- **Video display in the Gradio UI**: rewritten from polling (`gr.Image` +
  `gr.Timer`, capped at 2.5fps, dropped most frames) to push-based
  MJPEG-over-HTTP (`video_stream.MjpegServer` + an `<img>` tag). Verified
  end-to-end **offline** with a real HTTP client against real captured
  frame data — **not yet confirmed live in a browser against the actual
  drone.**
- **Reconnect robustness**: fixed a port-rebind bug (Stop→Start Video
  failing) and a use-after-close socket race (`WinError 10055` after
  several connect/disconnect cycles). Both verified via simulated
  reconnect cycles **offline** — **not yet confirmed live.**
- **Logging**: `applog.py` added, writes `app_logs/app.log` (rotating,
  persists across runs). Verified working.

**What still needs a live drone to confirm**: the video-display rewrite and
the reconnect-robustness fixes, ideally all exercised together in one
session (Connect → Start Video → confirm smooth-ish display → Stop → Start
again → confirm it still works → a few Disconnect/Connect cycles). If
anything breaks, `app_logs/app.log` and `video_debug/sessions/*.jsonl` will
have it — no need to paste console output back manually anymore.

## Control channel — CONFIRMED WORKING (`drone.py`)

- UDP to `192.168.169.1:8800`.
- Handshake: send `ef 00 04 00` five times (50ms apart), then a handful of
  zero-control init packets, then start a 50Hz send loop of the control
  packet (short 88-byte + long 124-byte variant alternating).
- Inner 8-byte control block sits at offset 18 in the 88/124-byte outer
  packet: `66 [roll][pitch][throttle][yaw][cmd][chk] 99`, checksum = XOR of
  roll/pitch/throttle/yaw/cmd.
- Axis neutral = `0x80` (128), range 0-255.
- `cmd` byte: `0x40`=armed/normal flight, `0x01`=takeoff/land toggle,
  `0x02`=emergency stop, `0x03`=land, `0x80`=gyro calibrate.
- This is implemented and tested-by-the-user in `drone.py` — **do not
  "improve" the byte values without a real capture to back it up**, they were
  reverse-engineered from an actual working session.
- `drone.py`'s send loop (`Drone._loop`) now catches `OSError` from `sendto`
  (e.g. "network unreachable" when not on the drone's WiFi), records it in
  `self.last_error`, and flips `self._armed = False` instead of dying
  silently in a background thread — added 2026-07-05 after this exact
  failure mode showed up live (was just testing on a normal network, not the
  drone's AP). `gradio_app.py`'s status log surfaces `drone.last_error` when
  a guarded command is ignored, so a dropped connection is now visible in
  the UI instead of the app quietly no-op'ing forever.
- **`disconnect()` used to leave `self._armed = True`**, and `connect()`
  reused the same (now-closed) socket object rather than creating a new one
  — so reconnecting after a disconnect crashed the first thing that touched
  the socket (`OSError: not a socket`, hit inside `VideoReceiver.start()`
  live on 2026-07-05). Both fixed: `disconnect()` now sets `_armed = False`,
  and `connect()` always creates a fresh `self.sock` before doing anything
  else.
- `gradio_app.py` now shows a persistent, highly visible status badge (green/
  orange/grey) for both connection state and video-live state, updated every
  0.4s via the same `gr.Timer` that polls the video frame — added after the
  UI gave zero indication of connection/video state, which contributed to
  the confusion above (clicking Disconnect twice not knowing it had already
  disconnected).
- `gradio_app.py` is a UI on top of this class; it does not duplicate the
  protocol.

## Video channel — port 1234, implemented in `video_stream.py`, LIVE-TESTED 2026-07-05

The drone streams JPEG-over-UDP video on port **1234**, arriving back on
whatever local ephemeral port the client's control socket used (in the
capture, both control-out-to-8800 and video-in-from-1234 shared local port
61934 — i.e. it's the *same socket*, no separate bind needed).

### Wire format (bytes, little-endian fields; confirmed against our own capture AND a live session)

- Byte 0: magic `0x93`.
- Byte 1: message type — `0x01` = JPEG fragment, `0x04` = info/handshake
  reply (ASCII payload `(NOFLOW)FLOW_720P_V1.4.12_20240309_Beta(B),I=22000...`
  — firmware/stream version string).
- Bytes 2-3: total UDP payload length (verified exact match: `38 04` LE =
  0x0438 = 1080 for full fragments, `39 01` LE = 0x0139 = 313 for the
  trailing short fragment of a frame).
- Bytes 8-15: frame ID (only the low byte varies frame-to-frame in the
  captured sample; live testing confirmed the drone also accepts/echoes
  arbitrary frame IDs requested by the client — see flow control below).
- Bytes 32-35: fragment index, 0-based (observed wire order in the capture
  was 1,2,3,4,5,6,0 for a 7-fragment frame — **fragment 0 is not necessarily
  first on the wire**, reassembly must sort by this field, not arrival order).
- Bytes 36-39: fragment count for the frame (constant `7` in our sample).
- Bytes 40-43: total frame body length across all fragments (matches the sum
  of each fragment's post-header bytes exactly — verified).
- Byte 48: quality parameter (present, not otherwise used yet).
- Payload (JPEG scan data) begins at byte 56.

### This is a FREE-RUNNING stream that occasionally stalls, recovered by a "kick" — CONFIRMED with real timestamps, twice over

**History of getting this wrong, so the next session doesn't repeat it:**
An early version of this code assumed (by porting turbodrone's *sibling*
drone implementation) that this is a request/ACK flow-controlled protocol —
send a per-frame ACK, drone sends the next frame, repeat. Live testing
seemed to half-confirm this (rapid bursts of 6-7 frames after connecting),
but every live test stalled after that initial burst regardless of how many
acks were sent afterward, and clicking Stop/Start again didn't reproduce
even the initial burst reliably. That ack mechanism was likely wrong for
this specific firmware and may have actively been interfering.

**What the timestamps in `wireshark_1.pcapng` actually show** (reanalyzed
with real EPB timestamps, not just capture order — see
`analyze_capture.py`-style parsing with proper `if_tsresol` handling):
- 80 frames complete in the first ~10.2 seconds of the capture (t=5.98s to
  t=16.13s), arriving continuously with small, irregular gaps (0.05-0.55s) —
  a real, sustained, multi-frame-per-second stream, achieved with **zero**
  request/ack packets of any kind sent during that window. This is
  free-running, not request-gated.
- Then a stall: no frame from t=16.13s to t=19.81s (3.68s gap). Right before
  the next frame arrives, at t=18.795-18.969s, the client sent the exact
  3-packet `ef 20 ...` triplet (below). New frame arrives ~0.84s after the
  triplet finishes.
- Same pattern again: stall from t=16.13s (frame 77) is followed later by
  another gap, another `ef 20` triplet at t=22.701-22.747s, and frame 79
  arriving at t=23.553s — again right after the triplet.
- **Every stall recovery in the reference capture is immediately preceded by
  this exact triplet, and never by anything resembling a per-frame ack.**
  There is no evidence in the capture of any per-frame ack mechanism at all.
- Separately, the capture also shows the client re-sending bursts of the
  plain `ef 00 04 00` handshake (6-9 repeats) later in the session (around
  t=28s and t=36s) — a second, heavier-handed recovery tier, though the
  capture ends shortly after without conclusively showing it reviving a
  stalled stream (may just be the original app being closed down instead).

**Current implementation** (`video_stream.py`, rewritten same session after
this analysis): no per-frame request/ack at all. `VideoReceiver` just
receives and reassembles fragments as they arrive. A watchdog thread detects
stalls (`_watchdog_loop`) and escalates:
1. Idle > 2.0s: resend the `ef 20 ...` triplet (`_send_wake`, matches every
   observed recovery in the reference capture).
2. Idle > 8.0s: also resend a 5x `ef 00 04 00` burst (`_send_handshake_resend`,
   matches the heavier recovery tier seen later in the capture).

**This has not yet been tested live** — the earlier ack-based version was
live-tested and found broken (that's what led to this rewrite); this
corrected free-running + kick-on-stall version needs a fresh live test.

The `ef 20 ...` triplet bytes (`VIDEO_WAKE_PACKETS`): `ef 20 06 00 01 65` and
two `ef 20 19 00 01 67 3c 69 3d 32 5e 62 66 5f 73 73 69 64 3d 63 6d 64 3d
3{2,3} 3e` packets (ASCII tail decodes to `...bf_ssid=cmd=2`/`cmd=3`) — exact
purpose still unconfirmed (looks like an SSID/config query, repurposed here
as a stall-kick because that's what the timing evidence shows it correlates
with), but replayed verbatim since they're known-real bytes from a working
session, not reconstructed/guessed.

### JPEG header reconstruction — the key finding from the offline-replay pass

**The drone does not send SOI/DQT/SOF/SOS/DHT JPEG header segments, only raw
entropy-coded scan bytes from byte 56 onward.** A decoder needs a
synthesized header prepended and an EOI (`ff d9`) appended before the bytes
mean anything. `video_stream.py` builds:
`SOI, DQT(luma id0), DQT(chroma id1), SOF0 (baseline, 3 components 4:4:4),
DHT(DC luma), DHT(AC luma), DHT(DC chroma), DHT(AC chroma), SOS` using the
**standard ITU T.81 Annex K quantization AND Huffman tables** — cheap
hardware JPEG encoders (this drone included) almost always use these canned
tables rather than optimized custom ones, so they don't bother transmitting
them.

**The DHT (Huffman table) segments are required — this was a real,
confirmed bug, not theoretical.** An earlier header (ported from a reference
that only had DQT/SOF/SOS, no DHT) produced JPEGs that PIL opened *without
raising an exception* but decoded to a flat, near-uniform image every single
time regardless of input — a false-positive "it works" signal. Adding the
standard DHT segments fixed this.

### How this was verified offline (before any live drone access)

`wireshark_1.pcapng` already contains a full, real, working video session.
The fragments for that session were fed directly through
`VideoReceiver._handle_packet` offline: 127 consecutive frames reassembled
and decoded with **zero PIL errors**, and critically, the decoded pixel
statistics (mean/stddev) **varied across the session** — most frames
near-black/flat (std ~1, consistent with a night-time capture), but one
stretch around frame ~101 spiked to std ~14 and decoded into a real,
recognizable image (looked like the FPV camera pointed at the drone's own
frame/props, blue+red colored parts visible). A structurally wrong header
produces the same degenerate output on every frame regardless of input;
getting *different, content-correlated* output across frames is strong
evidence the byte-offset model and header reconstruction are correct.

To reproduce: parse `wireshark_1.pcapng` with `analyze_capture.py`'s
`read_pcapng`/`decode` helpers, filter for `src=="192.168.169.1" and
sp==1234`, and feed each payload through
`VideoReceiver._handle_packet(payload, addr, header)` — no drone or network
required.

### Confirmed live 2026-07-05 (two rounds)

**Round 1** (ack-based version): decoded real frames from the actual drone
(13 `frame_decoded_ok` events across a 2-minute session), confirming the
byte-offset model and JPEG header reconstruction work against live
hardware, not just the old capture. But it stalled after an initial burst
every time, in a way re-requesting frames never fixed.

**Round 2** (still ack-based, after adding pid tagging to rule out a
multi-process mixup): confirmed single-process, and the exact same pattern
reproduced with real numbers — `pid=15676`, sub-session 0: 7 frames decoded
in 0.02s, then 35 re-requests over 25.7s with zero new fragments; clicking
Stop/Start again (sub-session 1) got zero frames at all over 20.7s despite
104 more requests. This ruled out "the ack request is basically working,
just needs a retry tweak" and prompted the full pcap timestamp reanalysis
above, which found the real mechanism (free-running + `ef 20` kick on
stall) and led to the rewrite in this section.

**The free-running + kick-on-stall version above has not been live-tested
yet.** If you're the one testing this next: check the current session's
`video_debug/sessions/packets_<timestamp>_pid<pid>.jsonl` (path also printed
in the Gradio status log and written to `video_debug/latest_session.txt`)
for a healthy, continuous `jpeg_fragment` → `frame_decoded_ok` cadence with
occasional `wake_sent`/`handshake_resent` events during gaps, rather than
one burst followed by silence.

### What's still unverified

- **Resolution.** 640x360 is a guess (turbodrone's default for a sibling
  drone in the same OEM app family); our own capture never states
  width/height explicitly. Both the offline replay and the live tests
  decoded successfully at this resolution, which is decent evidence it's
  right or close. If a live frame looks skewed/torn, try adjusting
  Width/Height/Color in the Gradio video panel first.
- **Whether the `ef 20` kick and/or handshake-resend actually clear a stall
  when triggered by our code**, as opposed to just correlating with recovery
  in the original app's traffic — this is inference from timing correlation
  in one capture, not a controlled experiment. If stalls still don't
  recover, check whether `wake_sent`/`handshake_resent` log events actually
  precede a resumed `jpeg_fragment`, or whether the drone seems to recover
  on its own regardless of what we send (in which case the real trigger is
  something else entirely, e.g. our own `Drone._loop` control packets doing
  something, or simply time-based).
- **Why the drone stalls at all** — root cause on the drone/firmware side is
  unknown (encoder buffer limit? thermal? WiFi congestion?). We can only
  react to it, not prevent it.

### Provenance / cross-reference

The wire format (fragment header byte offsets) matches an independent
reverse-engineering of the same OEM "WiFi UAV" app family (package
`com.lcfld.fldpublic`) by
[marshallrichards/turbodrone](https://github.com/marshallrichards/turbodrone):
`backend/protocols/wifi_uav_video_protocol.py` (fragment reassembly — same
offsets) and `backend/utils/wifi_uav_jpeg.py` (JPEG header synthesis —
their version omits DHT segments, which is the bug this session found and
fixed when porting the approach).

**Important caveat discovered this session: turbodrone's flow-control model
(per-frame request/ACK via `build_native_ack_packet`) does NOT apply to
this drone** — it was tried, live-tested, and found to not sustain a stream
past the first several frames; real capture timestamp analysis (above)
shows this drone's firmware is free-running with a stall/kick recovery
pattern instead. Their control-protocol command-byte assignment also
differs from ours (bit-flags vs. our single-byte enum). Treat turbodrone as
a useful reference for a **cousin** protocol/drone in the same app family,
not a byte-exact match — verify every borrowed piece against our own
capture or live behavior before trusting it, the way the DHT-tables and
flow-control assumptions both turned out to need correction.

Other reference material collected while researching this family of budget
WiFi FPV drones (not all directly used, but useful if this needs deeper
reverse-engineering later — e.g. porting turbodrone's fragment-retry path,
or if a future drone/firmware variant needs fresh RE from scratch):
- [guillesanbri/e58-drone-reversing](https://github.com/guillesanbri/e58-drone-reversing)
  and the accompanying writeup,
  ["Hacking an E58 Drone's Video Feed"](https://guillesanbri.com/drone-video/)
  — another cheap-drone JPEG-over-UDP reverse-engineering, useful for
  comparing header-reconstruction approaches.
- [FahrulRPutra/reversing-wifi-uav](https://github.com/FahrulRPutra/reversing-wifi-uav)
  — RE notes specifically for the WiFi UAV app family.
- ["Reverse engineering a drone's IP cam. stream"](https://forum.hackthebox.com/t/reverse-engineering-a-drones-ip-cam-stream/3800)
  (Hack The Box forum) and
  ["Reverse-engineering a Drone Camera Module"](https://medium.com/@meekworth/reverse-engineering-a-drone-camera-module-a17f7b6cdc04)
  (Medium) — general methodology for this class of problem.
- ["Wireshark to the rescue / figuring out a UDP data stream"](https://hackaday.io/project/19356-reverse-engineering-a-promark-vr-toy-drone/log/51984-wireshark-to-the-rescue-figuring-out-a-udp-data-stream)
  (Hackaday.io, Promark VR toy drone) — similar UDP-stream RE approach.
- ["Low-Level Wifi Protocol Reverse Engineering"](https://mavicpilots.com/threads/low-level-wifi-protocol-reverse-engineering.111099/page-2)
  (DJI Mavic/Air/Mini community forum) — background on WiFi FPV protocol RE
  in general, different (proprietary/higher-end) protocol family.
- The Play Store "WiFi UAV" app (package `com.lcfld.fldpublic`) is the
  likely origin of this exact protocol family — if this drone ever needs
  fresh RE (new firmware, different behavior), decompiling that app's APK is
  the fastest path back to ground truth, same as how turbodrone's authors
  derived their implementation.

### Logging — every run gets its own permanent file, nothing is ever deleted by this code

`PacketLogger` (in `video_stream.py`) writes each `VideoReceiver` session to
its own uniquely-named file: `video_debug/sessions/packets_<YYYYmmdd_HHMMSS>_pid<pid>.jsonl`,
plus a matching `video_debug/sessions/frames_<same-id>/` directory with the
first ~60 successful/failed frame reconstructions as `.jpg` files.
`video_debug/latest_session.txt` always points at the most recent one, and
the Gradio status log prints the exact path when you click Start Video.
**Nothing in `video_debug/` is ever deleted automatically** — old sessions
accumulate so every past run stays available for offline analysis. (An
earlier version of this code shared one flat `packets.jsonl` across all
runs and I manually `rm -rf`'d it between tests while iterating — don't do
that; if `video_debug/` ever needs real cleanup, that's a human decision,
not something to automate.)

Every logged record includes a `pid` field — if you ever see two different
`pid` values mixed together in what should be one session, that means two
processes were running against the drone at once (confirmed to happen once
this session, from an earlier `Ctrl+C` not fully killing the process) and
whatever you're seeing may be two sessions fighting each other, not a
protocol bug. The Gradio page header also shows the current process's PID.

### If video doesn't work live

1. Check the current session's JSONL (path in `video_debug/latest_session.txt`
   or the Gradio status log after clicking Start Video) and its matching
   `frames_*/` directory. These persist on disk and don't need live drone
   access or internet to inspect afterward.
2. Look for the cadence: continuous `jpeg_fragment`/`frame_decoded_ok`
   events is healthy. A burst followed by a long gap with only
   `wake_sent`/`handshake_resent` events and no new `jpeg_fragment` after
   them means the kick isn't actually working for this session — see "What's
   still unverified" above.
3. If `unknown_packets` is high or `frame_decode_failed` shows up a lot, the
   wire format may differ slightly from the captured session (different
   firmware version?) — compare the logged `hex_head` against the offsets
   table above.
4. If frames decode without error but look wrong, it's almost certainly the
   width/height guess — try the Gradio panel's Width/Height fields before
   touching `video_stream.py`.
5. Check `tasklist`/`netstat` (Windows) for more than one process bound to
   this app before assuming it's a protocol bug — see the `pid` note above.

## Getting a fresh packet capture (correct method)

The phone (running the real WiFi UAV app) and this laptop are different
devices — standard Wireshark-on-Windows can't see the phone's traffic to the
drone unless routed through something the laptop can see. The method that
actually works, confirmed 2026-07-05 (produced `newtest.pcapng`):

```
Phone ──WiFi──> Laptop (Mobile Hotspot)
                     |
                Windows routes/NATs
                     |
                Laptop (WiFi) ──WiFi──> Drone
```

The **drone stays exactly as-is** (its own AP, laptop connects to it
normally as a client — same as every other capture in this file). The
**laptop** does double duty: connected to the drone on its WiFi radio, and
broadcasting a *separate* hotspot for the phone to join, with Windows
routing between the two.

Steps:
1. Laptop WiFi connects to the drone's own hotspot (as always — verify with
   `ping 192.168.169.1`).
2. **Do NOT use the modern Settings → Mobile Hotspot toggle for this** — it
   silently refuses to share a connection Windows doesn't think has "real"
   internet (which the drone's network doesn't), even though the option may
   appear selectable. Use the classic `netsh` / Network Connections method
   instead:
   - Confirm hosted-network support: `netsh wlan show drivers` (look for
     "Hosted network supported: Yes").
   - `netsh wlan set hostednetwork mode=allow ssid=<name> key=<password>`
   - `netsh wlan start hostednetwork`
   - Open Network Connections (`ncpa.cpl`), right-click the WiFi adapter
     connected to the drone → Properties → Sharing tab → check "Allow other
     network users to connect..." → select the new hosted-network virtual
     adapter as the connection to share to.
3. Connect the phone to the new hosted-network SSID (not the drone's own).
4. Wireshark on the laptop, capture on the **WiFi** adapter (the one
   connected to the drone), filter `udp and ip.addr == 192.168.169.1`.
5. Use the real app on the phone normally — traffic is transparently NAT'd
   through, so captured packets show `192.168.169.1 <-> 192.168.169.2`(the
   laptop's own address on the drone's network, since NAT rewrites the
   phone's traffic to appear as coming from the laptop).

### `newtest.pcapng` findings (2026-07-05) — proof continuous streaming works

This capture (produced with the method above) shows **two genuinely
continuous, healthy video streams**, no protocol trick needed on the app's
part beyond normal operation:
- 38 frames over ~4.9s at the very start of the capture (already in
  progress when the capture began — frame_ids 154-191).
- After a ~5.3s gap (the app was reloaded — "loaded video twice"), a *new*
  session starts fresh at frame_id 1 and streams **95 frames continuously
  over 13.2 seconds** (~7.2 fps, no gaps over 0.3s).

This is dramatically better than anything achieved by our own
reimplementation so far (which has never exceeded ~12 frames before
stalling permanently, across every live test). Both the per-frame-ack
hypothesis and the `ef 20`-kick hypothesis were tested directly against the
live drone and disproven (see below).

### CONTROL-CHANNEL FINDING (2026-07-05) — the real app is DISARMED during sustained video, ours never is

Ran `pcap_analyzer.py control newtest.pcapng` (after fixing a stale-offset
bug the tool inherited from old `dump_drone_packets.py` — inner control
block is at offset 18, not 20; verified against `drone.py`'s own
`_build()` output before trusting the capture numbers). Result, decoding
roll/pitch/throttle/yaw/cmd/checksum from every client→8800 packet:

- **430 of 453** control packets in this capture — essentially the *entire*
  13-second healthy streaming window, including the very first packets of
  the file (the already-in-progress tail-end session) — have
  **roll=pitch=throttle=yaw=0, cmd=0, checksum=0**, with the packet
  **counter frozen at a constant value (11)**, never incrementing.
- Only **14 packets**, right after the `ef 00 04 00` reconnect handshake,
  show the "armed/neutral" values we assumed were the steady state:
  `roll=pitch=throttle=yaw=128, cmd=0x40, chk=0x40` (exactly what
  `drone.py` sends) — and the counter *does* increment normally (1,2,3...)
  during this brief window, for about 10 packets / ~0.5 seconds, before
  switching to the frozen all-zero state for the rest of the session.

**`drone.py`'s control loop does the opposite of what the healthy capture
shows**: it sends `cmd=CMD_ARMED` (0x40) with neutral (128) axes
*continuously, forever*, incrementing the counter every packet, 50 times a
second — i.e. it stays in the brief "armed" state the real app only uses
for half a second, and never reaches the frozen all-zero "idle/disarmed"
state the real app spends 95% of its time in during healthy streaming.

**This is the most concrete, testable lead so far** — the pattern is 100%
consistent within this capture (not just correlated with one event), and
gives a specific, mechanical hypothesis: sending `cmd=0` (disarmed) with
frozen/zero axes after the initial connect sequence, instead of
continuously re-sending `cmd=CMD_ARMED`, might be what actually lets the
drone's firmware sustain video encoding — perhaps continuous "armed" state
competes with the video pipeline for some shared resource, or the firmware
treats sustained-armed-with-no-throttle-change as some kind of degraded/
watchdog condition that (among other things) throttles the encoder.

**Implemented as `Drone.set_idle_mode(bool)` in `drone.py`** — additive, opt-in
(default off, existing flight behavior unchanged unless explicitly enabled).
When on, `_loop()` sends `cmd=0`/all-zero axes and freezes the packet
counter, matching the healthy-capture pattern exactly.

**First attempt (`gradio_test.pcapng`, 2026-07-05) — engaged too late,
inconclusive.** `gradio_app.py` originally only called `set_idle_mode(True)`
from the "Start Video" button. A live test (captured in `gradio_test.pcapng`)
showed: idle mode *did* engage correctly (1583 of 1820 control packets were
the idle pattern, vs 198 armed — confirmed via `pcap_analyzer.py control`),
but video had already stalled (last frame at t=4.0s) **over 2 seconds
before** idle mode was even applied (t=6.06s, when Start Video was actually
clicked). So this test doesn't confirm or deny the hypothesis — idle mode
was never active during the window that mattered.

**Root cause of the gap**: in the real app, connecting and going idle
happen together, within ~0.5s. In the Gradio UI, "Connect" and "Start Video"
are two separate manual clicks with an arbitrary delay between them — by
the time idle mode engaged, the stall may have already locked in.

**Fixed 2026-07-05**: `do_connect()` in `gradio_app.py` now calls
`drone.set_idle_mode(True)` immediately upon connecting, not waiting for
Start Video — closing the timing gap to match the real app. Since idle mode
overrides axes to zero unconditionally, every flight command (`_guarded`
wrapper, plus the manual axis panel's `do_set_axes`) now calls
`set_idle_mode(False)` first, so flying still works normally — idle mode
only stays engaged if you never touch the controls, exactly like the real
app only being "armed" while a joystick is actively touched.

This timing fix was tested live (`gradio_test2.pcapng`) — still stalled at 7
frames. But that same test surfaced the strongest lead of the whole session:

### `ctr2`/`ctr3` decoupling — likely the actual missing keepalive (2026-07-05)

Byte-position variance analysis (`bytelevel_diff.py` — compare every byte
position across all same-length control packets, not just the
roll/pitch/throttle/yaw/cmd columns already tracked) found something
completely missed until now: in the **124-byte "long" packet**, bytes
88-89 and 108-109 (`ctr2`/`ctr3`, part of the packet's tail suffix blocks)
**keep incrementing throughout the entire healthy 13-second stream in
`newtest.pcapng`, even while the main counter (`ctr1`, bytes 12-13) is
completely frozen at 11.** Precisely measured: 17 increments over 2.461s =
**~6.9 Hz** — suspiciously close to the ~7.2fps video frame rate in that
same window.

In our own captures (`gradio_test.pcapng`, `gradio_test2.pcapng`), `ctr2`/
`ctr3` barely move (1-6 total) because `drone.py`'s `_build()` always
derived them as `counter+1`/`counter+2` from the SAME counter used for
`ctr1` — so freezing `ctr1` for idle mode also froze `ctr2`/`ctr3`, which
the real app evidently does NOT do. This looks exactly like a per-frame
(or near-per-frame) client-side "still alive, still consuming" counter,
separate from the main control-state counter — plausibly what the drone's
firmware uses as proof the client hasn't gone away, independent of whether
the client is actively flying.

**Implemented**: `_build()` now takes an optional `long_counter` parameter
(defaults to old behavior — `counter+1`/`counter+2` — for backwards
compatibility). `Drone._loop()` maintains a new `self._long_counter` that
advances continuously at the measured 6.9 Hz using wall-clock elapsed time,
**regardless of idle_mode** (unlike `self._counter`/`ctr1`, which only
freezes during idle) — and passes it into the "long" packet build. Verified
the byte-level output is correct (`ctr1` frozen, `ctr2`/`ctr3` independently
advancing) and that the rate tracks real elapsed time correctly, both
without a live drone.

### CONFIRMED WORKING LIVE, then a state-management bug found (2026-07-05, `gradio_test3.pcapng`)

The idle-mode + decoupled-ctr2/ctr3 fix **worked** — a 168-second capture
of a real Gradio session shows one continuous run of **258 distinct video
frames over 37.7 seconds** (port 50768), and the app's own
`video_debug/sessions/` logs show several more successful runs within the
same process (524, 273, 124, 118 `frame_decoded_ok` events across separate
Start-Video clicks) — a huge, unambiguous improvement over the ~7-frame
ceiling every previous attempt hit.

But: reconnecting within the same running process (Disconnect → Connect →
Start Video again, without relaunching `python gradio_app.py`) intermittently
failed again, only reliably fixed by a full relaunch. Root cause, found by
comparing `ctr1`/`ctr2`/`ctr3` at the start of all 7 reconnect attempts in
the capture: **`Drone.connect()` never reset `self._counter`,
`self._long_counter`, or `self._last_long_tick`** — only `__init__` did. So
every reconnect within the same process carried over stale state from the
previous session. The symptom was stark: every *failed* reconnect showed
`ctr2` jump to a huge, discontinuous value (561, 752, 838, 977, 1050, 268)
right at session start, while the one session that happened to get a clean
reset (`ctr2=1`, matching what the real app always does on every connect)
was the 258-frame success. **Fixed**: `connect()` now explicitly resets all
three at the top, so every connect — not just process startup — starts
fresh exactly like the real app does.

**Not yet re-tested live since this reset fix landed.** Next test: several
Disconnect → Connect → Start Video cycles *within the same running process*
(no relaunch), checking that every cycle now streams well, not just the
first.

**Also worth adding later** (separate, lower-priority issue): fragment-level
retry for lossy WiFi — currently a single dropped fragment discards the
whole frame with no re-request, which would lower effective frame rate on a
noisy link. Doesn't explain the reconnect-state bug above, but would likely
improve overall frame rate/consistency once the state bug is confirmed
fixed.

### Live-tested and disproven hypotheses (2026-07-05)

Both were implemented in `video_stream.py`, tested directly against the
real drone (not just inferred from capture timing), and removed/reverted
after failing:
1. **Per-frame ACK/request** (ported from turbodrone): sent a
   `build_native_ack_packet`-style ACK after every completed frame,
   requesting the next one. Result: still stalled at ~6-8 frames every
   time, identical to no-ack behavior.
2. **`ef 20` kick on stall**: resent the exact captured wake triplet after
   2s of silence, escalating to a full handshake resend after 8s. Result:
   drone dutifully echoes an `info_packet` in response every time (proving
   it's received) but **never** resumes sending video fragments, even after
   25+ seconds and multiple escalating attempts.
3. **Missing 108-byte control packet type**: found via `pcap_analyzer.py
   control` that the real app also sends a third control-packet size (108
   bytes, header byte 8 = `0x01`) that `drone.py` never generates. Added it
   live (best-effort field reconstruction) alongside the normal 88/124-byte
   packets. Result: no change — still stalled at frame 7.

Also ruled out: WiFi signal quality (checked via `netsh wlan show
interfaces` during a live test — 100% signal, RSSI -35dBm, i.e. excellent;
not a link-quality issue).

### FLIGHT-WITH-VIDEO FINDING (2026-07-08, `flight_with_video.pcapng`) — the real root cause: PACKET RATE, not arm state

This is the most important video finding to date and it **corrects the
emphasis of the "CONTROL-CHANNEL FINDING" above.** Every prior "healthy
video" capture (`newtest.pcapng`, `gradio_test3.pcapng`) was shot with the
drone **grounded / sticks untouched**, so the app was idle/disarmed — which
made "disarmed" *look* like a video prerequisite. It isn't.

`flight_with_video.pcapng` is the first capture of the **real app streaming
video while actually flying** (laptop-relay method, phone running the real
WiFi UAV app: takeoff → move commands → land, video on throughout). It
contains **two clean, continuous flight-with-video windows** — t≈8–77s and
t≈270–330s, ~60–70s each, hundreds of complete JPEG frames per window with
no gap over ~0.7s. Decoded via `pcap_analyzer.py`:

- **The app is ARMED the entire flight and video is perfectly healthy.**
  During the healthy window `cmd=0x40` (armed) dominates (idle `cmd=0` only
  appears in the first few seconds pre-takeoff, exactly like our
  `set_idle_mode` grounded state). **Being continuously armed does NOT stall
  video.** The whole "must send `cmd=0`/disarmed to keep video alive" theory
  was an artifact of only ever having grounded captures.
- **The app sends 124-byte "long" packets EXCLUSIVELY during flight+video**
  — zero 88-byte short packets in either healthy window (across the whole
  capture 88/108/152-byte variants appear only during connect/reconnect
  storms, never in the streaming steady state).
- **Rate: ctr1 ≈ 17.6 Hz, ctr2/ctr3 ≈ 6.2 Hz (decoupled)** — i.e. ~18–20
  long packets/sec total.
- **Our `drone.py` 124-byte `_build(long=True)` output is byte-for-byte
  IDENTICAL to the real app's flight packet** (verified against a real armed
  mid-flight packet: zero differing offsets). The packet *content* was never
  wrong.

**So the actual cause of our "stalls after ~7 frames while armed" was packet
RATE/COUNT, not arm state and not packet content:** `drone.py`'s `_loop()`
sent an 88-byte + 124-byte pair *every* iteration at 50 Hz = ~100 packets/s,
~5.7× the real app's ~18 packets/s, flooding the shared control/video socket
and starving the video RX. (This also explains why `set_idle_mode` "helped"
earlier — cmd=0/frozen-ctr1 was incidental; what mattered was that the idle
path still only mattered while grounded, and the real differentiator we were
missing is here.)

**Fix (2026-07-08, `drone.py._loop`)**: send **only** the 124-byte long
packet, at **~18 Hz** (`time.sleep(0.055)`), matching the real app's flight
cadence exactly (verified offline: emits 124-byte-only at 18.7 Hz, ctr1 18.0
Hz, ctr2 6.7 Hz). No change to packet content, arm/cmd handling, or the
`set_idle_mode` machinery (idle-on-connect still harmlessly mirrors the real
app's grounded pre-takeoff state; it is simply **no longer required** for
video, and must never be force-restored mid-flight — that forces `cmd=0` =
disarm, the dangerous bug reverted in `circe_v1`). **Live-confirmed 2026-07-08**
(connect → takeoff → fly with video: no stall after commands, "works good
enough" per operator), and the same 124-byte-only/~18 Hz `_loop` change has now
been **ported to `circe_v1/optical_flow_control/drone.py`** (both `drone.py`
copies are byte-identical again).

### GYRO/TILT-MODE FINDING (2026-07-08, `flight_with_video.pcapng`) — the phone's smooth "gyro" control is CLIENT-SIDE ONLY

The DM002HW phone app has a gyro button: press it (does a gyro calibration),
then **tilt the phone to fly** — noticeably smoother, finer movements and much
better hover than the on-screen buttons. Question was whether that mode has a
distinct packet signature we were missing. Analysed the control channel of
`flight_with_video.pcapng` (the gyro-enabled capture) vs. our button-based
`gradio_test*.pcapng` with `pcap_analyzer.py`:

- **Continuous vs discrete axis values IS the whole story.** The gyro capture
  streams a near-continuous fine spread per axis — **roll 55, pitch 61,
  throttle 48, yaw 33 distinct values**, drifting **±1–3 every ~50 ms** around
  neutral 0x80 (e.g. pitch 109→110→111→106→114→106→104→103… over consecutive
  packets). Our app only ever sends the coarse discrete steps (96 / 128 / 160)
  from 1-second button pulses. That fine analog spread is the accelerometer-tilt
  fingerprint, and it is the *only* difference.
- **No drone-side mode byte.** The tilt packets use plain `cmd=0x40` (armed) —
  the identical byte we already send. No cmd value, outer-header byte, or packet
  subtype flips between "gyro on" and button mode. The header length byte just
  co-varies with packet size. There is no separate calibrate/mode packet type in
  the flight capture (no `cmd=0x80` calibrate event is even present in it).
- **Same rate/cadence** as any armed flight (~18–20 Hz, 124-byte long packets);
  the fine motion comes from the axis *values* changing packet-to-packet, not
  from any rate or counter change.

**Conclusion:** the phone maps its accelerometer tilt to fine analog axis bytes
and streams the **same protocol we already use** — there is nothing drone-side to
enable. **Replicated** in both apps by `analog_control.py`: a browser
virtual-joystick posts held stick positions to a local side-channel HTTP server
(`AnalogInputServer`, same idiom as `video_stream.MjpegServer`) at ~20 Hz, which
forwards them straight into `Drone.set_controls()`. A watchdog forces neutral if
the joystick stops posting. Composes with the generation-counter arbitration in
`drone.py` (see below) for free. Whether flight actually *feels* smoother, and
the axis sign convention (there are per-axis invert checkboxes for this), are the
only live-drone unknowns.

### STATE-ARBITRATION FIX (2026-07-08, `drone.py` generation counter)

A code review of the concurrent Gradio handlers found two safety bugs from
flight commands doing `set_controls(...); time.sleep(...); hover()` in the
handler thread: a move's terminal `hover()` (which sets `_cmd=CMD_ARMED`) could
fire *after* an E-STOP and re-arm the drone, and a concurrent move's `hover()`
could truncate a Land/Takeoff/Calibrate. Fixed with a **generation counter** in
`Drone`: every authoritative write (`set_controls`/`hover`/`takeoff`/`land`/
`calibrate`/`stop`) bumps `_gen` under `_state_lock` and returns a token; a timed
command captures its token before sleeping and only applies its terminal revert
if `_gen` is unchanged and still armed — so E-stop, a newer command, or a
disconnect cancels a stale revert. E-stop is an ungated bump (never blockable).
Verified offline (e-stop-vs-move, land-vs-move, disconnect-mid-move all pass).

## Analysis tools

All scripts below are tracked in git (not gitignored) specifically so a
future session can reproduce this analysis without redoing the RE work.
Raw capture files (`*.pcap`, `*.pcapng`) stay gitignored (potentially
sensitive traffic) — only the *tools* are tracked.

- **`pcap_analyzer.py`** — the main, consolidated, actively-maintained
  analyzer built during the 2026-07-05 session. Run
  `python pcap_analyzer.py <mode> [path.pcapng]`:
  - `summary` — flow overview (every UDP/TCP src/dst/port pair + packet
    counts). Good first look at an unfamiliar capture.
  - `control` — dumps every client→8800 control packet with decoded
    roll/pitch/throttle/yaw/cmd columns, flagging non-neutral rows (real
    flight commands) and any non-`ef 02` packets (handshake/wake/etc).
  - `video` — **the primary tool for video debugging.** Merges frame
    completions (with gap detection >0.3s) and every info/handshake/wake
    packet into one real-timestamp-ordered timeline, so you can see exactly
    what the client was sending around any stall or resume. This replaces
    all the ad-hoc one-off scripts written earlier in the session (properly
    handles multi-section pcapng files — see its docstring for a real
    timestamp bug this fixed).
  - `decode-frame N` — reassembles and decodes frame N via
    `video_stream.py`'s real pipeline, saves it as a PNG. Use to visually
    confirm a capture has real (not corrupted) video, or inspect a specific
    frame.
- **`analyze_capture.py`** — original first-pass flow dumper (groups all
  UDP/TCP flows, shows first 15 unique payloads per flow as hex). Superseded
  by `pcap_analyzer.py summary` for most uses but kept since
  `capture_analysis.txt` in this repo was generated by it and other notes
  reference its exact output format.
- **`dump_drone_packets.py`** — earlier, narrower version of
  `pcap_analyzer.py control` (control-channel-only, writes to
  `drone_packets.txt`). Kept for the same reason as above.
- **`discover.py`** — sends the known handshake and listens on a list of
  candidate UDP ports for any response/broadcast. Useful if reverse-engineering
  a *different* drone/firmware from scratch and you don't yet know which
  port(s) it uses.
- **`port_probe.py`** — sends a neutral control packet to a wider list of
  candidate ports and checks for replies. Same "starting from zero" use case
  as `discover.py`.
- **`tcp_scan.py`** — plain TCP port scan (1-10000) of the drone's IP.
  Useful for checking whether a drone/firmware variant exposes any TCP
  service (e.g. a web config UI, RTSP) in addition to the UDP control/video
  ports this one uses.

## `gradio_app.py`

Gradio 6 UI on top of `drone.py` + `video_stream.py`. Run with
`python gradio_app.py` from this directory (needs `pip install gradio`).
Two D-pad "gimbals" (left = throttle/yaw, right = pitch/roll) drive the
existing timed movement methods; an "Advanced" raw-axis panel exposes
roll/pitch/throttle/yaw sliders directly against `set_controls()` — since
the background send-loop in `Drone._loop` just keeps re-transmitting
whatever the current axis values are, leaving a slider at a non-neutral
position is equivalent to holding a real stick there (continuous
hover/movement, not just a timed tap). The video panel has Start/Stop
buttons and Width/Height/Color fields; the stats line (packet/frame/kick/
handshake-resend counters + PID) is still polled every 0.4s via `gr.Timer`,
but the video image itself is NOT polled — see below.

### Video display: MJPEG-over-HTTP push, not polling (2026-07-05)

Originally the video panel was a `gr.Image` updated by the same 0.4s
`gr.Timer` used for stats — this capped the displayed rate at 2.5fps and
silently dropped most successfully-decoded frames even when the underlying
stream was healthy (confirmed: a 6.8-7fps stream means most frames arrive
and get overwritten between one 0.4s poll and the next). Replaced with:

- `video_stream.py`'s `VideoReceiver` now has a `threading.Condition`
  (`_frame_ready`) + `wait_for_next_frame(after_version, timeout)` that
  blocks until a genuinely new frame exists — no fixed-interval loop.
- `video_stream.MjpegServer` — a small threaded HTTP server (stdlib
  `http.server`/`socketserver` only) exposing `/stream` as
  `multipart/x-mixed-replace` (the decades-old "IP camera" MJPEG-over-HTTP
  trick). Each connected viewer gets its own thread that blocks on
  `wait_for_next_frame` and pushes each new JPEG down the open connection
  the instant it's ready.
- The Gradio video panel is now a `gr.HTML` `<img src="http://127.0.0.1:8090/stream">`
  tag — the browser repaints on its own via the open HTTP connection, no
  Gradio-side polling/serialization/websocket round-trip per frame at all.

Verified end-to-end offline: fed real captured fragments from
`wireshark_1.pcapng` into a `VideoReceiver` on a background thread, started
an `MjpegServer`, and had a real `urllib` HTTP client read the multipart
stream — received valid JPEGs (correct SOI/EOI markers) matching the
source data.

**Why not WebRTC or `gr.Video(streaming=True)`**: both exist in Gradio 6
(the latter added in 5.0, chunk-based h.264/mp4; WebRTC via the separate
`fastrtc`/`gradio-webrtc` packages) and would work, but MJPEG-over-HTTP
needed zero new dependencies, no video-chunk encoding step, and is a much
smaller change for what's fundamentally just "push each already-decoded
JPEG out immediately" — the actual protocol from the drone was never
video in the RTSP/SRT/RTP sense to begin with, just a raw UDP JPEG-fragment
scheme (see the video channel section above), so treating the final output
as MJPEG matches what it actually is.

### Two real bugs found right after shipping the above (2026-07-05)

1. **`MjpegServer` couldn't restart on the same port.** `Stop Video` then
   `Start Video` again failed because `socketserver.ThreadingTCPServer`
   doesn't set `SO_REUSEADDR` by default — Windows briefly holds a just-closed
   port, so rebinding immediately after `stop()` raised `OSError`
   (uncaught, silent failure from the user's point of view). Fixed: a
   `_ReusableThreadingTCPServer` subclass with `allow_reuse_address = True`,
   plus `do_start_video` now catches bind failures and reports them in the
   status log instead of crashing silently. Also fixed: `daemon_threads=True`
   only reaps per-connection handler threads at process exit, not when
   `stop()` is called mid-run — added a `server.active` flag the handler's
   streaming loop checks every ~1s so old connections let go promptly
   instead of lingering.
2. **`WinError 10055` (socket buffer space exhausted) after several
   connect/disconnect cycles.** `Drone.disconnect()` closed the socket after
   waiting only up to 1s for the control thread to stop, regardless of
   whether it actually had — a real use-after-close race that could leave
   threads/sockets piling up across cycles. Hardened: now waits longer and
   verifies the thread actually stopped before closing the socket, logging
   clearly if it doesn't (rather than silently proceeding). Also fixed a gap
   where the thread's own `OSError` handler broke its loop but never cleared
   `self._running`, leaving `disconnect()`'s bookkeeping inconsistent.

### Proper logging added (2026-07-05)

Everything above was previously diagnosed purely from console `print()`
output the user had to copy-paste back — nothing was ever saved to disk for
non-video events (only `video_debug/sessions/*.jsonl` persisted). Added
`applog.py`: stdlib `logging` (no new dependency), one shared rotating file
`app_logs/app.log` (5MB x 5 backups, appends across runs — same "never
delete automatically" policy as `video_debug/`) plus console output, used by
`drone.py` (replacing its class-method `print()` calls — the
`if __name__ == "__main__":` demo block's prints were left alone, that's an
interactive CLI example, not part of the app) and `gradio_app.py` (`_note()`,
already called for every UI status message, now logs each one too — so
every connect/disconnect/video-start/error is captured with a real
timestamp automatically, with no per-call-site duplication needed).
