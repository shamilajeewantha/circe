# DM002HW protocol notes (for future agents/sessions)

Drone: DM002HW DIY WiFi+camera quad. AP at `192.168.169.1`, client gets
`192.168.169.3`. Reverse-engineered from `wireshark_1.pcapng` in this folder
(see `analyze_capture.py` / `capture_analysis.txt` / `dump_drone_packets.py`).
That capture appears to have been shot at night (most frames decode near-black).

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
buttons and Width/Height/Color fields, polled every 0.4s via `gr.Timer`, and
shows live packet/frame/kick/handshake-resend counters plus the process PID.
