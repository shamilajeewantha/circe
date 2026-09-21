# `src/` — CSI capture + stream

Two scripts, no dependencies beyond the Python standard library and `rpicam-vid` (already on the
Pi image — see [`../README.md`](../README.md)).

| Script | Runs on | Job |
|---|---|---|
| `stream_sender.py` | **the Pi Zero 2 W** | wraps `rpicam-vid`, pipes H.264 over TCP |
| `stream_receiver.py` | the RTX 4050 box (or the Uno Q) | writes the stream to a file and/or a player |

## Which way does the stream go?

Per [`circe_v1/docs/mothership-scout.md`](../../../circe_v1/docs/mothership-scout.md), the Pi's CSI
frames are bound for the **RTX 4050** — the Pi is "the sole path to the RTX 4050", and the Uno Q
relays drone frames *to* the Pi, not the other way. The **Uno Q ↔ RPi 2W physical link is still
undecided** (USB gadget-Ethernet vs UART).

So the destination is a CLI argument rather than a constant. Both candidate links are IP-based
(USB gadget mode gives the Pi `10.12.194.1/28`), so only the address changes:

```bash
python3 stream_sender.py connect 192.168.1.4  8888   # -> RTX 4050 over WiFi/LAN
python3 stream_sender.py connect 10.12.194.2  8888   # -> Uno Q over USB gadget-Ethernet
```

**UART would need a different transport entirely** — these scripts assume a socket. If the link
lands on serial, the `pump()`/`drain()` split is the seam to swap.

## Usage

Start the receiver first, then the sender.

```bash
# on the RTX 4050 box
python3 stream_receiver.py listen 8888 --out rover.h264 --play

# on the Pi
ssh pi
python3 stream_sender.py connect <receiver-ip> 8888
```

Reverse it if you'd rather have the viewer pull:

```bash
# on the Pi
python3 stream_sender.py listen 8888

# on the workstation
python3 stream_receiver.py connect raspberrypi.local 8888 --play
```

Capture options on the sender: `--width` `--height` `--fps` `--bitrate` (default
1280x720 @ 30 fps, 4 Mbit/s). The IMX219 sensor modes are listed by
`rpicam-hello --list-cameras`; 1640x1232 is the full-FoV binned mode and 1920x1080 is a crop.

`--play` needs a player on the receiving box. None is installed on the dev laptop as of writing:

```bash
sudo apt install mpv
```

Both scripts log throughput every 5 s and print a total on exit.

## Verification

No live-camera test has been run — that needs the Pi with the sensor attached. What *has* been
verified on the dev machine, with `rpicam-vid` faked by a stub on `PATH`:

| Test | Result |
|---|---|
| `py_compile` both scripts | OK |
| `--help` for every subcommand | OK |
| Sender refuses to run where `rpicam-vid` is absent | OK — exits 2 with a clear message |
| Receiver `listen` → 300 KB pushed → file on disk | **PASS**, sha256 identical |
| Sender `connect` → receiver `listen` → file on disk | **PASS**, 200000/200000 bytes byte-exact |

Reproduce the last two with the harness in the session scratchpad, or re-stub `rpicam-vid` as a
script that writes known bytes to stdout.

**Still to prove on real hardware:** actual H.264 output decodes, sustained bitrate over the Pi's
2.4 GHz link, and latency. `--inline` is set so a receiver joining mid-stream gets SPS/PPS headers
on every I-frame, but that has not been exercised against a real encoder.

## Notes

- The stream is a **raw H.264 elementary stream**, not a container. `mpv rover.h264` plays it;
  some tools want `-f h264` told explicitly. Wrap in MP4 with
  `ffmpeg -i rover.h264 -c copy rover.mp4` if you need seeking.
- **No authentication and no encryption.** The receiver binds `0.0.0.0` and accepts whoever
  connects first. Fine on the closed rover LAN; add a shared secret before it touches anything
  wider.
- `rpicam-vid` is used rather than `picamera2` because it is confirmed present on the image.
  If frame-level access is needed later (VGGT-SLAM, crack detection), `picamera2` is the better
  seam — but check it is installed first.
