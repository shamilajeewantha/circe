# circe

Python controller for the **DM002HW** drone (and compatible WiFi UAV family) — no official app required.

---

## Quick start

1. Power on the drone. It broadcasts its own WiFi hotspot.
2. Connect your laptop WiFi to the drone's network (SSID like `DM002_XXXXXX`).
3. Verify connection: `ping 192.168.169.1` should reply.
4. Run the demo flight:

```bash
python drone.py
```

---

## Usage

```python
from drone import Drone
import time

drone = Drone()
drone.connect()
time.sleep(2)

drone.takeoff()
time.sleep(4)

drone.forward(duration=2.0, power=160)
drone.turn_right(duration=0.8)
drone.land()

drone.disconnect()
```

### Available controls

| Method | Description |
|---|---|
| `takeoff()` | Auto take-off |
| `land()` | Auto land |
| `stop()` | Emergency stop — cuts motors immediately |
| `calibrate()` | Gyro calibration, keep drone flat |
| `hover()` | Return all axes to neutral |
| `up(duration, power)` | Ascend |
| `down(duration, power)` | Descend |
| `forward(duration, power)` | Pitch forward |
| `backward(duration, power)` | Pitch backward |
| `turn_left(duration, power)` | Yaw left |
| `turn_right(duration, power)` | Yaw right |
| `move_left(duration, power)` | Roll left |
| `move_right(duration, power)` | Roll right |
| `set_controls(roll, pitch, throttle, yaw)` | Direct axis control |

All axes: 0–255, neutral = 128.

---

## Protocol

The drone listens on **UDP port 8800** at `192.168.169.1`.

Each control tick sends two packets back-to-back at 50 Hz: a short (88-byte) and a long (124-byte) variant.

### Outer packet structure

```
Bytes  0–11   Header        ef 02 [58|7c] 00 02 02 00 01 [00|02] 00 00 00
Bytes 12–13   Counter       increments every tick
Bytes 14–17   Speed block   00 00 08 00
Bytes 18–25   Inner control (see below)
Bytes 26–81   Zero padding
Bytes 82–87   Fixed trailer 32 4b 14 2d 00 00
--- 124-byte only ---
Bytes 88–89   Counter2      (counter + 1)
Bytes 90–107  Counter2 suffix
Bytes 108–109 Counter3      (counter + 2)
Bytes 110–123 Counter3 suffix
```

### Inner 8-byte control block (at offset 18)

```
66 [roll] [pitch] [throttle] [yaw] [cmd] [checksum] 99
```

- All axes: 0–255, neutral = 0x80 (128)
- `checksum = roll ^ pitch ^ throttle ^ yaw ^ cmd`

### Command byte values

| Value | Meaning |
|---|---|
| `0x40` | Motors armed — normal flight state |
| `0x01` | Auto take-off / land toggle |
| `0x03` | Land |
| `0x02` | Emergency stop |
| `0x80` | Gyro calibration |

### Connection sequence

1. Send `ef 00 04 00` (4 bytes) to port 8800 — handshake
2. Send ~6 zero-control packets (arms the link)
3. Begin 50 Hz control loop with `cmd = 0x40` (armed)

---

## How the protocol was reverse-engineered

### Background

The DM002HW is controlled via the **WiFi UAV** Android app (`com.lcfld.fldpublic`). The drone creates its own WiFi hotspot; the phone connects to it and sends UDP control packets.

The goal was to replicate those packets from a laptop without the app.

### Step 1 — Find reference implementations

Searched GitHub for prior work on the same app and drone family:

- [FahrulRPutra/reversing-wifi-uav](https://github.com/FahrulRPutra/reversing-wifi-uav) — reversed the LSRC-S1S (same app, same protocol family). Contains working Python source.
- [marshallrichards/turbodrone](https://github.com/marshallrichards/turbodrone) — full library for WiFi UAV drones.
- [guillesanbri/e58-drone-reversing](https://github.com/guillesanbri/e58-drone-reversing) — E58 drone, same app.
- [blog.horner.tj](https://blog.horner.tj/hacking-chinese-drones-for-fun-and-no-profit/) — detailed protocol writeup.

From these sources we found:
- Drone IP: `192.168.169.1`
- Control port: UDP `8800`
- Handshake: `ef 00 04 00`
- General packet structure and command byte values

### Step 2 — Confirm with packet capture (laptop-as-relay method)

The reference code gave us the structure but not the exact byte values for this specific drone. We needed to capture the real app traffic.

**Problem with standard phone capture tools (PCAPdroid):**
PCAPdroid creates a local VPN on the Android device to intercept traffic. This VPN breaks UDP drone control — the drone never receives the packets, so the app never progresses past the handshake. The capture only showed repeated `ef 00 04 00` handshakes with no flight commands.

**Solution — laptop as a transparent relay:**

```
Phone  ──WiFi──>  Laptop (Mobile Hotspot)
                       |
                  Windows routes/NATs
                       |
                  Laptop (WiFi) ──WiFi──>  Drone
```

Setup on Windows 11:
1. Connect laptop WiFi to the drone's hotspot (`192.168.169.x` network).
2. Go to **Settings → Network & Internet → Mobile Hotspot**.
3. Set "Share my internet connection from" to the drone WiFi adapter.
4. Turn the hotspot on.
5. Connect the phone to the **laptop's** hotspot (not the drone directly).
6. Open Wireshark on the laptop, capture on the WiFi adapter, filter: `udp and ip.addr == 192.168.169.1`.
7. Use the WiFi UAV app normally on the phone — the drone actually flies because traffic routes through the laptop transparently.
8. Wireshark captures every packet the app sends to the drone.

This works because:
- The phone connects to the laptop's hotspot (a different SSID/network).
- The laptop forwards all traffic to the drone via its real WiFi connection.
- Since the laptop is physically routing the packets, Wireshark on the laptop's WiFi adapter sees everything.
- The drone responds normally — no VPN interference.

### Step 3 — Parse and analyze the capture programmatically

Wireshark's GUI is useful for exploration but not for systematic analysis of hundreds of packets. Instead, we wrote Python scripts from scratch to parse the raw `.pcapng` binary format and extract exactly what we needed.

#### Why custom scripts instead of scapy/dpkt

No external libraries — pure stdlib `struct` and `socket`. This avoids dependency issues and makes the parsing logic explicit and auditable.

#### pcapng binary format

A `.pcapng` file is a sequence of typed blocks. Each block has:

```
[Block Type: 4 bytes] [Block Length: 4 bytes] [Body...] [Block Length: 4 bytes]
```

The relevant block types:
- `0x0A0D0D0A` — Section Header Block: marks start of a capture section, contains byte-order magic
- `0x00000001` — Interface Description Block: declares a capture interface and its link-layer type
- `0x00000006` — Enhanced Packet Block: one captured frame, with timestamps and the raw frame bytes starting at body offset 20

The link-layer type from the Interface Description Block tells us how to strip the layer-2 header before reaching IP:
- Type 1 = Ethernet: skip 14 bytes
- Type 101 = Raw IP: no header to skip (Android captures use this)

#### Decoding frames to UDP payloads

For each Enhanced Packet Block:
1. Strip the link-layer header based on link type
2. Check IP version byte = 4, read IHL to find where the transport header starts
3. Check protocol byte = 17 (UDP) or 6 (TCP)
4. For UDP: extract source/dest IP, source/dest port, and payload from the UDP header
5. Filter to `dst == 192.168.169.1 AND dport == 8800` — these are drone control packets only

#### First pass — `analyze_capture.py`

Grouped all packets by `(src_ip, src_port, dst_ip, dst_port)` and printed every unique payload per flow. This immediately showed:

- **49 UDP flows** total — most were DNS queries and QUIC traffic from phone apps (Notion, WhatsApp, Google, etc.) — noise, not drone traffic
- **Key flow: `192.168.169.3:61934 → 192.168.169.1:8800` — 1197 packets** — this is the drone control channel
- **Video flow: `192.168.169.1:1234 → 192.168.169.3:61934` — 1061 packets** — drone camera stream going back to the phone
- Port 8801 only ever received `ef 00 04 00` — just a secondary handshake

The first-pass output also showed that `analyze_capture.py` was grouping by first 8 bytes only, so packets with different control values were being collapsed into the same entry — important limitation to fix next.

#### Second pass — `dump_drone_packets.py`

Focused exclusively on `192.168.169.1:8800` traffic. Two analysis modes:

**Mode 1 — Distinct full payloads**: grouped by complete payload content (not just header bytes). This revealed there are actually **365 distinct packets** across two structural sizes:
- 88-byte packets: `ef 02 58 00 ...`
- 124-byte packets: `ef 02 7c 00 ...`
- Plus 4-byte handshake and small `ef 20` command packets

The hex dump of each distinct packet showed the fixed vs variable byte positions clearly.

**Mode 2 — Control byte extraction**: extracted bytes at fixed offsets from every `ef 02` packet:

```python
roll, pitch, throttle, yaw = pay[20], pay[21], pay[22], pay[23]
cmd, headless               = pay[24], pay[25]
chk                         = pay[26]
```

Printed a table of every packet where any of these was non-zero. This showed:
- All flying packets have `roll=pitch=throttle=yaw=128`, `cmd=64 (0x40)`, `"headless"=153 (0x99)`, `chk=0`
- The `0x99` in the "headless" column is actually the **end footer** of the inner 8-byte block — meaning my byte offset was one position too high

#### Correcting the byte offset

With the offset shifted back by 2, the inner 8-byte control block at **payload bytes 18–25** became clear:

```
byte 18: 0x66  — inner block header (fixed)
byte 19: roll
byte 20: pitch
byte 21: throttle
byte 22: yaw
byte 23: cmd       (0x40 during normal flight)
byte 24: checksum  (= roll ^ pitch ^ throttle ^ yaw ^ cmd)
byte 25: 0x99      — inner block footer (fixed)
```

Verified: `0x80 ^ 0x80 ^ 0x80 ^ 0x80 ^ 0x40 = 0x40` ✓ — checksum matches every observed packet.

This inner 8-byte format matches the simpler protocol documented by [horner.tj](https://blog.horner.tj/hacking-chinese-drones-for-fun-and-no-profit/) exactly. The outer 88/124-byte wrapper is a framing layer added by the Lewei camera module's firmware.

**Key findings from the full analysis:**

- Two packet sizes interleaved every tick: **88 bytes** and **124 bytes**
- Inner control block `66 ... 99` sits at byte offset 18 of the outer wrapper
- Neutral for all axes = `0x80` (128)
- Normal flight command byte = `0x40` (motor armed/unlocked)
- Checksum = `roll ^ pitch ^ throttle ^ yaw ^ cmd`
- Counter field at bytes 12–13 increments every tick, shared across both packet sizes
- 124-byte variant adds two more counter blocks (counter+1 and counter+2) after byte 87

The FahrulRPutra source matched the captured structure almost exactly. The only difference was the speed byte at offset 16 (`0x08` observed vs `0x14` in source), which reflects the speed/sensitivity slider setting in the app at the time of capture.

### Step 4 — Implement and test

Rewrote `drone.py` using the exact packet format from the capture. First successful flight confirmed the protocol is correct.

---

## Files

| File | Purpose |
|---|---|
| `drone.py` | Main controller — import this |
| `README.md` | This file |

---

## Compatible drones

Any drone using the **WiFi UAV** app (`com.lcfld.fldpublic`) with IP `192.168.169.1` should work. Known compatible models include DM002, LSRC-S1S, E58, and various unbranded clones using the same Lewei camera/WiFi module.

If your drone uses a different IP, change `DRONE_IP` at the top of `drone.py`.
