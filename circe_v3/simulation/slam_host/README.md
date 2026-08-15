# slam_host — VGGT-SLAM server (runs on THIS laptop / the off-board SLAM box)

In the circe_v3 simulation the **rover's brain runs on a different machine** (the
Gazebo sim laptop, and on the real robot, the rover itself). VGGT-SLAM runs
**here**, off-board, and the rover talks to it over the network — the exact
project.md §9 split (rover ⇄ off-board RTX SLAM). Because VGGT-SLAM only ingests
frames through its `Camera` interface, the **same server runs unchanged from sim
to the real robot** — only the frame *source* differs.

```
[sim laptop / real rover]  ──POST /frames (JPEG)──▶  [THIS laptop: slam_server.py]
        rover brain        ◀──GET /map (poses+cloud)──   VGGT-SLAM 2.0 (GPU)
                                                          + viser raw-map viewer :8080
```

## What's here
| File | Purpose |
|------|---------|
| `network_camera.py` | `NetworkCamera(vggt_slam.cameras.Camera)` — frames arrive over HTTP instead of RealSense |
| `slam_server.py` | Loads VGGT + Solver, runs the submap loop, serves the HTTP API below |
| `requirements-lock.txt` | Exact verified deps frozen from the working WSL `vggt` env |
| `environment.yml` | Conda recreate spec |

**No edits are made to the vendored VGGT-SLAM source** — this imports the
installed `vggt_slam`, so `git pull`s of VGGT-SLAM stay clean.

## Environment (verified)
- **WSL `vggt` conda env**, Python **3.11.15**, torch **2.3.1+cu121**, CUDA 12.1, RTX 4050.
- `vggt_slam 2.0.0` is `pip -e` installed from `/mnt/d/others_github/VGGT-SLAM` (commit `35327ac`,
  which includes the released real-time code). Third-party: `salad@33ca9c0`, `vggt(VGGT_SPARK)@6e6e161`.
  `perception_models` + `sam3` are **not** installed / not needed (only for `--run_os`).
- `fastapi`, `uvicorn`, `python-multipart` are already present in the env.
- The VGGT-1B checkpoint (~5 GB) must be at `$TORCH_HOME/hub/checkpoints/model.pt`, and the SALAD
  loop-closure checkpoint at `$TORCH_HOME/hub/checkpoints/dino_salad.ckpt`. **On this machine `TORCH_HOME`
  is overridden** (`~/.bashrc` line 134: `export TORCH_HOME=/mnt/d/others_github/model_cache`), so the
  real paths are `/mnt/d/others_github/model_cache/hub/checkpoints/{model.pt,dino_salad.ckpt}`, **not**
  the torch default `~/.cache/torch/hub/...` — confirmed from a live run's log ("Loaded model from
  /mnt/d/others_github/model_cache/hub/checkpoints/dino_salad.ckpt"). Check `echo $TORCH_HOME` before
  assuming either path; `slam_server.py`'s `load_model()` falls back to downloading `model.pt` from the
  HF URL if it's missing wherever `TORCH_HOME` actually points.

## Run
```bash
# WSL, vggt env, GPU:
conda activate vggt
cd /mnt/d/my_github/circe/circe_v3/simulation/slam_host
python slam_server.py --port 8000 --submap_size 8 --vis_map
#   --vis_map opens VGGT-SLAM's own viser raw-map viewer at http://localhost:8080
```

### Logging (repo convention — see CLAUDE.md's "Verification" section)
Every run writes a timestamped UTF-8 log file to `slam_server_logs/` (gitignored, override with
`--log_dir`) — console output alone is never the record of a run. It captures: model load, the
worker's periodic `[frame N] keyframes_pending=X/target camera={...}` progress line (every 25 frames
— the loop has no fixed N, so this is the "don't go silent" signal for an unbounded stream), every
submap completion (`submap done (submaps=N loops=M)`), and uvicorn's own `POST /frames`/`GET /map`
access log lines. **Read this file after a run instead of scrolling/pasting terminal history** — it's
the actual record, and it's what tells you whether a submap really completed vs. the process being
killed (`Ctrl+C`) before one could finish.

### Sanity check (no rover needed) — replay a folder as if it were the rover
```bash
# from another shell in the vggt env: POST office_loop frames, then GET the map
python - <<'PY'
import glob, requests
files = sorted(glob.glob("/mnt/d/others_github/VGGT-SLAM/office_loop/*.jpg"))[:64]
for i in range(0, len(files), 8):
    fs = [("files", (f, open(f, "rb"), "image/jpeg")) for f in files[i:i+8]]
    print(requests.post("http://localhost:8000/frames", files=fs).json())
import time; time.sleep(20)
m = requests.get("http://localhost:8000/map", params={"after_submap": -1, "voxel": 0.02}).json()
print("submaps:", m["num_submaps"], "loops:", m["num_loops"], "cloud pts:", m["cloud"]["n"])
PY
```

## HTTP API
| Method / path | Body / query | Returns |
|---|---|---|
| `POST /session` | `{intrinsics, width, height, submap_size?}` | config ack (relative scale; restart for a fresh map) |
| `POST /frames` | multipart JPEG file(s) | `{received, queued}` |
| `GET /pose/latest` | — | latest camera pose `T_cam_world` (4×4, **relative scale**) |
| `GET /map` | `after_submap, known_loops, voxel, max_points` | `{full_refresh, num_submaps, num_loops, submaps:[{submap_id,poses}], cloud:{n,xyz_f32_b64,rgb_u8_b64}}` |
| `GET /status` | — | worker + camera counters |

**Loop closures:** a loop closure re-optimises *all* poses. When the loop count
grows past the client's `known_loops`, `/map` sets `full_refresh:true` and returns
the whole map so the client rebuilds and re-runs its motion self-calibration (§5).
Point cloud + poses are always **relative scale** — the rover resolves metric
motion locally via §5 self-calibration; never assume `/map` returns metric units.

## Networking: reaching this server from the sim laptop

`slam_server.py` binds `0.0.0.0`, but running it **inside WSL2** normally puts it on
WSL's own NAT'd subnet, unreachable from a second LAN machine by default. **This is
now set up and verified working, on this machine, as follows** (2026-08-15):

### 1. WSL mirrored networking (done)
`C:\Users\<you>\.wslconfig`:
```ini
[wsl2]
memory=12GB
networkingMode=mirrored
```
Applied via `wsl --shutdown` + restart. Confirmed: `wsl hostname -I` now returns the
**same IP as the Windows host's LAN adapter** (verified `192.168.1.8` on both sides) —
no separate WSL-only subnet anymore. The sim laptop connects to
`http://192.168.1.8:8000` (or whatever this laptop's current LAN IP is — check both
`wsl hostname -I` and the Windows host's IP match before trusting this address; DHCP
can reassign it after a reboot).

### 2. Firewall rules (done — two separate layers, both required)
Mirrored networking alone does **not** open the port — verified both layers
default-deny inbound on this machine (`Get-NetFirewallHyperVVMSetting` showed
`DefaultInboundAction: Block`; `Get-NetFirewallProfile -Name Public` also showed
`DefaultInboundAction: Block`, since this Wi-Fi network is categorized `Public`, not
`Private`). Two rules were added, **each scoped to TCP port 8000 only** — run in an
**elevated** PowerShell (Claude does not and will not run these itself — see the
repo's `CLAUDE.md` hard rule on this):

```powershell
# Hyper-V firewall — governs WSL/mirrored-mode traffic specifically.
# {40E0AC32-46A5-438A-A0B2-2B479E8F2E90} is WSL's fixed VMCreatorId
# (from Get-NetFirewallHyperVVMCreator).
New-NetFirewallHyperVRule -Name "CirceSlamServer" -DisplayName "Circe VGGT-SLAM server (port 8000)" `
    -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8000

# Regular Windows Firewall — governs the Windows host itself.
New-NetFirewallRule -Name "CirceSlamServer" -DisplayName "Circe VGGT-SLAM server (port 8000)" `
    -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Both confirmed `Enabled: True`, `Action: Allow`, scoped to TCP/8000 only — nothing
broader was opened. To remove later: `Remove-NetFirewallHyperVRule -Name
"CirceSlamServer"` and `Remove-NetFirewallRule -Name "CirceSlamServer"` (elevated).

### What's NOT yet verified
Actual cross-machine reachability — the above proves the Windows/WSL side is
correctly configured to *accept* the connection, but there's no second device on
this LAN yet to confirm a real inbound connection succeeds end-to-end. That
confirms itself the first time `circe_vggt_client` on the sim laptop successfully
hits `POST /frames`.

### Security note
This exposes `slam_server.py` (which has **no authentication** — anyone reaching
port 8000 can `POST /frames` or `GET /map`) to the LAN. Accepted deliberately for a
trusted home network with no other devices on it (2026-08-15 decision) — re-evaluate
if that assumption ever changes (e.g. guests, IoT devices, a shared/office network).

**On the real robot this whole section disappears** — the off-board box is native
Linux on the LAN; only the frame source (Gazebo camera vs real RPi camera) differs.

## Re-verifying after an upstream VGGT-SLAM pull

This is a **big external dependency, pinned by commit** (see "Environment" above) — it is
not vendored, so a `git pull` on `/mnt/d/others_github/VGGT-SLAM` can silently move you
past the verified commit. Do NOT assume it still works; re-verify:

1. **See what actually changed:**
   ```bash
   cd /mnt/d/others_github/VGGT-SLAM
   git log --oneline 35327ac28b7d193df9ccc39ba6346052bb6f1207..HEAD
   ```
2. **Check whether the pinned sub-dependencies moved.** `setup.sh` pins `salad` and
   `vggt` (VGGT_SPARK) to their own commits — diff its git URLs/commit hashes against
   the ones frozen in `requirements-lock.txt` (currently `salad@33ca9c0`, `vggt@6e6e161`).
   If either moved, re-`pip install -e` that one too.
3. **Check `main_realtime.py` for API drift.** `slam_server.py`'s `_process_submap` /
   `slam_worker` mirror its call sequence (`run_predictions` → `add_points` →
   `graph.optimize()`, `flow_tracker.compute_disparity` keyframe gate). If the upstream
   signatures changed, `slam_server.py` needs matching edits.
4. **Re-run the smoke test** (the "Sanity check" section above) — replay `office_loop`,
   confirm `GET /map` still returns real poses and a nonzero point cloud (`cloud.n > 0`).
   Don't trust a clean import; only a real submap counts as verified.
5. **Re-freeze:** from the WSL `vggt` env, `pip freeze > requirements-lock.txt`, diff
   against the previous version, and note anything unexpected.
6. **Update the pinned commit everywhere it's written down** — this file (line 29-30,
   line 17-19 in `requirements-lock.txt`'s header) and `circe_v3/simulation/BUILD_GUIDE.md`
   (§3 evidence table, §4 "Verified deps + checkpoints", §8 "Settled"). All three must
   agree on the same commit hash — that's the whole point of pinning it in three places.
7. The two checkpoints (`model.pt`, `dino_salad.ckpt`) do **not** need re-fetching for a
   `vggt_slam` code pull — they're stable released model weights, versioned separately
   from the SLAM code. Only re-fetch if VGGT-1B or SALAD itself gets a new release.
