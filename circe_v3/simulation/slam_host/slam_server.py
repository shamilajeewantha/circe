"""VGGT-SLAM host server — runs the SLAM on THIS laptop, exposes a small HTTP API.

Topology (see circe_v3/simulation/BUILD_GUIDE.md and the plan): the rover's brain
runs elsewhere (the Gazebo sim laptop, or the real robot) and talks to this
off-board SLAM over the network — exactly the project.md §9 split. This server:

  * builds one VGGT + Solver (same sequence as VGGT-SLAM's main_realtime.py),
  * feeds it frames through a NetworkCamera backend (frames arrive over HTTP),
  * runs the incremental submap loop in a background thread,
  * serves poses + point cloud back out.

Because VGGT-SLAM ingests frames only through its Camera interface, the SAME
server runs unchanged whether frames come from Gazebo or a real RPi camera.

Sessions: one SLAM run = one session. POST /session ends the current run
(archived, listed at GET /sessions) and starts an empty map, so relaunching the
sim no longer fuses a new scene into the previous run's geometry.

Endpoints
---------
  POST /session   {intrinsics, width, height, submap_size?, label?, reset?}
                                                             -> new session (reset=false: config only)
  GET  /sessions                                             -> current run + archived past runs
  POST /frames    multipart JPEG file(s)                     -> {received, queued}
  GET  /pose/latest                                          -> latest camera pose (rel scale)
  GET  /map?after_submap=&known_loops=                       -> session_id + poses + cloud
  GET  /status                                               -> session, solver + camera counters

Run (WSL `vggt` env, GPU):
  python slam_server.py --port 8000 --submap_size 8 --vis_map
The Solver also opens VGGT-SLAM's own viser raw-map viewer (default :8080).

This file imports the installed `vggt_slam` and does NOT edit vendored source, so
`git pull`s of VGGT-SLAM stay clean.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Version + running-source identity.
#
# Python does not hot-reload: editing this file does NOTHING to an already
# running server, and that has burned us — a fix was "applied" on disk while the
# live process kept executing the old code path, and we only noticed by watching
# the old buggy behaviour happen (a session reset firing minutes late). So
# /status reports which build is actually SERVING, not which build is on disk:
#
#   version          -> bump by hand on any wire-visible change
#   src_sha          -> sha256 of this file as read AT STARTUP (the running code)
#   src_sha_on_disk  -> sha256 of this file right now
#   stale            -> the two differ => the file changed since launch,
#                       i.e. RESTART REQUIRED for the edit to take effect
# --------------------------------------------------------------------------- #
SERVER_VERSION = "1.3.0"      # sessions + fail-fast /session (503) + stage counters


def _src_sha() -> str:
    """sha256 (first 12 hex) of this source file, or 'unknown' if unreadable."""
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        return "unknown"


# Captured at import time on purpose: this is the fingerprint of the code the
# interpreter actually loaded. Computing it per-request would re-read whatever is
# on disk now and could never reveal a stale process.
_SRC_SHA_AT_START = _src_sha()
_PROCESS_STARTED_AT = time.time()

import cv2
import numpy as np
import torch

log = logging.getLogger("slam_server")


def _setup_logging(log_dir: str, debug: bool = False) -> None:
    """Console + a UTF-8 log file per run (repo convention: never print-only for
    a long-running loop, always leave a file a session can be reconstructed from)."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(log_dir, f"slam_server_{ts}.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)
    log.info("Logging to %s (debug=%s)", path, debug)

# Register the network backend BEFORE anything reads BACKENDS.
from vggt_slam.cameras import BACKENDS
from network_camera import NetworkCamera

BACKENDS["network"] = NetworkCamera

from vggt_slam.solver import Solver          # noqa: E402
from vggt.models.vggt import VGGT            # noqa: E402

from fastapi import FastAPI, UploadFile, File, Query   # noqa: E402
from fastapi.responses import JSONResponse, Response   # noqa: E402
import uvicorn                                          # noqa: E402


# --------------------------------------------------------------------------- #
# Shared SLAM state (one Solver for the process lifetime; viser binds a port
# once, so we never rebuild it — same constraint the repo's gradio_demo.py notes)
# --------------------------------------------------------------------------- #
solver_lock = threading.Lock()   # only one submap inference in flight
data_lock = threading.Lock()     # protects solver map/graph reads & writes

_model: Optional[VGGT] = None
_solver: Optional[Solver] = None
_camera: Optional[NetworkCamera] = None
_cfg = argparse.Namespace()
_worker_started = threading.Event()
_stop = threading.Event()
# _worker_started is one-shot (set once, never cleared), so on its own it can't
# tell /status apart from "worker is fine" vs "worker's thread died an hour ago".
# Track real liveness separately: bumped every loop iteration, plus the last
# per-frame exception (if any) so a crash is visible over HTTP, not just in
# whatever terminal happens to be watching this process's stdout.
_worker_last_beat = 0.0
_worker_last_error: Optional[str] = None
# Pipeline-stage counters, so "no submaps" can be diagnosed over HTTP instead of
# only from the server console. They separate the three ways it can stall:
#   frames_seen == 0            -> frames aren't reaching the worker at all
#   keyframes_pending stuck 0   -> the disparity gate is rejecting everything
#                                  (rover not moving enough / min_disparity too high)
#   keyframes_pending hits N,   -> submaps ARE being attempted; read submap_last_error
#   then resets, submaps == 0
_frames_seen = 0
_keyframes_pending = 0
_keyframes_total = 0
# _process_submap runs in its own thread and only logged failures; a submap that
# throws every time was invisible in /status while everything else looked healthy.
_submap_last_error: Optional[str] = None
_submaps_failed = 0
# submap_id -> the on-disk keyframe paths that made up that submap. Populated in
# _process_submap; read (under data_lock) by GET /submap/{id}/frames, which the
# diagnostic viewer uses to build each ledger row's real image gallery.
_submap_frames: dict = {}
# Plain-int mirrors of the solver's submap/loop counts, refreshed inside
# _process_submap's critical section. /status reads THESE, never the solver —
# so a health check can't block behind a long submap optimisation.
_n_submaps = 0
_n_loops = 0

# --- session state (one SLAM run = one session) ----------------------------- #
# Without this the server had no notion of a "run": every sim relaunch kept
# appending into the same map, so a fresh Gazebo world (new coordinate frame,
# new scene) got fused on top of the previous run's geometry and the client's
# submap cursor (after_submap=N) silently filtered out the whole new map.
# POST /session now ends the current run and starts an empty one.
_session_id = 0
_session_label: Optional[str] = None
_session_started_at = time.time()
_past_runs: List[dict] = []
# Bumped on every reset. slam_worker keeps its own copy and, when they differ,
# drops its pending keyframes and renumbers — otherwise keyframes captured
# just before the reset would leak into the new session's first submap.
_reset_generation = 0


def load_model(device: str) -> VGGT:
    """Load VGGT-1B — from the torch.hub checkpoint if present, else the HF URL
    (mirrors main.py's cached path with main_realtime.py's URL as fallback)."""
    model = VGGT()
    cached = os.path.join(torch.hub.get_dir(), "checkpoints/model.pt")
    if os.path.exists(cached):
        # VGGT-1B is a pure state_dict → weights_only=True (safe unpickler).
        state = torch.load(cached, map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(state)
        del state
    else:
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    model = model.to(torch.bfloat16).to(device)
    return model


# How long POST /session waits for an in-flight submap before giving up. Must stay
# comfortably BELOW the client's read timeout so the caller gets a real answer
# ("busy, nothing changed") instead of a silent late reset. circe_vggt_client uses
# 10s, so this leaves it room to see the 503.
_RESET_LOCK_WAIT_S = 6.0


class SessionBusy(RuntimeError):
    """Raised when a reset cannot safely run right now. Surfaces as HTTP 503."""


def _session_keyframe_dir(session_id: int) -> str:
    """Per-session keyframe folder. Sessions restart frame numbering at 1, so a
    single shared folder would have each new run overwrite the previous run's
    frame_%06d.png — which would make "past run" a lie the moment you relaunch."""
    return os.path.join(_cfg.keyframe_folder, f"session_{session_id:03d}")


def _start_new_session(label: Optional[str] = None) -> dict:
    """End the current run and start a fresh, empty map. Returns the new session.

    Resets exactly the stateful pieces ``Solver.__init__`` builds, and nothing
    else (all three take no constructor args — verified against upstream
    ``vggt_slam/solver.py``). Deliberately NOT rebuilt:

    * ``solver.viewer`` — ``Viewer.__init__`` binds viser's port (8080); a second
      instance in the same process would fail to bind. It also exposes no clear
      method upstream, so old geometry lingers in viser until the new run's
      submaps overwrite it by id. That's cosmetic and viser-only; ``/map`` (what
      the rover actually consumes) is genuinely empty after this call.
    * ``solver.image_retrieval`` — verified stateless upstream (it holds only the
      loaded SALAD model + an image transform, no descriptor database), so there
      is nothing to clear and rebuilding it would reload a ~336 MB checkpoint
      onto the GPU on every reset. The retrieval database lives in ``map``,
      which IS rebuilt below.

    They're rebuilt via ``type(x)()`` rather than importing ``GraphMap`` /
    ``PoseGraph`` / ``FrameTracker`` directly: this module's contract is that it
    imports the *installed* ``vggt_slam`` and never couples to vendored internals
    (so upstream ``git pull``s stay clean), and those classes' import paths are an
    upstream detail this file otherwise never depends on.
    """
    global _session_id, _session_label, _session_started_at, _reset_generation
    global _n_submaps, _n_loops, _worker_last_error
    global _submap_last_error, _submaps_failed

    # Block until any in-flight submap inference finishes. Without this, a
    # _process_submap already running would call add_points() AFTER we swap in
    # the empty map — silently seeding the new run with the old run's geometry.
    #
    # The wait is SHORT and failure is reported, never silently deferred: this
    # originally waited 180s, far longer than any client read timeout. A client
    # calling POST /session while a big submap was optimising (measured: 46
    # submaps / 21 loops held both locks for minutes) timed out at 10s, gave up,
    # and started streaming — and then the reset landed anyway, wiping the map
    # mid-run under a client that believed its session had never started. A reset
    # that arrives after the caller stopped waiting is worse than no reset.
    acquired = solver_lock.acquire(timeout=_RESET_LOCK_WAIT_S)
    if not acquired:
        raise SessionBusy(
            f"SLAM busy (submap in flight) — no reset performed after "
            f"{_RESET_LOCK_WAIT_S:.0f}s. Retry; the map is untouched.")
    try:
        with data_lock:
            now = time.time()
            finished = {
                "session_id": _session_id,
                "label": _session_label,
                "started_at": _session_started_at,
                "ended_at": now,
                "duration_s": round(now - _session_started_at, 1),
                "num_submaps": _n_submaps,
                "num_loops": _n_loops,
                "submaps_failed": _submaps_failed,
                "frames_received": _camera.stats()["received"] if _camera else 0,
                "keyframe_dir": _session_keyframe_dir(_session_id),
            }
            _past_runs.append(finished)

            _solver.map = type(_solver.map)()
            _solver.graph = type(_solver.graph)()
            _solver.flow_tracker = type(_solver.flow_tracker)()
            _solver.current_working_submap = None

            _submap_frames.clear()
            _n_submaps = 0
            _n_loops = 0
            _worker_last_error = None
            _submap_last_error = None   # a past run's failures aren't this run's
            _submaps_failed = 0

            # Drop frames still queued from the previous run so they can't be
            # consumed into the new session's first submap.
            if _camera is not None:
                _camera.drain()

            _session_id += 1
            _session_label = label
            _session_started_at = now
            os.makedirs(_session_keyframe_dir(_session_id), exist_ok=True)
            _reset_generation += 1
    finally:
        if acquired:
            solver_lock.release()

    log.info("new session %d (label=%s); archived run %d (%d submaps, %d loops)",
             _session_id, label, finished["session_id"],
             finished["num_submaps"], finished["num_loops"])
    return finished


def _process_submap(frames: List[str]) -> None:
    """Run VGGT + graph optimisation for one submap (background thread)."""
    global _n_submaps, _n_loops, _submap_last_error, _submaps_failed
    try:
        preds = _solver.run_predictions(frames, _model, _cfg.max_loops, None, None)
        t0 = time.monotonic()
        with data_lock:
            _solver.add_points(preds)
            _solver.graph.optimize()
            new_submap_id = int(list(_solver.map.ordered_submaps_by_key())[-1].get_id())
            _submap_frames[new_submap_id] = list(frames)
            # Cheap counters published for /status, which must never take
            # data_lock (see the status endpoint for why).
            _n_submaps = int(_solver.map.get_num_submaps())
            _n_loops = int(_solver.graph.get_num_loops())
        held = time.monotonic() - t0

        # Viser visualisation is DELIBERATELY outside data_lock. update_all_submap_vis()
        # re-pushes every submap's cloud to viser and grows with map size — measured
        # holding the lock 90+s at 32 submaps/9 loops, which starved GET /status and
        # GET /map into client-side read timeouts (the diagnostic viewer showed
        # "unreachable" while SLAM was actually healthy). Viser reads the solver's own
        # state; a concurrent reader here is no worse than the viser thread already is.
        t1 = time.monotonic()
        if _cfg.vis_map:
            if len(preds.get("detected_loops", [])) > 0:
                _solver.update_all_submap_vis()
            else:
                _solver.update_latest_submap_vis()
        log.info("submap done (submaps=%d loops=%d) lock_held=%.2fs vis=%.2fs",
                 _n_submaps, _n_loops, held, time.monotonic() - t1)
    except Exception as e:  # keep the loop alive on a bad submap
        _submap_last_error = f"{type(e).__name__}: {e}"
        _submaps_failed += 1
        log.exception("submap processing failed (%d total failures)", _submaps_failed)
    finally:
        if solver_lock.locked():
            solver_lock.release()


def slam_worker() -> None:
    """The capture -> keyframe-gate -> submap loop (mirrors main_realtime.py)."""
    _camera.start()
    subset: List[str] = []
    frame_count = 0
    target = _cfg.submap_size + _cfg.overlapping_window_size
    my_generation = _reset_generation
    kf_dir = _session_keyframe_dir(_session_id)
    os.makedirs(kf_dir, exist_ok=True)
    _worker_started.set()
    log.info("worker started (session %d); waiting for frames...", _session_id)

    global _worker_last_beat, _worker_last_error
    global _frames_seen, _keyframes_pending, _keyframes_total
    while not _stop.is_set():
        _worker_last_beat = time.monotonic()
        try:
            # A /session reset landed: drop keyframes staged for the old run and
            # renumber into the new session's folder. Without this the first
            # submap of a new run would be built partly from the previous run's
            # frames — i.e. exactly the cross-run contamination sessions exist
            # to prevent.
            if _reset_generation != my_generation:
                my_generation = _reset_generation
                subset = []
                frame_count = 0
                kf_dir = _session_keyframe_dir(_session_id)
                os.makedirs(kf_dir, exist_ok=True)
                _frames_seen = 0
                _keyframes_pending = 0
                _keyframes_total = 0
                log.info("worker: switched to session %d, pending keyframes dropped",
                         _session_id)

            img = _camera.capture(timeout=1.0)
            if img is None:
                continue
            frame_count += 1
            _frames_seen = frame_count
            if _solver.flow_tracker.compute_disparity(img, _cfg.min_disparity, False):
                path = os.path.join(kf_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(path, img)
                subset.append(path)
                _keyframes_total += 1
            _keyframes_pending = len(subset)

            if frame_count % 25 == 0:      # mandatory progress signal for a loop with no fixed N
                log.info("[session %d][frame %d] keyframes_pending=%d/%d camera=%s",
                          _session_id, frame_count, len(subset), target, _camera.stats())

            if len(subset) >= target:
                if solver_lock.acquire(blocking=False):
                    t = threading.Thread(target=_process_submap, args=(list(subset),), daemon=True)
                    t.start()
                    subset = subset[-_cfg.overlapping_window_size:]
                elif len(subset) > target * 2:      # SLAM busy; bound the backlog
                    subset = subset[-target:]
        except Exception as e:  # a single bad frame must not silently kill this thread
            _worker_last_error = f"{type(e).__name__}: {e}"
            log.exception("slam_worker: frame %d failed, skipping", frame_count)


# --------------------------------------------------------------------------- #
# Egress helpers — read the map through the same public getters main.py /
# gradio_demo.py use (confirmed present at commit 35327ac).
# --------------------------------------------------------------------------- #
def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _collect(submaps, voxel: float, max_points: int):
    poses_out, xyz_all, rgb_all = [], [], []
    for sm in submaps:
        poses = sm.get_all_poses_world(_solver.graph)          # (F,4,4) rel-scale
        poses_out.append({"submap_id": int(sm.get_id()),
                          "loop_closure": bool(sm.get_lc_status()),
                          "poses": poses.astype(np.float32).tolist()})
        pts = sm.get_points_in_world_frame(_solver.graph)
        cols = sm.get_points_colors()
        if pts is not None and len(pts) > 0:
            xyz_all.append(np.asarray(pts).reshape(-1, 3))
            rgb_all.append(np.asarray(cols).reshape(-1, 3))

    if xyz_all:
        xyz = np.concatenate(xyz_all, 0).astype(np.float32)
        rgb = np.concatenate(rgb_all, 0).astype(np.uint8)
        if voxel > 0:                                          # cheap grid dedupe
            keys = np.floor(xyz / voxel).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            xyz, rgb = xyz[idx], rgb[idx]
        if max_points and len(xyz) > max_points:
            sel = np.linspace(0, len(xyz) - 1, max_points).astype(int)
            xyz, rgb = xyz[sel], rgb[sel]
        cloud = {"n": int(len(xyz)), "xyz_f32_b64": _b64(xyz), "rgb_u8_b64": _b64(rgb)}
    else:
        cloud = {"n": 0, "xyz_f32_b64": "", "rgb_u8_b64": ""}
    return poses_out, cloud


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #
app = FastAPI(title="circe VGGT-SLAM host")


@app.post("/session")
def session(payload: dict):
    """Start a NEW run: archive the current map as a past run, reset to empty.

    This is the run boundary — the rover client calls it once at startup, so
    relaunching the sim (new world, new coordinate frame) no longer fuses into
    the previous run's map. A server restart is no longer required for a fresh
    map; past runs stay listed at ``GET /sessions`` and their keyframes stay on
    disk under ``<keyframe_folder>/session_<id>/``.

    Pass ``{"reset": false}`` to only update config (intrinsics/size) without
    ending the current run — e.g. a client reconnecting mid-run that must not
    destroy the map it is already mapping into.
    """
    _cfg.intrinsics = payload.get("intrinsics")
    _cfg.width = payload.get("width")
    _cfg.height = payload.get("height")
    if payload.get("submap_size"):
        _cfg.submap_size = int(payload["submap_size"])

    if not payload.get("reset", True):
        return {"ok": True, "reset": False, "session_id": _session_id,
                "submap_size": _cfg.submap_size,
                "note": "config only; current run left intact (relative scale)"}

    try:
        archived = _start_new_session(label=payload.get("label"))
    except SessionBusy as e:
        # 503 + Retry-After: the map is untouched, the caller can simply try again.
        log.warning("POST /session refused: %s", e)
        return JSONResponse({"ok": False, "reset": False, "busy": True,
                             "session_id": _session_id, "error": str(e)},
                            status_code=503, headers={"Retry-After": "5"})
    return {"ok": True, "reset": True, "session_id": _session_id,
            "submap_size": _cfg.submap_size, "archived_run": archived,
            "note": "fresh empty map; relative scale"}


@app.get("/sessions")
def sessions():
    """Past runs (archived, newest last) + the run currently building."""
    return {"current": {"session_id": _session_id, "label": _session_label,
                        "started_at": _session_started_at,
                        "num_submaps": _n_submaps, "num_loops": _n_loops,
                        "keyframe_dir": _session_keyframe_dir(_session_id)},
            "past_runs": _past_runs}


@app.post("/frames")
async def frames(files: List[UploadFile] = File(...)):
    """Receive JPEG keyframes from the rover and push them to the SLAM loop."""
    n = 0
    for f in files:
        buf = np.frombuffer(await f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # BGR uint8
        if img is not None:
            _camera.push(img)
            n += 1
    return {"received": n, "queued": _camera.stats()["queued"]}


@app.get("/frame/latest")
def frame_latest():
    """The most recently POSTed frame, as JPEG — ground truth for a diagnostic
    viewer to prove real frames are arriving (not a claim, an actual image)."""
    img = _camera.get_latest_frame() if _camera else None
    if img is None:
        return JSONResponse({"available": False}, status_code=404)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return JSONResponse({"available": False}, status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/submap/{submap_id}/frames")
def submap_frames(submap_id: int):
    """The on-disk keyframe paths that made up this submap — for diag_viewer.py's
    per-ledger-row image gallery. Paths are on THIS machine (slam_host runs the
    viewer in-process), not served as bytes here."""
    # Lock-free on purpose: taking data_lock for a single dict lookup would make
    # this hang for the whole duration of a submap optimisation (same bug /status
    # had). A plain dict .get() is atomic under CPython, and the only race is
    # reading a key mid-insert — which just yields a 404 the client retries.
    paths = _submap_frames.get(submap_id)
    if paths is None:
        return JSONResponse({"available": False}, status_code=404)
    return {"available": True, "submap_id": submap_id, "paths": paths}


@app.get("/pose/latest")
def pose_latest():
    with data_lock:
        if _solver.map.get_num_submaps() == 0:
            return JSONResponse({"available": False})
        last = list(_solver.map.ordered_submaps_by_key())[-1]
        poses = last.get_all_poses_world(_solver.graph)
        return {"available": True, "submap_id": int(last.get_id()),
                "T_cam_world": poses[-1].astype(np.float32).tolist(),
                "num_submaps": int(_solver.map.get_num_submaps()),
                "num_loops": int(_solver.graph.get_num_loops())}


@app.get("/map")
def get_map(after_submap: int = Query(-1), known_loops: int = Query(0),
            voxel: float = Query(0.0), max_points: int = Query(400000)):
    """Return poses + cloud. A loop closure re-optimises *all* poses, so if the
    loop count grew we flag ``full_refresh`` and return the whole map for the
    client to rebuild (and re-run its motion self-calibration); otherwise we
    return only submaps newer than ``after_submap``."""
    with data_lock:
        num_loops = int(_solver.graph.get_num_loops())
        num_submaps = int(_solver.map.get_num_submaps())
        all_submaps = list(_solver.map.ordered_submaps_by_key())
        full_refresh = num_loops > known_loops
        chosen = all_submaps if full_refresh else \
            [sm for sm in all_submaps if int(sm.get_id()) > after_submap]
        poses_out, cloud = _collect(chosen, voxel, max_points)
    # session_id is load-bearing for the client, not informational: submap ids
    # restart at 0 each run, so a client still holding after_submap=N from the
    # previous run would filter out the entire new map and see it as empty.
    # On a change, the client must reset its cursor and rebuild (§5 self-cal too).
    return {"session_id": _session_id,
            "full_refresh": full_refresh, "num_submaps": num_submaps,
            "num_loops": num_loops, "submaps": poses_out, "cloud": cloud}


@app.get("/status")
def status():
    """Health check — MUST be lock-free and instant.

    This deliberately reads the plain-int _n_submaps/_n_loops mirrors instead of
    calling into the solver under data_lock. Taking data_lock here was a real bug:
    a submap optimisation can hold it for tens of seconds, so every health poll
    blocked and the diagnostic viewer reported "slam_server unreachable
    (ReadTimeout)" while SLAM was in fact healthy and building submaps. A health
    endpoint that hangs exactly when the system is busiest is worse than useless.
    """
    # worker_started is one-shot (set once at thread launch) and stays True even
    # if the thread has since died — worker_alive is the real liveness signal:
    # False if slam_worker hasn't looped in >5s (it loops at least once/sec via
    # _camera.capture(timeout=1.0), so a stalled/dead thread shows up within 5s).
    alive = _worker_started.is_set() and (time.monotonic() - _worker_last_beat) < 5.0
    on_disk = _src_sha()
    return {"version": SERVER_VERSION,
            "src_sha": _SRC_SHA_AT_START,        # what is actually RUNNING
            "src_sha_on_disk": on_disk,          # what is in the file now
            "stale": on_disk != _SRC_SHA_AT_START and on_disk != "unknown",
            "pid": os.getpid(),
            "uptime_s": round(time.time() - _PROCESS_STARTED_AT, 1),
            "worker_started": _worker_started.is_set(), "worker_alive": alive,
            "worker_last_error": _worker_last_error,
            "session_id": _session_id, "session_label": _session_label,
            "past_runs": len(_past_runs),
            # pipeline stages — see the _frames_seen block for how to read these
            "frames_seen": _frames_seen,
            "keyframes_pending": _keyframes_pending,
            "keyframes_needed": _cfg.submap_size + _cfg.overlapping_window_size,
            "keyframes_total": _keyframes_total,
            "submaps_failed": _submaps_failed,
            "submap_last_error": _submap_last_error,
            "min_disparity": _cfg.min_disparity,
            "num_submaps": _n_submaps, "num_loops": _n_loops,
            "submap_in_flight": solver_lock.locked(),
            "map_lock_busy": data_lock.locked(),
            "camera": _camera.stats() if _camera else None}


def main() -> None:
    global _model, _solver, _camera, _session_started_at
    p = argparse.ArgumentParser(description="circe VGGT-SLAM host server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--submap_size", type=int, default=8)
    p.add_argument("--overlapping_window_size", type=int, default=1)
    p.add_argument("--max_loops", type=int, default=1)
    p.add_argument("--min_disparity", type=float, default=50.0)
    p.add_argument("--conf_threshold", type=float, default=25.0)
    p.add_argument("--lc_thres", type=float, default=0.95)
    p.add_argument("--vis_voxel_size", type=float, default=None)
    p.add_argument("--keyframe_folder", default="slam_host_keyframes")
    p.add_argument("--log_dir", default="slam_server_logs")
    args = p.parse_args()
    _cfg.__dict__.update(vars(args))
    # Never runs in production — this is a dev/research server. Every diagnostic
    # feature is unconditionally on, always, no opt-out: viser raw-map viewer,
    # the built-in Gradio ground-truth viewer, DEBUG logging, FastAPI tracebacks.
    _cfg.vis_map = True
    GRADIO_PORT = 7861

    _setup_logging(args.log_dir, debug=True)
    app.debug = True

    log.info("slam_server v%s (src_sha=%s pid=%d) — /status reports `stale` if this "
             "file changes after launch", SERVER_VERSION, _SRC_SHA_AT_START, os.getpid())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s; loading VGGT...", device)
    _model = load_model(device)
    _solver = Solver(init_conf_threshold=args.conf_threshold, lc_thres=args.lc_thres,
                     vis_voxel_size=args.vis_voxel_size)
    _camera = NetworkCamera()
    _session_started_at = time.time()   # real run-0 start, not module-import time
    log.info("model + solver ready (session %d).", _session_id)

    threading.Thread(target=slam_worker, daemon=True).start()
    _worker_started.wait(timeout=10)

    # The diagnostic Gradio UI lives in-process here (not a separate script to
    # remember to launch) — it polls this same server's own HTTP API on
    # localhost, so what it shows is ground truth off the wire, same as any
    # other client. prevent_thread_lock=True so it doesn't block uvicorn below.
    from diag_viewer import make_app
    gradio_demo = make_app(f"http://localhost:{args.port}", poll_period=2.0)
    gradio_demo.launch(server_port=GRADIO_PORT, prevent_thread_lock=True,
                        quiet=True, show_error=True)
    log.info("diagnostic Gradio viewer at http://localhost:%d", GRADIO_PORT)

    log.info("serving on %s:%d", args.host, args.port)
    # log_config=None is load-bearing, not cosmetic: uvicorn.run() otherwise calls
    # logging.config.dictConfig() with its own default config, which — per Python's
    # dictConfig(disable_existing_loggers=True) default — SILENCES this module's
    # pre-existing `log` logger the instant uvicorn starts. Confirmed by evidence:
    # the log file went dark right after "serving on..." with zero [frame N]
    # progress lines despite frames actively arriving. log_config=None skips that
    # dictConfig call entirely, so our root handlers (console+file) stay live and
    # uvicorn's own access/error loggers propagate up into the same file for free.
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug", log_config=None)


if __name__ == "__main__":
    main()
