"""diag_viewer — live incremental Gradio viewer for slam_server.py.

Rebuilt to match the richness of the two reference Gradio apps on this machine
(both read in full before writing this):
  - D:\\others_github\\VGGT-SLAM\\gradio_demo.py                — generator-yield-per-submap pattern
  - D:\\my_github\\circe\\my_slam_vggt_omega\\{main.py,pipeline.py,glb_builder.py,logger.py}
    + D:\\others_github\\vggt-omega\\visual_util.py             — twin viewers, cone cameras, ledger,
                                                                   past-runs, JSONL logging (richer)

What's reused verbatim vs. adapted (see the approved plan for the full table):
  - Camera-cone mesh + OpenGL scene alignment (_integrate_camera_into_scene / _get_opengl_conversion_matrix
    / _apply_scene_alignment / _transform_points / _compute_camera_faces below) — copied from
    visual_util.py essentially verbatim; pure geometry, no VGGT-Omega-specific dependency.
  - Packed-radio ledger, arrow-key JS, fullscreen JS, instrument-panel CSS, past-runs scan/load — adapted
    from main.py's build_ui/_pack/_unpack/_scan_past_runs/_load_run.
  - RunLogger — same JSONL + human `display` line shape as logger.py.
  - NOT reused: revisit/recent/new/cube role coloring — that's VGGT-Omega's cube-covisibility algorithm,
    which our backend (VGGT-SLAM's fixed-size submap loop) has no equivalent of. Our real distinction is
    new-submap / loop-closure-submap / older-accumulated, which gradio_demo.py's simpler model already
    matches — extended here with a lime "just-added-to-global-map" highlight (that idea IS reused from
    pipeline.py's `_build_global_map(highlight_frame_indices=...)`).

Unlike both references (batch tools that process a static uploaded folder and need a Start button), this
watches an already-running live server — there's nothing to click to start it. A gr.Timer polls
GET /status continuously; the moment num_submaps increases, a new ledger row is built automatically from
GET /map + the new GET /submap/{id}/frames endpoint (real keyframe images, not placeholders).

Run (WSL `vggt` env) — auto-launched by slam_server.py, or standalone:
  python diag_viewer.py --slam_url http://localhost:8000 --port 7861
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import requests
import trimesh
from scipy.spatial.transform import Rotation

_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_viewer_outputs")

# Camera-cone colors (RGB)
_COLOR_NEW = (60, 220, 60)        # green — this submap's own cameras (Latest Submap viewer)
_COLOR_LOOP = (230, 60, 60)       # red   — a loop-closure submap
_COLOR_OLDER = (140, 140, 140)    # gray  — older submaps in the Global Map viewer
_COLOR_HIGHLIGHT = (170, 255, 60) # lime  — just added/refreshed this poll (Global Map viewer)


# --------------------------------------------------------------------------- #
# Reused near-verbatim from D:\others_github\vggt-omega\visual_util.py —
# pure geometry, no dependency on VGGT-Omega's prediction-tensor shapes.
# --------------------------------------------------------------------------- #
def _get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def _transform_points(transformation: np.ndarray, points: np.ndarray, dim: int | None = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def _compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    faces = []
    num_vertices = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_o, v2_o, v3_o = face + num_vertices
        v1_o2, v2_o2, v3_o2 = face + 2 * num_vertices
        faces.extend([
            (v1, v2, v2_o), (v1, v1_o, v3), (v3_o, v2, v3),
            (v1, v2, v2_o2), (v1, v1_o2, v3), (v3_o2, v2, v3),
        ])
    faces += [(v3, v2, v1) for v1, v2, v3 in faces]
    return np.array(faces)


def _integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray,
                                  face_colors: tuple, scene_scale: float) -> None:
    """A solid colored cone mesh for one camera pose — richer than a flat wireframe."""
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45 = np.eye(4)
    rot_45[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45[2, 3] = -cam_height

    complete_transform = transform @ _get_opengl_conversion_matrix() @ rot_45
    cone = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rot = np.eye(4)
    slight_rot[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices = np.concatenate([
        cone.vertices, 0.95 * cone.vertices, _transform_points(slight_rot, cone.vertices),
    ])
    vertices = _transform_points(complete_transform, vertices)

    mesh = trimesh.Trimesh(vertices=vertices, faces=_compute_camera_faces(cone))
    mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(mesh)


def _apply_scene_alignment(scene: trimesh.Scene, first_pose_cam_to_world: np.ndarray) -> trimesh.Scene:
    world_to_cam0 = np.linalg.inv(first_pose_cam_to_world)
    scene.apply_transform(world_to_cam0 @ _get_opengl_conversion_matrix())
    return scene


# --------------------------------------------------------------------------- #
# JSONL run logger — same shape as my_slam_vggt_omega/logger.py.
# --------------------------------------------------------------------------- #
class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, "run_log.jsonl")
        os.makedirs(run_dir, exist_ok=True)

    def log_submap(self, entry: dict) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        loop = " [LOOP CLOSURE]" if entry.get("loop_closure") else ""
        msg = (f"[{ts}] [SUBMAP {entry['submap_id']:04d}]{loop} "
               f"frames={entry.get('n_frames', '?')} "
               f"total_submaps={entry.get('num_submaps')} total_loops={entry.get('num_loops')}")
        entry["ts"] = datetime.now().isoformat()
        entry["display"] = msg
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return msg

    def info(self, msg: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [INFO] {msg}"
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"info": msg, "display": line, "ts": datetime.now().isoformat()}) + "\n")
        return line


# --------------------------------------------------------------------------- #
# HTTP polling against slam_server.py's own API.
# --------------------------------------------------------------------------- #
def _get_json(url: str, timeout: float = 4.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# GET /status is lock-free and answers instantly — a short timeout there is a real
# health signal. GET /map still serialises the whole map under data_lock and can
# legitimately take tens of seconds on a large map, so it gets a much longer budget:
# with one shared 4s timeout, a merely-slow /map made the banner scream "server
# unreachable" while SLAM was perfectly healthy. Separate budgets, separate verdicts.
_STATUS_TIMEOUT = 4.0
_MAP_TIMEOUT = 60.0

# Hard ceiling on accumulated global-map points held in RAM. This viewer runs
# in-process with slam_server (which already holds VGGT ~5 GB + every submap's
# cloud), and the server was observed dying at 8.1 GB against WSL's 12 GB cap —
# so unbounded accumulation here is not a cosmetic concern.
_MAX_GLOBAL_POINTS = 600_000


def _get_frame_image(slam_url: str):
    try:
        r = requests.get(f"{slam_url}/frame/latest", timeout=3.0)
        if r.status_code != 200:
            return None
        buf = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img[:, :, ::-1] if img is not None else None  # BGR -> RGB
    except Exception:
        return None


def _decode_cloud(cloud: dict):
    n = cloud.get("n", 0)
    if n == 0:
        return None, None
    xyz = np.frombuffer(base64.b64decode(cloud["xyz_f32_b64"]), np.float32).reshape(n, 3).astype(np.float64)
    rgb = np.frombuffer(base64.b64decode(cloud["rgb_u8_b64"]), np.uint8).reshape(n, 3)
    return xyz, rgb


def _pack(a: str | None, b: str | None) -> str:
    return f"{a or ''}||{b or ''}"


def _unpack(value: str | None):
    if not value:
        return None, None
    parts = value.split("||", 1)
    a = parts[0] or None
    b = parts[1] if len(parts) > 1 and parts[1] else None
    return a, b


# --------------------------------------------------------------------------- #
# Live state — accumulated across polls (this run session).
# --------------------------------------------------------------------------- #
class _State:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.logger = RunLogger(run_dir)
        self.known_loops = 0
        self.last_submap_id = -1
        self.global_points: list[np.ndarray] = []
        self.global_colors: list[np.ndarray] = []
        self.global_cams: list[tuple[np.ndarray, tuple, bool]] = []  # (pose, base_color, is_newest)
        self.first_pose: np.ndarray | None = None
        self.ledger: list[tuple[str, str]] = []   # (label, packed_value)
        self.frame_strip: list[np.ndarray] = []
        self.seen_submap_ids: set[int] = set()
        self.last_global_glb: str | None = None
        self.tick = 0


def _scene_scale(xyz: np.ndarray) -> float:
    if xyz is None or len(xyz) < 2:
        return 1.0
    lo, hi = np.percentile(xyz, 5, axis=0), np.percentile(xyz, 95, axis=0)
    s = float(np.linalg.norm(hi - lo))
    return s if s > 0 else 1.0


def _submap_scene(xyz, rgb, poses_list, cam_colors, first_pose) -> trimesh.Scene | None:
    if xyz is None:
        return None
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=xyz, colors=rgb))
    scale = _scene_scale(xyz)
    for pose, color in zip(poses_list, cam_colors):
        _integrate_camera_into_scene(scene, pose, color, scale)
    if first_pose is not None:
        _apply_scene_alignment(scene, first_pose)
    return scene


def _fetch_submap_gallery(slam_url: str, submap_id: int):
    data, _ = _get_json(f"{slam_url}/submap/{submap_id}/frames")
    if not data or not data.get("available"):
        return []
    return [(p, f"submap {submap_id} keyframe") for p in data["paths"] if os.path.isfile(p)]


def _process_new_submaps(state: _State, slam_url: str, map_json: dict):
    """One ledger row per newly-seen submap_id in this poll's response."""
    new_rows = []
    is_full_refresh = bool(map_json.get("full_refresh"))
    xyz, rgb = _decode_cloud(map_json.get("cloud", {}))

    if is_full_refresh:
        # Loop closure re-optimised everything — the returned cloud supersedes
        # our accumulated history rather than appending onto it.
        state.global_points, state.global_colors, state.global_cams = [], [], []
        if xyz is not None:
            state.global_points.append(xyz)
            state.global_colors.append(rgb)
    elif xyz is not None:
        state.global_points.append(xyz)
        state.global_colors.append(rgb)

    # Fade last poll's "just added" highlight to older before this poll's own
    # cams get appended below as the new highlight.
    state.global_cams = [(p, c, False) for p, c, _ in state.global_cams]

    submaps = map_json.get("submaps", [])
    for sm in submaps:
        sid = int(sm["submap_id"])
        poses = np.array(sm["poses"]) if sm["poses"] else np.zeros((0, 4, 4))
        loop = bool(sm.get("loop_closure"))
        base_color = _COLOR_LOOP if loop else _COLOR_HIGHLIGHT
        for p in poses:
            state.global_cams.append((p, base_color, True))
        if state.first_pose is None and len(poses):
            state.first_pose = poses[0]
        if sid > state.last_submap_id:
            state.last_submap_id = sid

        # A full_refresh (loop closure) re-sends EVERY submap, not just new
        # ones — skip re-creating a ledger row/gallery/log entry for a submap
        # already recorded; its points/cameras above still get folded into the
        # (just-reset) accumulated state, since that data is authoritative.
        if sid in state.seen_submap_ids:
            continue
        state.seen_submap_ids.add(sid)

        # This submap's own scene (Latest Submap viewer) — if >1 submap arrived
        # in one poll, they share the merged cloud this poll returned (the /map
        # contract doesn't split cloud per-submap); each still gets its own
        # labeled ledger row and its own real keyframe gallery.
        sm_color = _COLOR_LOOP if loop else _COLOR_NEW
        submap_scene = _submap_scene(xyz, rgb, poses, [sm_color] * len(poses), poses[0] if len(poses) else None)
        submap_glb = os.path.join(state.run_dir, f"submap_{sid:04d}.glb")
        if submap_scene is not None:
            submap_scene.export(submap_glb)
        else:
            submap_glb = None

        n_frames = len(poses)
        label = f"submap {sid:04d}  frames={n_frames}" + ("  [LOOP]" if loop else "")
        gallery = _fetch_submap_gallery(slam_url, sid)
        if gallery and submap_glb:
            sidecar = submap_glb.replace(".glb", "_images.json")
            with open(sidecar, "w") as f:
                json.dump([{"path": p, "caption": c} for p, c in gallery], f)

        state.logger.log_submap({
            "submap_id": sid, "loop_closure": loop, "n_frames": n_frames,
            "num_submaps": map_json.get("num_submaps"), "num_loops": map_json.get("num_loops"),
            "submap_glb_path": submap_glb,
        })
        new_rows.append((label, sid, submap_glb, gallery))

    state.known_loops = map_json.get("num_loops", state.known_loops)
    return new_rows


def _build_global_scene(state: _State) -> str | None:
    if not state.global_points:
        return None
    scene = trimesh.Scene()
    xyz = np.concatenate(state.global_points, axis=0)
    rgb = np.concatenate(state.global_colors, axis=0)
    if len(xyz) > _MAX_GLOBAL_POINTS:
        idx = np.linspace(0, len(xyz) - 1, _MAX_GLOBAL_POINTS).astype(int)
        xyz, rgb = xyz[idx], rgb[idx]
        # Collapse the accumulated per-poll chunks down to this one decimated
        # array. Without this the raw lists keep growing for the life of the
        # process — and this viewer runs IN-PROCESS with slam_server, which was
        # observed dying at 8.1 GB of WSL's 12 GB ceiling. Decimating only the
        # render while retaining every raw chunk in memory would be a slow leak.
        state.global_points = [xyz]
        state.global_colors = [rgb]
    scene.add_geometry(trimesh.PointCloud(vertices=xyz, colors=rgb))
    scale = _scene_scale(xyz)
    for pose, base_color, is_newest in state.global_cams:
        color = base_color if is_newest else _COLOR_OLDER
        _integrate_camera_into_scene(scene, pose, color, scale)
    if state.first_pose is not None:
        _apply_scene_alignment(scene, state.first_pose)
    out_path = os.path.join(state.run_dir, f"global_tick_{state.tick:05d}.glb")
    scene.export(out_path)
    state.last_global_glb = out_path
    return out_path


def _status_banner(status: dict | None, err: str | None, map_err: str | None = None) -> str:
    if err is not None:
        return f"## 🔴 slam_server unreachable — {err}"
    alive = status.get("worker_alive")
    dot = "🟢" if alive else "🔴"
    cam = status.get("camera") or {}
    last_err = status.get("worker_last_error")
    busy = status.get("submap_in_flight") or status.get("map_lock_busy")
    lines = [
        f"## {dot} worker_alive={alive}  |  **submaps={status.get('num_submaps')}  "
        f"loops={status.get('num_loops')}**"
        + ("  ⏳ *submap in flight*" if busy else ""),
        f"camera: received={cam.get('received')} dropped={cam.get('dropped')} "
        f"queued={cam.get('queued')}",
    ]
    if last_err:
        lines.append(f"🔴 **worker_last_error:** `{last_err}`")
    if map_err:
        # /status answered, so the server is alive — only the (expensive) map
        # fetch failed. Say that precisely instead of implying the server is down.
        lines.append(f"⚠️ map fetch failed this tick (server is alive): `{map_err}`")
    return "  \n".join(lines)


# --------------------------------------------------------------------------- #
# Past runs — adapted from main.py's _scan_past_runs / _load_run.
# --------------------------------------------------------------------------- #
def _scan_past_runs() -> list:
    dirs = [d for d in glob.glob(os.path.join(_OUTPUTS_DIR, "run_*")) if os.path.isdir(d)]
    dirs.sort(reverse=True)
    return [(os.path.basename(d), d) for d in dirs]


def _load_run(run_dir: str):
    if not run_dir or not os.path.isdir(run_dir):
        return "No run selected.", gr.update(choices=[], value=None), None, None, []

    log_path = os.path.join(run_dir, "run_log.jsonl")
    ledger, log_lines = [], []
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "display" in entry:
                    log_lines.append(entry["display"])
                if "submap_id" in entry:
                    sid = entry["submap_id"]
                    label = f"submap {sid:04d}  frames={entry.get('n_frames', '?')}" + \
                            ("  [LOOP]" if entry.get("loop_closure") else "")
                    glb = entry.get("submap_glb_path")
                    global_glbs = sorted(glob.glob(os.path.join(run_dir, "global_tick_*.glb")))
                    global_glb = global_glbs[-1] if global_glbs else None
                    ledger.append((label, _pack(glb, global_glb)))

    last_value = ledger[-1][1] if ledger else None
    submap_glb, global_glb = _unpack(last_value)
    log_text = "\n".join(log_lines[-100:]) if log_lines else f"(no run_log.jsonl in {run_dir})"
    return (log_text, gr.update(choices=ledger, value=last_value),
            submap_glb if submap_glb and os.path.isfile(submap_glb) else None,
            global_glb if global_glb and os.path.isfile(global_glb) else None,
            [])


def _load_ledger_row(value: str):
    submap_glb, global_glb = _unpack(value)
    gallery = []
    if submap_glb:
        sidecar = submap_glb.replace(".glb", "_images.json")
        if os.path.isfile(sidecar):
            with open(sidecar) as f:
                entries = json.load(f)
            gallery = [(e["path"], e.get("caption", "")) for e in entries if os.path.isfile(e["path"])]
    submap_view = submap_glb if submap_glb and os.path.isfile(submap_glb) else None
    global_view = global_glb if global_glb and os.path.isfile(global_glb) else None
    return submap_view, global_view, gallery


# --------------------------------------------------------------------------- #
# UI — instrument-panel layout adapted from my_slam_vggt_omega/main.py.
# --------------------------------------------------------------------------- #
_MONO = "ui-monospace, 'Cascadia Mono', Consolas, 'Courier New', monospace"

_CSS = f"""
.gradio-container {{ max-width: 100% !important; padding: 6px 14px !important; }}
#hdr {{ display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap;
    padding: 2px 2px 6px 2px; border-bottom: 1px solid rgba(128,128,128,.25); margin-bottom: 6px; }}
#hdr .t {{ font-size: 17px; font-weight: 700; letter-spacing: .04em; }}
#hdr .chip {{ font-family: {_MONO}; font-size: 12px; opacity: .9; }}
#hdr .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px;
    margin-right: 5px; vertical-align: baseline; }}
#log-box textarea {{ font-family: {_MONO}; font-size: 11.5px; line-height: 1.5; }}
#model3d_viewer:fullscreen, #global_map_viewer:fullscreen {{
    width: 100vw !important; height: 100vh !important; background: #000; }}
.glb-list .wrap {{ display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important;
    gap: 3px !important; max-height: 260px; overflow-y: auto; padding-right: 6px; }}
.glb-list .wrap label {{ width: 100% !important; border: 1px solid rgba(128,128,128,.22) !important;
    border-left: 4px solid rgba(128,128,128,.35) !important; border-radius: 5px !important;
    padding: 5px 12px !important; font-family: {_MONO}; font-size: 12.5px; }}
.glb-list .wrap label:hover {{ border-left-color: #4c8dff !important; }}
.glb-list .wrap label.selected, .glb-list .wrap label:has(input:checked) {{
    border-left-color: #4c8dff !important; background: rgba(76, 141, 255, .12) !important; }}
.glb-list .wrap label:has(input[value*="LOOP"]) {{ border-left-color: #e63c3c !important; font-weight: 700; }}
"""

_ARROW_KEY_JS = """
() => {
    const list = document.querySelector('.glb-list');
    if (!list) return;
    let hovering = false;
    list.addEventListener('mouseenter', () => { hovering = true; });
    list.addEventListener('mouseleave', () => { hovering = false; });
    document.addEventListener('keydown', (e) => {
        if (!hovering) return;
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
        e.preventDefault();
        const inputs = Array.from(list.querySelectorAll('input[type=radio]'));
        if (inputs.length === 0) return;
        let idx = inputs.findIndex((i) => i.checked);
        if (idx === -1) idx = 0;
        else if (e.key === 'ArrowDown') idx = Math.min(idx + 1, inputs.length - 1);
        else idx = Math.max(idx - 1, 0);
        inputs[idx].click();
        inputs[idx].scrollIntoView({ block: 'nearest' });
    });
}
"""


def _fullscreen_js(elem_id: str) -> str:
    return f"""
    () => {{
        const el = document.getElementById('{elem_id}');
        if (el) (el.requestFullscreen || el.webkitRequestFullscreen)?.call(el);
    }}
    """


def make_app(slam_url: str, poll_period: float) -> gr.Blocks:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_OUTPUTS_DIR, f"run_{ts}")
    state = _State(run_dir)

    with gr.Blocks(title="circe SLAM diagnostic viewer", theme=gr.themes.Soft(), css=_CSS) as demo:
        gr.HTML(
            "<div id='hdr'>"
            "<span class='t'>circe SLAM · LIVE DIAGNOSTIC VIEWER</span>"
            f"<span class='chip'>polling <code>{slam_url}</code> every {poll_period:.0f}s</span>"
            "<span class='chip'><span class='dot' style='background:#3cdc3c'></span>new submap</span>"
            "<span class='chip'><span class='dot' style='background:#e63c3c'></span>loop closure</span>"
            "<span class='chip'><span class='dot' style='background:#aaff3c'></span>just added (global)</span>"
            "<span class='chip'><span class='dot' style='background:#8c8c8c'></span>older</span>"
            "</div>"
        )
        banner = gr.Markdown("connecting...")

        with gr.Row():
            past_run_dropdown = gr.Dropdown(label="Past Runs", choices=_scan_past_runs(), value=None, scale=3)
            refresh_runs_btn = gr.Button("↻ Refresh", scale=0)
            load_run_btn = gr.Button("Load Run", variant="secondary", scale=0)

        with gr.Row():
            with gr.Column(scale=0, min_width=280):
                frame_img = gr.Image(label="Live incoming frame", type="numpy", height=180)
                frame_strip = gr.Gallery(label="Recent frames", columns=3, height=180,
                                          object_fit="contain", preview=True)
                raw_status = gr.JSON(label="GET /status (raw)")

            with gr.Column(scale=1):
                submap_gallery = gr.Gallery(label="Keyframes in selected submap", columns=8, rows=1,
                                             height=180, object_fit="contain", elem_id="submap-gallery")
                with gr.Row(equal_height=True):
                    model3d = gr.Model3D(label="Latest Submap", height=560, elem_id="model3d_viewer")
                    global_map_viewer = gr.Model3D(label="Global Map (lime = just added)", height=560,
                                                    elem_id="global_map_viewer")
                with gr.Row():
                    fullscreen_btn = gr.Button("⛶ Fullscreen Submap", scale=0)
                    fullscreen_global_btn = gr.Button("⛶ Fullscreen Global Map", scale=0)
                    gr.Markdown("**Submap ledger** — click a row (or hover + ↑/↓) to browse")
                glb_radio = gr.Radio(label="", choices=[], value=None, interactive=True,
                                      elem_classes=["glb-list"])
                log_box = gr.Textbox(label="Submap Log", interactive=False, lines=10, max_lines=10,
                                      elem_id="log-box")

        def refresh():
            state.tick += 1
            status, status_err = _get_json(f"{slam_url}/status", timeout=_STATUS_TIMEOUT)
            map_json, map_err = _get_json(
                f"{slam_url}/map?after_submap={state.last_submap_id}"
                f"&known_loops={state.known_loops}&voxel=0.02&max_points=200000",
                timeout=_MAP_TIMEOUT)

            frame = _get_frame_image(slam_url)
            if frame is not None:
                state.frame_strip.append(frame)
                state.frame_strip = state.frame_strip[-30:]

            # Default to "leave unchanged" (gr.update()), not None — most polls
            # find no new submap, and None would clear these back to empty every
            # idle tick instead of keeping the last real data on screen.
            new_rows, submap_view, gallery_view = [], gr.update(), gr.update()
            if map_json is not None and map_json.get("submaps"):
                new_rows = _process_new_submaps(state, slam_url, map_json)
                if new_rows:
                    # One global-map export covers every row this poll produced —
                    # each row still gets its own submap GLB + gallery, just the
                    # "current global map" snapshot is shared for this tick.
                    global_glb = _build_global_scene(state)
                    for label, sid, glb, gal in new_rows:
                        state.ledger.append((label, _pack(glb, global_glb)))
                    submap_view = new_rows[-1][2]
                    gallery_view = new_rows[-1][3]

            global_view = state.last_global_glb
            radio_value = state.ledger[-1][1] if state.ledger else None
            log_tail = ""
            if os.path.isfile(state.logger.log_path):
                with open(state.logger.log_path) as f:
                    lines = [json.loads(l).get("display", "") for l in f if l.strip()]
                log_tail = "\n".join(lines[-40:])

            return (
                _status_banner(status, status_err, map_err), frame, list(reversed(state.frame_strip)),
                (status or {}), submap_view, global_view,
                gr.update(choices=state.ledger, value=radio_value), gallery_view,
                log_tail,
            )

        refresh_outputs = [banner, frame_img, frame_strip, raw_status, model3d, global_map_viewer,
                            glb_radio, submap_gallery, log_box]
        timer = gr.Timer(poll_period)
        timer.tick(refresh, outputs=refresh_outputs)
        demo.load(refresh, outputs=refresh_outputs)

        glb_radio.change(fn=_load_ledger_row, inputs=[glb_radio],
                          outputs=[model3d, global_map_viewer, submap_gallery], show_progress="hidden")

        refresh_runs_btn.click(fn=lambda: gr.update(choices=_scan_past_runs()), outputs=[past_run_dropdown])
        load_run_btn.click(fn=_load_run, inputs=[past_run_dropdown],
                            outputs=[log_box, glb_radio, model3d, global_map_viewer, submap_gallery])

        fullscreen_btn.click(fn=None, js=_fullscreen_js("model3d_viewer"))
        fullscreen_global_btn.click(fn=None, js=_fullscreen_js("global_map_viewer"))
        demo.load(fn=None, js=_ARROW_KEY_JS)

    demo.queue(default_concurrency_limit=10)
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Live incremental diagnostic viewer for slam_server.py")
    p.add_argument("--slam_url", default="http://localhost:8000")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--poll_period", type=float, default=2.0)
    args = p.parse_args()

    demo = make_app(args.slam_url, args.poll_period)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
