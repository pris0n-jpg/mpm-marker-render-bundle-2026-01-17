"""
Triplet runner: A/B/C apples-to-apples output directories.

Goal
----
Generate 3 comparable result directories under a single run root:
- A: legacy MPM RGB (from example/mpm_fem_rgb_compare.py, mpm_*.png)
- B: legacy FEM RGB (from example/mpm_fem_rgb_compare.py, fem_*.png)
- C: MPM → xensim RGB + marker (from intermediate height/uv → xensim SimMeshItem + MarkerTextureCamera)

Each directory contains:
- rgb_*.png
- (optional) marker_*.png
- run_manifest.json

Additionally, the run root contains a summary.md that lists directories and key parameters.

Run (PowerShell)
---------------
  conda run -n xengym python example/mpm_xensim_triplet_runner.py --lockfile example/mpm_xensim_baseline_lockfile.json

Preflight (PowerShell)
---------------------
  conda run -n xengym python example/preflight_mpm_xensim_render.py --calibrate-file xensim/examples/calib_table.npz
  # 若 preflight 失败，请先按输出提示修复 OpenGL/glfw/驱动/环境问题，再跑 triplet。

Notes
-----
- This script writes to output/ only; never commit output/.
- It reuses the intermediate arrays exported by mpm_fem_rgb_compare.py (height_field_mm, uv_disp_mm).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure repo root is importable even when running from outside the repo directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xengym.marker_appearance import resolve_marker_appearance_config  # noqa: E402

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def _run_cmd(cmd: List[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit={proc.returncode}): {' '.join(cmd)}")


def _git_head(repo_dir: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    head = (proc.stdout or "").strip()
    return head or None


def _resolve_repo_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def _load_lockfile(lockfile_path: Path) -> Dict[str, object]:
    # JSON lockfile: 固化 baseline 对比输入（calib_table/fem_file/preset/steps/record_interval/keyframes）。
    try:
        payload = json.loads(lockfile_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"failed to read lockfile: {lockfile_path}") from e
    if not isinstance(payload, dict):
        raise ValueError("lockfile must be a JSON object")
    return dict(payload)


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _parse_keep_frames_arg(value: str) -> Optional[Set[int]]:
    s = str(value).strip()
    if not s:
        return None
    if s.lower() == "all":
        return None
    out: Set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    if not out:
        raise ValueError("Invalid --keep-frames (expected 'all' or comma-separated integers)")
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _materialize_intermediate(raw_dir: Path, dst_dir: Path, *, keep_frames: Optional[Set[int]] = None) -> None:
    """
    Ensure `<dst_dir>/intermediate/frame_*.npz` exists for downstream analyze scripts.

    We prefer hardlinks (fast + no extra disk usage) and fall back to copying when unavailable.
    """
    src_int = raw_dir / "intermediate"
    if not src_int.exists():
        return
    dst_int = dst_dir / "intermediate"
    dst_int.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_int.glob("frame_*.npz")):
        try:
            frame_id = int(src.stem.split("_")[-1])
        except Exception:
            continue
        if keep_frames is not None and frame_id not in keep_frames:
            continue
        _link_or_copy(src, dst_int / src.name)


def _prune_indexed_files(dir_path: Path, pattern: str, *, keep_frames: Set[int]) -> None:
    if not dir_path.exists():
        return
    for p in sorted(dir_path.glob(pattern)):
        try:
            frame_id = int(p.stem.split("_")[-1])
        except Exception:
            continue
        if frame_id in keep_frames:
            continue
        try:
            p.unlink()
        except Exception:
            # Best-effort cleanup only; do not fail the whole triplet run.
            pass


def _prune_raw_dir(raw_dir: Path, *, keep_frames: Set[int]) -> None:
    _prune_indexed_files(raw_dir / "intermediate", "frame_*.npz", keep_frames=keep_frames)
    _prune_indexed_files(raw_dir, "mpm_*.png", keep_frames=keep_frames)
    _prune_indexed_files(raw_dir, "fem_*.png", keep_frames=keep_frames)


def _build_variant_manifest(
    raw_manifest: Dict[str, object],
    *,
    variant: str,
    raw_dir: Path,
    files: List[str],
    marker_files: Optional[List[str]] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Build a run_manifest.json compatible with existing rgb_compare analyze/eval scripts.

    Most tooling expects top-level keys like `run_context`, `scene_params`, `trajectory`, etc.
    For triplet outputs, we embed extra audit metadata under `triplet`.
    """
    manifest = dict(raw_manifest)
    triplet_meta: Dict[str, object] = {
        "variant": str(variant),
        "raw_dir": _posix(raw_dir),
        "files": list(files),
        "marker_files": list(marker_files) if marker_files else [],
    }
    if extra:
        triplet_meta.update(dict(extra))
    manifest["variant"] = str(variant)
    manifest["triplet"] = triplet_meta
    return manifest


def _infer_rgb_size_from_png(dir_path: Path, prefix: str) -> Tuple[int, int]:
    if cv2 is None:
        raise RuntimeError("cv2 is required to infer image size")
    cand = sorted(dir_path.glob(f"{prefix}_*.png"))
    if not cand:
        raise FileNotFoundError(f"no {prefix}_*.png found under {dir_path}")
    img = cv2.imread(str(cand[0]), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read image: {cand[0]}")
    h, w = int(img.shape[0]), int(img.shape[1])
    return w, h


def _copy_rgb_series(
    src_dir: Path,
    dst_dir: Path,
    *,
    src_prefix: str,
    keep_frames: Optional[Set[int]] = None,
) -> List[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_files: List[str] = []
    for src in sorted(src_dir.glob(f"{src_prefix}_*.png")):
        # Keep frame index audit-friendly by preserving the numeric suffix.
        stem = src.stem  # e.g. mpm_0000
        idx_str = stem.split("_")[-1]
        try:
            frame_id = int(idx_str)
        except Exception:
            continue
        if keep_frames is not None and frame_id not in keep_frames:
            continue
        dst = dst_dir / f"rgb_{idx_str}.png"
        _link_or_copy(src, dst)
        out_files.append(dst.name)
    return out_files


def _build_vertices_from_fields(
    *,
    height_field_mm: np.ndarray,
    uv_disp_mm: np.ndarray,
    gel_w_mm: float,
    gel_h_mm: float,
) -> np.ndarray:
    h = int(height_field_mm.shape[0])
    w = int(height_field_mm.shape[1])
    if uv_disp_mm.shape[:2] != (h, w) or uv_disp_mm.shape[2] != 2:
        raise ValueError("uv_disp_mm must be (H,W,2) and match height_field_mm")

    cell_w = float(gel_w_mm) / max(w, 1)
    cell_h = float(gel_h_mm) / max(h, 1)
    x_centers = (np.arange(w, dtype=np.float32) + 0.5) * np.float32(cell_w) - np.float32(gel_w_mm / 2.0)
    y_centers = (np.arange(h, dtype=np.float32) + 0.5) * np.float32(cell_h)
    xx, yy = np.meshgrid(x_centers, y_centers)

    v = np.empty((h, w, 3), dtype=np.float32)
    v[..., 0] = xx.astype(np.float32, copy=False) + uv_disp_mm[..., 0].astype(np.float32, copy=False)
    v[..., 1] = yy.astype(np.float32, copy=False) + uv_disp_mm[..., 1].astype(np.float32, copy=False)
    v[..., 2] = height_field_mm.astype(np.float32, copy=False)
    return v


def _np_percentile_safe(values: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return float("nan")
    try:
        return float(np.percentile(v, q))
    except Exception:
        return float("nan")


def _compute_motion_diagnostics_from_npz(npz: object) -> Dict[str, object]:
    """
    Compute lightweight diagnostics to debug coverage holes and pseudo-holes.

    Why:
    - 当 footprint 内出现 pseudo-hole（uv_cnt>0 但 |Δu| 近零）时，会表现为“压头下不动 / 前动后不动”。
    - 把这些信号在 C 输出侧落盘，方便直接定位是“渲染/坐标”还是“surface field”问题。
    """
    if not hasattr(npz, "__getitem__") or not hasattr(npz, "files"):
        return {}
    files = set(getattr(npz, "files", []) or [])

    def _get(name: str):
        return npz[name] if name in files else None

    footprint = _get("footprint_mask_u8")
    uv_cnt = _get("uv_cnt_i32")
    contact = _get("contact_mask")
    if footprint is None or uv_cnt is None:
        return {}

    footprint = np.asarray(footprint, dtype=np.uint8)
    uv_cnt = np.asarray(uv_cnt, dtype=np.int32)
    fp = footprint > 0
    fp_area = int(fp.sum())

    out: Dict[str, object] = {"footprint_area_px": int(fp_area)}
    if fp_area > 0:
        hole_area = int(((uv_cnt <= 0) & fp).sum())
        out["uv_hole_area_px"] = int(hole_area)
        out["uv_hole_ratio"] = float(hole_area) / float(fp_area)
    else:
        out["uv_hole_area_px"] = 0
        out["uv_hole_ratio"] = 0.0

    if "uv_hole_count_i32" in files:
        try:
            out["uv_hole_count"] = int(_get("uv_hole_count_i32"))
        except Exception:
            pass
    if "uv_pseudohole_count_i32" in files:
        try:
            out["uv_pseudohole_count"] = int(_get("uv_pseudohole_count_i32"))
        except Exception:
            pass

    if "uv_pseudohole_mask_u8" in files:
        m = np.asarray(_get("uv_pseudohole_mask_u8"), dtype=np.uint8) > 0
        ph = int((m & fp).sum())
        out["uv_pseudohole_area_px"] = int(ph)
        out["uv_pseudohole_ratio"] = float(ph) / float(fp_area) if fp_area > 0 else 0.0

    if "uv_du_mm" in files:
        du = np.asarray(_get("uv_du_mm"), dtype=np.float32)
        if du.ndim == 3 and du.shape[2] == 2 and fp_area > 0:
            mag = np.sqrt(np.square(du[..., 0]) + np.square(du[..., 1]))
            mag_fp = mag[fp]
            out["uv_du_abs_p50_mm"] = _np_percentile_safe(mag_fp, 50)
            out["uv_du_abs_p95_mm"] = _np_percentile_safe(mag_fp, 95)
            out["uv_du_abs_max_mm"] = float(np.nanmax(mag_fp)) if mag_fp.size else float("nan")

            if contact is not None:
                cm = np.asarray(contact, dtype=np.uint8) > 0
                mag_c = mag[cm]
                out["uv_du_abs_p50_mm_contact"] = _np_percentile_safe(mag_c, 50)
                out["uv_du_abs_p95_mm_contact"] = _np_percentile_safe(mag_c, 95)

    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_calib = repo_root / "xensim" / "examples" / "calib_table.npz"
    default_fem = repo_root / "xengym" / "assets" / "data" / "fem_data_gel_2035.npz"

    parser = argparse.ArgumentParser(description="Generate A/B/C triplet outputs for MPM vs FEM vs MPM→xensim")
    parser.add_argument("--out-root", type=str, default="output/mpm_xensim_triplet", help="Output root directory")
    parser.add_argument(
        "--reuse-run-dir",
        type=str,
        default="",
        help="Reuse an existing run_dir (must contain raw/run_manifest.json + raw/intermediate). "
        "When set, skip exporter and only (re)render C into the selected C subdir.",
    )
    parser.add_argument(
        "--c-out-subdir",
        type=str,
        default="",
        help="C output directory name under run_dir. "
        "Default: 'C' for new runs; for --reuse-run-dir defaults to 'C_<xensim_marker_mode>' (e.g. C_advect).",
    )
    parser.add_argument(
        "--lockfile",
        type=str,
        default="",
        help="Baseline lockfile JSON. When set, overrides calibrate/fem/preset/steps/record_interval for reproducibility.",
    )
    parser.add_argument("--calibrate-file", type=str, default=str(default_calib), help="xensim calib_table.npz")
    parser.add_argument("--fem-file", type=str, default=str(default_fem), help="FEM npz used by mpm_fem_rgb_compare")
    parser.add_argument("--steps", type=int, default=6, help="Steps (smoke default 6; publish keyframes usually need >=90)")
    parser.add_argument("--record-interval", type=int, default=1, help="Record every N steps")
    parser.add_argument(
        "--preset",
        type=str,
        default="publish_v1",
        help="mpm_fem_rgb_compare preset (recommended: publish_v1 for publish gate; override explicitly to reproduce legacy outputs)",
    )
    # Pass-through flags for legacy exporter (mpm_fem_rgb_compare). Keep defaults empty to avoid changing existing behavior.
    parser.add_argument(
        "--mpm-disable-light-file",
        choices=["on", "off"],
        default="",
        help="Pass-through: disable loading xengym/assets/data/light.txt for A/B (default: preset/default behavior).",
    )
    parser.add_argument(
        "--mpm-render-shadow",
        choices=["on", "off"],
        default="",
        help="Pass-through: enable/disable LineLight shadows for A/B (default: preset/default behavior).",
    )
    parser.add_argument(
        "--mpm-marker",
        choices=["off", "static", "warp", "advect"],
        default="",
        help="Pass-through: marker mode for A (default: preset/default behavior).",
    )
    parser.add_argument(
        "--fem-marker",
        choices=["on", "off"],
        default="",
        help="Pass-through: marker on/off for B (default: preset/default behavior).",
    )
    parser.add_argument(
        "--marker-appearance-mode",
        choices=["grid", "random_ellipses"],
        default="",
        help="Marker appearance (initial texture/dot set): grid|random_ellipses. "
        "Default: grid. When --reuse-run-dir is set, this applies to C only (A/B are not rerendered).",
    )
    parser.add_argument(
        "--marker-appearance-seed",
        type=str,
        default="",
        help="Marker appearance seed (int). Use -1 for random (resolved seed recorded in manifest). "
        "When --reuse-run-dir is set, this applies to C only (A/B are not rerendered).",
    )
    parser.add_argument(
        "--keep-frames",
        type=str,
        default="",
        help="Keep only selected frame ids in A/B/C + intermediate (comma-separated, or 'all'). "
        "Default: when lockfile provides keyframes, keep those; otherwise keep all.",
    )
    parser.add_argument(
        "--prune-raw",
        choices=["on", "off"],
        default="on",
        help="When --keep-frames is active, prune raw/ intermediate + raw images to kept frames (default: on).",
    )
    parser.add_argument(
        "--xensim-bg-mode",
        choices=["ref", "flat"],
        default="",
        help="xensim(C) background mode: `ref` uses calib_table ref texture; `flat` uses constant gray. "
        "Default: lockfile.xensim_bg_mode if present, otherwise `ref`.",
    )
    parser.add_argument(
        "--xensim-color-scale",
        type=str,
        default="",
        help="xensim(C) lighting scale (>=0). Set to 0 to disable lighting and render background-only (ablation). "
        "Default: lockfile.xensim_color_scale if present, otherwise 1.0.",
    )
    parser.add_argument(
        "--xensim-post-gamma",
        type=str,
        default="",
        help="xensim(C) post gamma correction (>0). Applied after rendering (ablation/alignment knob). "
        "Default: lockfile.xensim_post_gamma if present, otherwise 1.0.",
    )
    parser.add_argument(
        "--xensim-marker-mode",
        choices=["advect", "mesh_warp"],
        default="",
        help="xensim(C) marker mode: `advect` moves dot centers (discrete markers, better continuity); "
        "`mesh_warp` projects the marker texture onto the deformed mesh (legacy xensim). "
        "Default: lockfile.xensim_marker_mode if present, otherwise `advect`.",
    )
    parser.add_argument(
        "--xensim-marker",
        choices=["on", "off"],
        default="",
        help="xensim(C) marker overlay flag: `off` disables marker contribution in final RGB (ablation/debug). "
        "Default: `on`.",
    )
    parser.add_argument(
        "--xensim-indenter-overlay",
        choices=["on", "off"],
        default="",
        help="xensim(C) debug overlay for indenter/contact region (draws a red ellipse + center cross on top of RGB). "
        "Default: lockfile.xensim_indenter_overlay if present, otherwise `off`.",
    )
    parser.add_argument(
        "--xensim-render-mode",
        choices=["sim_mesh", "sensor_scene"],
        default="",
        help="xensim(C) renderer mode: `sim_mesh` uses SimMeshItem+calib_table (legacy); "
        "`sensor_scene` uses SensorScene/GLSurfMeshItem style (closer to FEM). "
        "Default: lockfile.xensim_render_mode if present, otherwise `sim_mesh`.",
    )
    args = parser.parse_args()

    lockfile_meta: Optional[Dict[str, object]] = None
    lockfile_keyframes: List[int] = []
    xensim_bg_mode = str(getattr(args, "xensim_bg_mode", "") or "").strip().lower()
    if not xensim_bg_mode:
        xensim_bg_mode = "ref"
    xensim_color_scale_arg = str(getattr(args, "xensim_color_scale", "") or "").strip()
    xensim_color_scale = float(xensim_color_scale_arg) if xensim_color_scale_arg else 1.0
    xensim_post_gamma_arg = str(getattr(args, "xensim_post_gamma", "") or "").strip()
    xensim_post_gamma = float(xensim_post_gamma_arg) if xensim_post_gamma_arg else 1.0
    xensim_marker_mode = str(getattr(args, "xensim_marker_mode", "") or "").strip().lower()
    if not xensim_marker_mode:
        xensim_marker_mode = "advect"
    xensim_marker_flag = str(getattr(args, "xensim_marker", "") or "").strip().lower()
    xensim_marker_enabled = True if not xensim_marker_flag else (xensim_marker_flag == "on")
    xensim_indenter_overlay_flag = str(getattr(args, "xensim_indenter_overlay", "") or "").strip().lower()
    xensim_indenter_overlay = False if not xensim_indenter_overlay_flag else (xensim_indenter_overlay_flag == "on")
    xensim_render_mode = str(getattr(args, "xensim_render_mode", "") or "").strip().lower()
    if not xensim_render_mode:
        xensim_render_mode = "sim_mesh"

    marker_appearance_mode_arg = str(getattr(args, "marker_appearance_mode", "") or "").strip().lower()
    marker_appearance_seed_arg = str(getattr(args, "marker_appearance_seed", "") or "").strip()
    marker_appearance_seed_i = None
    if marker_appearance_seed_arg:
        marker_appearance_seed_i = int(marker_appearance_seed_arg)
    marker_appearance_cfg = resolve_marker_appearance_config(
        mode=marker_appearance_mode_arg if marker_appearance_mode_arg else None,
        seed=marker_appearance_seed_i,
    )
    marker_appearance_manifest = {
        "mode": marker_appearance_cfg.mode,
        "seed": marker_appearance_cfg.seed,
        "config": {k: v for k, v in marker_appearance_cfg.to_manifest().items() if k not in {"mode", "seed"}},
    }
    if args.lockfile:
        lockfile_path = Path(args.lockfile)
        if not lockfile_path.is_absolute():
            lockfile_path = repo_root / lockfile_path
        lock = _load_lockfile(lockfile_path)

        lf_calib = lock.get("calibrate_file", None)
        lf_fem = lock.get("fem_file", None)
        lf_preset = lock.get("preset", None)
        lf_steps = lock.get("steps", None)
        lf_rec = lock.get("record_interval", None)
        lf_keyframes = lock.get("keyframes", [])
        lf_bg = lock.get("xensim_bg_mode", None)
        lf_color_scale = lock.get("xensim_color_scale", None)
        lf_post_gamma = lock.get("xensim_post_gamma", None)
        lf_render_mode = lock.get("xensim_render_mode", None)
        lf_marker_mode = lock.get("xensim_marker_mode", None)
        lf_indenter_overlay = lock.get("xensim_indenter_overlay", None)

        if isinstance(lf_calib, str) and lf_calib.strip():
            args.calibrate_file = lf_calib.strip()
        if isinstance(lf_fem, str) and lf_fem.strip():
            args.fem_file = lf_fem.strip()
        if isinstance(lf_preset, str) and lf_preset.strip():
            args.preset = lf_preset.strip()
        if lf_steps is not None:
            args.steps = int(lf_steps)
        if lf_rec is not None:
            args.record_interval = int(lf_rec)
        if str(getattr(args, "xensim_bg_mode", "") or "").strip() == "" and isinstance(lf_bg, str) and lf_bg.strip():
            xensim_bg_mode = str(lf_bg).strip().lower()
        if str(getattr(args, "xensim_color_scale", "") or "").strip() == "" and lf_color_scale is not None:
            try:
                xensim_color_scale = float(lf_color_scale)
            except Exception:
                pass
        if str(getattr(args, "xensim_post_gamma", "") or "").strip() == "" and lf_post_gamma is not None:
            try:
                xensim_post_gamma = float(lf_post_gamma)
            except Exception:
                pass
        if (
            str(getattr(args, "xensim_render_mode", "") or "").strip() == ""
            and isinstance(lf_render_mode, str)
            and lf_render_mode.strip()
        ):
            xensim_render_mode = str(lf_render_mode).strip().lower()
        if (
            str(getattr(args, "xensim_marker_mode", "") or "").strip() == ""
            and isinstance(lf_marker_mode, str)
            and lf_marker_mode.strip()
        ):
            xensim_marker_mode = str(lf_marker_mode).strip().lower()
        if str(getattr(args, "xensim_indenter_overlay", "") or "").strip() == "" and lf_indenter_overlay is not None:
            if isinstance(lf_indenter_overlay, bool):
                xensim_indenter_overlay = bool(lf_indenter_overlay)
            elif isinstance(lf_indenter_overlay, str) and lf_indenter_overlay.strip():
                xensim_indenter_overlay = str(lf_indenter_overlay).strip().lower() == "on"

        if isinstance(lf_keyframes, list):
            for x in lf_keyframes:
                try:
                    lockfile_keyframes.append(int(x))
                except Exception:
                    continue
            lockfile_keyframes = sorted(set(lockfile_keyframes))

        # Validate key paths exist (relative paths are resolved against repo_root).
        calib_p = _resolve_repo_path(repo_root, str(args.calibrate_file))
        fem_p = _resolve_repo_path(repo_root, str(args.fem_file))
        if not calib_p.exists():
            raise FileNotFoundError(f"lockfile calibrate_file not found: {args.calibrate_file}")
        if not fem_p.exists():
            raise FileNotFoundError(f"lockfile fem_file not found: {args.fem_file}")

        try:
            lockfile_rel = _posix(lockfile_path.resolve().relative_to(repo_root.resolve()))
        except Exception:
            lockfile_rel = _posix(lockfile_path)
        lockfile_meta = {
            "path": lockfile_rel,
            "calibrate_file": str(args.calibrate_file).replace("\\", "/"),
            "fem_file": str(args.fem_file).replace("\\", "/"),
            "preset": str(args.preset),
            "xensim_bg_mode": str(xensim_bg_mode),
            "xensim_color_scale": float(xensim_color_scale),
            "xensim_post_gamma": float(xensim_post_gamma),
            "xensim_render_mode": str(xensim_render_mode),
            "xensim_marker_mode": str(xensim_marker_mode),
            "xensim_indenter_overlay": bool(xensim_indenter_overlay),
            "steps": int(args.steps),
            "record_interval": int(args.record_interval),
            "keyframes": list(lockfile_keyframes),
            "expected": lock.get("expected", {}),
        }

    if xensim_bg_mode not in {"ref", "flat"}:
        raise ValueError("--xensim-bg-mode must be one of: ref, flat")
    if not np.isfinite(float(xensim_color_scale)) or float(xensim_color_scale) < 0.0:
        raise ValueError("--xensim-color-scale must be a finite number >= 0")
    if not np.isfinite(float(xensim_post_gamma)) or float(xensim_post_gamma) <= 0.0:
        raise ValueError("--xensim-post-gamma must be a finite number > 0")
    if xensim_render_mode not in {"sim_mesh", "sensor_scene"}:
        raise ValueError("--xensim-render-mode must be one of: sim_mesh, sensor_scene")
    if xensim_marker_mode not in {"advect", "mesh_warp"}:
        raise ValueError("--xensim-marker-mode must be one of: advect, mesh_warp")

    keep_frames_arg = str(getattr(args, "keep_frames", "") or "").strip()
    keep_frames_src = "all"
    keep_frames: Optional[Set[int]] = None
    if keep_frames_arg:
        keep_frames = _parse_keep_frames_arg(keep_frames_arg)
        keep_frames_src = f"cli:{keep_frames_arg}"
    elif lockfile_keyframes:
        keep_frames = set(int(x) for x in lockfile_keyframes)
        keep_frames_src = "lockfile.keyframes"
    prune_raw = (str(getattr(args, "prune_raw", "on")).lower().strip() == "on") and (keep_frames is not None)

    out_root = Path(args.out_root)
    reuse_run_dir_arg = str(getattr(args, "reuse_run_dir", "") or "").strip()
    reuse_run_dir: Optional[Path] = None
    if reuse_run_dir_arg:
        reuse_run_dir = _resolve_repo_path(repo_root, reuse_run_dir_arg)
        if not reuse_run_dir.exists():
            raise FileNotFoundError(f"--reuse-run-dir not found: {reuse_run_dir}")
        if prune_raw:
            raise ValueError("--prune-raw on is not allowed with --reuse-run-dir (would modify an existing run_dir)")
        if str(getattr(args, "mpm_disable_light_file", "") or "").strip() or str(getattr(args, "mpm_render_shadow", "") or "").strip():
            raise ValueError("--mpm-disable-light-file/--mpm-render-shadow are not supported with --reuse-run-dir (A/B are not rerendered)")
        if str(getattr(args, "mpm_marker", "") or "").strip() or str(getattr(args, "fem_marker", "") or "").strip():
            raise ValueError("--mpm-marker/--fem-marker are not supported with --reuse-run-dir (A/B are not rerendered)")

    # Keep naming simple & monotonic using filesystem timestamp (avoid locale issues).
    if reuse_run_dir is None:
        run_dir = out_root / f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = reuse_run_dir

    raw_dir = run_dir / "raw"
    if reuse_run_dir is not None and not raw_dir.exists():
        raise FileNotFoundError(f"missing raw/ under --reuse-run-dir: {raw_dir}")

    a_dir = run_dir / "A"
    b_dir = run_dir / "B"
    c_out_subdir = str(getattr(args, "c_out_subdir", "") or "").strip()
    if c_out_subdir:
        c_dir = run_dir / c_out_subdir
    else:
        if reuse_run_dir is None:
            c_dir = run_dir / "C"
        elif xensim_render_mode == "sensor_scene":
            c_dir = run_dir / "C_sensor_scene"
        else:
            c_dir = run_dir / f"C_{xensim_marker_mode}"

    cmd: Optional[List[str]] = None
    if reuse_run_dir is None:
        # 1) Run legacy exporter once (produces both mpm_*.png and fem_*.png + intermediate).
        cmd = [
            sys.executable,
            "example/mpm_fem_rgb_compare.py",
            "--mode",
            "raw",
            "--preset",
            str(args.preset),
            "--steps",
            str(int(args.steps)),
            "--record-interval",
            str(int(args.record_interval)),
            "--export-intermediate",
            "--export-intermediate-every",
            "1",
            "--save-dir",
            str(raw_dir),
            "--fem-file",
            str(args.fem_file),
        ]
        mpm_disable_light_file = str(getattr(args, "mpm_disable_light_file", "") or "").strip().lower()
        if mpm_disable_light_file:
            cmd += ["--mpm-disable-light-file", mpm_disable_light_file]
        mpm_render_shadow = str(getattr(args, "mpm_render_shadow", "") or "").strip().lower()
        if mpm_render_shadow:
            cmd += ["--mpm-render-shadow", mpm_render_shadow]
        mpm_marker = str(getattr(args, "mpm_marker", "") or "").strip().lower()
        if mpm_marker:
            cmd += ["--mpm-marker", mpm_marker]
        fem_marker = str(getattr(args, "fem_marker", "") or "").strip().lower()
        if fem_marker:
            cmd += ["--fem-marker", fem_marker]
        marker_appearance_mode = str(getattr(args, "marker_appearance_mode", "") or "").strip().lower()
        marker_appearance_seed = str(getattr(args, "marker_appearance_seed", "") or "").strip()
        if marker_appearance_mode or marker_appearance_seed:
            # Resolve seed=-1 once here so A/C can share the same concrete seed.
            cmd += ["--marker-appearance-mode", str(marker_appearance_cfg.mode)]
            if marker_appearance_cfg.seed is not None:
                cmd += ["--marker-appearance-seed", str(int(marker_appearance_cfg.seed))]
        _run_cmd(cmd, cwd=repo_root)

    raw_manifest_path = raw_dir / "run_manifest.json"
    if not raw_manifest_path.exists():
        raise FileNotFoundError(f"missing run_manifest.json: {raw_manifest_path}")
    raw_manifest = _read_json(raw_manifest_path)

    if reuse_run_dir is None:
        # 2) Split A/B directories (rename to rgb_*.png).
        a_files = _copy_rgb_series(raw_dir, a_dir, src_prefix="mpm", keep_frames=keep_frames)
        b_files = _copy_rgb_series(raw_dir, b_dir, src_prefix="fem", keep_frames=keep_frames)

        _materialize_intermediate(raw_dir, a_dir, keep_frames=keep_frames)
        _materialize_intermediate(raw_dir, b_dir, keep_frames=keep_frames)

        a_manifest = _build_variant_manifest(raw_manifest, variant="A_legacy_mpm", raw_dir=raw_dir, files=a_files)
        b_manifest = _build_variant_manifest(raw_manifest, variant="B_legacy_fem", raw_dir=raw_dir, files=b_files)
        if lockfile_meta is not None:
            a_manifest = _build_variant_manifest(
                raw_manifest,
                variant="A_legacy_mpm",
                raw_dir=raw_dir,
                files=a_files,
                extra={"lockfile": lockfile_meta},
            )
            b_manifest = _build_variant_manifest(
                raw_manifest,
                variant="B_legacy_fem",
                raw_dir=raw_dir,
                files=b_files,
                extra={"lockfile": lockfile_meta},
            )
        _write_json(a_dir / "run_manifest.json", a_manifest)
        _write_json(b_dir / "run_manifest.json", b_manifest)

    # 3) Render C using xensim pipeline driven by intermediate height/uv fields.
    # Import the renderer from MXR-050 implementation (keeps this runner compact).
    import importlib.util

    mod_path = repo_root / "example" / "mpm_xensim_render_adapter.py"
    module_name = "mpm_xensim_render_adapter_mod"
    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # required for dataclasses/type resolution
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    MPMXensimRenderer = getattr(mod, "MPMXensimRenderer")
    MPMSensorSceneRenderer = getattr(mod, "MPMSensorSceneRenderer", None)

    # Extract shared params for auditability.
    run_context = raw_manifest.get("run_context", {})
    resolved = (run_context.get("resolved", {}) if isinstance(run_context, dict) else {})
    scale = (resolved.get("scale", {}) if isinstance(resolved, dict) else {})
    gel_size_mm = scale.get("gel_size_mm", None)
    if not (isinstance(gel_size_mm, list) and len(gel_size_mm) == 2):
        raise ValueError("raw run_manifest.json missing run_context.scale.gel_size_mm")
    gel_w_mm, gel_h_mm = float(gel_size_mm[0]), float(gel_size_mm[1])
    conventions = (resolved.get("conventions", {}) if isinstance(resolved, dict) else {})
    if not isinstance(conventions, dict):
        conventions = {}
    warp_flip_x = bool(conventions.get("mpm_warp_flip_x", False))
    warp_flip_y = bool(conventions.get("mpm_warp_flip_y", True))

    indenter_radius_mm: Optional[float] = None
    indenter = (resolved.get("indenter", {}) if isinstance(resolved, dict) else {})
    if isinstance(indenter, dict):
        mpm_ind = indenter.get("mpm", {})
        if isinstance(mpm_ind, dict):
            size_mm = mpm_ind.get("size_mm", {})
            if isinstance(size_mm, dict):
                r_mm = size_mm.get("radius_mm", None)
                try:
                    if r_mm is not None:
                        indenter_radius_mm = float(r_mm)
                except Exception:
                    indenter_radius_mm = None

    marker_grid = (resolved.get("marker_grid", {}) if isinstance(resolved, dict) else {})
    tex_size_wh = marker_grid.get("tex_size_wh", [320, 560])
    if not (isinstance(tex_size_wh, list) and len(tex_size_wh) == 2):
        tex_size_wh = [320, 560]
    marker_tex_w, marker_tex_h = int(tex_size_wh[0]), int(tex_size_wh[1])

    rgb_w, rgb_h = _infer_rgb_size_from_png(raw_dir, prefix="mpm")
    intermediate_dir = raw_dir / "intermediate"
    frames_npz = sorted(intermediate_dir.glob("frame_*.npz"))
    if keep_frames is not None:
        filtered: List[Path] = []
        for p in frames_npz:
            try:
                frame_id = int(p.stem.split("_")[-1])
            except Exception:
                continue
            if frame_id in keep_frames:
                filtered.append(p)
        frames_npz = filtered
    if not frames_npz:
        raise FileNotFoundError(f"no intermediate frames found under {intermediate_dir}")

    # Determine mesh shape from the first available height_field_mm.
    first = np.load(frames_npz[0])
    if "height_field_mm" not in first.files or "uv_disp_mm" not in first.files:
        raise ValueError("intermediate frame missing height_field_mm/uv_disp_mm")
    mesh_shape = (int(first["height_field_mm"].shape[0]), int(first["height_field_mm"].shape[1]))
    first.close()

    if xensim_render_mode == "sensor_scene":
        if MPMSensorSceneRenderer is None:
            raise RuntimeError("MPMSensorSceneRenderer not available in mpm_xensim_render_adapter.py")
        renderer = MPMSensorSceneRenderer(
            gel_w_mm=gel_w_mm,
            gel_h_mm=gel_h_mm,
            mesh_shape=mesh_shape,
            marker_tex_size=(marker_tex_w, marker_tex_h),
            rgb_size=(rgb_w, rgb_h),
            marker_grid=(int(marker_grid.get("rows", 20)), int(marker_grid.get("cols", 11))),
            marker_radius_px=int(marker_grid.get("radius_px", 2)),
            marker_mode=str(xensim_marker_mode),
            marker_enabled=bool(xensim_marker_enabled),
            warp_flip_x=bool(warp_flip_x),
            warp_flip_y=bool(warp_flip_y),
            xensim_post_gamma=float(xensim_post_gamma),
            indenter_overlay=bool(xensim_indenter_overlay),
            indenter_radius_mm=indenter_radius_mm,
            visible=False,
        )
    else:
        renderer = MPMXensimRenderer(
            calibrate_file=str(args.calibrate_file),
            gel_w_mm=gel_w_mm,
            gel_h_mm=gel_h_mm,
            mesh_shape=mesh_shape,
            marker_tex_size=(marker_tex_w, marker_tex_h),
            rgb_size=(rgb_w, rgb_h),
            marker_grid=(int(marker_grid.get("rows", 20)), int(marker_grid.get("cols", 11))),
            marker_radius_px=int(marker_grid.get("radius_px", 2)),
            marker_mode=str(xensim_marker_mode),
            marker_enabled=bool(xensim_marker_enabled),
            marker_appearance_mode=str(marker_appearance_cfg.mode),
            marker_appearance_seed=marker_appearance_cfg.seed,
            warp_flip_x=bool(warp_flip_x),
            warp_flip_y=bool(warp_flip_y),
            xensim_bg_mode=str(xensim_bg_mode),
            xensim_color_scale=float(xensim_color_scale),
            xensim_post_gamma=float(xensim_post_gamma),
            indenter_overlay=bool(xensim_indenter_overlay),
            indenter_radius_mm=indenter_radius_mm,
            visible=False,
        )

    c_dir.mkdir(parents=True, exist_ok=True)
    c_files: List[str] = []
    c_marker: List[str] = []
    c_diag_frames: List[Dict[str, object]] = []
    for npz_path in frames_npz:
        frame_id = npz_path.stem.split("_")[-1]  # frame_0000xx
        data = np.load(npz_path)
        if "height_field_mm" not in data.files or "uv_disp_mm" not in data.files:
            continue
        height = data["height_field_mm"].astype(np.float32, copy=False)
        uv = data["uv_disp_mm"].astype(np.float32, copy=False)
        uv_du = data["uv_du_mm"].astype(np.float32, copy=False) if "uv_du_mm" in data.files else None

        # Per-frame diagnostics are computed from intermediate signals (uv_cnt/hole/pseudohole/Δu stats).
        try:
            diag = _compute_motion_diagnostics_from_npz(data)
            diag["frame"] = int(frame_id)
            c_diag_frames.append(diag)
        except Exception:
            pass
        data.close()

        if xensim_render_mode == "sensor_scene":
            rgb_u8, marker_u8 = renderer.render_frame(height_field_mm=height, uv_disp_mm=uv, uv_du_mm=uv_du)
        else:
            v = _build_vertices_from_fields(height_field_mm=height, uv_disp_mm=uv, gel_w_mm=gel_w_mm, gel_h_mm=gel_h_mm)
            rgb_u8, marker_u8 = renderer.render_frame(v, uv_disp_mm=uv, uv_du_mm=uv_du)

        if cv2 is not None:
            rgb_path = c_dir / f"rgb_{frame_id}.png"
            marker_path = c_dir / f"marker_{frame_id}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(marker_path), marker_u8)
            c_files.append(rgb_path.name)
            c_marker.append(marker_path.name)

    _materialize_intermediate(raw_dir, c_dir, keep_frames=keep_frames)

    # Per-run diagnostics summary (cheap; helps debug frozen-footprint failures).
    c_diag_summary: Dict[str, object] = {}
    if c_diag_frames:
        try:
            ph = [float(d.get("uv_pseudohole_ratio", 0.0)) for d in c_diag_frames if d.get("uv_pseudohole_ratio") is not None]
            holes = [float(d.get("uv_hole_ratio", 0.0)) for d in c_diag_frames if d.get("uv_hole_ratio") is not None]
            c_diag_summary = {
                "n_frames": int(len(c_diag_frames)),
                "max_uv_pseudohole_ratio": float(max(ph)) if ph else 0.0,
                "max_uv_hole_ratio": float(max(holes)) if holes else 0.0,
            }
        except Exception:
            c_diag_summary = {"n_frames": int(len(c_diag_frames))}
    diag_path = c_dir / "motion_diagnostics.json"
    _write_json(diag_path, {"summary": c_diag_summary, "frames": c_diag_frames})

    c_manifest = _build_variant_manifest(
        raw_manifest,
        variant="C_mpm_to_xensim_sensor_scene" if xensim_render_mode == "sensor_scene" else "C_mpm_to_xensim",
        raw_dir=raw_dir,
        files=c_files,
        marker_files=c_marker,
        extra={
            "calibrate_file": str(args.calibrate_file).replace("\\", "/"),
            "mesh_shape": list(mesh_shape),
            "rgb_size": [int(rgb_w), int(rgb_h)],
            "marker_tex_size": [int(marker_tex_w), int(marker_tex_h)],
            "xensim_bg_mode": str(xensim_bg_mode),
            "xensim_color_scale": float(xensim_color_scale),
            "xensim_post_gamma": float(xensim_post_gamma),
            "xensim_render_mode": str(xensim_render_mode),
            "xensim_marker_mode": str(xensim_marker_mode),
            "xensim_marker_enabled": bool(xensim_marker_enabled),
            "xensim_indenter_overlay": bool(xensim_indenter_overlay),
            "indenter_radius_mm": indenter_radius_mm,
            "marker_appearance": marker_appearance_manifest,
            "warp_flip_x": bool(warp_flip_x),
            "warp_flip_y": bool(warp_flip_y),
            "motion_diagnostics": {"file": diag_path.name, "summary": c_diag_summary},
            # Resolved conventions/scale for auditability (A/C alignment debugging).
            "resolved_conventions": {
                "uv_disp_mm_axes": {"u": "+x (right)", "v": "+y (up)"},
                "image_axes": {"x": "+col (right)", "y": "+row (down)"},
                # Intermediate exports are in solver/grid orientation (no render-layer flips applied).
                "intermediate_height_field_flip_x": False,
                "intermediate_uv_disp_flip_x": False,
                # C uses positive indentation zmap for stability (>=0).
                "zmap_convention": "indentation (zmap_mm=clip(-height_field_mm,0,inf))",
                # Marker texture mapping uses gel_size_mm (mm→px), then applies warp_flip_x/y.
                "marker_mm_to_px_source": "gel_size_mm",
                "marker_mm_to_px_formula": {
                    "dx_px": "(u_mm/gel_w_mm)*tex_w",
                    "dy_px": "(v_mm/gel_h_mm)*tex_h",
                    "flip_x": bool(warp_flip_x),
                    "flip_y": bool(warp_flip_y),
                },
                # Mesh coordinate ranges differ by renderer mode; recording helps spot y-origin shifts.
                "mesh_xy_range_mm": (
                    {
                        "x_range": [-float(gel_w_mm) / 2.0, float(gel_w_mm) / 2.0],
                        "y_range": [-float(gel_h_mm) / 2.0, float(gel_h_mm) / 2.0],
                    }
                    if xensim_render_mode != "sensor_scene"
                    else {
                        "x_range": [float(gel_w_mm) / 2.0, -float(gel_w_mm) / 2.0],
                        "y_range": [float(gel_h_mm), 0.0],
                    }
                ),
            },
            "resolved_scale": {
                "gel_size_mm": [float(gel_w_mm), float(gel_h_mm)],
                "rgb_size_wh": [int(rgb_w), int(rgb_h)],
                "marker_tex_size_wh": [int(marker_tex_w), int(marker_tex_h)],
                "marker_tex_px_per_mm": [
                    float(marker_tex_w) / max(float(gel_w_mm), 1e-6),
                    float(marker_tex_h) / max(float(gel_h_mm), 1e-6),
                ],
                "rgb_px_per_mm": [
                    float(rgb_w) / max(float(gel_w_mm), 1e-6),
                    float(rgb_h) / max(float(gel_h_mm), 1e-6),
                ],
            },
            "warnings": (
                []
                if not isinstance(scale, dict)
                else (
                    [f"gel_size_mm != cam_view_mm (delta_mm={scale.get('delta_mm', None)}); avoid mixing mm→px sources"]
                    if not bool(scale.get("consistent", True))
                    else []
                )
            ),
            "xensim_git_head": _git_head(repo_root / "xensim"),
            "lockfile": lockfile_meta,
            "renderer_imports": getattr(renderer, "imports", {}),
        },
    )
    _write_json(c_dir / "run_manifest.json", c_manifest)

    if prune_raw and keep_frames is not None:
        _prune_raw_dir(raw_dir, keep_frames=keep_frames)

    if cmd is not None:
        # 4) Summary markdown (only for freshly-generated runs).
        calibrate_file_str = str(args.calibrate_file).replace("\\", "/")
        fem_file_str = str(args.fem_file).replace("\\", "/")
        xensim_head = _git_head(repo_root / "xensim")
        summary = run_dir / "summary.md"
        summary.write_text(
            "\n".join(
                [
                    "# Triplet Summary",
                    "",
                    f"- raw: `{_posix(raw_dir)}`",
                    f"- A (legacy MPM): `{_posix(a_dir)}`",
                    f"- B (legacy FEM): `{_posix(b_dir)}`",
                    f"- C (MPM→xensim): `{_posix(c_dir)}`",
                    "",
                    "## Key Params",
                    f"- calibrate_file: `{calibrate_file_str}`",
                    f"- fem_file: `{fem_file_str}`",
                    f"- steps: {int(args.steps)}",
                    f"- record_interval: {int(args.record_interval)}",
                    f"- lockfile: `{lockfile_meta.get('path')}`"
                    if lockfile_meta and isinstance(lockfile_meta.get("path"), str)
                    else "- lockfile: (none)",
                    f"- keyframes: {lockfile_keyframes}" if lockfile_keyframes else "- keyframes: (auto from frames)",
                    f"- keep_frames: {sorted(keep_frames) if keep_frames is not None else 'all'} ({keep_frames_src}; prune_raw={'on' if prune_raw else 'off'})",
                    f"- xensim_bg_mode: {xensim_bg_mode}",
                    f"- xensim_marker_mode: {xensim_marker_mode}",
                    f"- marker_appearance: {marker_appearance_cfg.mode} (seed={marker_appearance_cfg.seed})",
                    f"- xensim_indenter_overlay: {'on' if xensim_indenter_overlay else 'off'}",
                    f"- gel_size_mm: [{gel_w_mm}, {gel_h_mm}]",
                    f"- rgb_size: [{rgb_w}, {rgb_h}]",
                    f"- marker_tex_size: [{marker_tex_w}, {marker_tex_h}]",
                    f"- xensim_git_head: {xensim_head}" if xensim_head else "- xensim_git_head: (unavailable)",
                    "",
                    "## Commands",
                    "```bash",
                    " ".join(cmd),
                    "```",
                ]
            ),
            encoding="utf-8",
        )

    print(f"[mpm_xensim_triplet_runner] run_dir={_posix(run_dir)} c_dir={_posix(c_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
