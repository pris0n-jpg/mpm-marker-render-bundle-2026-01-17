"""
MPM vs FEM Sensor RGB Comparison

This script compares MPM and FEM simulations using the visuotactile sensor
RGB rendering approach. Both methods render to the same sensor image format
for direct visual comparison.

Features:
- FEM path: Uses existing VecTouchSim rendering pipeline
- MPM path: Extracts top-surface height field and renders with matching style
- Side-by-side visualization with raw/diff modes
- Configurable press + slide trajectory

Usage:
    python example/mpm_fem_rgb_compare.py --mode raw
    python example/mpm_fem_rgb_compare.py --mode diff --press-mm 1.5
    python example/mpm_fem_rgb_compare.py --save-dir output/rgb_compare

    # Recommended baseline (stable, auditable output):
    # NOTE: When running batch mode (--save-dir without --interactive), the script applies
    # quality defaults (dx=0.2mm, gap=0mm) unless explicitly overridden (or disabled).
    python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/baseline

Output (when --save-dir is set):
    - fem_XXXX.png / mpm_XXXX.png
    - run_manifest.json (resolved params + frame->phase mapping)

Requirements:
    - xengym conda environment with Taichi installed
    - FEM data file (default: assets/data/fem_data_gel_2035.npz)
"""
import argparse
import math
import csv
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence
import time
import sys
import ast
import re
import datetime
import struct
import platform
import subprocess

# Add project root to path for standalone execution
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Repo-local xensesdk lives under ./xensesdk/xensesdk, so we also need the outer
# ./xensesdk on sys.path for `import xensesdk.ezgl` to work.
_XENSESDK_ROOT = _PROJECT_ROOT / "xensesdk"
if _XENSESDK_ROOT.exists() and str(_XENSESDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_XENSESDK_ROOT))

# Project imports
try:
    from xengym import PROJ_DIR
except ImportError:
    PROJ_DIR = _PROJECT_ROOT / "xengym"

from xengym.marker_appearance import (
    generate_random_ellipses_attenuation_texture_u8,
    resolve_marker_appearance_config,
)

# Attempt optional imports
try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False
    print("Warning: Taichi not available, MPM mode disabled")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Attempt ezgl/render imports
try:
    from xensesdk.ezgl import tb, Matrix4x4
    from xensesdk.ezgl.items.scene import Scene
    from xensesdk.ezgl.experimental.GLSurfMeshItem import GLSurfMeshItem
    from xensesdk.ezgl.items import (
        GLModelItem,
        GLBoxItem,
        GLMeshItem,
        DepthCamera,
        RGBCamera,
        PointLight,
        LineLight,
        GLAxisItem,
        Texture2D,
        Material,
    )
    from xensesdk.ezgl.items.MeshData import sphere as ezgl_mesh_sphere
    from xengym.render import VecTouchSim
    from xengym import ASSET_DIR
    HAS_EZGL = True
except Exception as e:
    HAS_EZGL = False
    # Define placeholder classes when ezgl is not available
    class Scene:
        """Placeholder Scene class when ezgl is not available"""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ezgl not available")
    class Matrix4x4:
        """Placeholder Matrix4x4 class"""
        pass
    class GLSurfMeshItem:
        """Placeholder GLSurfMeshItem class"""
        pass
    class GLModelItem:
        """Placeholder GLModelItem class"""
        pass
    class GLBoxItem:
        """Placeholder GLBoxItem class"""
        pass
    class GLMeshItem:
        """Placeholder GLMeshItem class"""
        pass
    class DepthCamera:
        """Placeholder DepthCamera class"""
        pass
    class RGBCamera:
        """Placeholder RGBCamera class"""
        pass
    class PointLight:
        """Placeholder PointLight class"""
        pass
    class LineLight:
        """Placeholder LineLight class"""
        pass
    class Texture2D:
        """Placeholder Texture2D class"""
        pass
    class Material:
        """Placeholder Material class"""
        pass
    tb = None
    ASSET_DIR = Path(".")
    print(f"Warning: ezgl/xengym render not available ({e})")
    print("Hint: repo-local xensesdk is under ./xensesdk; try setting PYTHONPATH to include it, "
          "e.g. PowerShell: $env:PYTHONPATH=\"$PWD\\xensesdk;$env:PYTHONPATH\"")


# ==============================================================================
# Scene Parameters
# ==============================================================================
# 默认 marker 策略属于用户可见行为；如需调整默认值，请先按 `openspec/AGENTS.md` 创建 proposal，
# 以保证向后兼容并补齐迁移说明。
DEFAULT_MPM_MARKER_MODE = "warp"

SCENE_PARAMS = {
    # Gel geometry (match VecTouchSim defaults)
    'gel_size_mm': (17.3, 29.15),       # width (x), height (y) in mm
    'gel_thickness_mm': 5.0,            # depth (z) in mm

    # Height field grid resolution (matches SensorScene)
    'height_grid_shape': (140, 80),     # (n_row, n_col)

    # Height-field postprocess (MPM -> render)
    'mpm_height_fill_holes': True,      # fill empty cells via diffusion (recommended to avoid hard-edge artifacts)
    'mpm_height_fill_holes_iters': 10,  # iterations for hole filling
    'mpm_height_smooth': True,          # apply box smoothing before rendering
    'mpm_height_smooth_iters': 2,       # matches previous hardcoded behavior
    'mpm_height_reference_edge': True,  # use edge-region baseline alignment (preserve slide motion)
    # UV displacement postprocess (MPM -> marker warp)
    # UV smoothing (box blur). Keep defaults consistent with previous hardcoded behavior.
    'mpm_uv_smooth': True,
    'mpm_uv_smooth_iters': 1,
    # UV hole filling (diffusion) for cells without particle coverage.
    'mpm_uv_fill_holes': True,
    'mpm_uv_fill_holes_iters': 10,
    # Optional: use pre-clamp height_field as the per-cell reference for selecting top-surface particles
    # when extracting uv_disp_mm. This avoids the common failure mode where height_field is clamped to a flat
    # indenter surface (constant), causing `z >= ref - band` to reject the true surface particles on the
    # trailing side and leaving uv_cnt==0 inside footprint.
    # Default OFF to preserve legacy output unless explicitly requested (publish_v1 turns it on).
    'mpm_uv_ref_preclamp_height': False,
    # Optional: fill UV holes *inside* indenter footprint using footprint median displacement.
    # Why: when height_field is clamped to indenter surface but local particle coverage is missing,
    # the default diffusion fill can make the trailing-side UV collapse to near-zero, causing
    # the visible “under-indenter marker freezes while leading edge moves” artifact.
    # Default OFF to preserve baseline behavior unless explicitly requested (publish_v1 turns it on).
    'mpm_uv_fill_footprint_holes': False,
    # Optional: bin uv_disp_mm by particles' initial (reference) XY to form a Lagrangian displacement field u(X).
    # Why: When using marker_mode=advect (default for publish_v1), sampling an Eulerian u(x) at reference dot
    #      centers can create the classic "front moves / back stays" artifact during sliding. This mode aligns
    #      UV extraction with advection: marker dot at X moves to X + u(X).
    # Default OFF to preserve baseline behavior unless explicitly requested (publish_v1 turns it on).
    'mpm_uv_bin_init_xy': False,
    # NOTE: 默认关闭以保持基线行为不变；仅在明确开启时用于抑制接触边界的尖峰位移（常见于洞填充/插值/采样伪影）。
    'mpm_uv_despike': False,
    'mpm_uv_despike_scope': 'boundary',  # boundary|footprint
    'mpm_uv_despike_abs_mm': 0.8,
    'mpm_uv_despike_cap_mm': None,      # None => abs_mm-1e-3
    'mpm_uv_despike_boundary_iters': 2,
    # Optional: mask uv_disp outside indenter footprint (set to 0).
    # Why: prevent hole-fill/smoothing artifacts from spreading into non-contact areas, which can
    #      cause visible marker "smear" and local spikes when warping high-frequency textures.
    # Default OFF to preserve baseline behavior.
    'mpm_uv_mask_footprint': False,
    # Default is a conservative dilation to avoid introducing a hard discontinuity inside the broad height-derived
    # `contact_mask` used by offline metrics (see analyze_rgb_compare_smear_metrics.py).
    'mpm_uv_mask_footprint_dilate_iters': 8,
    # Extra box blur iterations applied after masking (soften boundary discontinuity).
    'mpm_uv_mask_footprint_blur_iters': 6,
    # Optional: hard cap |uv| magnitude (mm). Last-resort knob to prevent rare extremes from dominating warp.
    # Default None (disabled) to preserve baseline/publish behavior unless explicitly requested.
    'mpm_uv_cap_mm': None,
    # UV 位移缩放系数（在所有 UV 后处理步骤之后生效）。用于对齐 MPM uv_disp_mm 与 FEM marker 位移量级。
    'mpm_uv_scale': 1.0,
    # Height-field outlier suppression (MPM -> render)
    # NOTE: 仅作为“最后一道防线”，避免 footprint 外的异常深值把整块区域渲染成暗盘/彩虹 halo。
    # 默认关闭以保持基线行为不变；建议与 fill_holes 联用。
    'mpm_height_clip_outliers': False,
    'mpm_height_clip_outliers_min_mm': 2.0,  # 超过该负向深度（mm）的值会被视作离群并置为 NaN
    # IMPORTANT: MPM 接触采用 penalty 形式时，压头可能“穿透”粒子表面并导致高度场过深，
    # 从而出现非物理的“整块发黑/暗盘”。该开关会把 height_field 限制在“不低于压头表面”。
    'mpm_height_clamp_indenter': True,

    # Render conventions (MPM -> RGB)
    # NOTE: 为了与 FEM(SensorScene) 的最终输出保持“同帧同侧”，这里默认不在 field 层额外做 X 翻转。
    # 若需要复现旧输出（legacy），可用 CLI 显式打开。
    'mpm_render_flip_x': False,
    # Mesh Z convention (MPM height_field -> ezgl GLSurfMeshItem zmap)
    # - sensor_depth: keep SensorScene convention (indentation is negative). May be clipped by camera near/far.
    # - indentation: convert to positive indentation zmap (>=0), aligning with xensim (recommended).
    'mpm_zmap_convention': 'sensor_depth',  # sensor_depth|indentation
    # Marker warp convention: flip dx/dy in texture space to match mesh/texcoords.
    # NOTE: 默认让 marker 位移方向与 UV 场一致（不额外翻 X）。
    'mpm_warp_flip_x': False,
    'mpm_warp_flip_y': True,
    # Lighting configuration (MPM -> RGB).
    # Default keeps legacy behavior: use in-code defaults, then (unless disabled) load repo-local light.txt if present.
    'mpm_light_profile': 'default',      # default|publish_v1
    'mpm_disable_light_file': False,     # True => do NOT load xengym/assets/data/light.txt
    'mpm_render_shadow': True,           # True => LineLight.render_shadow (can be downgraded for publish stability)
    # Marker grid/layout parameters (for audit + offline analysis consistency)
    # NOTE: marker_uv_compare 需要 dx/dy(mm) 来将 FEM marker 网格映射到 gel 平面坐标。
    'marker_grid_rows': 20,
    'marker_grid_cols': 14,
    'marker_dx_mm': 1.31,
    'marker_dy_mm': 1.31,
    'marker_radius_px': 3,
    # Marker texture size used by MPMSensorScene._make_marker_texture (W,H).
    'marker_tex_size_wh': (320, 560),

    # Trajectory parameters
    # NOTE: 默认压深建议保持在 FEM 数据覆盖范围内（通常 <= 1mm），否则 MPM 侧容易出现“广域下陷”
    # 进而放大渲染伪影（暗块/halo），导致 MPM vs FEM 观感不可直接对比。
    'press_depth_mm': 1.0,              # target indentation depth (mm)
    'slide_distance_mm': 3.0,           # tangential travel (x direction)
    'press_steps': 150,                 # steps to reach press depth
    'slide_steps': 240,                 # steps for sliding phase
    'hold_steps': 40,                   # steps to hold at end

    # MPM simulation parameters
    'mpm_dt': 2e-5,                     # Reduced for stability with higher stiffness
    'mpm_grid_dx_mm': 0.4,              # grid spacing in mm
    'mpm_particles_per_cell': 2,        # particles per cell per dimension      

    # MPM grid padding (cells) to keep particles away from sticky boundaries.
    # Grid boundary clamp is a legacy behavior from MPMSolver.grid_op.
    # Defaults preserve previous hardcoded behavior (enabled, width=3).
    'mpm_sticky_boundary': True,
    'mpm_sticky_boundary_width': 3,

    # MPM grid padding (cells) to keep particles away from sticky boundaries.
    # MPMSolver.grid_op clamps grid velocities when sticky boundary is enabled.
    # Using >=6 padding cells provides a safety margin for contact/friction (default width=3).
    'mpm_grid_padding_cells_xy': 6,
    'mpm_grid_padding_cells_z_bottom': 6,
    'mpm_grid_padding_cells_z_top': 20,

    # Material (soft gel)
    'density': 1000.0,                  # kg/m³
    'ogden_mu': [2500.0],               # Pa
    'ogden_alpha': [2.0],
    # NOTE: kappa 控制体积压缩性（kappa >> mu 时近似不可压缩）。
    # 早期的 2.5e4 Pa 会导致明显“广域下陷带”，MPM vs FEM 无法直接对比。
    # 这里默认对齐到 xengym MPM demo 常用量级（~3e5 Pa），并保留 CLI 覆盖入口。
    'ogden_kappa': 300000.0,            # Pa

    # Indenter (sphere)
    'indenter_radius_mm': 4.0,
    'indenter_start_gap_mm': 0.5,       # initial clearance above gel
    # NOTE: audit-friendly alias; keep in sync with indenter_start_gap_mm
    'mpm_indenter_gap_mm': 0.5,
    # Indenter (sphere/cylinder/box)
    # - cylinder: flat round pad, matches circle_r4.STL (tip face) better than box
    'indenter_type': 'cylinder',        # 'sphere' | 'cylinder' | 'box'
    'indenter_cylinder_half_height_mm': None,  # Optional half height for cylinder; default uses radius
    'indenter_half_extents_mm': None,   # Optional (x,y,z) for box mode; overrides indenter_radius_mm

    # Contact / friction (explicit defaults for auditability)
    'fem_fric_coef': 0.4,               # FEM fric_coef (single coefficient)
    'mpm_mu_s': 2.0,                    # MPM static friction (mu_s)
    'mpm_mu_k': 1.5,                    # MPM kinetic friction (mu_k)     
    'mpm_contact_stiffness_normal': 8e2,
    'mpm_contact_stiffness_tangent': 4e2,

    # Optional Kelvin-Voigt bulk viscosity (damping) for MPM
    'mpm_enable_bulk_viscosity': False,
    'mpm_bulk_viscosity': 0.0,          # Pa·s

    # Depth camera settings (for FEM path)
    'depth_img_size': (100, 175),       # matches demo_simple_sensor
    # IMPORTANT: For FEM depth->gel mapping, the depth camera ortho view should match gel_size_mm.
    # Keep these consistent to avoid implicit scaling mismatch between FEM and MPM.
    'cam_view_width_m': 0.0173,         # 17.3 mm (match gel_size_mm[0])
    'cam_view_height_m': 0.02915,       # 29.15 mm (match gel_size_mm[1])

    # Debug settings
    'debug_verbose': False,             # Enable verbose per-frame logging
}


def _infer_square_size_mm_from_stl_path(object_file: Optional[str]) -> Optional[float]:
    """
    从资产命名中推断正方形压头边长（mm）。

    约定：xengym/assets/obj/square_d6.STL -> d=6mm。
    """
    if not object_file:
        return None
    name = Path(object_file).name.lower()
    m = re.search(r"square_d(\d+)", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _analyze_binary_stl_endfaces_mm(stl_path: Path) -> Optional[Dict[str, object]]:
    """
    Analyze a binary STL and return simple end-face extents (mm) at y_min / y_max.

    This is used as a lightweight verification tool for assets like circle_r4.STL
    whose bottom (y_min) and top (y_max) faces may have very different contact footprints.
    """
    try:
        with stl_path.open("rb") as f:
            header = f.read(80)
            if len(header) != 80:
                return None
            tri_count_bytes = f.read(4)
            if len(tri_count_bytes) != 4:
                return None
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            if tri_count <= 0:
                return None
            raw = f.read(50 * tri_count)
            if len(raw) != 50 * tri_count:
                return None

        tri_dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("v3", "<f4", (3,)),
                ("attr", "<u2"),
            ]
        )
        if tri_dtype.itemsize != 50:
            return None

        tris = np.frombuffer(raw, dtype=tri_dtype, count=tri_count)
        verts = np.concatenate([tris["v1"], tris["v2"], tris["v3"]], axis=0).astype(np.float64)
        if verts.size == 0:
            return None

        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        y_min = float(bbox_min[1])
        y_max = float(bbox_max[1])
        height = max(y_max - y_min, 0.0)

        tol = max(1e-6, height * 1e-4)
        mask_min = verts[:, 1] <= (y_min + tol)
        mask_max = verts[:, 1] >= (y_max - tol)
        if not mask_min.any() or not mask_max.any():
            return None

        vmin = verts[mask_min]
        vmax = verts[mask_max]

        def _xz_extents_mm(v: np.ndarray) -> Dict[str, float]:
            x_min_m = float(v[:, 0].min())
            x_max_m = float(v[:, 0].max())
            z_min_m = float(v[:, 2].min())
            z_max_m = float(v[:, 2].max())
            return {
                "x_min_mm": x_min_m * 1000.0,
                "x_max_mm": x_max_m * 1000.0,
                "z_min_mm": z_min_m * 1000.0,
                "z_max_mm": z_max_m * 1000.0,
                "size_x_mm": (x_max_m - x_min_m) * 1000.0,
                "size_z_mm": (z_max_m - z_min_m) * 1000.0,
            }

        return {
            "path": str(stl_path).replace("\\", "/"),
            "triangles": int(tri_count),
            "bbox_min_mm": (bbox_min * 1000.0).tolist(),
            "bbox_max_mm": (bbox_max * 1000.0).tolist(),
            "y_min_mm": y_min * 1000.0,
            "y_max_mm": y_max * 1000.0,
            "height_mm": height * 1000.0,
            "endfaces_mm": {
                "y_min": _xz_extents_mm(vmin),
                "y_max": _xz_extents_mm(vmax),
            },
        }
    except Exception:
        return None


def _box_blur_2d_xy(values: np.ndarray, iterations: int = 1) -> np.ndarray:
    """对 (H,W,2) 的位移场做轻量 3x3 box blur。"""
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("values must be (H,W,2)")
    result = values.astype(np.float32, copy=True)
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        result = (
            padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
            padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    return result


def _fill_uv_holes(
    uv: np.ndarray,
    valid_mask: np.ndarray,
    max_iterations: int = 10,
    *,
    return_filled_mask: bool = False,
) -> object:
    """
    使用扩散法填充 UV 场中的空洞（无粒子覆盖的网格单元）。

    Args:
        uv: (H,W,2) UV 位移场
        valid_mask: (H,W) bool，True 表示该单元有有效数据
        max_iterations: 最大扩散迭代次数

    Returns:
        填充后的 UV 场 (H,W,2)
    """
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError("uv must be (H,W,2)")
    if valid_mask.shape != uv.shape[:2]:
        raise ValueError("valid_mask shape must match uv[:,:,0]")

    result = uv.astype(np.float32, copy=True)
    filled = valid_mask.copy()

    for _ in range(max_iterations):
        if filled.all():
            break  # 全部填充完成

        # 找到未填充但有邻居已填充的单元
        padded_filled = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        neighbor_count = (
            padded_filled[0:-2, 0:-2] + padded_filled[0:-2, 1:-1] + padded_filled[0:-2, 2:] +
            padded_filled[1:-1, 0:-2] +                            padded_filled[1:-1, 2:] +
            padded_filled[2:, 0:-2]   + padded_filled[2:, 1:-1]   + padded_filled[2:, 2:]
        )

        # 可填充的单元：未填充 且 至少有一个邻居已填充
        can_fill = (~filled) & (neighbor_count > 0)

        if not can_fill.any():
            break

        # 计算邻居的加权平均
        padded_uv = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0)
        padded_mask = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)

        neighbor_sum = np.zeros_like(result)
        neighbor_weight = np.zeros(result.shape[:2], dtype=np.float32)

        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            i_slice = slice(1 + di, result.shape[0] + 1 + di)
            j_slice = slice(1 + dj, result.shape[1] + 1 + dj)
            w = padded_mask[i_slice, j_slice]
            neighbor_sum += padded_uv[i_slice, j_slice] * w[..., None]
            neighbor_weight += w

        # 填充
        fill_mask = can_fill & (neighbor_weight > 0)
        result[fill_mask] = neighbor_sum[fill_mask] / neighbor_weight[fill_mask][..., None]
        filled[fill_mask] = True

    if return_filled_mask:
        return result, filled
    return result


def _fill_uv_holes_in_mask(
    uv: np.ndarray,
    valid_mask: np.ndarray,
    fill_mask: np.ndarray,
    max_iterations: int = 10,
    *,
    return_filled_mask: bool = False,
) -> object:
    """
    填充 UV 场中的“空洞”，但仅在指定区域内进行扩散（避免从区域外泄漏）。

    Notes:
    - `valid_mask` 仅在 `fill_mask` 内生效；区域外单元既不参与扩散，也不会被写入。
    - 该函数用于 footprint 内的 UV inpaint：既可用于 uv_cnt==0 的空洞，也可用于“幅值近零”的伪空洞。
    """
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError("uv must be (H,W,2)")
    if valid_mask.shape != uv.shape[:2]:
        raise ValueError("valid_mask shape must match uv[:,:,0]")
    if fill_mask.shape != uv.shape[:2]:
        raise ValueError("fill_mask shape must match uv[:,:,0]")

    region = fill_mask.astype(np.bool_, copy=False)
    result = uv.astype(np.float32, copy=True)
    filled = (valid_mask & region).copy()

    if not bool(region.any()):
        if return_filled_mask:
            return result, filled
        return result

    for _ in range(max_iterations):
        # Only care about completion inside region.
        if bool(filled[region].all()):
            break

        # Find unfilled cells inside region with at least one filled neighbor.
        padded_filled = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        neighbor_count = (
            padded_filled[0:-2, 0:-2] + padded_filled[0:-2, 1:-1] + padded_filled[0:-2, 2:] +
            padded_filled[1:-1, 0:-2] +                            padded_filled[1:-1, 2:] +
            padded_filled[2:, 0:-2]   + padded_filled[2:, 1:-1]   + padded_filled[2:, 2:]
        )
        can_fill = (~filled) & region & (neighbor_count > 0)
        if not bool(can_fill.any()):
            break

        padded_uv = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0)
        padded_mask = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)

        neighbor_sum = np.zeros_like(result)
        neighbor_weight = np.zeros(result.shape[:2], dtype=np.float32)

        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            i_slice = slice(1 + di, result.shape[0] + 1 + di)
            j_slice = slice(1 + dj, result.shape[1] + 1 + dj)
            w = padded_mask[i_slice, j_slice]
            neighbor_sum += padded_uv[i_slice, j_slice] * w[..., None]
            neighbor_weight += w

        fill_mask2 = can_fill & (neighbor_weight > 0)
        result[fill_mask2] = neighbor_sum[fill_mask2] / neighbor_weight[fill_mask2][..., None]
        filled[fill_mask2] = True

    if return_filled_mask:
        return result, filled
    return result


def _contact_boundary_mask(contact_mask_u8: np.ndarray) -> np.ndarray:
    """
    Compute a 4-neighborhood boundary mask of a binary contact/footprint mask.

    Notes:
    - Pure numpy; edges treated as non-contact neighbors.
    - Boundary here means "in mask but not a strict 4-neighbor interior".
    """
    m = (contact_mask_u8 > 0).astype(np.bool_)
    up = np.roll(m, 1, axis=0)
    down = np.roll(m, -1, axis=0)
    left = np.roll(m, 1, axis=1)
    right = np.roll(m, -1, axis=1)
    interior = up & down & left & right
    boundary = m & (~interior)
    boundary[0, :] = m[0, :]
    boundary[-1, :] = m[-1, :]
    boundary[:, 0] = m[:, 0]
    boundary[:, -1] = m[:, -1]
    return boundary.astype(np.uint8, copy=False)


def _dilate_mask_3x3(mask_u8: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    """3x3 binary dilation (numpy-only). Boundary treated as non-contact."""
    m = (mask_u8 > 0).astype(np.bool_)
    iters = max(0, int(iterations))
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        m = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return m.astype(np.uint8, copy=False)


def _stack_3x3_edge(values: np.ndarray) -> np.ndarray:
    """Return a 3x3 shifted stack (9,H,W) with edge padding (numpy-only)."""
    if values.ndim != 2:
        raise ValueError(f"values must be HxW, got {values.shape}")
    h, w = int(values.shape[0]), int(values.shape[1])
    p = np.pad(values, ((1, 1), (1, 1)), mode="edge")
    shifts = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifts.append(p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w])
    return np.stack(shifts, axis=0)


def _despike_uv_disp_mm(
    uv_disp_mm: np.ndarray,
    *,
    gate_mask_u8: np.ndarray,
    abs_mm: float,
    cap_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Despike UV field in an artifact-prone band using 3x3 median replacement + magnitude capping.

    - Only operates where `gate_mask_u8>0` (typically a dilated footprint boundary band).
    - For |uv| >= abs_mm: replace u/v with 3x3 nanmedian.
    - Then cap |uv| <= cap_mm (preserve direction).
    """
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[-1] != 2:
        raise ValueError(f"uv_disp_mm must be HxWx2, got shape={uv_disp_mm.shape}")
    if gate_mask_u8.ndim != 2 or gate_mask_u8.shape != uv_disp_mm.shape[:2]:
        raise ValueError(f"gate_mask_u8 must be HxW, got shape={gate_mask_u8.shape} vs uv={uv_disp_mm.shape[:2]}")

    uv = uv_disp_mm.astype(np.float32, copy=False)
    out = np.array(uv, copy=True)

    gate = (gate_mask_u8 > 0).astype(np.bool_)
    abs_thr = float(abs_mm)
    cap = float(cap_mm)
    if cap <= 0.0 or (not gate.any()):
        z = np.zeros(uv.shape[:2], dtype=np.bool_)
        return out, z, z

    mag = np.linalg.norm(uv, axis=2)
    finite = gate & np.isfinite(mag)
    replaced = finite & (mag >= abs_thr)

    if replaced.any():
        u = uv[..., 0]
        v = uv[..., 1]
        med_u = np.nanmedian(_stack_3x3_edge(u), axis=0)
        med_v = np.nanmedian(_stack_3x3_edge(v), axis=0)
        out[..., 0][replaced] = med_u[replaced]
        out[..., 1][replaced] = med_v[replaced]

    mag2 = np.linalg.norm(out, axis=2)
    capped = gate & np.isfinite(mag2) & (mag2 > cap)
    if capped.any():
        s = (cap / (mag2[capped] + 1e-12)).astype(np.float32, copy=False)
        out[..., 0][capped] *= s
        out[..., 1][capped] *= s

    return out, replaced, capped


def _cap_uv_disp_mm(uv_disp_mm: np.ndarray, *, cap_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hard cap |uv| <= cap_mm (mm), preserving direction.

    Returns:
    - uv_capped: capped uv field (float32)
    - capped_mask: bool mask where capping was applied (H,W)
    """
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[-1] != 2:
        raise ValueError(f"uv_disp_mm must be HxWx2, got shape={uv_disp_mm.shape}")
    cap = float(cap_mm)
    if cap <= 0.0:
        z = np.zeros(uv_disp_mm.shape[:2], dtype=np.bool_)
        return uv_disp_mm.astype(np.float32, copy=False), z

    uv = uv_disp_mm.astype(np.float32, copy=False)
    out = np.array(uv, copy=True)
    mag = np.linalg.norm(out, axis=2)
    capped = np.isfinite(mag) & (mag > cap)
    if capped.any():
        s = (cap / (mag[capped] + 1e-12)).astype(np.float32, copy=False)
        out[..., 0][capped] *= s
        out[..., 1][capped] *= s
    return out, capped


def _build_indenter_footprint_mask_u8(
    n_row: int,
    n_col: int,
    *,
    gel_size_mm: Tuple[float, float],
    indenter_type: str,
    indenter_radius_mm: float,
    indenter_half_extents_mm: Optional[Sequence[float]],
    indenter_center_m: Tuple[float, float, float],
    ref_x_center_mm: float,
    ref_y_min_mm: float,
) -> np.ndarray:
    gel_w_mm, gel_h_mm = gel_size_mm
    if n_row <= 0 or n_col <= 0:
        raise ValueError(f"Invalid grid shape: {(n_row, n_col)}")
    if gel_w_mm <= 0 or gel_h_mm <= 0:
        raise ValueError(f"Invalid gel_size_mm: {gel_size_mm}")

    cell_w = float(gel_w_mm) / float(n_col)
    cell_h = float(gel_h_mm) / float(n_row)

    cx_mm = float(indenter_center_m[0]) * 1000.0 - float(ref_x_center_mm)
    # 约定：MPM solver 的平面坐标使用 (x, y)；z 为高度。
    # 因此 footprint 的中心应使用 (center_x, center_y) 映射到 sensor 平面。
    cy_mm = float(indenter_center_m[1]) * 1000.0 - float(ref_y_min_mm)

    x_centers = (np.arange(n_col, dtype=np.float32) + 0.5) * np.float32(cell_w) - np.float32(gel_w_mm / 2.0)
    y_centers = (np.arange(n_row, dtype=np.float32) + 0.5) * np.float32(cell_h)
    xx, yy = np.meshgrid(x_centers, y_centers)

    ind_type = str(indenter_type or "box").lower().strip()
    if ind_type in {"sphere", "cylinder"}:
        r_mm = float(indenter_radius_mm)
        inside = ((xx - np.float32(cx_mm)) ** 2 + (yy - np.float32(cy_mm)) ** 2) <= np.float32(r_mm) ** 2
    else:
        half_extents = None
        if indenter_half_extents_mm is not None:
            vals = list(indenter_half_extents_mm)
            if len(vals) == 3:
                try:
                    half_extents = (float(vals[0]), float(vals[1]), float(vals[2]))
                except Exception:
                    half_extents = None
        if half_extents is not None:
            hx_mm, hy_mm, _hz_mm = half_extents
        else:
            r_mm = float(indenter_radius_mm)
            hx_mm = hy_mm = float(r_mm)
        inside = (np.abs(xx - np.float32(cx_mm)) <= np.float32(hx_mm)) & (np.abs(yy - np.float32(cy_mm)) <= np.float32(hy_mm))

    return inside.astype(np.uint8, copy=False)


def _fill_height_holes(height_mm: np.ndarray, valid_mask: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """
    使用扩散法填充高度场中的空洞（无粒子覆盖的网格单元）。

    Args:
        height_mm: (H,W) 高度场（mm，<=0 表示压入）
        valid_mask: (H,W) bool，True 表示该单元有有效数据
        max_iterations: 最大扩散迭代次数

    Returns:
        填充后的高度场 (H,W)
    """
    if height_mm.ndim != 2:
        raise ValueError("height_mm must be (H,W)")
    if valid_mask.shape != height_mm.shape:
        raise ValueError("valid_mask shape must match height_mm")

    # NOTE: height_mm may contain NaN for missing cells; keep NaN but ensure
    # they do not pollute neighbor aggregation (use nan_to_num when padding).
    result = height_mm.astype(np.float32, copy=True)
    filled = valid_mask.copy()

    for _ in range(max(max_iterations, 0)):
        if filled.all():
            break

        padded_filled = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        neighbor_count = (
            padded_filled[0:-2, 0:-2] + padded_filled[0:-2, 1:-1] + padded_filled[0:-2, 2:] +
            padded_filled[1:-1, 0:-2] +                            padded_filled[1:-1, 2:] +
            padded_filled[2:, 0:-2]   + padded_filled[2:, 1:-1]   + padded_filled[2:, 2:]
        )

        can_fill = (~filled) & (neighbor_count > 0)
        if not can_fill.any():
            break

        padded_h = np.pad(
            np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0),
            ((1, 1), (1, 1)),
            mode="constant",
            constant_values=0,
        )
        padded_mask = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)

        neighbor_sum = np.zeros_like(result)
        neighbor_weight = np.zeros_like(result, dtype=np.float32)

        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            i_slice = slice(1 + di, result.shape[0] + 1 + di)
            j_slice = slice(1 + dj, result.shape[1] + 1 + dj)
            w = padded_mask[i_slice, j_slice]
            neighbor_sum += padded_h[i_slice, j_slice] * w
            neighbor_weight += w

        fill_mask = can_fill & (neighbor_weight > 0)
        result[fill_mask] = neighbor_sum[fill_mask] / neighbor_weight[fill_mask]
        filled[fill_mask] = True

    # Remaining holes (if any) are set to 0mm (flat), consistent with previous behavior.
    result[~filled] = 0.0
    return result


def _upsample_uv_disp_to_hw(uv_disp_mm: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Upsample (Ny,Nx,2) uv_disp_mm to (out_h,out_w,2) using bilinear interpolation."""
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[2] != 2:
        raise ValueError("uv_disp_mm must be (Ny,Nx,2)")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("out_h/out_w must be positive")

    src_h, src_w = int(uv_disp_mm.shape[0]), int(uv_disp_mm.shape[1])
    if HAS_CV2:
        return cv2.resize(uv_disp_mm, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # Numpy fallback: bilinear upsampling
    row_scale = (src_h - 1) / max(out_h - 1, 1)
    col_scale = (src_w - 1) / max(out_w - 1, 1)
    row_coords = np.arange(out_h) * row_scale
    col_coords = np.arange(out_w) * col_scale
    r0 = np.floor(row_coords).astype(np.int32)
    c0 = np.floor(col_coords).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, src_h - 1)
    c1 = np.clip(c0 + 1, 0, src_w - 1)
    wr = (row_coords - r0).astype(np.float32)
    wc = (col_coords - c0).astype(np.float32)
    # Bilinear interpolation for each channel
    uv_up = np.zeros((out_h, out_w, 2), dtype=np.float32)
    for ch in range(2):
        src = uv_disp_mm[..., ch]
        top = src[r0[:, None], c0[None, :]] * (1 - wc[None, :]) + src[r0[:, None], c1[None, :]] * wc[None, :]
        bot = src[r1[:, None], c0[None, :]] * (1 - wc[None, :]) + src[r1[:, None], c1[None, :]] * wc[None, :]
        uv_up[..., ch] = top * (1 - wr[:, None]) + bot * wr[:, None]
    return uv_up


def _marker_grid_centers_px(
    tex_w: int,
    tex_h: int,
    *,
    n_cols: int = 14,
    n_rows: int = 20,
    margin_x: int = 20,
    margin_y: int = 20,
) -> List[Tuple[int, int]]:
    """Return marker dot centers in texture pixel coordinates, matching _make_marker_texture()."""
    if tex_w <= 0 or tex_h <= 0:
        return []
    if n_cols <= 0 or n_rows <= 0:
        return []
    centers: List[Tuple[int, int]] = []
    if n_cols == 1:
        xs = [int(tex_w // 2)]
    else:
        xs = [int(margin_x + c * (tex_w - 2 * margin_x) / (n_cols - 1)) for c in range(n_cols)]
    if n_rows == 1:
        ys = [int(tex_h // 2)]
    else:
        ys = [int(margin_y + r * (tex_h - 2 * margin_y) / (n_rows - 1)) for r in range(n_rows)]
    for y in ys:
        for x in xs:
            centers.append((x, y))
    return centers


def warp_marker_texture(
    base_tex: np.ndarray,
    uv_disp_mm: np.ndarray,
    gel_size_mm: Tuple[float, float],
    flip_x: bool,
    flip_y: bool,
    stats_out: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    使用面内位移场对 marker 纹理做非均匀 warp，使点阵体现拉伸/压缩/剪切。

    - uv_disp_mm: (Ny,Nx,2)，单位 mm，u=+x 向右，v=+y 向上（以“传感器平面坐标”约定）
    - 采用逆向映射：输出像素 (x,y) 从输入 base_tex 采样 (x - dx, y - dy)
    - flip_x/flip_y 用于处理 texcoords / mesh 的翻转约定差异
    - stats_out 可选：返回 remap 出界统计（oob_px/oob_ratio），用于诊断“短横线/拉丝”是否由出界采样引起
    """
    if base_tex.ndim != 3 or base_tex.shape[2] != 3:
        raise ValueError("base_tex must be (H,W,3)")
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[2] != 2:
        raise ValueError("uv_disp_mm must be (Ny,Nx,2)")

    tex_h, tex_w = base_tex.shape[0], base_tex.shape[1]
    gel_w_mm, gel_h_mm = gel_size_mm

    # Upsample uv field to texture resolution using bilinear interpolation
    uv_up = _upsample_uv_disp_to_hw(uv_disp_mm, out_h=tex_h, out_w=tex_w)

    # mm -> px
    dx_px = (uv_up[..., 0] / max(gel_w_mm, 1e-6)) * tex_w
    dy_px = (uv_up[..., 1] / max(gel_h_mm, 1e-6)) * tex_h

    # Mesh/texcoords 翻转修正
    if flip_x:
        dx_px = -dx_px
    if flip_y:
        dy_px = -dy_px

    xx, yy = np.meshgrid(np.arange(tex_w, dtype=np.float32), np.arange(tex_h, dtype=np.float32))
    map_x = xx - dx_px.astype(np.float32)
    map_y = yy - dy_px.astype(np.float32)

    # 诊断：统计 remap 出界比例（不做 clip，以避免“边缘反射/夹断”造成的拉丝拖影）。
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    oob = (~finite) | (map_x < 0.0) | (map_x > float(tex_w - 1)) | (map_y < 0.0) | (map_y > float(tex_h - 1))
    if stats_out is not None:
        stats_out["oob_px"] = float(np.sum(oob))
        stats_out["oob_ratio"] = float(np.mean(oob))

    if HAS_CV2:
        # 与 numpy fallback 保持一致：对出界坐标使用 constant border，而不是 clip/reflect。
        # Why：clip 会把大量出界采样“挤”到边缘像素，形成短横线/拉丝；reflect 会引入镜像拖影。
        map_x = np.nan_to_num(map_x, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32, copy=False)
        map_y = np.nan_to_num(map_y, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32, copy=False)
        border_value = tuple(int(x) for x in base_tex[0, 0].tolist())
        return cv2.remap(
            base_tex,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    # Numpy fallback: bilinear sampling + constant border for out-of-bounds.
    map_x = np.nan_to_num(map_x, nan=-1.0, posinf=-1.0, neginf=-1.0)
    map_y = np.nan_to_num(map_y, nan=-1.0, posinf=-1.0, neginf=-1.0)
    map_x = np.clip(map_x, 0.0, tex_w - 1.001)
    map_y = np.clip(map_y, 0.0, tex_h - 1.001)
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, tex_w - 1)
    y1 = np.clip(y0 + 1, 0, tex_h - 1)
    wx = (map_x - x0).astype(np.float32)[..., None]
    wy = (map_y - y0).astype(np.float32)[..., None]

    Ia = base_tex[y0, x0].astype(np.float32)
    Ib = base_tex[y0, x1].astype(np.float32)
    Ic = base_tex[y1, x0].astype(np.float32)
    Id = base_tex[y1, x1].astype(np.float32)
    out = (Ia * (1 - wx) * (1 - wy) +
           Ib * wx * (1 - wy) +
           Ic * (1 - wx) * wy +
           Id * wx * wy)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if oob.any():
        out_u8[oob] = base_tex[0, 0].astype(np.uint8, copy=False)
    return out_u8


def _bilinear_sample_vec2_px(field_xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """
    Bilinear sample a (H,W,2) vector field at subpixel points (N,2) in pixel coordinates.

    Returns:
        (N,2) float32
    """
    if field_xy.ndim != 3 or field_xy.shape[2] != 2:
        raise ValueError("field_xy must be (H,W,2)")
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must be (N,2)")
    h, w = int(field_xy.shape[0]), int(field_xy.shape[1])
    if h <= 0 or w <= 0 or points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    x = points_xy[:, 0].astype(np.float32, copy=False)
    y = points_xy[:, 1].astype(np.float32, copy=False)
    x = np.clip(x, 0.0, float(w - 1.001))
    y = np.clip(y, 0.0, float(h - 1.001))

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (x - x0.astype(np.float32)).astype(np.float32)
    fy = (y - y0.astype(np.float32)).astype(np.float32)

    f00 = field_xy[y0, x0].astype(np.float32, copy=False)
    f10 = field_xy[y0, x1].astype(np.float32, copy=False)
    f01 = field_xy[y1, x0].astype(np.float32, copy=False)
    f11 = field_xy[y1, x1].astype(np.float32, copy=False)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy
    out = (
        f00 * w00[:, None]
        + f10 * w10[:, None]
        + f01 * w01[:, None]
        + f11 * w11[:, None]
    )
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32, copy=False)


def _render_advect_points_texture(
    base_tex: np.ndarray,
    points_xy: np.ndarray,
    ellipse_axes_angle: Optional[np.ndarray],
    *,
    marker_color_bgr: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Render marker dots as ellipses (or circles) at subpixel positions.

    Args:
        base_tex: (H,W,3) reference texture; only base_tex[0,0] is used as background color.
        points_xy: (N,2) float32 pixel coords (x,y).
        ellipse_axes_angle: Optional (N,3) float32: (axis_a_px, axis_b_px, angle_deg).
            If None, render circles with axis_a==axis_b.
    """
    if base_tex.ndim != 3 or base_tex.shape[2] != 3:
        raise ValueError("base_tex must be (H,W,3)")
    tex_h, tex_w = int(base_tex.shape[0]), int(base_tex.shape[1])
    bg = base_tex[0, 0].astype(np.uint8, copy=True)
    out = np.empty_like(base_tex)
    out[...] = bg
    if points_xy.size == 0:
        return out

    if HAS_CV2:
        # Use fixed-point shift to preserve subpixel motion (reduces “snapping”).
        shift = 4  # 1/16 px
        scale = 1 << shift
        color = tuple(int(c) for c in marker_color_bgr)
        for i in range(int(points_xy.shape[0])):
            x = float(points_xy[i, 0])
            y = float(points_xy[i, 1])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if ellipse_axes_angle is None:
                a = 3.0
                b = 3.0
                ang = 0.0
            else:
                a = float(ellipse_axes_angle[i, 0])
                b = float(ellipse_axes_angle[i, 1])
                ang = float(ellipse_axes_angle[i, 2])
            if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(ang)):
                continue
            if a <= 0.5 or b <= 0.5:
                continue
            cx = int(round(x * scale))
            cy = int(round(y * scale))
            ax = int(round(a * scale))
            by = int(round(b * scale))
            try:
                cv2.ellipse(
                    out,
                    (cx, cy),
                    (ax, by),
                    ang,
                    0.0,
                    360.0,
                    color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                    shift=shift,
                )
            except Exception:
                # Best-effort: skip invalid ellipses (e.g., overflow).
                continue
        return out

    # Numpy fallback: circles only (elliptical AA is intentionally omitted to keep KISS).
    r = 3
    for i in range(int(points_xy.shape[0])):
        x = int(round(float(points_xy[i, 0])))
        y = int(round(float(points_xy[i, 1])))
        if x < -r or x > tex_w - 1 + r or y < -r or y > tex_h - 1 + r:
            continue
        x_lo = max(0, x - r)
        x_hi = min(tex_w, x + r + 1)
        y_lo = max(0, y - r)
        y_hi = min(tex_h, y + r + 1)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        out[y_lo:y_hi, x_lo:x_hi][mask] = 0
    return out


def advect_marker_texture(
    base_tex: np.ndarray,
    uv_disp_mm: np.ndarray,
    gel_size_mm: Tuple[float, float],
    flip_x: bool,
    flip_y: bool,
    *,
    marker_radius_px: int = 3,
    marker_grid_cols: int = 14,
    marker_grid_rows: int = 20,
    stats_out: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Legacy advect mode (stateless): place marker dots at (x0 + u(x0)).

    NOTE:
    - This is kept for backward compatibility and as a fallback when advect_points state is unavailable.
    - New publish presets are expected to prefer advect_points integration inside DepthRenderScene.
    """
    if base_tex.ndim != 3 or base_tex.shape[2] != 3:
        raise ValueError("base_tex must be (H,W,3)")
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[2] != 2:
        raise ValueError("uv_disp_mm must be (Ny,Nx,2)")
    tex_h, tex_w = int(base_tex.shape[0]), int(base_tex.shape[1])
    gel_w_mm, gel_h_mm = gel_size_mm

    uv_up = _upsample_uv_disp_to_hw(uv_disp_mm, out_h=tex_h, out_w=tex_w)
    dx_px = (uv_up[..., 0] / max(gel_w_mm, 1e-6)) * tex_w
    dy_px = (uv_up[..., 1] / max(gel_h_mm, 1e-6)) * tex_h
    if flip_x:
        dx_px = -dx_px
    if flip_y:
        dy_px = -dy_px

    centers = _marker_grid_centers_px(
        tex_w,
        tex_h,
        n_cols=max(1, int(marker_grid_cols)),
        n_rows=max(1, int(marker_grid_rows)),
    )
    if not centers:
        out = np.empty_like(base_tex)
        out[...] = base_tex[0, 0].astype(np.uint8, copy=True)
        return out
    pts = np.asarray([(float(x), float(y)) for (x, y) in centers], dtype=np.float32)
    d = np.stack([dx_px[pts[:, 1].astype(np.int32), pts[:, 0].astype(np.int32)],
                  dy_px[pts[:, 1].astype(np.int32), pts[:, 0].astype(np.int32)]], axis=1).astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    pts_adv = pts + d
    if stats_out is not None and d.size:
        mag = np.sqrt(np.sum(d * d, axis=1))
        stats_out["disp_px_p50"] = float(np.percentile(mag, 50))
        stats_out["disp_px_p90"] = float(np.percentile(mag, 90))
        stats_out["disp_px_p99"] = float(np.percentile(mag, 99))
    return _render_advect_points_texture(base_tex, pts_adv, ellipse_axes_angle=None)


def _mpm_flip_x_field(field: np.ndarray) -> np.ndarray:
    """Apply MPM->render horizontal flip (x axis) to match mesh x_range convention."""
    return field[:, ::-1]


def _mpm_flip_x_mm(x_mm: float) -> float:
    """Apply MPM->render horizontal flip (x axis) for scalar coordinates (mm)."""
    return -float(x_mm)


def _compute_rgb_diff_metrics(a_rgb: np.ndarray, b_rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute simple per-frame RGB difference metrics for audit/regression.

    Returns:
        dict with keys: mae, mae_r, mae_g, mae_b, max_abs, p50, p90, p99.
    """
    if a_rgb is None or b_rgb is None:
        raise ValueError("RGB inputs must not be None")
    if a_rgb.shape != b_rgb.shape:
        raise ValueError(f"RGB shape mismatch: {a_rgb.shape} vs {b_rgb.shape}")

    a16 = a_rgb.astype(np.int16, copy=False)
    b16 = b_rgb.astype(np.int16, copy=False)
    abs_diff = np.abs(a16 - b16).astype(np.float32, copy=False)

    p50, p90, p99 = [float(x) for x in np.percentile(abs_diff, [50, 90, 99]).tolist()]
    return {
        "mae": float(abs_diff.mean()),
        "mae_r": float(abs_diff[..., 0].mean()),
        "mae_g": float(abs_diff[..., 1].mean()),
        "mae_b": float(abs_diff[..., 2].mean()),
        "max_abs": float(abs_diff.max()),
        "p50": p50,
        "p90": p90,
        "p99": p99,
    }


def _sanitize_run_context_for_manifest(run_context: Dict[str, object]) -> Dict[str, object]:
    """
    Keep run_manifest diff-friendly by removing volatile/sensitive fields from run_context.

    Notes:
    - argv is already recorded in run_manifest.json; avoid duplicating save_dir or other
      environment-dependent paths in run_context.args.
    - Keep resolved.* for audit / alignment checks.
    """
    if not isinstance(run_context, dict):
        return {}
    sanitized: Dict[str, object] = {}
    resolved = run_context.get("resolved")
    if isinstance(resolved, dict):
        sanitized["resolved"] = resolved
    args = run_context.get("args")
    if isinstance(args, dict):
        args_copy = dict(args)
        args_copy.pop("save_dir", None)
        sanitized["args"] = args_copy
    return sanitized


def _sanitize_argv_for_manifest(argv: Sequence[str]) -> List[str]:
    """
    Keep argv diff-friendly by stripping volatile output paths.

    Note: the run directory is already implicit (this manifest lives under save_dir),
    so recording --save-dir adds noise without improving reproducibility.
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        a = str(argv[i])
        if a == "--save-dir":
            i += 2
            continue
        if a.startswith("--save-dir="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


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


def _collect_manifest_deps(repo_root: Path) -> Dict[str, object]:
    deps: Dict[str, object] = {
        "has_taichi": bool(HAS_TAICHI),
        "has_ezgl": bool(HAS_EZGL),
        "has_cv2": bool(HAS_CV2),
        "python": str(sys.version.split()[0]) if sys.version else None,
        "platform": platform.platform(),
        "git_head": _git_head(repo_root),
        # Seed note: this pipeline does not currently use an explicit RNG seed.
        # We still record it to make future nondeterminism triage easier.
        "seed": None,
    }

    try:
        deps["numpy"] = str(np.__version__)
    except Exception:
        deps["numpy"] = None

    if bool(HAS_TAICHI):
        try:
            import taichi as _ti  # local import to avoid hard dependency

            deps["taichi"] = str(getattr(_ti, "__version__", None))
        except Exception:
            deps["taichi"] = None

    if bool(HAS_CV2):
        try:
            import cv2 as _cv2  # local import to avoid hard dependency

            deps["cv2"] = str(getattr(_cv2, "__version__", None))
        except Exception:
            deps["cv2"] = None

    return deps


def _write_tuning_notes(
    save_dir: Path,
    *,
    record_interval: int,
    total_frames: int,
    run_context: Dict[str, object],
    overwrite: bool = False,
    reason: Optional[str] = None,
) -> None:
    """
    Write a stable, human-editable tuning_notes.md alongside run_manifest.json.

    The file is intended to be diff-friendly (avoid timestamps) and can be edited
    by humans after the run without being clobbered by default.
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    notes_path = save_dir / "tuning_notes.md"
    if notes_path.exists() and not overwrite:
        return

    press_steps = int(SCENE_PARAMS["press_steps"])
    slide_steps = int(SCENE_PARAMS["slide_steps"])
    hold_steps = int(SCENE_PARAMS["hold_steps"])

    def _phase_for_step(step: int) -> str:
        if step < press_steps:
            return "press"
        if step < press_steps + slide_steps:
            return "slide"
        return "hold"

    frame_to_step = [int(i * int(record_interval)) for i in range(int(total_frames))]
    frame_to_phase = [_phase_for_step(step) for step in frame_to_step]

    phase_ranges: Dict[str, Dict[str, int]] = {}
    for i, phase in enumerate(frame_to_phase):
        if phase not in phase_ranges:
            phase_ranges[phase] = {"start_frame": i, "end_frame": i}
        else:
            phase_ranges[phase]["end_frame"] = i

    def _pick_mid(start: int, end: int) -> int:
        return int((int(start) + int(end)) // 2)

    key_frames: Dict[str, Optional[int]] = {
        "press_end": None,
        "slide_mid": None,
        "hold_end": None,
    }
    if "press" in phase_ranges:
        key_frames["press_end"] = int(phase_ranges["press"]["end_frame"])
    if "slide" in phase_ranges:
        key_frames["slide_mid"] = _pick_mid(phase_ranges["slide"]["start_frame"], phase_ranges["slide"]["end_frame"])
    if "hold" in phase_ranges:
        key_frames["hold_end"] = int(phase_ranges["hold"]["end_frame"])

    resolved = run_context.get("resolved") if isinstance(run_context, dict) else None
    resolved_dict: Dict[str, object] = resolved if isinstance(resolved, dict) else {}

    def _dump(obj: object) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

    lines: List[str] = []
    lines.append("# Tuning Notes (mpm_fem_rgb_compare)")
    lines.append("")
    lines.append("## Baseline command")
    lines.append("```json")
    lines.append(_dump(list(sys.argv)))
    lines.append("```")
    if reason:
        lines.append(f"- reason: `{str(reason)}`")
    lines.append("")
    lines.append("## Key frames (frame_id)")
    for k in ["press_end", "slide_mid", "hold_end"]:
        lines.append(f"- {k}: `{key_frames.get(k)}`")
    lines.append("")
    lines.append("## Resolved (for diff/audit)")
    for section in ["friction", "scale", "indenter", "conventions", "render", "export", "contact"]:
        if section in resolved_dict:
            lines.append(f"### {section}")
            lines.append("```json")
            lines.append(_dump(resolved_dict.get(section)))
            lines.append("```")
            lines.append("")
    lines.append("## Conclusion")
    lines.append("- (fill in) What changed, what improved, what to try next.")
    lines.append("")
    lines.append("## Repro / analysis")
    lines.append(f"- Run intermediate analysis: `python example/analyze_rgb_compare_intermediate.py --save-dir {save_dir}`")
    lines.append("")

    try:
        notes_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"Warning: failed to write tuning_notes.md: {e}")


def _write_preflight_run_manifest(
    save_dir: Path,
    record_interval: int,
    total_frames: int,
    run_context: Dict[str, object],
    *,
    reason: Optional[str] = None,
) -> None:
    """
    Write a minimal run_manifest.json before entering the heavy render/sim path.

    This keeps outputs auditable even when optional dependencies (ezgl/taichi)
    are missing in the current environment.
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    press_steps = int(SCENE_PARAMS["press_steps"])
    slide_steps = int(SCENE_PARAMS["slide_steps"])
    hold_steps = int(SCENE_PARAMS["hold_steps"])
    total_steps = press_steps + slide_steps + hold_steps

    def _phase_for_step(step: int) -> str:
        if step < press_steps:
            return "press"
        if step < press_steps + slide_steps:
            return "slide"
        return "hold"

    frame_to_step = [int(i * record_interval) for i in range(int(total_frames))]
    frame_to_phase = [_phase_for_step(step) for step in frame_to_step]
    phase_ranges: Dict[str, Dict[str, int]] = {}
    for i, phase in enumerate(frame_to_phase):
        if phase not in phase_ranges:
            phase_ranges[phase] = {"start_frame": i, "end_frame": i}
        else:
            phase_ranges[phase]["end_frame"] = i

    sanitized_run_context = _sanitize_run_context_for_manifest(run_context)
    marker_appearance = None
    if isinstance(sanitized_run_context, dict):
        resolved = sanitized_run_context.get("resolved")
        if isinstance(resolved, dict):
            marker_appearance = resolved.get("marker_appearance")

    manifest: Dict[str, object] = {
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "argv": _sanitize_argv_for_manifest(sys.argv),
        "run_context": sanitized_run_context,
        "marker_appearance": marker_appearance,
        "scene_params": dict(SCENE_PARAMS),
        "deps": _collect_manifest_deps(_PROJECT_ROOT),
        "execution": {
            "stage": "preflight",
            "note": "Written before running; may be overwritten by the runtime manifest.",
            "reason": str(reason) if reason else None,
        },
        "trajectory": {
            "press_steps": press_steps,
            "slide_steps": slide_steps,
            "hold_steps": hold_steps,
            "total_steps": total_steps,
            "record_interval": int(record_interval),
            "total_frames": int(total_frames),
            "phase_ranges_frames": phase_ranges,
            "frame_to_step": frame_to_step,
            "frame_to_phase": frame_to_phase,
            "frame_controls": None,
        },
        "outputs": {
            "frames_glob": {
                "fem": "fem_*.png",
                "mpm": "mpm_*.png",
            },
            "run_manifest": "run_manifest.json",
            "metrics": {
                "csv": "metrics.csv",
                "json": "metrics.json",
            },
            "intermediate": {
                "dir": "intermediate",
                "frames_glob": "intermediate/frame_*.npz",
            },
        },
    }

    manifest_path = save_dir / "run_manifest.json"
    try:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Warning: failed to write preflight run manifest: {e}")
        return

    try:
        _write_tuning_notes(
            save_dir,
            record_interval=int(record_interval),
            total_frames=int(total_frames),
            run_context=manifest.get("run_context") if isinstance(manifest, dict) else {},
            overwrite=False,
            reason=reason,
        )
    except Exception as e:
        print(f"Warning: failed to write tuning notes: {e}")


# ==============================================================================
# FEM RGB Renderer (Reuses VecTouchSim)
# ==============================================================================
class FEMRGBRenderer:
    """FEM sensor RGB rendering using existing VecTouchSim pipeline"""

    def __init__(
        self,
        fem_file: str,
        object_file: Optional[str] = None,
        visible: bool = False,
        indenter_face: str = "tip",
        indenter_geom: str = "stl",
    ):
        if not HAS_EZGL:
            raise RuntimeError("ezgl not available for FEM rendering")   

        self.fem_file = Path(fem_file)
        self.object_file = object_file
        self.visible = visible
        self.indenter_face = indenter_face
        self.indenter_geom = indenter_geom

        # Create depth scene for object rendering
        self.depth_scene = self._create_depth_scene()

        # Create VecTouchSim for FEM sensor rendering
        self.sensor_sim = VecTouchSim(
            depth_size=SCENE_PARAMS['depth_img_size'],
            fem_file=str(fem_file),
            visible=visible,
            title="FEM Sensor"
        )
        self.sensor_sim.set_friction_coefficient(float(SCENE_PARAMS.get("fem_fric_coef", 0.4)))

        # Object pose (indenter position)
        self._object_y = 0.02  # initial y position (m)
        self._object_z = 0.0   # will be set by trajectory

    def _create_depth_scene(self) -> 'DepthRenderScene':
        """Create depth rendering scene for the indenter"""
        return DepthRenderScene(
            object_file=self.object_file,
            visible=self.visible,
            indenter_face=self.indenter_face,
            indenter_geom=self.indenter_geom,
        )

    def set_indenter_pose(self, x_mm: float, y_mm: float, z_mm: float):
        """Set indenter position in mm (sensor coordinates)"""
        # Convert to meters for depth scene
        x_m = x_mm * 1e-3
        y_m = y_mm * 1e-3
        z_m = z_mm * 1e-3
        self.depth_scene.set_object_pose(x_m, y_m, z_m)

    def step(self) -> np.ndarray:
        """Run one step and return RGB image"""
        # Update depth scene to render object
        self.depth_scene.update()

        # Render depth map
        depth = self.depth_scene.get_depth()

        # Get poses
        sensor_pose = self.depth_scene.get_sensor_pose()
        object_pose = self.depth_scene.get_object_pose()

        # Step FEM simulation
        self.sensor_sim.step(object_pose, sensor_pose, depth)

        # Update sensor scene rendering
        self.sensor_sim.update()

        return self.get_image()

    def get_image(self) -> np.ndarray:
        """Get current RGB image (H, W, 3) uint8"""
        return self.sensor_sim.get_image()

    def get_diff_image(self) -> np.ndarray:
        """Get diff image relative to reference"""
        return self.sensor_sim.get_diff_image()

    def update(self):
        """Update visualization windows"""
        if self.visible:
            self.depth_scene.update()
            self.sensor_sim.update()


class DepthRenderScene(Scene):
    """Simple depth rendering scene for indenter object"""

    def __init__(
        self,
        object_file: Optional[str] = None,
        visible: bool = True,
        indenter_face: str = "tip",
        indenter_geom: str = "stl",
    ):
        # Note: visible=True is required for proper OpenGL context initialization
        super().__init__(600, 400, visible=visible, title="Depth Render")
        self.cameraLookAt((0.1, 0.1, 0.1), (0, 0, 0), (0, 1, 0))

        # Camera view parameters
        self.cam_view_width = SCENE_PARAMS['cam_view_width_m']
        self.cam_view_height = SCENE_PARAMS['cam_view_height_m']

        self._indenter_geom = str(indenter_geom).strip().lower()
        if self._indenter_geom not in {"stl", "box", "sphere"}:
            self._indenter_geom = "stl"

        # Create indenter object
        stl_path: Optional[Path] = None
        if self._indenter_geom == "box":
            half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
            if half_extents_mm is not None:
                hx_mm, hy_mm, hz_mm = half_extents_mm
            else:
                r_mm = float(SCENE_PARAMS["indenter_radius_mm"])
                hx_mm = hy_mm = hz_mm = r_mm
            size_m = (float(hx_mm) * 2e-3, float(hy_mm) * 2e-3, float(hz_mm) * 2e-3)
            self.object = GLBoxItem(size=size_m, glOptions="translucent")
        elif self._indenter_geom == "sphere":
            r_m = float(SCENE_PARAMS["indenter_radius_mm"]) * 1e-3
            verts, faces = ezgl_mesh_sphere(radius=r_m, rows=24, cols=24)
            self.object = GLMeshItem(
                vertexes=verts,
                indices=faces,
                lights=PointLight(),
                glOptions="translucent",
            )
        else:
            # STL path: prefer explicit object_file if provided
            if object_file and Path(object_file).exists():
                stl_path = Path(object_file)
            else:
                stl_path = ASSET_DIR / "obj/circle_r4.STL"
            self.object = GLModelItem(
                str(stl_path),
                glOptions="translucent",
                lights=PointLight(),
            )

        # Depth camera
        self.depth_cam = DepthCamera(
            self,
            eye=(0, 0, 0),
            center=(0, 1, 0),
            up=(0, 0, 1),
            img_size=SCENE_PARAMS['depth_img_size'],
            proj_type="ortho",
            ortho_space=(
                -self.cam_view_width / 2, self.cam_view_width / 2,
                -self.cam_view_height / 2, self.cam_view_height / 2,
                -0.005, 0.1
            ),
            frustum_visible=True,
            actual_depth=True
        )

        # IMPORTANT: setTransform() will overwrite any constructor-time rotate/translate.
        # 所以“固定旋转/偏移”必须显式地与每帧 pose 合成，再一次性 setTransform()，避免隐式状态。
        self._indenter_face = indenter_face
        self._stl_endfaces_mm: Optional[Dict[str, object]] = None
        self._stl_height_m = 0.0
        if self._indenter_geom == "stl" and stl_path and stl_path.suffix.lower() == ".stl" and stl_path.exists():
            stats = _analyze_binary_stl_endfaces_mm(stl_path)
            if stats is not None:
                self._stl_endfaces_mm = stats
                height_mm = float(stats.get("height_mm", 0.0))
                self._stl_height_m = max(height_mm * 1e-3, 0.0)

        self._object_fixed_tf = Matrix4x4()     # local->parent 固定变换（轴对齐/朝向/模型原点偏移等）
        self._object_pose_raw = Matrix4x4()     # 每帧输入 pose（仅平移/旋转控制量）
        self._object_pose = Matrix4x4()         # 最终用于渲染+FEM 的世界变换（raw * fixed）
        self._apply_indenter_face()

    def _apply_indenter_face(self) -> None:
        if self._indenter_geom != "stl":
            self._object_fixed_tf = Matrix4x4()
            return
        face = str(self._indenter_face).strip().lower()
        if face not in {"base", "tip"}:
            face = "base"
        self._indenter_face = face

        self._object_fixed_tf = Matrix4x4()
        if face == "tip":
            # 目标：把 STL 的 y_max 端面翻到 y_min 方向，确保“tip”端面可用于接触对齐验证。
            # 使用 y 轴翻转（绕 X 轴 180°），并用 STL 高度做一次平移以保持 tip 端面仍位于局部 y≈0 平面。
            if self._stl_height_m > 0:
                self._object_fixed_tf.translate(0, float(self._stl_height_m), 0)
            self._object_fixed_tf.rotate(180, 1, 0, 0)

    def set_object_pose(self, x: float, y: float, z: float):
        """Set object position in meters"""
        self._object_pose_raw = Matrix4x4.fromVector6d(x, y, z, 0, 0, 0)
        final_tf = Matrix4x4(self._object_pose_raw) * self._object_fixed_tf
        self._object_pose = Matrix4x4(final_tf)
        self.object.setTransform(self._object_pose)

        if SCENE_PARAMS.get("debug_verbose", False):
            try:
                raw_xyz = self._object_pose_raw.xyz.tolist()
                fixed_xyz = self._object_fixed_tf.xyz.tolist()
                final_xyz = self._object_pose.xyz.tolist()
                raw_euler = getattr(self._object_pose_raw, "euler", None)
                fixed_euler = getattr(self._object_fixed_tf, "euler", None)
                final_euler = getattr(self._object_pose, "euler", None)
                print(
                    "[DepthRenderScene] "
                    f"raw_xyz={raw_xyz}, fixed_xyz={fixed_xyz}, final_xyz={final_xyz}; "
                    f"raw_euler={raw_euler}, fixed_euler={fixed_euler}, final_euler={final_euler}"
                )
            except Exception:
                # debug 模式下也不应因为日志失败而影响渲染
                pass

    def get_object_pose(self) -> Matrix4x4:
        return self._object_pose

    def get_object_pose_raw(self) -> Matrix4x4:
        return self._object_pose_raw

    def get_sensor_pose(self) -> Matrix4x4:
        return self.depth_cam.transform(local=False)

    def get_depth(self) -> np.ndarray:
        depth = self.depth_cam.render()

        # CRITICAL: Filter out background (far plane) values
        # Background has depth = far = 0.1m
        # FEM contact detection: z_gel < p_gel[:, 2]
        # After FEM processing: depth_map = depth * 0.4 * 1000 (mm)
        # Background 0.1m -> 40mm, which may still trigger contact if gel z < 40mm
        # Solution: Set background to a very large value (e.g., 1000mm) so it never triggers contact
        far_value = 0.1
        background_mask = depth >= (far_value - 0.001)  # Pixels at or near far plane
        depth_filtered = depth.copy()
        depth_filtered[background_mask] = 10.0  # 10m = 10000mm after processing, definitely no contact

        return depth_filtered


# ==============================================================================
# MPM Height Field Renderer
# ==============================================================================
class MPMHeightFieldRenderer:
    """Renders MPM particle data as sensor RGB images via height field extraction"""

    def __init__(self, visible: bool = False):
        if not HAS_EZGL:
            raise RuntimeError("ezgl not available for MPM rendering")

        self.visible = visible
        self.gel_size_mm = SCENE_PARAMS['gel_size_mm']
        self.grid_shape = SCENE_PARAMS['height_grid_shape']

        # Create rendering scene
        self.scene = self._create_render_scene()

        # Reference height field (flat surface)
        self._ref_height = np.zeros(self.grid_shape, dtype=np.float32)

        # Cached mapping derived from initial particle positions (stable across frames)
        self._is_configured = False
        self._x_center_mm = 0.0
        self._y_min_mm = 0.0
        self._surface_indices: Optional[np.ndarray] = None
        self._initial_positions_m: Optional[np.ndarray] = None
        self._last_height_reference_z_mm = 0.0
        # Cached pre-clamp height field (after reference_z alignment, before indenter clamp). Used to select
        # top-surface particles for uv_disp extraction when enabled.
        self._last_height_field_preclamp_mm: Optional[np.ndarray] = None
        # UV despike diagnostics (set by extract_surface_fields when enabled)
        self._last_uv_despike_footprint_mask_u8: Optional[np.ndarray] = None
        self._last_uv_despike_gate_mask_u8: Optional[np.ndarray] = None
        self._last_uv_despike_replaced_mask_u8: Optional[np.ndarray] = None
        self._last_uv_despike_capped_mask_u8: Optional[np.ndarray] = None
        # UV mask diagnostics (set by extract_surface_fields when enabled)
        self._last_uv_mask_footprint_mask_u8: Optional[np.ndarray] = None
        # UV coverage diagnostics (set by extract_surface_fields)
        self._last_uv_cnt_i32: Optional[np.ndarray] = None
        self._last_uv_nonzero_mask_u8: Optional[np.ndarray] = None
        self._last_uv_hole_count_i32: Optional[np.ndarray] = None
        # UV pseudo-hole diagnostics: uv_cnt>0 but |uv|≈0 inside footprint (a common cause of frozen markers)
        self._last_uv_pseudohole_mask_u8: Optional[np.ndarray] = None
        self._last_uv_pseudohole_count_i32: Optional[np.ndarray] = None
        # Surface tangential motion field diagnostics (Δu per recorded frame, derived from particle velocities)
        self._last_uv_du_mm: Optional[np.ndarray] = None
        self._last_uv_du_cnt_i32: Optional[np.ndarray] = None
        self._last_uv_du_nonzero_mask_u8: Optional[np.ndarray] = None
        self._last_uv_du_hole_count_i32: Optional[np.ndarray] = None

    def _create_render_scene(self) -> 'MPMSensorScene':
        """Create rendering scene matching SensorScene style"""
        return MPMSensorScene(
            gel_size_mm=self.gel_size_mm,
            grid_shape=self.grid_shape,
            visible=self.visible
        )

    def configure_from_initial_positions(self, initial_positions_m: np.ndarray, initial_top_z_m: float) -> None:
        """
        Cache a stable mapping from MPM solver coordinates to sensor grid coordinates.

        - Avoids per-frame recentring (which cancels slide motion and causes jitter)
        - Derives a thin top-surface particle mask to produce a proper height field
        """
        pos_mm = initial_positions_m * 1000.0
        self._x_center_mm = float((pos_mm[:, 0].min() + pos_mm[:, 0].max()) / 2.0)
        self._y_min_mm = float(pos_mm[:, 1].min())

        dx_m = float(SCENE_PARAMS['mpm_grid_dx_mm']) * 1e-3
        particles_per_cell = float(SCENE_PARAMS['mpm_particles_per_cell'])
        particle_spacing_m = dx_m / max(particles_per_cell, 1.0)
        surface_band_m = 2.0 * particle_spacing_m

        z_threshold_m = float(initial_top_z_m) - surface_band_m
        surface_mask = initial_positions_m[:, 2] >= z_threshold_m
        self._surface_indices = np.nonzero(surface_mask)[0].astype(np.int32)
        self._initial_positions_m = initial_positions_m.copy()
        self._is_configured = True

    def extract_height_field(
        self,
        positions_m: np.ndarray,
        initial_top_z_m: float,
        indenter_center_m: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        Extract top-surface height field from MPM particles

        Args:
            positions_m: Particle positions (N, 3) in meters
            initial_top_z_m: Initial top surface z coordinate in meters  
            indenter_center_m: Optional indenter center (x,y,z) in meters, used to clamp
                height_field not below indenter surface (avoid penalty penetration artifacts).

        Returns:
            height_field_mm: (n_row, n_col) array, negative = indentation
        """
        n_row, n_col = self.grid_shape
        gel_w_mm, gel_h_mm = self.gel_size_mm

        if not self._is_configured:
            self.configure_from_initial_positions(positions_m, initial_top_z_m)

        # NOTE: 这里不能只取“初始顶面一层粒子”。在平底压头（cylinder/box）接触中，
        # 初始顶面粒子可能会被挤压下沉并被更深层粒子“顶替”成为新表面；
        # 若仅追踪初始顶面索引，会把“已下沉的旧表面”误当成当前表面，导致高度场过深，
        # 进而在渲染中出现非物理的“整块变暗/发脏”。
        pos_mm = positions_m * 1000.0
        z_top_init_mm = initial_top_z_m * 1000.0

        # Map to sensor grid using cached reference frame:
        # x ∈ [-gel_w/2, gel_w/2], y ∈ [0, gel_h]
        pos_sensor = pos_mm.copy()
        pos_sensor[:, 0] -= self._x_center_mm
        pos_sensor[:, 1] -= self._y_min_mm

        # Grid cell dimensions
        cell_w = gel_w_mm / n_col
        cell_h = gel_h_mm / n_row

        # Build height field using a small neighborhood splat to reduce holes.
        # The render grid resolution is close to particle spacing, so strict per-cell binning
        # creates pepper noise (empty cells) which becomes hard edges after shading.
        x_mm = pos_sensor[:, 0].astype(np.float32, copy=False)
        y_mm = pos_sensor[:, 1].astype(np.float32, copy=False)
        z_mm = pos_sensor[:, 2].astype(np.float32, copy=False)
        z_disp = (z_mm - np.float32(z_top_init_mm)).astype(np.float32, copy=False)  # <= 0

        col_f = (x_mm + np.float32(gel_w_mm / 2.0)) / np.float32(cell_w) - np.float32(0.5)
        row_f = y_mm / np.float32(cell_h) - np.float32(0.5)
        col0 = np.floor(col_f).astype(np.int32)
        row0 = np.floor(row_f).astype(np.int32)

        # Initialize with -inf so we can take max z_disp per cell (top surface).
        height_field = np.full((n_row, n_col), -np.inf, dtype=np.float32)
        for di in (0, 1):
            rr = row0 + di
            for dj in (0, 1):
                cc = col0 + dj
                m = (rr >= 0) & (rr < n_row) & (cc >= 0) & (cc < n_col)
                if m.any():
                    np.maximum.at(height_field, (rr[m], cc[m]), z_disp[m])

        height_field[~np.isfinite(height_field)] = np.nan
        valid_mask = np.isfinite(height_field)

        reference_z = 0.0
        if bool(SCENE_PARAMS.get("mpm_height_reference_edge", True)):    
            # CRITICAL: Use EDGE regions as fixed spatial reference for better contrast.
            # This preserves slide motion (unlike global percentile which cancels it).
            # Use original valid cells for baseline to avoid bias from hole filling.
            edge_margin = max(3, n_row // 20)  # ~5% margin or at least 3 rows/cols
            edge_mask = np.zeros_like(valid_mask, dtype=bool)
            edge_mask[:edge_margin, :] = True
            edge_mask[-edge_margin:, :] = True
            if edge_margin * 2 < n_row and edge_margin * 2 < n_col:
                edge_mask[edge_margin:-edge_margin, :edge_margin] = True
                edge_mask[edge_margin:-edge_margin, -edge_margin:] = True

            edge_values = height_field[edge_mask & valid_mask]
            edge_valid = edge_values[np.isfinite(edge_values) & (edge_values > -10)]  # filter outliers
            if edge_valid.size > 0:
                reference_z = float(np.median(edge_valid))  # Median is robust to outliers
                height_field = height_field - reference_z  # Now center region depression is negative

        # Cache the per-run reference used by extract_surface_fields() so UV selection is consistent.
        self._last_height_reference_z_mm = float(reference_z)
        # Cache the pre-clamp height field for UV top-surface selection (when enabled).
        # This must be captured AFTER reference_z alignment, and BEFORE indenter clamp / hole fill / smoothing.
        self._last_height_field_preclamp_mm = height_field.astype(np.float32, copy=True)

        # IMPORTANT: Clamp to indenter surface to suppress over-penetration artifacts.
        if (
            indenter_center_m is not None
            and bool(SCENE_PARAMS.get("mpm_height_clamp_indenter", True))
        ):
            try:
                cx_mm = float(indenter_center_m[0]) * 1000.0 - float(self._x_center_mm)
                cy_mm = float(indenter_center_m[1]) * 1000.0 - float(self._y_min_mm)
                cz_mm = float(indenter_center_m[2]) * 1000.0

                x_centers = (np.arange(n_col, dtype=np.float32) + 0.5) * np.float32(cell_w) - np.float32(gel_w_mm / 2.0)
                y_centers = (np.arange(n_row, dtype=np.float32) + 0.5) * np.float32(cell_h)
                xx, yy = np.meshgrid(x_centers, y_centers)

                clamp_field = np.full((n_row, n_col), np.nan, dtype=np.float32)
                indenter_type = str(SCENE_PARAMS.get("indenter_type", "box")).lower().strip()

                if indenter_type == "sphere":
                    r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                    rr = np.sqrt((xx - np.float32(cx_mm)) ** 2 + (yy - np.float32(cy_mm)) ** 2)
                    inside = rr <= np.float32(r_mm)
                    # Sphere lower hemisphere: z = cz - sqrt(R^2 - r^2)
                    dz = np.sqrt(np.maximum(np.float32(r_mm) ** 2 - rr ** 2, 0.0))
                    z_surface_mm = np.float32(cz_mm) - dz
                    surface_disp = (z_surface_mm - np.float32(z_top_init_mm)) - np.float32(reference_z)
                    clamp_field[inside] = surface_disp[inside].astype(np.float32, copy=False)
                    clamp_field[~(clamp_field < 0.0)] = np.nan
                elif indenter_type == "cylinder":
                    r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                    half_h_mm = SCENE_PARAMS.get("indenter_cylinder_half_height_mm", None)
                    half_h_mm = float(r_mm if half_h_mm is None else float(half_h_mm))
                    inside = ((xx - np.float32(cx_mm)) ** 2 + (yy - np.float32(cy_mm)) ** 2) <= np.float32(r_mm) ** 2
                    surface_disp = (np.float32(cz_mm) - np.float32(half_h_mm) - np.float32(z_top_init_mm)) - np.float32(reference_z)
                    if float(surface_disp) < 0.0:
                        clamp_field[inside] = np.float32(surface_disp)
                else:
                    half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
                    if half_extents_mm is not None and len(half_extents_mm) == 3:
                        hx_mm, hy_mm, hz_mm = [float(v) for v in half_extents_mm]
                    else:
                        r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                        hx_mm = hy_mm = hz_mm = float(r_mm)
                    inside = (np.abs(xx - np.float32(cx_mm)) <= np.float32(hx_mm)) & (np.abs(yy - np.float32(cy_mm)) <= np.float32(hy_mm))
                    surface_disp = (np.float32(cz_mm) - np.float32(hz_mm) - np.float32(z_top_init_mm)) - np.float32(reference_z)
                    if float(surface_disp) < 0.0:
                        clamp_field[inside] = np.float32(surface_disp)

                # Use fmax to keep NaNs outside indenter footprint.
                height_field = np.fmax(height_field, clamp_field)
                valid_mask = np.isfinite(height_field)
            except Exception as e:
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM HEIGHT] clamp_to_indenter failed: {e}")

        # 可选：footprint 外离群深值裁剪（避免“深坑”把整块区域渲染成暗盘/彩虹 halo）
        if bool(SCENE_PARAMS.get("mpm_height_clip_outliers", False)):
            clip_min = float(SCENE_PARAMS.get("mpm_height_clip_outliers_min_mm", 0.0))
            if clip_min > 0.0:
                floor_mm = -abs(clip_min)
                footprint_mask = None
                if indenter_center_m is not None:
                    try:
                        cx_mm = float(indenter_center_m[0]) * 1000.0 - float(self._x_center_mm)
                        cy_mm = float(indenter_center_m[1]) * 1000.0 - float(self._y_min_mm)

                        x_centers = (np.arange(n_col, dtype=np.float32) + 0.5) * np.float32(cell_w) - np.float32(gel_w_mm / 2.0)
                        y_centers = (np.arange(n_row, dtype=np.float32) + 0.5) * np.float32(cell_h)
                        xx, yy = np.meshgrid(x_centers, y_centers)

                        indenter_type = str(SCENE_PARAMS.get("indenter_type", "box")).lower().strip()
                        if indenter_type in {"sphere", "cylinder"}:
                            r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                            footprint_mask = ((xx - np.float32(cx_mm)) ** 2 + (yy - np.float32(cy_mm)) ** 2) <= np.float32(r_mm) ** 2
                        else:
                            half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
                            if half_extents_mm is not None and len(half_extents_mm) == 3:
                                hx_mm, hy_mm, _ = [float(v) for v in half_extents_mm]
                            else:
                                r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                                hx_mm = hy_mm = float(r_mm)
                            footprint_mask = (np.abs(xx - np.float32(cx_mm)) <= np.float32(hx_mm)) & (np.abs(yy - np.float32(cy_mm)) <= np.float32(hy_mm))
                    except Exception as e:
                        footprint_mask = None
                        if SCENE_PARAMS.get("debug_verbose", False):
                            print(f"[MPM HEIGHT] footprint_mask failed: {e}")

                try:
                    if footprint_mask is not None and isinstance(footprint_mask, np.ndarray):
                        outliers = (~footprint_mask) & np.isfinite(height_field) & (height_field < floor_mm)
                    else:
                        outliers = np.isfinite(height_field) & (height_field < floor_mm)
                    if outliers.any():
                        height_field[outliers] = np.nan
                        valid_mask = np.isfinite(height_field)
                except Exception as e:
                    if SCENE_PARAMS.get("debug_verbose", False):
                        print(f"[MPM HEIGHT] clip_outliers failed: {e}")

        # Fill holes after baseline alignment to avoid converting holes into small positive bumps
        # (which get clamped to 0 and create hard edges / rainbow halos after shading).
        if bool(SCENE_PARAMS.get("mpm_height_fill_holes", False)):
            iters = int(SCENE_PARAMS.get("mpm_height_fill_holes_iters", 10))
            if iters > 0:
                try:
                    height_field = _fill_height_holes(height_field, valid_mask, max_iterations=iters)
                except Exception:
                    height_field = np.nan_to_num(height_field, nan=0.0)
            else:
                height_field = np.nan_to_num(height_field, nan=0.0)
        else:
            height_field = np.nan_to_num(height_field, nan=0.0)

        # Debug: check height field statistics
        valid_mask = height_field < -0.05  # Only count significant depressions
        if valid_mask.any():
            # Find center of mass of the depression
            rows, cols = np.where(valid_mask)
            if len(rows) > 0:
                center_row = np.average(rows, weights=-height_field[rows, cols])
                center_col = np.average(cols, weights=-height_field[rows, cols])
                center_x_mm = (center_col - n_col/2) * cell_w
                center_y_mm = center_row * cell_h

                # Debug: show actual surface particle x positions
                x_positions = pos_sensor[:, 0]
                if SCENE_PARAMS.get('debug_verbose', False):
                    print(f"[MPM HEIGHT] min={height_field.min():.2f}mm, cells={valid_mask.sum()}, "
                          f"center=({center_x_mm:.1f}, {center_y_mm:.1f})mm | "
                          f"particles: x_range=[{x_positions.min():.1f}, {x_positions.max():.1f}]mm")
        else:
            if SCENE_PARAMS.get('debug_verbose', False):
                print(f"[MPM HEIGHT] No significant deformation (threshold=0.05mm), particles={len(pos_sensor)}")

        return height_field

    def extract_surface_fields(
        self,
        positions_m: np.ndarray,
        initial_top_z_m: float,
        indenter_center_m: Optional[Tuple[float, float, float]] = None,
        smooth_uv: bool = True,
        *,
        velocities_m: Optional[np.ndarray] = None,
        frame_dt_s: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时提取顶面高度场与面内位移场（u,v）。

        Returns:
            height_field_mm: (Ny,Nx), <= 0 表示压入
            uv_disp_mm: (Ny,Nx,2), 单位 mm
        """
        height_field = self.extract_height_field(
            positions_m,
            initial_top_z_m,
            indenter_center_m=indenter_center_m,
        )

        if not self._is_configured:
            self.configure_from_initial_positions(positions_m, initial_top_z_m)

        # Reset UV diagnostics (avoid stale values when early-returning).
        self._last_uv_cnt_i32 = None
        self._last_uv_nonzero_mask_u8 = None
        self._last_uv_hole_count_i32 = None
        self._last_uv_pseudohole_mask_u8 = None
        self._last_uv_pseudohole_count_i32 = None
        self._last_uv_du_mm = None
        self._last_uv_du_cnt_i32 = None
        self._last_uv_du_nonzero_mask_u8 = None
        self._last_uv_du_hole_count_i32 = None
        if self._initial_positions_m is None:
            n_row, n_col = self.grid_shape
            uv = np.zeros((n_row, n_col, 2), dtype=np.float32)
            self._last_uv_cnt_i32 = np.zeros((n_row, n_col), dtype=np.int32)
            self._last_uv_nonzero_mask_u8 = np.zeros((n_row, n_col), dtype=np.uint8)
            self._last_uv_hole_count_i32 = np.array(int(n_row * n_col), dtype=np.int32)
            return height_field, uv

        n_row, n_col = self.grid_shape
        gel_w_mm, gel_h_mm = self.gel_size_mm

        pos_mm_all = positions_m * 1000.0
        init_mm_all = self._initial_positions_m * 1000.0

        # NOTE: 同 extract_height_field()，这里也必须使用“当前顶面”而不是“初始顶面一层”。
        # 否则在平底压头接触/滑移时，UV 会接近 0，marker 看起来像贴在屏幕上不动。

        pos_sensor = pos_mm_all.copy()
        init_sensor = init_mm_all.copy()
        pos_sensor[:, 0] -= self._x_center_mm
        pos_sensor[:, 1] -= self._y_min_mm
        init_sensor[:, 0] -= self._x_center_mm
        init_sensor[:, 1] -= self._y_min_mm

        cell_w = gel_w_mm / n_col
        cell_h = gel_h_mm / n_row

        uv_sum = np.zeros((n_row, n_col, 2), dtype=np.float32)
        uv_cnt = np.zeros((n_row, n_col), dtype=np.int32)

        disp_x = (pos_sensor[:, 0] - init_sensor[:, 0]).astype(np.float32, copy=False)
        disp_y = (pos_sensor[:, 1] - init_sensor[:, 1]).astype(np.float32, copy=False)

        # NOTE: marker_mode=advect expects u(X) on a reference grid (Lagrangian),
        # while marker_mode=warp expects u(x) on the current grid (Eulerian).
        # Keep both behind an explicit switch for backward compatibility.
        if bool(SCENE_PARAMS.get("mpm_uv_bin_init_xy", False)):
            # Lagrangian UV u(X): select *current* top-surface particles, but accumulate their displacement
            # onto bins defined by their *initial* XY (reference) coordinates.
            #
            # Why: marker advection moves dot at X to X + u(X). If we bin by current XY (Eulerian),
            # then sample at reference dot centers, sliding can look like “front moves / back stays”.
            x_mm = pos_sensor[:, 0].astype(np.float32, copy=False)
            y_mm = pos_sensor[:, 1].astype(np.float32, copy=False)
            z_mm = pos_sensor[:, 2].astype(np.float32, copy=False)
            z_top_init_mm = np.float32(initial_top_z_m * 1000.0)
            z_disp = (z_mm - z_top_init_mm).astype(np.float32, copy=False)
            z_disp = z_disp - np.float32(getattr(self, "_last_height_reference_z_mm", 0.0))

            # Height reference for surface selection (same as Eulerian path).
            height_ref = height_field
            if bool(SCENE_PARAMS.get("mpm_uv_ref_preclamp_height", False)):
                pre = getattr(self, "_last_height_field_preclamp_mm", None)
                if isinstance(pre, np.ndarray) and pre.shape == height_field.shape:
                    height_ref = pre

            dx_m = float(SCENE_PARAMS['mpm_grid_dx_mm']) * 1e-3
            particles_per_cell = float(SCENE_PARAMS['mpm_particles_per_cell'])
            particle_spacing_m = dx_m / max(particles_per_cell, 1.0)
            surface_band_mm = np.float32(2.0 * particle_spacing_m * 1000.0)

            # Current-cell (nearest) for top-surface selection.
            col_f_cur = (x_mm + np.float32(gel_w_mm / 2.0)) / np.float32(cell_w) - np.float32(0.5)
            row_f_cur = y_mm / np.float32(cell_h) - np.float32(0.5)
            col_cur = np.rint(col_f_cur).astype(np.int32)
            row_cur = np.rint(row_f_cur).astype(np.int32)
            in_grid = (row_cur >= 0) & (row_cur < n_row) & (col_cur >= 0) & (col_cur < n_col)
            if in_grid.any():
                rr = row_cur[in_grid]
                cc = col_cur[in_grid]
                ref = height_ref[rr, cc].astype(np.float32, copy=False)
                z_m = z_disp[in_grid]
                top = np.isfinite(ref) & (z_m >= (ref - surface_band_mm))
                if top.any():
                    sel = np.nonzero(in_grid)[0][top]

                    # Initial-cell (nearest) for Lagrangian binning.
                    x0_mm = init_sensor[sel, 0].astype(np.float32, copy=False)
                    y0_mm = init_sensor[sel, 1].astype(np.float32, copy=False)
                    col_f0 = (x0_mm + np.float32(gel_w_mm / 2.0)) / np.float32(cell_w) - np.float32(0.5)
                    row_f0 = y0_mm / np.float32(cell_h) - np.float32(0.5)
                    col0 = np.rint(col_f0).astype(np.int32)
                    row0 = np.rint(row_f0).astype(np.int32)
                    m0 = (row0 >= 0) & (row0 < n_row) & (col0 >= 0) & (col0 < n_col)
                    if m0.any():
                        rr0 = row0[m0]
                        cc0 = col0[m0]
                        np.add.at(uv_sum[..., 0], (rr0, cc0), disp_x[sel][m0])
                        np.add.at(uv_sum[..., 1], (rr0, cc0), disp_y[sel][m0])
                        np.add.at(uv_cnt, (rr0, cc0), 1)
        else:
            # Eulerian UV: bin by particles' current XY and keep only top-surface particles per cell.
            # Vectorized per-cell surface displacement:
            # - reuse the same 4-neighbor splat as height extraction
            # - only keep particles near each cell's top surface (within `surface_band_mm`)
            x_mm = pos_sensor[:, 0].astype(np.float32, copy=False)
            y_mm = pos_sensor[:, 1].astype(np.float32, copy=False)
            z_mm = pos_sensor[:, 2].astype(np.float32, copy=False)
            z_top_init_mm = np.float32(initial_top_z_m * 1000.0)
            z_disp = (z_mm - z_top_init_mm).astype(np.float32, copy=False)
            z_disp = z_disp - np.float32(getattr(self, "_last_height_reference_z_mm", 0.0))

            # Choose the per-cell height reference used for selecting "top surface" particles.
            # Using the post-clamp height_field can be overly optimistic (flat indenter clamp),
            # rejecting true surface particles and causing uv_cnt==0 in parts of the footprint.
            height_ref = height_field
            if bool(SCENE_PARAMS.get("mpm_uv_ref_preclamp_height", False)):
                pre = getattr(self, "_last_height_field_preclamp_mm", None)
                if isinstance(pre, np.ndarray) and pre.shape == height_field.shape:
                    height_ref = pre

            col_f = (x_mm + np.float32(gel_w_mm / 2.0)) / np.float32(cell_w) - np.float32(0.5)
            row_f = y_mm / np.float32(cell_h) - np.float32(0.5)
            col0 = np.floor(col_f).astype(np.int32)
            row0 = np.floor(row_f).astype(np.int32)

            dx_m = float(SCENE_PARAMS['mpm_grid_dx_mm']) * 1e-3
            particles_per_cell = float(SCENE_PARAMS['mpm_particles_per_cell'])
            particle_spacing_m = dx_m / max(particles_per_cell, 1.0)
            surface_band_mm = np.float32(2.0 * particle_spacing_m * 1000.0)

            for di in (0, 1):
                rr = row0 + di
                for dj in (0, 1):
                    cc = col0 + dj
                    m = (rr >= 0) & (rr < n_row) & (cc >= 0) & (cc < n_col)
                    if not m.any():
                        continue

                    rr_m = rr[m]
                    cc_m = cc[m]
                    ref = height_ref[rr_m, cc_m].astype(np.float32, copy=False)
                    z_m = z_disp[m]
                    top = np.isfinite(ref) & (z_m >= (ref - surface_band_mm))
                    if not top.any():
                        continue

                    rr_t = rr_m[top]
                    cc_t = cc_m[top]
                    np.add.at(uv_sum[..., 0], (rr_t, cc_t), disp_x[m][top])
                    np.add.at(uv_sum[..., 1], (rr_t, cc_t), disp_y[m][top])
                    np.add.at(uv_cnt, (rr_t, cc_t), 1)

        uv = np.zeros((n_row, n_col, 2), dtype=np.float32)
        nonzero = uv_cnt > 0
        uv[nonzero] = uv_sum[nonzero] / uv_cnt[nonzero][..., None]

        # Fill holes in UV field using diffusion from neighboring cells
        # This prevents discontinuities in areas without particle coverage
        hole_count_raw = int((~nonzero).sum())
        # Cache UV coverage diagnostics (pre-fill).
        self._last_uv_cnt_i32 = uv_cnt.astype(np.int32, copy=False)
        self._last_uv_nonzero_mask_u8 = nonzero.astype(np.uint8, copy=False)
        hole_count_effective = hole_count_raw
        if bool(SCENE_PARAMS.get("mpm_uv_fill_holes", True)) and hole_count_raw > 0:
            fill_iters = int(SCENE_PARAMS.get("mpm_uv_fill_holes_iters", 10))
            if fill_iters > 0:
                try:
                    uv_filled, filled_mask = _fill_uv_holes(
                        uv,
                        nonzero,
                        max_iterations=fill_iters,
                        return_filled_mask=True,
                    )
                    uv = uv_filled
                    hole_count_effective = int((~filled_mask).sum())
                except Exception:
                    # Keep best-effort behavior; fall back to raw holes count.
                    hole_count_effective = hole_count_raw
        self._last_uv_hole_count_i32 = np.array(int(hole_count_effective), dtype=np.int32)

        # Surface tangential motion field (Δu per recorded frame) derived from particle velocities.
        #
        # Why:
        # - Texture warp/remap is very sensitive to u(x)/u(X) semantic mismatch and coverage artefacts.
        # - Marker material-point advection prefers a semantically consistent motion field (v or Δu) on the render grid.
        #
        # Scope:
        # - This computes an Eulerian Δu(x) on the current grid, using the same "top surface" selection strategy
        #   as the Eulerian UV path (height_ref + surface_band).
        if velocities_m is not None and frame_dt_s is not None:
            try:
                if isinstance(velocities_m, np.ndarray) and velocities_m.shape == positions_m.shape:
                    dt_s = float(frame_dt_s)
                    if dt_s > 0.0:
                        du_sum = np.zeros((n_row, n_col, 2), dtype=np.float32)
                        du_cnt = np.zeros((n_row, n_col), dtype=np.int32)
                        du_x = (velocities_m[:, 0] * np.float32(1000.0) * np.float32(dt_s)).astype(np.float32, copy=False)
                        du_y = (velocities_m[:, 1] * np.float32(1000.0) * np.float32(dt_s)).astype(np.float32, copy=False)

                        x_mm = pos_sensor[:, 0].astype(np.float32, copy=False)
                        y_mm = pos_sensor[:, 1].astype(np.float32, copy=False)
                        z_mm = pos_sensor[:, 2].astype(np.float32, copy=False)
                        z_top_init_mm = np.float32(initial_top_z_m * 1000.0)
                        z_disp = (z_mm - z_top_init_mm).astype(np.float32, copy=False)
                        z_disp = z_disp - np.float32(getattr(self, "_last_height_reference_z_mm", 0.0))

                        height_ref = height_field
                        if bool(SCENE_PARAMS.get("mpm_uv_ref_preclamp_height", False)):
                            pre = getattr(self, "_last_height_field_preclamp_mm", None)
                            if isinstance(pre, np.ndarray) and pre.shape == height_field.shape:
                                height_ref = pre

                        col_f = (x_mm + np.float32(gel_w_mm / 2.0)) / np.float32(cell_w) - np.float32(0.5)
                        row_f = y_mm / np.float32(cell_h) - np.float32(0.5)
                        col0 = np.floor(col_f).astype(np.int32)
                        row0 = np.floor(row_f).astype(np.int32)

                        dx_m = float(SCENE_PARAMS['mpm_grid_dx_mm']) * 1e-3
                        particles_per_cell = float(SCENE_PARAMS['mpm_particles_per_cell'])
                        particle_spacing_m = dx_m / max(particles_per_cell, 1.0)
                        surface_band_mm = np.float32(2.0 * particle_spacing_m * 1000.0)

                        for di in (0, 1):
                            rr = row0 + di
                            for dj in (0, 1):
                                cc = col0 + dj
                                m = (rr >= 0) & (rr < n_row) & (cc >= 0) & (cc < n_col)
                                if not m.any():
                                    continue

                                rr_m = rr[m]
                                cc_m = cc[m]
                                ref = height_ref[rr_m, cc_m].astype(np.float32, copy=False)
                                z_m = z_disp[m]
                                top = np.isfinite(ref) & (z_m >= (ref - surface_band_mm))
                                if not top.any():
                                    continue

                                rr_t = rr_m[top]
                                cc_t = cc_m[top]
                                np.add.at(du_sum[..., 0], (rr_t, cc_t), du_x[m][top])
                                np.add.at(du_sum[..., 1], (rr_t, cc_t), du_y[m][top])
                                np.add.at(du_cnt, (rr_t, cc_t), 1)

                        du = np.zeros((n_row, n_col, 2), dtype=np.float32)
                        du_nonzero = du_cnt > 0
                        du[du_nonzero] = du_sum[du_nonzero] / du_cnt[du_nonzero][..., None]
                        self._last_uv_du_mm = du
                        self._last_uv_du_cnt_i32 = du_cnt.astype(np.int32, copy=False)
                        self._last_uv_du_nonzero_mask_u8 = du_nonzero.astype(np.uint8, copy=False)
                        self._last_uv_du_hole_count_i32 = np.array(int((~du_nonzero).sum()), dtype=np.int32)
                        if SCENE_PARAMS.get("debug_verbose", False):
                            mag = np.linalg.norm(du.astype(np.float64, copy=False), axis=-1)
                            mag_nz = mag[du_nonzero & np.isfinite(mag)]
                            p50 = float(np.percentile(mag_nz, 50.0)) if mag_nz.size else 0.0
                            p90 = float(np.percentile(mag_nz, 90.0)) if mag_nz.size else 0.0
                            print(
                                f"[MPM DU] dt_s={dt_s:.3g} cnt_nz={int(np.sum(du_nonzero))} "
                                f"hole={int(np.sum(~du_nonzero))} mag_p50={p50:.3g}mm mag_p90={p90:.3g}mm"
                            )
            except Exception as e:
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM DU] extract failed: {e}")

        # Optional: restrict uv_disp to indenter footprint (mask outside to 0) and/or despike UV.
        # Keep defaults OFF to preserve baseline behavior.
        self._last_uv_despike_footprint_mask_u8 = None
        self._last_uv_despike_gate_mask_u8 = None
        self._last_uv_despike_replaced_mask_u8 = None
        self._last_uv_despike_capped_mask_u8 = None
        self._last_uv_mask_footprint_mask_u8 = None
        need_footprint = (
            bool(SCENE_PARAMS.get("mpm_uv_despike", False))
            or bool(SCENE_PARAMS.get("mpm_uv_mask_footprint", False))
            or bool(SCENE_PARAMS.get("mpm_uv_fill_footprint_holes", False))
        )
        footprint = None
        uv_mask_outside = None
        if need_footprint and indenter_center_m is not None:
            try:
                footprint = _build_indenter_footprint_mask_u8(
                    n_row=n_row,
                    n_col=n_col,
                    gel_size_mm=(gel_w_mm, gel_h_mm),
                    indenter_type=str(SCENE_PARAMS.get("indenter_type", "box")),
                    indenter_radius_mm=float(SCENE_PARAMS.get("indenter_radius_mm", 4.0)),
                    indenter_half_extents_mm=SCENE_PARAMS.get("indenter_half_extents_mm", None),
                    indenter_center_m=(float(indenter_center_m[0]), float(indenter_center_m[1]), float(indenter_center_m[2])),
                    ref_x_center_mm=float(self._x_center_mm),
                    ref_y_min_mm=float(self._y_min_mm),
                )
                self._last_uv_despike_footprint_mask_u8 = footprint
            except Exception as e:
                footprint = None
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM UV] footprint_mask failed: {e}")
        elif need_footprint and indenter_center_m is None:
            if SCENE_PARAMS.get("debug_verbose", False):
                print("[MPM UV] footprint-based postprocess enabled but indenter_center_m missing; skipped")

        # Optional: inpaint UV inside the indenter footprint to improve continuity under flat indenters.
        #
        # Why:
        # - In press phase, uv_cnt can be non-zero but many footprint cells still end up with near-zero |uv|
        #   (selection/averaging artifacts). This makes under-indenter markers appear frozen vs FEM.
        # - A simple footprint-median vector fill works for sliding (mostly uniform tangential motion),
        #   but fails for radial press fields where the median can be ~0.
        #
        # Strategy (numpy-only, deterministic):
        # - Treat cells inside footprint with sufficiently large |uv| as seeds.
        # - Diffuse/inpaint the rest inside footprint via neighbor averaging (Laplace-style).
        # - Run BEFORE smoothing/despike so spatial blending remains FEM-like.
        if bool(SCENE_PARAMS.get("mpm_uv_fill_footprint_holes", False)) and footprint is not None:
            try:
                fp = (footprint > 0)
                if bool(fp.any()):
                    uv_f64 = uv.astype(np.float64, copy=False)
                    mag = np.linalg.norm(uv_f64, axis=-1)
                    seed_mag = mag[(uv_cnt > 0) & fp & np.isfinite(mag)]
                    # Robust inpaint seeding:
                    # - Exclude near-zero vectors (common inside footprint in press, makes markers freeze).
                    # - Exclude extreme outliers (rare spikes) so they don't dominate the diffusion.
                    min_mag = 0.03  # mm, pre-scale; matches ~0.02mm after publish_v1 uv_scale=0.62

                    # Diagnostics: pseudo-holes are cells with uv_cnt>0 but |uv| is near-zero.
                    # These are often the root cause of "under-indenter markers freeze" (A/C press late frames).
                    try:
                        pseudoholes = (uv_cnt > 0) & fp & np.isfinite(mag) & (mag < float(min_mag))
                        self._last_uv_pseudohole_mask_u8 = pseudoholes.astype(np.uint8, copy=False)
                        self._last_uv_pseudohole_count_i32 = np.array(int(np.sum(pseudoholes)), dtype=np.int32)
                    except Exception:
                        self._last_uv_pseudohole_mask_u8 = None
                        self._last_uv_pseudohole_count_i32 = None
                    max_mag = None
                    if seed_mag.size:
                        p90 = float(np.percentile(seed_mag, 90.0))
                        p99 = float(np.percentile(seed_mag, 99.0))
                        # Cap seeds to avoid propagating spikes; still allow broad slide motion.
                        max_mag = max(min_mag, min(p99, 3.0 * max(p90, 0.0)))

                    seeds = (uv_cnt > 0) & fp & np.isfinite(mag) & (mag >= float(min_mag))
                    if max_mag is not None:
                        seeds = seeds & (mag <= float(max_mag))
                    holes = fp & (~seeds)
                    if bool(holes.any()) and bool(seeds.any()):
                        iters = int(SCENE_PARAMS.get("mpm_uv_fill_holes_iters", 10))
                        iters = max(int(iters), 12)
                        uv_filled, filled_mask = _fill_uv_holes_in_mask(
                            uv,
                            valid_mask=seeds,
                            fill_mask=fp,
                            max_iterations=iters,
                            return_filled_mask=True,
                        )
                        uv = uv_filled
                        if SCENE_PARAMS.get("debug_verbose", False):
                            filled_fp = bool((filled_mask | (~fp)).all())
                            print(
                                f"[MPM UV] fill_footprint_inpaint: fp_px={int(np.sum(fp))} "
                                f"seeds={int(np.sum(seeds))} holes={int(np.sum(holes))} "
                                f"pseudo_holes={int(np.sum(pseudoholes)) if 'pseudoholes' in locals() else -1} "
                                f"min_mag={float(min_mag):.3g}mm max_mag={float(max_mag) if max_mag is not None else float('nan'):.3g}mm "
                                f"iters={int(iters)} filled_fp={filled_fp}"
                            )
            except Exception as e:
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM UV] fill_footprint_inpaint failed: {e}")

        if bool(SCENE_PARAMS.get("mpm_uv_mask_footprint", False)) and footprint is not None:
            try:
                dilate_iters = int(SCENE_PARAMS.get("mpm_uv_mask_footprint_dilate_iters", 0))
                if dilate_iters > 0:
                    fp = _dilate_mask_3x3(footprint, iterations=dilate_iters)
                else:
                    fp = footprint
                self._last_uv_mask_footprint_mask_u8 = fp.astype(np.uint8, copy=False)
                uv_mask_outside = (fp <= 0)
                # Pre-mask before smoothing so boundary can be softly attenuated by blur (while keeping outside zero after re-mask).
                if uv_mask_outside.any():
                    uv[uv_mask_outside] = 0.0
            except Exception as e:
                uv_mask_outside = None
                self._last_uv_mask_footprint_mask_u8 = None
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM UV] mask_footprint (pre) failed: {e}")

        if smooth_uv:
            if bool(SCENE_PARAMS.get("mpm_uv_smooth", True)):
                base_iters = max(0, int(SCENE_PARAMS.get("mpm_uv_smooth_iters", 1)))
                blur_iters = base_iters
                if uv_mask_outside is not None:
                    blur_iters = max(
                        base_iters,
                        max(0, int(SCENE_PARAMS.get("mpm_uv_mask_footprint_blur_iters", 6))),
                    )
                if int(blur_iters) > 0:
                    uv = _box_blur_2d_xy(uv, iterations=int(blur_iters))

        if bool(SCENE_PARAMS.get("mpm_uv_despike", False)) and footprint is not None:
            try:
                abs_mm = float(SCENE_PARAMS.get("mpm_uv_despike_abs_mm", 0.8))
                cap_cfg = SCENE_PARAMS.get("mpm_uv_despike_cap_mm", None)
                if cap_cfg is None:
                    cap_mm = max(abs_mm - 1e-3, 0.0)
                else:
                    cap_mm = max(float(cap_cfg), 0.0)
                boundary_iters = int(SCENE_PARAMS.get("mpm_uv_despike_boundary_iters", 2))

                scope = str(SCENE_PARAMS.get("mpm_uv_despike_scope", "boundary")).lower().strip()
                if scope == "footprint":
                    gate = footprint.astype(np.uint8, copy=False)
                else:
                    boundary = _contact_boundary_mask(footprint)
                    gate = _dilate_mask_3x3(boundary, iterations=boundary_iters)

                uv_despike, replaced, capped = _despike_uv_disp_mm(
                    uv,
                    gate_mask_u8=gate,
                    abs_mm=abs_mm,
                    cap_mm=cap_mm,
                )
                uv = uv_despike
                self._last_uv_despike_gate_mask_u8 = gate
                self._last_uv_despike_replaced_mask_u8 = replaced.astype(np.uint8, copy=False)
                self._last_uv_despike_capped_mask_u8 = capped.astype(np.uint8, copy=False)
                if SCENE_PARAMS.get("debug_verbose", False):
                    gate_px = int(np.sum(gate > 0))
                    print(
                        f"[MPM UV] despike: scope={scope} abs={abs_mm:.3g}mm cap={cap_mm:.3g}mm "
                        f"band_iters={boundary_iters} gate_px={gate_px} "
                        f"replaced_px={int(np.sum(replaced))} capped_px={int(np.sum(capped))}"
                    )
            except Exception as e:
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM UV] despike failed: {e}")

        if uv_mask_outside is not None:
            try:
                # Re-mask after smoothing/despike to guarantee "outside footprint uv == 0".
                if uv_mask_outside.any():
                    uv[uv_mask_outside] = 0.0
            except Exception as e:
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM UV] mask_footprint (post) failed: {e}")

        cap_cfg = SCENE_PARAMS.get("mpm_uv_cap_mm", None)
        if cap_cfg is not None:
            try:
                cap_mm = max(float(cap_cfg), 0.0)
                if cap_mm > 0.0:
                    uv_capped, capped = _cap_uv_disp_mm(uv, cap_mm=cap_mm)
                    uv = uv_capped
                    if SCENE_PARAMS.get("debug_verbose", False):
                        print(f"[MPM UV] cap: cap_mm={cap_mm:.3g} capped_px={int(np.sum(capped))}")
            except Exception as e:
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM UV] cap failed: {e}")

        scale_cfg = SCENE_PARAMS.get("mpm_uv_scale", 1.0)
        if scale_cfg is not None:
            try:
                scale = float(scale_cfg)
                if scale != 1.0:
                    uv = uv * np.float32(scale)
                    if SCENE_PARAMS.get("debug_verbose", False):
                        print(f"[MPM UV] scale: scale={scale:.6g}")
            except Exception as e:
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM UV] scale failed: {e}")

        return height_field, uv

    def render(self, height_field_mm: np.ndarray) -> np.ndarray:
        """
        Render height field to RGB image

        Args:
            height_field_mm: (n_row, n_col) height displacement in mm

        Returns:
            rgb_image: (H, W, 3) uint8
        """
        # Debug: verify height field values before rendering
        neg_mask = height_field_mm < 0
        if SCENE_PARAMS.get('debug_verbose', False):
            if neg_mask.any():
                print(f"[MPM RENDER] height_field: min={height_field_mm.min():.2f}mm, "
                      f"shape={height_field_mm.shape}, negative_cells={neg_mask.sum()}")
            else:
                print(f"[MPM RENDER] WARNING: No negative values in height_field! "
                      f"range=[{height_field_mm.min():.4f}, {height_field_mm.max():.4f}]")
        smooth = bool(SCENE_PARAMS.get("mpm_height_smooth", True))
        self.scene.set_height_field(height_field_mm, smooth=smooth)
        return self.scene.get_image()

    def get_diff_image(self, height_field_mm: np.ndarray) -> np.ndarray:
        """Get diff image relative to flat reference"""
        smooth = bool(SCENE_PARAMS.get("mpm_height_smooth", True))
        self.scene.set_height_field(height_field_mm, smooth=smooth)
        return self.scene.get_diff_image()

    def update(self):
        if self.visible:
            self.scene.update()


# ------------------------------------------------------------------------------
# Lighting profiles
# ------------------------------------------------------------------------------
# NOTE: 这些 profile 仅用于“可复现/可审计”的渲染基线。
# `publish_v1` 以 repo 内 light.txt 的当前配置为基准固化到代码中，避免环境差异。
PUBLISH_V1_LIGHT_DICTS: List[Dict[str, Any]] = [
    {
        "position": [0.029999999329447746, -1.0499999523162842, 1.0],
        "ambient": [0.0, 0.05, 0.0],
        "diffuse": [0.07, 0.16, 0.0],
        "specular": [0, 0, 0],
        "constant": 1.0,
        "linear": 0.059,
        "quadratic": 0,
        "visible": False,
        "directional": False,
        "render_shadow": False,
        "shadow_size": [1600, 1600],
        "light_frustum_visible": False,
    },
    {
        "position": [2.490000009536743, -3.609999895095825, 0.5700000524520874],
        "ambient": [0.1868, 0.0, 0.0],
        "diffuse": [0.9085098039215686, 0.006901960784313727, 0.0],
        "specular": [0.18999999999999995, 0.0007843137254901962, 0.0],
        "constant": 1,
        "linear": 0.10300000000000001,
        "quadratic": 0.029000000000000012,
        "visible": 0,
        "directional": False,
        "render_shadow": False,
        "shadow_size": [1600, 1600],
        "light_frustum_visible": False,
        "position2": [1.8499999999999999, 2.6399999999999997, 0.5499999999999999],
    },
    {
        "position": [-2.419999837875366, -3.8899998664855957, 0.5299999713897705],
        "ambient": [0.0017647058823529412, 0.1129411764705882, 0.042352941176470586],
        "diffuse": [0.0, 0.956, 0.04831372549019608],
        "specular": [0.0007843137254901962, 0.16999999999999993, 0.005490196078431373],
        "constant": 1.0,
        "linear": 0.02100000000000002,
        "quadratic": 0.022,
        "visible": 0,
        "directional": False,
        "render_shadow": False,
        "shadow_size": [1600, 1600],
        "light_frustum_visible": False,
        "position2": [-2.11, 2.9499999523162845, 0.44999999999999996],
    },
    {
        "position": [-2.049999952316284, -3.200000047683716, 1.0],
        "ambient": [0.001764705882352941, 0.0, 0.17],
        "diffuse": [0.010352941176470589, 0.07690196078431373, 0.99],
        "specular": [0.0011764705882352942, 0.0007843137254901962, 0.24333333333333337],
        "constant": 1.0,
        "linear": 0.05600000000000001,
        "quadratic": 0,
        "visible": 0,
        "directional": False,
        "render_shadow": False,
        "shadow_size": [1600, 1600],
        "light_frustum_visible": False,
        "position2": [2.0, -3.200000047683716, 1.0],
    },
    {
        "position": [-1.7000000476837158, 3.200000047683716, 1.0],
        "ambient": [0.12176470588235294, 0.21999999999999997, 0.2],
        "diffuse": [0.0, 0.0269019607843137, 0.07],
        "specular": [0.0011764705882352942, 0.010784313725490196, 0.09333333333333335],
        "constant": 1.0,
        "linear": 0.042,
        "quadratic": 0.0,
        "visible": 0,
        "directional": False,
        "render_shadow": False,
        "shadow_size": [1600, 1600],
        "light_frustum_visible": False,
        "position2": [1.7000000476837158, 3.200000047683716, 1.0],
    },
]


class MPMSensorScene(Scene):
    """Minimal rendering scene matching SensorScene visual style"""

    def __init__(
        self,
        gel_size_mm: Tuple[float, float],
        grid_shape: Tuple[int, int],
        visible: bool = False
    ):
        super().__init__(win_height=630, win_width=375, visible=visible, title="MPM Sensor")

        self.gel_width_mm, self.gel_height_mm = gel_size_mm
        self.n_row, self.n_col = grid_shape

        self.align_view()

        # Scale factor to match SensorScene
        scale_ratio = 4 / self.gel_width_mm
        base_tf = (Matrix4x4.fromScale(scale_ratio, scale_ratio, scale_ratio)
                   .translate(0, -self.gel_height_mm / 2, 0, True)
                   .rotate(180, 0, 1, 0, True))

        # Lights matching SensorScene
        self.light_white = PointLight(
            pos=(0, 0, 1), ambient=(0.1, 0.1, 0.1), diffuse=(0.1, 0.1, 0.1),
            specular=(0, 0, 0), visible=True, directional=False, render_shadow=False
        )
        self.light_r = LineLight(
            pos=np.array([2, -3.0, 1.5]), pos2=np.array([2, 3.0, 1.3]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        self.light_g = LineLight(
            pos=np.array([-2, -3.2, 1.5]), pos2=np.array([-2, 3.2, 1.3]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        self.light_b = LineLight(
            pos=np.array([-2, -3.2, 1]), pos2=np.array([2, -3.2, 1]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        self.light_b2 = LineLight(
            pos=np.array([-1.7, 3.2, 1]), pos2=np.array([1.7, 3.2, 1]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        lights = [self.light_white, self.light_r, self.light_g, self.light_b, self.light_b2]

        # Apply explicit lighting profile first, then (optionally) let light.txt override.
        light_profile = str(SCENE_PARAMS.get("mpm_light_profile", "default")).lower().strip()
        disable_light_file = bool(SCENE_PARAMS.get("mpm_disable_light_file", False))
        if light_profile == "publish_v1":
            try:
                self._apply_light_dicts(PUBLISH_V1_LIGHT_DICTS)
            except Exception:
                pass

        light_file = ASSET_DIR / "data/light.txt"
        self._light_file_loaded = False
        if (not disable_light_file) and light_file.exists():
            self.loadLight(str(light_file))
            self._light_file_loaded = True
        # Finalize shadow toggle (may override profile/file defaults).
        shadow_on = bool(SCENE_PARAMS.get("mpm_render_shadow", True))
        for lt in (self.light_r, self.light_g, self.light_b, self.light_b2):
            try:
                lt.render_shadow = bool(shadow_on)
            except Exception:
                try:
                    lt.loadDict({"render_shadow": bool(shadow_on)})
                except Exception:
                    pass

        # Create marker texture (static grid pattern matching SensorScene style)
        tex_size = SCENE_PARAMS.get("marker_tex_size_wh", (320, 560))
        try:
            tex_w = int(tex_size[0])
            tex_h = int(tex_size[1])
        except Exception:
            tex_w, tex_h = 320, 560
        marker_radius_px = int(SCENE_PARAMS.get("marker_radius_px", 3))
        self._marker_radius_px = max(marker_radius_px, 0)
        self.marker_tex_np = self._make_marker_texture(tex_size=(tex_w, tex_h), marker_radius=self._marker_radius_px)
        self.white_tex_np = np.full((tex_h, tex_w, 3), 255, dtype=np.uint8)
        self.marker_tex = Texture2D(self.marker_tex_np)
        self._show_marker = True
        self._marker_mode = "static"  # off|static|warp|advect
        self._uv_disp_mm: Optional[np.ndarray] = None
        self._cached_warped_tex: Optional[np.ndarray] = None  # Cache to avoid double remap per frame
        self._depth_tint_enabled = True
        self._depth_tint_mode = "frame"  # off|fixed|frame
        self._depth_tint_fixed_max_mm = float(SCENE_PARAMS.get("press_depth_mm", 1.0))
        # NOTE: SensorScene.depth_mesh texcoords: u_range=(0,1), v_range=(1,0)；MPM 侧跟随此约定。
        self._warp_flip_x = bool(SCENE_PARAMS.get("mpm_warp_flip_x", False))
        self._warp_flip_y = bool(SCENE_PARAMS.get("mpm_warp_flip_y", True))
        self._render_flip_x = bool(SCENE_PARAMS.get("mpm_render_flip_x", False))
        self._zmap_convention = str(SCENE_PARAMS.get("mpm_zmap_convention", "sensor_depth")).strip().lower()

        # Surface mesh
        self.surf_mesh = GLSurfMeshItem(
            (self.n_row, self.n_col),
            x_range=(self.gel_width_mm / 2, -self.gel_width_mm / 2),
            y_range=(self.gel_height_mm, 0),
            lights=lights,
            material=Material(
                ambient=(1, 1, 1), diffuse=(1, 1, 1), specular=(1, 1, 1),
                textures=[self.marker_tex]
            ),
        )
        self.surf_mesh.applyTransform(base_tf)

        # Set texture coordinates (CRITICAL for marker display!)
        texcoords = self._gen_texcoords(self.n_row, self.n_col, v_range=(1, 0))
        self.surf_mesh.mesh_item.setData(texcoords=texcoords)

        # RGB camera
        w = (self.gel_width_mm - 1) / 2 * scale_ratio
        h = (self.gel_height_mm - 1) / 2 * scale_ratio
        self.rgb_camera = RGBCamera(
            self, img_size=(400, 700), eye=(0, 0, 10 * scale_ratio), up=(0, 1, 0),
            ortho_space=(-w, w, -h, h, 0, 10),
            frustum_visible=False
        )
        self.rgb_camera.render_group.update(self.surf_mesh)

        # Reference image for diff mode
        self._ref_image = None
        self._indenter_overlay_enabled = False
        self._indenter_center_mm = (0.0, 0.0)
        self._indenter_square_size_mm: Optional[float] = None
        self._debug_overlay = "off"  # off|uv|warp
        # Cached depth field (mm, indentation is negative). Used only for debug overlays.
        self._last_depth_mm: Optional[np.ndarray] = None

        # advect_points state (material-point marker advection)
        self._uv_du_mm: Optional[np.ndarray] = None
        self._advect_points_xy: Optional[np.ndarray] = None
        self._advect_pending_du_px: Optional[np.ndarray] = None
        self._advect_prev_uv_disp_mm: Optional[np.ndarray] = None
        self._advect_last_ellipse_axes_angle: Optional[np.ndarray] = None

    def align_view(self):
        self.cameraLookAt([0, 0, 8.15], [0, 0, 0], [0, 1, 0])

    @staticmethod
    def _gen_texcoords(n_row: int, n_col: int, u_range: Tuple[float, float] = (0, 1),
                       v_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Generate texture coordinates for mesh grid"""
        tex_u = np.linspace(*u_range, n_col)
        tex_v = np.linspace(*v_range, n_row)
        return np.stack(np.meshgrid(tex_u, tex_v), axis=-1).reshape(-1, 2)

    def _apply_light_dicts(self, dicts: List[Dict[str, Any]]) -> None:
        """Apply 5-line light dicts (point + 4 line lights), matching loadLight() order."""
        if len(dicts) != 5:
            raise ValueError("lighting profile must have 5 dicts")
        self.light_white.loadDict(dict(dicts[0]))
        self.light_r.loadDict(dict(dicts[1]))
        self.light_g.loadDict(dict(dicts[2]))
        self.light_b.loadDict(dict(dicts[3]))
        self.light_b2.loadDict(dict(dicts[4]))

    def loadLight(self, file_path: str):
        """Load light configuration from file"""
        try:
            with open(file_path, "r") as f:
                self.light_white.loadDict(ast.literal_eval(f.readline()))
                self.light_r.loadDict(ast.literal_eval(f.readline()))
                self.light_g.loadDict(ast.literal_eval(f.readline()))
                self.light_b.loadDict(ast.literal_eval(f.readline()))
                self.light_b2.loadDict(ast.literal_eval(f.readline()))
        except Exception:
            pass

    def _make_marker_texture(self, tex_size: Tuple[int, int], marker_radius: int = 3) -> np.ndarray:
        """
        Create static marker texture with uniform grid pattern

        Args:
            tex_size: (width, height) of texture
            marker_radius: radius of marker dots

        Returns:
            Texture array (H, W, 3) uint8
        """
        tex_w, tex_h = tex_size
        tex = np.full((tex_h, tex_w, 3), 255, dtype=np.uint8)

        # Create uniform marker grid matching typical sensor pattern
        n_cols = max(1, int(SCENE_PARAMS.get("marker_grid_cols", 14)))
        n_rows = max(1, int(SCENE_PARAMS.get("marker_grid_rows", 20)))
        margin_x, margin_y = 20, 20

        dx = 0.0 if n_cols <= 1 else float(tex_w - 2 * margin_x) / float(n_cols - 1)
        dy = 0.0 if n_rows <= 1 else float(tex_h - 2 * margin_y) / float(n_rows - 1)
        centers = [(float(margin_x + col * dx), float(margin_y + row * dy)) for row in range(n_rows) for col in range(n_cols)]

        appearance_mode = str(SCENE_PARAMS.get("marker_appearance_mode", "grid") or "grid").strip().lower()
        appearance_seed = SCENE_PARAMS.get("marker_appearance_seed", None)
        if appearance_mode == "random_ellipses":
            cfg = resolve_marker_appearance_config(
                mode=appearance_mode,
                seed=(int(appearance_seed) if appearance_seed is not None else None),
            )
            return generate_random_ellipses_attenuation_texture_u8(
                tex_size_wh=(int(tex_w), int(tex_h)),
                centers_xy=np.asarray(centers, dtype=np.float32),
                cfg=cfg,
            )

        for x_f, y_f in centers:
            x = int(round(x_f))
            y = int(round(y_f))
            if HAS_CV2:
                cv2.ellipse(tex, (x, y), (marker_radius, marker_radius), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
            else:
                # Fallback: draw filled circle with numpy
                yy, xx = np.ogrid[:tex_h, :tex_w]
                mask = (xx - x) ** 2 + (yy - y) ** 2 <= marker_radius**2
                tex[mask] = 0

        return tex

    def set_show_marker(self, show: bool):
        """Toggle marker visibility"""
        self._show_marker = show
        self._update_marker_texture()

    def set_marker_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        # Alias: advect_points is the spec-facing name; keep advect for backward compatibility.
        if mode == "advect_points":
            mode = "advect"
        if mode not in ("off", "static", "warp", "advect"):
            raise ValueError("marker mode must be one of: off|static|warp|advect|advect_points")
        self._marker_mode = mode
        if self._marker_mode != "advect":
            # Avoid stale state when toggling modes.
            self._advect_points_xy = None
            self._advect_pending_du_px = None
            self._advect_prev_uv_disp_mm = None
            self._advect_last_ellipse_axes_angle = None
        self._update_marker_texture()

    def set_depth_tint_config(self, mode: str, *, max_mm: Optional[float] = None) -> None:
        """Configure depth tint normalization mode (off|fixed|frame)."""
        m = str(mode).lower().strip()
        if m not in ("off", "fixed", "frame"):
            raise ValueError("depth tint mode must be one of: off|fixed|frame")
        self._depth_tint_mode = m
        self._depth_tint_enabled = bool(m != "off")
        if max_mm is not None:
            self._depth_tint_fixed_max_mm = max(float(max_mm), 0.0)
        self._update_marker_texture()

    def set_depth_tint_enabled(self, enabled: bool) -> None:
        """(Legacy) Toggle depth tint overlay (on -> frame mode). Prefer set_depth_tint_config()."""
        self.set_depth_tint_config("frame" if bool(enabled) else "off")

    def set_uv_displacement(
        self,
        uv_disp_mm: Optional[np.ndarray],
        *,
        uv_du_mm: Optional[np.ndarray] = None,
    ) -> None:
        """设置当前帧的面内位移场/增量位移场（单位 mm）。"""
        if uv_disp_mm is None:
            self._uv_disp_mm = None
        else:
            # Legacy option: apply horizontal flip for render alignment (see --mpm-render-flip-x).
            # When enabled, keep UV consistent with height_field flip.
            uv_flipped = _mpm_flip_x_field(uv_disp_mm).copy() if self._render_flip_x else uv_disp_mm
            # NOTE: u 分量的“方向反转”由 warp 的 flip_x 统一处理，避免同一轴被多处重复修正。
            self._uv_disp_mm = uv_flipped.astype(np.float32, copy=False)

        if uv_du_mm is None:
            self._uv_du_mm = None
        else:
            du_flipped = _mpm_flip_x_field(uv_du_mm).copy() if self._render_flip_x else uv_du_mm
            self._uv_du_mm = du_flipped.astype(np.float32, copy=False)

        # 在 warp/advect 模式下，每帧都需要更新纹理
        if self._marker_mode in ("warp", "advect"):
            self._update_marker_texture()

    def set_indenter_overlay(self, enabled: bool, square_size_mm: Optional[float] = None) -> None:
        self._indenter_overlay_enabled = bool(enabled)
        self._indenter_square_size_mm = square_size_mm

    def set_indenter_center(self, x_mm: float, y_mm: float) -> None:
        self._indenter_center_mm = (float(x_mm), float(y_mm))

    def set_debug_overlay(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in ("off", "uv", "warp"):
            raise ValueError("debug overlay must be one of: off|uv|warp")
        self._debug_overlay = mode

    def _ensure_advect_points_state(self) -> None:
        """Initialize advect_points state on first use."""
        if self._advect_points_xy is not None:
            return
        centers = _marker_grid_centers_px(
            int(self.marker_tex_np.shape[1]),
            int(self.marker_tex_np.shape[0]),
            n_cols=max(1, int(SCENE_PARAMS.get("marker_grid_cols", 14))),
            n_rows=max(1, int(SCENE_PARAMS.get("marker_grid_rows", 20))),
        )
        self._advect_points_xy = np.asarray([(float(x), float(y)) for (x, y) in centers], dtype=np.float32)
        self._advect_pending_du_px = None
        self._advect_prev_uv_disp_mm = None
        self._advect_last_ellipse_axes_angle = None

    def reset_advect_points(self) -> None:
        """Reset advect_points state (used when the UI loops back to frame 0)."""
        self._advect_points_xy = None
        self._advect_pending_du_px = None
        self._advect_prev_uv_disp_mm = None
        self._advect_last_ellipse_axes_angle = None

    def _compute_du_px_for_advect(self) -> Optional[np.ndarray]:
        """
        Compute per-frame incremental displacement Δu in pixel coordinates.

        Prefers reconstructed Δu (uv_du_mm). When unavailable, advect_points falls back to
        using Δu ≈ uv_disp_mm - prev_uv_disp_mm *for the current frame* (see _advect_points_update_and_render).
        """
        if self._uv_disp_mm is None:
            return None
        tex_h, tex_w = int(self.marker_tex_np.shape[0]), int(self.marker_tex_np.shape[1])
        gel_w_mm, gel_h_mm = float(self.gel_width_mm), float(self.gel_height_mm)

        du_mm = self._uv_du_mm
        if du_mm is None:
            return None

        du_up = _upsample_uv_disp_to_hw(du_mm, out_h=tex_h, out_w=tex_w)
        du_px = np.empty_like(du_up, dtype=np.float32)
        du_px[..., 0] = (du_up[..., 0] / max(gel_w_mm, 1e-6)) * float(tex_w)
        du_px[..., 1] = (du_up[..., 1] / max(gel_h_mm, 1e-6)) * float(tex_h)
        if self._warp_flip_x:
            du_px[..., 0] = -du_px[..., 0]
        if self._warp_flip_y:
            du_px[..., 1] = -du_px[..., 1]
        return np.nan_to_num(du_px, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _advect_points_update_and_render(self) -> np.ndarray:
        """
        advect_points（material-point）marker：按 Δu 时间积分点中心，并用局部 ∇Δu 渲染椭圆。

        关键对齐：使用“上一帧缓存的 Δu”推进点中心，再渲染当前帧；当前帧 Δu 作为下一帧的 pending。
        """
        self._ensure_advect_points_state()
        pts = self._advect_points_xy
        if pts is None or pts.size == 0:
            return self.marker_tex_np

        ellipse = None
        du_apply = self._advect_pending_du_px
        if (
            du_apply is None
            and self._uv_du_mm is None
            and self._advect_prev_uv_disp_mm is not None
            and self._uv_disp_mm is not None
        ):
            # Fallback: Δu ≈ u_i - u_{i-1} (apply for the *current* frame).
            try:
                tex_h, tex_w = int(self.marker_tex_np.shape[0]), int(self.marker_tex_np.shape[1])
                gel_w_mm, gel_h_mm = float(self.gel_width_mm), float(self.gel_height_mm)
                du_mm = (self._uv_disp_mm - self._advect_prev_uv_disp_mm).astype(np.float32, copy=False)
                du_up = _upsample_uv_disp_to_hw(du_mm, out_h=tex_h, out_w=tex_w)
                du_px = np.empty_like(du_up, dtype=np.float32)
                du_px[..., 0] = (du_up[..., 0] / max(gel_w_mm, 1e-6)) * float(tex_w)
                du_px[..., 1] = (du_up[..., 1] / max(gel_h_mm, 1e-6)) * float(tex_h)
                if self._warp_flip_x:
                    du_px[..., 0] = -du_px[..., 0]
                if self._warp_flip_y:
                    du_px[..., 1] = -du_px[..., 1]
                du_apply = np.nan_to_num(du_px, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            except Exception:
                du_apply = None

        if du_apply is not None:
            du_prev = du_apply
            step = _bilinear_sample_vec2_px(du_prev, pts)
            # Cap rare spikes to avoid runaway points (deterministic clamp).
            max_step = 30.0  # px
            mag = np.sqrt(np.sum(step * step, axis=1))
            scale = np.ones_like(mag, dtype=np.float32)
            mask = mag > float(max_step)
            scale[mask] = (float(max_step) / (mag[mask] + 1e-6)).astype(np.float32)
            step = step * scale[:, None]
            pts = (pts + step).astype(np.float32, copy=False)
            self._advect_points_xy = pts

            # Ellipse proxy from local linearization A = I + J (J=∇Δu).
            try:
                du_x = du_prev[..., 0].astype(np.float32, copy=False)
                du_y = du_prev[..., 1].astype(np.float32, copy=False)
                ddux_dy, ddux_dx = np.gradient(du_x)
                dduy_dy, dduy_dx = np.gradient(du_y)

                xi = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, du_prev.shape[1] - 1)
                yi = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, du_prev.shape[0] - 1)

                r0 = float(getattr(self, "_marker_radius_px", 3))
                s_min, s_max = 0.5, 2.5
                ellipse = np.empty((pts.shape[0], 3), dtype=np.float32)
                for i in range(int(pts.shape[0])):
                    j11 = float(ddux_dx[yi[i], xi[i]])
                    j12 = float(ddux_dy[yi[i], xi[i]])
                    j21 = float(dduy_dx[yi[i], xi[i]])
                    j22 = float(dduy_dy[yi[i], xi[i]])
                    a = np.array([[1.0 + j11, j12], [j21, 1.0 + j22]], dtype=np.float32)
                    try:
                        u, s, _ = np.linalg.svd(a)
                        s0 = float(np.clip(float(s[0]), s_min, s_max))
                        s1 = float(np.clip(float(s[1]), s_min, s_max))
                        ang = math.degrees(math.atan2(float(u[1, 0]), float(u[0, 0])))
                    except Exception:
                        s0, s1, ang = 1.0, 1.0, 0.0
                    ellipse[i, 0] = float(r0) * s0
                    ellipse[i, 1] = float(r0) * s1
                    ellipse[i, 2] = float(ang)

                if SCENE_PARAMS.get("debug_verbose", False) and ellipse.size:
                    try:
                        ax = ellipse[:, 0].astype(np.float32, copy=False)
                        by = ellipse[:, 1].astype(np.float32, copy=False)
                        ratio = np.maximum(ax, by) / np.maximum(np.minimum(ax, by), 1e-6)
                        step_mag = np.sqrt(np.sum(step * step, axis=1)).astype(np.float32, copy=False)
                        print(
                            f"[MPM MARKER] advect_points: step_px_p50={float(np.percentile(step_mag, 50)):.3g} "
                            f"step_px_p90={float(np.percentile(step_mag, 90)):.3g} "
                            f"axis_ratio_p50={float(np.percentile(ratio, 50)):.3g} "
                            f"axis_ratio_p90={float(np.percentile(ratio, 90)):.3g}"
                        )
                    except Exception:
                        pass
            except Exception:
                ellipse = None

        self._advect_last_ellipse_axes_angle = ellipse
        tex = _render_advect_points_texture(self.marker_tex_np, pts, ellipse_axes_angle=ellipse)

        # Store Δu for next frame (best-effort).
        self._advect_pending_du_px = self._compute_du_px_for_advect()
        # Store uv_disp for Δu fallback on next frame.
        if self._uv_disp_mm is not None:
            self._advect_prev_uv_disp_mm = self._uv_disp_mm.astype(np.float32, copy=True)

        return tex

    def _update_marker_texture(self) -> None:
        if not self._show_marker or self._marker_mode == "off":
            self.marker_tex.setTexture(self.white_tex_np)
            self._cached_warped_tex = None
            return
        if self._marker_mode == "static" or self._uv_disp_mm is None:
            self.marker_tex.setTexture(self.marker_tex_np)
            self._cached_warped_tex = None
            return
        if self._marker_mode == "warp":
            warped = warp_marker_texture(
                self.marker_tex_np,
                self._uv_disp_mm,
                gel_size_mm=(self.gel_width_mm, self.gel_height_mm),
                flip_x=self._warp_flip_x,
                flip_y=self._warp_flip_y,
            )
        else:
            warped = self._advect_points_update_and_render()
        self._cached_warped_tex = warped  # Cache for reuse in _update_depth_tint_texture
        self.marker_tex.setTexture(warped)

    @staticmethod
    def _box_blur_2d(values: np.ndarray, iterations: int = 1) -> np.ndarray:
        """轻量平滑：纯 numpy 3x3 box blur，避免引入 SciPy 依赖。"""
        result = values.astype(np.float32, copy=True)
        for _ in range(max(iterations, 0)):
            padded = np.pad(result, ((1, 1), (1, 1)), mode="edge")
            result = (
                padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
                padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
        return result

    def _update_depth_tint_texture(self, depth_mm: np.ndarray) -> None:
        """
        MPM 表层按压深度着色：压得越深越红，增强反馈可见性。

        做法：在 marker/white 底图上叠加红色热度（不改变 marker warp）。
        """
        if not self._show_marker or self._marker_mode == "off":
            base = self.white_tex_np
        elif self._marker_mode == "static" or self._uv_disp_mm is None:
            base = self.marker_tex_np
        else:
            # Use cached warped texture to avoid double remap per frame
            if self._cached_warped_tex is not None:
                base = self._cached_warped_tex
            else:
                # Fallback: compute if cache is missing (shouldn't happen in normal flow)
                if self._marker_mode == "warp":
                    base = warp_marker_texture(
                        self.marker_tex_np,
                        self._uv_disp_mm,
                        gel_size_mm=(self.gel_width_mm, self.gel_height_mm),
                        flip_x=self._warp_flip_x,
                        flip_y=self._warp_flip_y,
                    )
                else:
                    # advect_points: do NOT advance state here (avoid double-stepping).
                    try:
                        self._ensure_advect_points_state()
                        pts = self._advect_points_xy
                        base = _render_advect_points_texture(
                            self.marker_tex_np,
                            pts if pts is not None else np.zeros((0, 2), dtype=np.float32),
                            ellipse_axes_angle=self._advect_last_ellipse_axes_angle,
                        )
                    except Exception:
                        # Conservative fallback: stateless placement (no integration/ellipse).
                        base = advect_marker_texture(
                            self.marker_tex_np,
                            self._uv_disp_mm,
                            gel_size_mm=(self.gel_width_mm, self.gel_height_mm),
                            flip_x=self._warp_flip_x,
                            flip_y=self._warp_flip_y,
                            marker_radius_px=int(getattr(self, "_marker_radius_px", 3)),
                            marker_grid_cols=int(SCENE_PARAMS.get("marker_grid_cols", 14)),
                            marker_grid_rows=int(SCENE_PARAMS.get("marker_grid_rows", 20)),
                        )

        depth_pos = np.clip(-depth_mm, 0.0, None)  # mm, >=0
        if depth_pos.max() <= 1e-6:
            self.marker_tex.setTexture(base)
            return

        # NOTE: frame-wise normalization causes visible "breathing"/halo. Use fixed max when requested.
        if self._depth_tint_mode == "fixed":
            denom = max(float(getattr(self, "_depth_tint_fixed_max_mm", 0.0)), 1e-6)
        else:
            denom = float(depth_pos.max()) + 1e-6
        depth_norm = np.clip(depth_pos / denom, 0.0, 1.0)
        tex_h, tex_w = base.shape[0], base.shape[1]
        src_h, src_w = depth_norm.shape
        row_idx = (np.linspace(0, src_h - 1, tex_h)).astype(np.int32)
        col_idx = (np.linspace(0, src_w - 1, tex_w)).astype(np.int32)
        upsampled = depth_norm[row_idx][:, col_idx]

        tinted = base.astype(np.float32)
        tinted[..., 0] = np.clip(tinted[..., 0] + 150.0 * upsampled, 0, 255)
        tinted[..., 1] = np.clip(tinted[..., 1] * (1.0 - 0.45 * upsampled), 0, 255)
        tinted[..., 2] = np.clip(tinted[..., 2] * (1.0 - 0.45 * upsampled), 0, 255)

        self.marker_tex.setTexture(tinted.astype(np.uint8))

    def set_height_field(self, height_field_mm: np.ndarray, smooth: bool = True):
        """Update surface mesh with height field data"""
        if smooth:
            iters = int(SCENE_PARAMS.get("mpm_height_smooth_iters", 2))
            if iters > 0:
                height_field_mm = self._box_blur_2d(height_field_mm, iterations=iters)

        # Legacy option: flip horizontally for render alignment (see --mpm-render-flip-x).
        # Height field: col=0 is x=-gel_w/2 (left)
        # Mesh x_range: (gel_w/2, -gel_w/2) means col=0 is x=+gel_w/2 (right)
        height_field_mm = _mpm_flip_x_field(height_field_mm) if self._render_flip_x else height_field_mm

        # Ensure negative values for indentation (SensorScene convention)
        depth = np.minimum(height_field_mm, 0)
        # Keep a copy for debug overlays (best-effort; never affect render output).
        try:
            self._last_depth_mm = depth.astype(np.float32, copy=True)
        except Exception:
            self._last_depth_mm = None

        # Debug: verify depth values being sent to mesh
        neg_count = (depth < -0.01).sum()
        if neg_count > 0 and SCENE_PARAMS.get('debug_verbose', False):
            print(f"[MESH UPDATE] depth: min={depth.min():.2f}mm, neg_cells(>0.01mm)={neg_count}")

        # GLSurfMeshItem uses zmap as vertex z directly. Keep a switch for backward compatibility.
        # NOTE: For xensim-style rendering, prefer positive indentation zmap (>=0).
        mesh_z = depth
        if self._zmap_convention == "indentation":
            mesh_z = np.clip(-depth, 0.0, None)
        self.surf_mesh.setData(mesh_z, smooth)
        if self._depth_tint_enabled:
            self._update_depth_tint_texture(depth)

    def get_image(self) -> np.ndarray:
        """Render and return RGB image"""
        image = (self.rgb_camera.render() * 255).astype(np.uint8)

        if self._debug_overlay in ("uv", "warp") and self._uv_disp_mm is not None:
            if self._debug_overlay == "uv":
                field = np.sqrt(np.sum(self._uv_disp_mm**2, axis=-1))  # mm
            else:
                # Approximate warp magnitude in pixels at texture resolution
                tex_h, tex_w = self.marker_tex_np.shape[0], self.marker_tex_np.shape[1]
                gel_w_mm = max(self.gel_width_mm, 1e-6)
                gel_h_mm = max(self.gel_height_mm, 1e-6)
                dx_px = (self._uv_disp_mm[..., 0] / gel_w_mm) * tex_w
                dy_px = (self._uv_disp_mm[..., 1] / gel_h_mm) * tex_h
                # Keep consistent with warp_marker_texture(): apply flip_x/flip_y in texture space.
                if self._warp_flip_x:
                    dx_px = -dx_px
                if self._warp_flip_y:
                    dy_px = -dy_px
                field = np.sqrt(dx_px**2 + dy_px**2)

            if field.max() > 1e-6:
                norm = field / (field.max() + 1e-6)
                h, w = image.shape[0], image.shape[1]
                src_h, src_w = norm.shape
                row_idx = (np.linspace(0, src_h - 1, h)).astype(np.int32)
                col_idx = (np.linspace(0, src_w - 1, w)).astype(np.int32)
                ov = norm[row_idx][:, col_idx]
                image = image.copy()
                image[..., 0] = np.clip(image[..., 0] + 120.0 * ov, 0, 255)
                image[..., 1] = np.clip(image[..., 1] * (1.0 - 0.35 * ov), 0, 255)
                image[..., 2] = np.clip(image[..., 2] * (1.0 - 0.35 * ov), 0, 255)

            # Direction/scale cue: draw a small arrow of the median Δu inside the contact region.
            # This helps verify sign conventions (flip, y-up/y-down) without digging into npz arrays.
            try:
                uv = self._uv_disp_mm.astype(np.float32, copy=False)
                mask = None
                if (
                    self._last_depth_mm is not None
                    and isinstance(self._last_depth_mm, np.ndarray)
                    and self._last_depth_mm.shape == uv.shape[:2]
                ):
                    mask = self._last_depth_mm < -0.01
                if mask is None:
                    # Fallback: use non-trivial displacement as ROI (still better than nothing).
                    mask = (np.sqrt(np.sum(uv * uv, axis=-1)) > 1e-4)

                if bool(np.any(mask)):
                    u_med = float(np.median(uv[..., 0][mask]))
                    v_med = float(np.median(uv[..., 1][mask]))
                else:
                    u_med, v_med = 0.0, 0.0

                tex_h, tex_w = int(self.marker_tex_np.shape[0]), int(self.marker_tex_np.shape[1])
                gel_w_mm = max(float(self.gel_width_mm), 1e-6)
                gel_h_mm = max(float(self.gel_height_mm), 1e-6)
                dx_tex = (u_med / gel_w_mm) * float(tex_w)
                dy_tex = (v_med / gel_h_mm) * float(tex_h)
                if self._warp_flip_x:
                    dx_tex = -dx_tex
                if self._warp_flip_y:
                    dy_tex = -dy_tex

                rgb_h, rgb_w = int(image.shape[0]), int(image.shape[1])
                dx_rgb = dx_tex * (float(rgb_w) / max(float(tex_w), 1.0))
                dy_rgb = dy_tex * (float(rgb_h) / max(float(tex_h), 1.0))

                mag = float(np.hypot(dx_rgb, dy_rgb))
                if math.isfinite(mag) and mag > 1e-6:
                    max_len = 60.0  # px, keep overlay readable even when |u| grows over time
                    s = min(1.0, max_len / (mag + 1e-6))
                    dx_d = int(round(dx_rgb * s))
                    dy_d = int(round(dy_rgb * s))
                else:
                    dx_d = dy_d = 0

                ox, oy = 18, 18
                x0, y0 = int(np.clip(ox, 0, rgb_w - 1)), int(np.clip(oy, 0, rgb_h - 1))
                x1 = int(np.clip(x0 + dx_d, 0, rgb_w - 1))
                y1 = int(np.clip(y0 + dy_d, 0, rgb_h - 1))

                if HAS_CV2:
                    cv2.arrowedLine(image, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
                    txt = f"uv_med=({u_med:+.2f},{v_med:+.2f})mm dx=({dx_tex:+.1f},{dy_tex:+.1f})px"
                    cv2.putText(image, txt, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    # Numpy fallback: draw a thin line (no arrow head / text).
                    rr = np.linspace(y0, y1, num=max(abs(y1 - y0), abs(x1 - x0), 1) + 1).astype(np.int32)
                    cc = np.linspace(x0, x1, num=rr.size).astype(np.int32)
                    image[rr, cc, :3] = (0, 255, 255)
            except Exception:
                pass

        if self._indenter_overlay_enabled:
            x_mm, y_mm = self._indenter_center_mm
            cell_w = self.gel_width_mm / self.n_col
            cell_h = self.gel_height_mm / self.n_row
            # Keep overlay consistent with MPM render flip convention (see _mpm_flip_x_field).
            x_mm = _mpm_flip_x_mm(x_mm) if self._render_flip_x else x_mm
            col = int((x_mm + self.gel_width_mm / 2.0) / cell_w)
            row = int(y_mm / cell_h)
            col = int(np.clip(col, 0, self.n_col - 1))
            row = int(np.clip(row, 0, self.n_row - 1))

            px = int(col / max(self.n_col - 1, 1) * (image.shape[1] - 1))
            py = int(row / max(self.n_row - 1, 1) * (image.shape[0] - 1))

            size_mm = self._indenter_square_size_mm if self._indenter_square_size_mm is not None else 6.0
            half_cols = int((size_mm / 2.0) / cell_w)
            half_rows = int((size_mm / 2.0) / cell_h)
            px0 = int(np.clip((col - half_cols) / max(self.n_col - 1, 1) * (image.shape[1] - 1), 0, image.shape[1] - 1))
            px1 = int(np.clip((col + half_cols) / max(self.n_col - 1, 1) * (image.shape[1] - 1), 0, image.shape[1] - 1))
            py0 = int(np.clip((row - half_rows) / max(self.n_row - 1, 1) * (image.shape[0] - 1), 0, image.shape[0] - 1))
            py1 = int(np.clip((row + half_rows) / max(self.n_row - 1, 1) * (image.shape[0] - 1), 0, image.shape[0] - 1))

            if HAS_CV2:
                cv2.rectangle(image, (px0, py0), (px1, py1), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(image, (px, py), 3, (255, 255, 0), -1, cv2.LINE_AA)
            else:
                image[py0:py0+2, px0:px1] = (255, 255, 0)
                image[py1-2:py1, px0:px1] = (255, 255, 0)
                image[py0:py1, px0:px0+2] = (255, 255, 0)
                image[py0:py1, px1-2:px1] = (255, 255, 0)
                image[max(py-1,0):py+2, max(px-1,0):px+2] = (255, 255, 0)

        return image

    def get_diff_image(self) -> np.ndarray:
        """Get difference image relative to reference"""
        if self._ref_image is None:
            # Capture reference on first call
            self.set_height_field(np.zeros((self.n_row, self.n_col)))
            self._ref_image = self.get_image()

        cur_image = self.get_image()
        return np.clip((cur_image.astype(np.int16) - self._ref_image.astype(np.int16)) + 110, 0, 255).astype(np.uint8)


# ==============================================================================
# MPM Simulation Adapter
# ==============================================================================
class MPMSimulationAdapter:
    """Adapter for running MPM simulation with press+slide trajectory"""

    def __init__(self):
        if not HAS_TAICHI:
            raise RuntimeError("Taichi not available for MPM simulation")

        self.solver = None
        self.positions_history: List[np.ndarray] = []
        # Per-frame particle velocities (same length as positions_history).
        # Prefer solver-provided v; fall back to Δx/Δt if v is unavailable.
        self.velocities_history: List[np.ndarray] = []
        self.velocities_source: str = "unknown"  # solver|delta|none
        self.frame_controls: List[Tuple[float, float]] = []  # [(press_amount_m, slide_amount_m)]
        self.frame_indenter_centers_m: List[Tuple[float, float, float]] = []  # [(x,y,z)] at recorded frames
        self.frame_dt_s: float = 0.0  # physical time between recorded frames
        self.initial_top_z_m = 0.0
        self.initial_positions_m: Optional[np.ndarray] = None
        self._base_indices_np: Optional[np.ndarray] = None
        self._base_fixer = None

    def setup(self):
        """Initialize MPM solver"""
        ti.init(arch=ti.cpu)

        from xengym.mpm import (
            MPMConfig, GridConfig, TimeConfig, OgdenConfig,
            MaterialConfig, ContactConfig, OutputConfig, MPMSolver, SDFConfig
        )

        # Gel dimensions in meters
        gel_w_m = SCENE_PARAMS['gel_size_mm'][0] * 1e-3
        gel_h_m = SCENE_PARAMS['gel_size_mm'][1] * 1e-3
        gel_t_m = SCENE_PARAMS['gel_thickness_mm'] * 1e-3

        dx = SCENE_PARAMS['mpm_grid_dx_mm'] * 1e-3

        # Grid size (with padding)
        pad_xy = int(SCENE_PARAMS.get("mpm_grid_padding_cells_xy", 6))
        pad_z_bottom = int(SCENE_PARAMS.get("mpm_grid_padding_cells_z_bottom", 6))
        pad_z_top = int(SCENE_PARAMS.get("mpm_grid_padding_cells_z_top", 20))
        grid_extent = [
            int(np.ceil(gel_w_m / dx)) + pad_xy * 2,
            int(np.ceil(gel_h_m / dx)) + pad_xy * 2,
            int(np.ceil(gel_t_m / dx)) + pad_z_bottom + pad_z_top,
        ]

        # Create particles
        n_particles = self._create_particles(gel_w_m, gel_h_m, gel_t_m, dx)

        # Indenter setup
        indenter_r = SCENE_PARAMS['indenter_radius_mm'] * 1e-3
        indenter_gap = SCENE_PARAMS['indenter_start_gap_mm'] * 1e-3
        indenter_type = SCENE_PARAMS.get('indenter_type', 'box')

        # Determine the effective z half-height for indenter placement
        # - sphere: use radius
        # - cylinder: use half_height (defaults to radius)
        # - box: use half_extents[2]
        if indenter_type == 'sphere':
            indenter_z_half = indenter_r
            half_extents = (indenter_r, 0, 0)  # sphere uses radius in first component
        elif indenter_type == 'cylinder':
            half_h_mm = SCENE_PARAMS.get("indenter_cylinder_half_height_mm", None)
            if half_h_mm is None:
                half_h = float(indenter_r)
            else:
                half_h = float(half_h_mm) * 1e-3
            indenter_z_half = float(half_h)
            half_extents = (indenter_r, indenter_r, float(half_h))  # (radius, radius, half_height)
        else:
            # Box (flat bottom): use half_extents for z calculation
            half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
            if half_extents_mm is not None:
                hx, hy, hz = half_extents_mm
                half_extents = (float(hx) * 1e-3, float(hy) * 1e-3, float(hz) * 1e-3)
            else:
                half_extents = (indenter_r, indenter_r, indenter_r)
            indenter_z_half = half_extents[2]  # Use actual z half-height for box

        # Calculate indenter center with correct z based on indenter type
        indenter_center = (
            float(self._particle_center[0]),
            float(self._particle_center[1]),
            float(self.initial_top_z_m + indenter_z_half + indenter_gap),
        )

        obstacles = [
            SDFConfig(sdf_type="plane", center=(0, 0, 0), normal=(0, 0, 1)),
        ]

        # Add indenter based on type
        if indenter_type == 'sphere':
            obstacles.append(SDFConfig(
                sdf_type="sphere",
                center=indenter_center,
                half_extents=half_extents
            ))
        elif indenter_type == 'cylinder':
            obstacles.append(SDFConfig(
                sdf_type="cylinder",
                center=indenter_center,
                half_extents=half_extents
            ))
        else:
            obstacles.append(SDFConfig(
                sdf_type="box",
                center=indenter_center,
                half_extents=half_extents
            ))

        print(f"[MPM] Indenter: type={indenter_type}, z_half={indenter_z_half*1000:.1f}mm, "
              f"center=({indenter_center[0]*1000:.1f}, {indenter_center[1]*1000:.1f}, {indenter_center[2]*1000:.1f})mm")

        mpm_mu_s = float(SCENE_PARAMS.get("mpm_mu_s", 2.0))
        mpm_mu_k = float(SCENE_PARAMS.get("mpm_mu_k", 1.5))
        k_n = float(SCENE_PARAMS.get("mpm_contact_stiffness_normal", 8e2))
        k_t = float(SCENE_PARAMS.get("mpm_contact_stiffness_tangent", 4e2))
        print(f"[MPM] Contact: k_n={k_n:.3g}, k_t={k_t:.3g}, mu_s={mpm_mu_s:.3g}, mu_k={mpm_mu_k:.3g}")

        enable_bulk_visc = bool(SCENE_PARAMS.get("mpm_enable_bulk_viscosity", False))
        eta_bulk = float(SCENE_PARAMS.get("mpm_bulk_viscosity", 0.0))
        print(
            f"[MPM] Material: ogden_mu={SCENE_PARAMS.get('ogden_mu')}, "
            f"ogden_alpha={SCENE_PARAMS.get('ogden_alpha')}, "
            f"ogden_kappa={SCENE_PARAMS.get('ogden_kappa')}, "
            f"bulk_viscosity={'on' if enable_bulk_visc else 'off'} (eta={eta_bulk:g})"
        )

        config = MPMConfig(
            grid=GridConfig(
                grid_size=grid_extent,
                dx=dx,
                sticky_boundary=bool(SCENE_PARAMS.get("mpm_sticky_boundary", True)),
                sticky_boundary_width=int(SCENE_PARAMS.get("mpm_sticky_boundary_width", 3)),
            ),
            time=TimeConfig(dt=SCENE_PARAMS['mpm_dt'], num_steps=1),      
            material=MaterialConfig(
                density=SCENE_PARAMS['density'],
                ogden=OgdenConfig(
                    mu=SCENE_PARAMS['ogden_mu'],
                    alpha=SCENE_PARAMS['ogden_alpha'],
                    kappa=SCENE_PARAMS['ogden_kappa']
                ),
                maxwell_branches=[],
                enable_bulk_viscosity=enable_bulk_visc,
                bulk_viscosity=eta_bulk,
            ),
            contact=ContactConfig(
                enable_contact=True,
                contact_stiffness_normal=k_n,
                contact_stiffness_tangent=k_t,
                mu_s=mpm_mu_s,  # static friction
                mu_k=mpm_mu_k,  # kinetic friction
                obstacles=obstacles,
            ),
            output=OutputConfig()
        )

        self.solver = MPMSolver(config, n_particles)
        self._indenter_center0 = np.array(indenter_center, dtype=np.float32)

        # Disable gravity to better match the quasi-static FEM use case in this demo.
        try:
            self.solver.gravity = ti.Vector([0.0, 0.0, 0.0])
        except Exception:
            pass

        self._setup_base_fixer()

        print(f"MPM solver initialized: {n_particles} particles")

    def _setup_base_fixer(self) -> None:
        """Fix a thin bottom layer of particles to emulate gel bonded to a rigid sensor base."""
        if self._base_indices_np is None or len(self._base_indices_np) == 0:
            self._base_fixer = None
            return
        if self.solver is None:
            self._base_fixer = None
            return

        base_indices = ti.field(dtype=ti.i32, shape=int(self._base_indices_np.shape[0]))
        base_indices.from_numpy(self._base_indices_np.astype(np.int32))

        base_init_pos = ti.Vector.field(3, dtype=ti.f32, shape=int(self._base_indices_np.shape[0]))
        base_init_pos.from_numpy(self._initial_positions[self._base_indices_np].astype(np.float32))

        @ti.kernel
        def apply_fix():
            for k in range(base_indices.shape[0]):
                p = base_indices[k]
                self.solver.fields.x[p] = base_init_pos[k]
                self.solver.fields.v[p] = ti.Vector([0.0, 0.0, 0.0])
                self.solver.fields.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                self.solver.fields.F[p] = ti.Matrix.identity(ti.f32, 3)

        self._base_fixer = apply_fix

    def _create_particles(self, gel_w: float, gel_h: float, gel_t: float, dx: float) -> int:
        """Create particle positions filling gel volume"""
        spacing = dx / SCENE_PARAMS['mpm_particles_per_cell']

        nx = int(np.ceil(gel_w / spacing))
        ny = int(np.ceil(gel_h / spacing))
        nz = int(np.ceil(gel_t / spacing))

        x = np.linspace(-gel_w / 2, gel_w / 2, nx)
        y = np.linspace(0, gel_h, ny)
        z = np.linspace(0, gel_t, nz)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

        # Shift to positive domain with padding
        # MPM solver has sticky boundary at I[d] < 3 or I[d] >= grid_size[d]-3.
        # Keep particles away from boundary nodes to avoid artificial clamping.
        pad_xy = int(SCENE_PARAMS.get("mpm_grid_padding_cells_xy", 6))
        pad_z_bottom = int(SCENE_PARAMS.get("mpm_grid_padding_cells_z_bottom", 6))
        padding_vec = np.array([pad_xy * dx, pad_xy * dx, pad_z_bottom * dx], dtype=np.float32)
        min_pos = positions.min(axis=0)
        positions += (padding_vec - min_pos)

        self._initial_positions = positions.copy()
        self.initial_positions_m = self._initial_positions.copy()
        self._particle_center = positions.mean(axis=0)
        self.initial_top_z_m = float(positions[:, 2].max())

        # Bottom fixation indices (2*dx thick layer)
        z_min = float(positions[:, 2].min())
        base_thickness = 2.0 * float(dx)
        base_mask = positions[:, 2] <= (z_min + base_thickness)
        self._base_indices_np = np.nonzero(base_mask)[0].astype(np.int32)

        return len(positions)

    def run_trajectory(self, record_interval: int = 10) -> List[np.ndarray]:
        """
        Run press + slide trajectory and record particle positions

        Args:
            record_interval: Record positions every N steps

        Returns:
            List of position arrays
        """
        if self.solver is None:
            self.setup()

        # Initialize particles
        velocities = np.zeros_like(self._initial_positions, dtype=np.float32)
        self.solver.initialize_particles(self._initial_positions, velocities)

        # Trajectory parameters
        press_steps = SCENE_PARAMS['press_steps']
        slide_steps = SCENE_PARAMS['slide_steps']
        hold_steps = SCENE_PARAMS['hold_steps']
        total_steps = press_steps + slide_steps + hold_steps

        press_depth_m = SCENE_PARAMS['press_depth_mm'] * 1e-3
        slide_dist_m = SCENE_PARAMS['slide_distance_mm'] * 1e-3

        self.positions_history = []
        self.velocities_history = []
        self.velocities_source = "unknown"
        self.frame_controls = []
        self.frame_indenter_centers_m = []
        # Best-effort dt between recorded frames; used as fallback when solver velocity is unavailable.
        try:
            self.frame_dt_s = float(SCENE_PARAMS.get("mpm_dt", 0.0)) * float(record_interval)
        except Exception:
            self.frame_dt_s = 0.0

        print(f"Running MPM trajectory: {total_steps} steps")
        start_time = time.time()

        for step in range(total_steps):
            # Compute indenter position
            if step < press_steps:
                # Press phase
                t = step / max(press_steps, 1)
                dz = press_depth_m * t
                dx_slide = 0.0
            elif step < press_steps + slide_steps:
                # Slide phase
                t = (step - press_steps) / max(slide_steps, 1)
                dz = press_depth_m
                dx_slide = slide_dist_m * t
            else:
                # Hold phase
                dz = press_depth_m
                dx_slide = slide_dist_m

            # Update indenter position
            new_center = np.array([
                float(self._indenter_center0[0] + dx_slide),
                float(self._indenter_center0[1]),
                float(self._indenter_center0[2] - dz),
            ], dtype=np.float32)

            # Use numpy interface for reliable Taichi field update
            centers_np = self.solver.obstacle_centers.to_numpy()
            centers_np[1] = new_center
            self.solver.obstacle_centers.from_numpy(centers_np)

            # Verify update took effect
            if step % 50 == 0 and SCENE_PARAMS.get('debug_verbose', False):
                actual = self.solver.obstacle_centers[1]
                print(f"[INDENTER UPDATE] step={step}: target_x={new_center[0]*1000:.2f}mm, actual_x={actual[0]*1000:.2f}mm")

            # Step simulation
            self.solver.step()
            if self._base_fixer is not None:
                try:
                    self._base_fixer()
                except Exception:
                    pass

            # Record positions
            if step % record_interval == 0:
                particle_data = self.solver.get_particle_data()
                pos = particle_data['x'].copy()
                self.positions_history.append(pos)
                self.frame_controls.append((float(dz), float(dx_slide)))
                self.frame_indenter_centers_m.append((float(new_center[0]), float(new_center[1]), float(new_center[2])))

                # Record velocities for later surface-motion reconstruction.
                # Why: marker advection prefers a semantically consistent v_surface(x) over using u(x) for remap.
                vel = particle_data.get('v', None)
                if vel is not None:
                    self.velocities_history.append(vel.copy())
                    if self.velocities_source == "unknown":
                        self.velocities_source = "solver"
                else:
                    # Fallback: estimate v via Δx/Δt across recorded frames.
                    # Note: this is noisier than solver v but is stable and does not rely on magic constants.
                    self.velocities_source = "delta"
                    if len(self.positions_history) >= 2 and float(self.frame_dt_s) > 0.0:
                        prev = self.positions_history[-2]
                        v_est = (pos - prev) / float(self.frame_dt_s)
                    else:
                        v_est = np.zeros_like(pos, dtype=np.float32)
                        if float(self.frame_dt_s) <= 0.0:
                            self.velocities_source = "none"
                    self.velocities_history.append(v_est.astype(np.float32, copy=False))

                # Debug: check particle displacement
                z_displacements = pos[:, 2] - self._initial_positions[:, 2]
                x_displacements = pos[:, 0] - self._initial_positions[:, 0]
                max_indent = -z_displacements.min()
                max_x_slide = x_displacements.max()

                # Check for ground penetration
                min_z = pos[:, 2].min()
                below_ground = (pos[:, 2] < 0).sum()

                # Verify actual indenter position from solver
                actual_center = self.solver.obstacle_centers[1]
                if SCENE_PARAMS.get('debug_verbose', False):
                    print(f"[MPM SIM] Step {step}: dz={dz*1000:.1f}mm, dx={dx_slide*1000:.1f}mm | "
                          f"indent={max_indent*1000:.1f}mm, x_slide={max_x_slide*1000:.1f}mm | "
                          f"min_z={min_z*1000:.1f}mm, below_ground={below_ground}")

        elapsed = time.time() - start_time
        print(f"MPM trajectory complete: {elapsed:.2f}s, {len(self.positions_history)} frames")

        return self.positions_history


# ==============================================================================
# Comparison Engine
# ==============================================================================
class RGBComparisonEngine:
    """Engine for comparing FEM and MPM sensor RGB outputs"""

    def __init__(
        self,
        fem_file: str,
        object_file: Optional[str] = None,
        mode: str = 'raw',
        visible: bool = True,
        save_dir: Optional[str] = None,
        fem_indenter_face: str = "tip",
        fem_indenter_geom: str = "auto",
    ):
        self.mode = mode
        self.visible = visible
        self.save_dir = Path(save_dir) if save_dir else None
        self.run_context: Dict[str, object] = {}
        self.fem_indenter_face = fem_indenter_face
        self.fem_indenter_geom = fem_indenter_geom

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize renderers
        self.fem_renderer = None
        self.mpm_renderer = None
        self.mpm_sim = None

        if HAS_EZGL:
            self.fem_renderer = FEMRGBRenderer(
                fem_file,
                object_file=object_file,
                visible=False,
                indenter_face=self.fem_indenter_face,
                indenter_geom=self.fem_indenter_geom,
            )
            self.mpm_renderer = MPMHeightFieldRenderer(visible=False)    

        if HAS_TAICHI:
            self.mpm_sim = MPMSimulationAdapter()
        self.fem_show_marker = True
        self.mpm_marker_mode = "warp"
        self.mpm_depth_tint = True
        self.mpm_depth_tint_mode = "frame"  # off|fixed|frame
        self.mpm_depth_tint_max_mm: Optional[float] = None
        self.mpm_show_indenter = False
        self.mpm_debug_overlay = "off"
        self.indenter_square_size_mm = _infer_square_size_mm_from_stl_path(object_file)

        # Export / audit outputs (only active when --save-dir is set)
        self.export_intermediate = False
        self.export_intermediate_every = 1
        self._exported_image_frames = set()
        self._exported_metrics_frames = set()
        self._exported_intermediate_frames = set()
        self._metrics_rows = []
        self._frame_to_phase: Optional[List[str]] = None

    def set_fem_show_marker(self, show: bool) -> None:
        self.fem_show_marker = bool(show)
        if self.fem_renderer is None:
            return
        sim = getattr(self.fem_renderer, "sensor_sim", None)
        if sim is None:
            return
        try:
            sim.set_show_marker(self.fem_show_marker)
        except Exception:
            pass

    def _write_run_manifest(self, record_interval: int, total_frames: int) -> None:
        if not self.save_dir:
            return

        # Record the exact solver->sensor frame reference used by MPM extraction.
        # Why: footprint inside/outside stats must use the same x_center_mm/y_min_mm,
        # otherwise the mask can be silently wrong when the indenter center is not
        # aligned to the gel center in solver coordinates.
        try:
            mpm = self.mpm_renderer
            if mpm is not None and bool(getattr(mpm, "_is_configured", False)):
                x_center_mm = float(getattr(mpm, "_x_center_mm", 0.0))
                y_min_mm = float(getattr(mpm, "_y_min_mm", 0.0))

                if isinstance(self.run_context, dict):
                    resolved = self.run_context.get("resolved")
                    if not isinstance(resolved, dict):
                        resolved = {}
                        self.run_context["resolved"] = resolved
                    scale = resolved.get("scale")
                    if not isinstance(scale, dict):
                        scale = {}
                        resolved["scale"] = scale

                    scale["x_center_mm"] = x_center_mm
                    scale["y_min_mm"] = y_min_mm
                    scale["sensor_frame_source"] = "mpm_renderer.configure_from_initial_positions"
        except Exception:
            pass

        # Resolve and self-check key coordinate conventions (auditability for A/C alignment).
        try:
            resolved = self.run_context.get("resolved") if isinstance(self.run_context, dict) else None
            if not isinstance(resolved, dict):
                resolved = {}
                if isinstance(self.run_context, dict):
                    self.run_context["resolved"] = resolved
            conv = resolved.get("conventions")
            if not isinstance(conv, dict):
                conv = {}
                resolved["conventions"] = conv
            scale = resolved.get("scale")
            if not isinstance(scale, dict):
                scale = {}
                resolved["scale"] = scale

            warnings: List[str] = []

            # 1) Flip conventions (field flip vs warp flip must be coherent).
            render_flip_x = bool(conv.get("mpm_height_field_flip_x", False))
            warp_flip_x = bool(conv.get("mpm_warp_flip_x", False))
            warp_flip_y = bool(conv.get("mpm_warp_flip_y", True))
            if render_flip_x != warp_flip_x:
                warnings.append(
                    f"mpm_render_flip_x({render_flip_x}) != mpm_warp_flip_x({warp_flip_x}); "
                    "this often indicates a double/half X flip in the UV→marker mapping"
                )
            if not warp_flip_y:
                warnings.append("mpm_warp_flip_y is off; v axis is likely inverted (uv v=+y up vs image y=down)")

            # 2) mm→px scales (explicitly record both marker_tex and RGB image).
            gel_size = scale.get("gel_size_mm", None)
            gel_w_mm = float(gel_size[0]) if isinstance(gel_size, list) and len(gel_size) == 2 else float(SCENE_PARAMS.get("gel_size_mm", (0.0, 0.0))[0])
            gel_h_mm = float(gel_size[1]) if isinstance(gel_size, list) and len(gel_size) == 2 else float(SCENE_PARAMS.get("gel_size_mm", (0.0, 0.0))[1])
            tex_size = None
            marker_grid = resolved.get("marker_grid")
            if isinstance(marker_grid, dict):
                tex_size = marker_grid.get("tex_size_wh", None)
            if isinstance(tex_size, list) and len(tex_size) == 2:
                tex_w, tex_h = int(tex_size[0]), int(tex_size[1])
            else:
                tex_w, tex_h = 320, 560

            # RGB camera size is currently fixed in MPMSensorScene (w,h).
            rgb_w, rgb_h = 400, 700
            scale.setdefault("mpm_rgb_camera_size_wh", [int(rgb_w), int(rgb_h)])
            scale.setdefault(
                "mpm_marker_tex_px_per_mm",
                [float(tex_w) / max(float(gel_w_mm), 1e-6), float(tex_h) / max(float(gel_h_mm), 1e-6)],
            )
            scale.setdefault(
                "mpm_rgb_px_per_mm",
                [float(rgb_w) / max(float(gel_w_mm), 1e-6), float(rgb_h) / max(float(gel_h_mm), 1e-6)],
            )
            scale.setdefault("mpm_marker_mm_to_px_source", "gel_size_mm")
            scale.setdefault("mpm_rgb_mm_to_px_source", "gel_size_mm")

            # 3) Explicit axis conventions (y-up/y-down; depth sign).
            conv.setdefault("uv_disp_mm_axes", {"u": "+x (right)", "v": "+y (up)"})
            conv.setdefault("image_axes", {"x": "+col (right)", "y": "+row (down)"})
            conv.setdefault("mpm_depth_sign", "indentation_negative_mm (height_field_mm<=0)")

            # 4) gel_size vs cam_view mismatch (common cause of wrong mm→px mapping when mixing sources).
            if not bool(scale.get("consistent", True)):
                try:
                    delta = scale.get("delta_mm", None)
                    warnings.append(f"gel_size_mm != cam_view_mm (delta_mm={delta}); avoid mixing mm→px sources")
                except Exception:
                    warnings.append("gel_size_mm != cam_view_mm; avoid mixing mm→px sources")

            if warnings:
                diag = resolved.get("diagnostics")
                if not isinstance(diag, dict):
                    diag = {}
                    resolved["diagnostics"] = diag
                diag.setdefault("warnings", [])
                if isinstance(diag.get("warnings"), list):
                    for w in warnings:
                        if w not in diag["warnings"]:
                            diag["warnings"].append(w)
        except Exception:
            # Never fail manifest generation due to diagnostics.
            pass

        press_steps = int(SCENE_PARAMS["press_steps"])
        slide_steps = int(SCENE_PARAMS["slide_steps"])
        hold_steps = int(SCENE_PARAMS["hold_steps"])
        total_steps = press_steps + slide_steps + hold_steps

        def _phase_for_step(step: int) -> str:
            if step < press_steps:
                return "press"
            if step < press_steps + slide_steps:
                return "slide"
            return "hold"

        frame_to_step = [int(i * record_interval) for i in range(int(total_frames))]
        frame_to_phase = [_phase_for_step(step) for step in frame_to_step]
        self._frame_to_phase = list(frame_to_phase)
        phase_ranges: Dict[str, Dict[str, int]] = {}
        for i, phase in enumerate(frame_to_phase):
            if phase not in phase_ranges:
                phase_ranges[phase] = {"start_frame": i, "end_frame": i}
            else:
                phase_ranges[phase]["end_frame"] = i

        frame_controls = None
        if self.mpm_sim and self.mpm_sim.frame_controls:
            frame_controls = [
                {
                    "frame": int(i),
                    "press_amount_m": float(press_m),
                    "slide_amount_m": float(slide_m),
                }
                for i, (press_m, slide_m) in enumerate(self.mpm_sim.frame_controls)
            ]

        frame_indenter_centers = None
        if self.mpm_sim and getattr(self.mpm_sim, "frame_indenter_centers_m", None):
            centers = self.mpm_sim.frame_indenter_centers_m
            if centers:
                frame_indenter_centers = [
                    {
                        "frame": int(i),
                        "center_m": [float(cx), float(cy), float(cz)],
                    }
                    for i, (cx, cy, cz) in enumerate(centers)
                ]

        sanitized_run_context = _sanitize_run_context_for_manifest(self.run_context)
        marker_appearance = None
        if isinstance(sanitized_run_context, dict):
            resolved = sanitized_run_context.get("resolved")
            if isinstance(resolved, dict):
                marker_appearance = resolved.get("marker_appearance")

        manifest = {
            "created_at": datetime.datetime.now().astimezone().isoformat(),
            "argv": _sanitize_argv_for_manifest(sys.argv),
            "run_context": sanitized_run_context,
            "marker_appearance": marker_appearance,
            "scene_params": dict(SCENE_PARAMS),
            "deps": _collect_manifest_deps(_PROJECT_ROOT),
            "trajectory": {
                "press_steps": press_steps,
                "slide_steps": slide_steps,
                "hold_steps": hold_steps,
                "total_steps": total_steps,
                "record_interval": int(record_interval),
                "total_frames": int(total_frames),
                "phase_ranges_frames": phase_ranges,
                "frame_to_step": frame_to_step,
                "frame_to_phase": frame_to_phase,
                "frame_controls": frame_controls,
                "frame_indenter_centers_m": frame_indenter_centers,
            },
            "outputs": {
                "frames_glob": {
                    "fem": "fem_*.png",
                    "mpm": "mpm_*.png",
                },
                "run_manifest": "run_manifest.json",
                "metrics": {
                    "csv": "metrics.csv",
                    "json": "metrics.json",
                },
                "intermediate": {
                    "dir": "intermediate",
                    "frames_glob": "intermediate/frame_*.npz",
                },
            },
        }

        manifest_path = self.save_dir / "run_manifest.json"
        try:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Warning: failed to write run manifest: {e}")
            return

        try:
            _write_tuning_notes(
                self.save_dir,
                record_interval=int(record_interval),
                total_frames=int(total_frames),
                run_context=manifest.get("run_context") if isinstance(manifest, dict) else {},
                overwrite=False,
                reason="runtime",
            )
        except Exception as e:
            print(f"Warning: failed to write tuning notes (runtime): {e}")

    def _write_metrics_files(self) -> None:
        if not self.save_dir or not self._metrics_rows:
            return

        metrics_csv_path = self.save_dir / "metrics.csv"
        metrics_json_path = self.save_dir / "metrics.json"
        fieldnames = [
            "frame",
            "phase",
            "mode",
            "mae",
            "mae_r",
            "mae_g",
            "mae_b",
            "max_abs",
            "p50",
            "p90",
            "p99",
        ]

        try:
            with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self._metrics_rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
        except Exception as e:
            print(f"Warning: failed to write metrics.csv: {e}")

        try:
            mae_values = [float(row.get("mae", 0.0)) for row in self._metrics_rows]
            summary = {"frames": int(len(self._metrics_rows))}
            if mae_values:
                mae_arr = np.array(mae_values, dtype=np.float32)
                summary.update(
                    {
                        "mae_mean": float(mae_arr.mean()),
                        "mae_p50": float(np.percentile(mae_arr, 50)),
                        "mae_p90": float(np.percentile(mae_arr, 90)),
                        "mae_max": float(mae_arr.max()),
                    }
                )
            payload = {
                "created_at": datetime.datetime.now().astimezone().isoformat(),
                "rows": list(self._metrics_rows),
                "summary": summary,
            }
            metrics_json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Warning: failed to write metrics.json: {e}")

    def _compute_fem_contact_mask_u8(self) -> Optional[np.ndarray]:
        if not self.fem_renderer:
            return None
        sim = getattr(self.fem_renderer, "sensor_sim", None)
        fem_sim = getattr(sim, "fem_sim", None)
        if fem_sim is None:
            return None
        try:
            contact_idx = fem_sim.contact_state.contact_idx()
            n_top = int(len(fem_sim.top_nodes))
            mask_flat = np.zeros((n_top,), dtype=np.uint8)
            if contact_idx is not None and contact_idx.shape[1] > 0:
                top_idx = contact_idx[0].astype(np.int64, copy=False)
                mask_flat[top_idx] = 1
            return mask_flat.reshape(fem_sim.mesh_shape)
        except Exception:
            return None

    def _export_intermediate_frame(
        self,
        frame: int,
        mpm_height_field_mm: Optional[np.ndarray],
        mpm_uv_disp_mm: Optional[np.ndarray],
        mpm_uv_cnt_i32: Optional[np.ndarray] = None,
        mpm_uv_nonzero_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_hole_count_i32: Optional[np.ndarray] = None,
        mpm_uv_pseudohole_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_pseudohole_count_i32: Optional[np.ndarray] = None,
        mpm_uv_du_mm: Optional[np.ndarray] = None,
        mpm_uv_du_cnt_i32: Optional[np.ndarray] = None,
        mpm_uv_du_nonzero_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_du_hole_count_i32: Optional[np.ndarray] = None,
        mpm_uv_mask_footprint_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_despike_footprint_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_despike_gate_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_despike_replaced_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_despike_capped_mask_u8: Optional[np.ndarray] = None,
        fem_depth_mm: Optional[np.ndarray] = None,
        fem_marker_disp: Optional[np.ndarray] = None,
        fem_contact_mask_u8: Optional[np.ndarray] = None,
    ) -> None:
        if not self.save_dir:
            return
        intermediate_dir = self.save_dir / "intermediate"
        try:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        payload: Dict[str, object] = {"frame": np.array(int(frame), dtype=np.int32)}
        if mpm_height_field_mm is not None:
            height_field_mm = mpm_height_field_mm.astype(np.float32, copy=False)
            payload["height_field_mm"] = height_field_mm
            payload["contact_mask"] = (height_field_mm < -0.01).astype(np.uint8, copy=False)
        if mpm_uv_disp_mm is not None:
            payload["uv_disp_mm"] = mpm_uv_disp_mm.astype(np.float32, copy=False)
        if mpm_uv_cnt_i32 is not None:
            payload["uv_cnt_i32"] = mpm_uv_cnt_i32.astype(np.int32, copy=False)
        if mpm_uv_nonzero_mask_u8 is not None:
            payload["uv_nonzero_mask_u8"] = mpm_uv_nonzero_mask_u8.astype(np.uint8, copy=False)
        if mpm_uv_hole_count_i32 is not None:
            payload["uv_hole_count_i32"] = mpm_uv_hole_count_i32.astype(np.int32, copy=False)
        if mpm_uv_pseudohole_mask_u8 is not None:
            payload["uv_pseudohole_mask_u8"] = mpm_uv_pseudohole_mask_u8.astype(np.uint8, copy=False)
        if mpm_uv_pseudohole_count_i32 is not None:
            payload["uv_pseudohole_count_i32"] = mpm_uv_pseudohole_count_i32.astype(np.int32, copy=False)
        if mpm_uv_du_mm is not None:
            payload["uv_du_mm"] = mpm_uv_du_mm.astype(np.float32, copy=False)
        if mpm_uv_du_cnt_i32 is not None:
            payload["uv_du_cnt_i32"] = mpm_uv_du_cnt_i32.astype(np.int32, copy=False)
        if mpm_uv_du_nonzero_mask_u8 is not None:
            payload["uv_du_nonzero_mask_u8"] = mpm_uv_du_nonzero_mask_u8.astype(np.uint8, copy=False)
        if mpm_uv_du_hole_count_i32 is not None:
            payload["uv_du_hole_count_i32"] = mpm_uv_du_hole_count_i32.astype(np.int32, copy=False)
        if mpm_uv_mask_footprint_mask_u8 is not None:
            payload["uv_mask_footprint_u8"] = mpm_uv_mask_footprint_mask_u8.astype(np.uint8, copy=False)
        if mpm_uv_despike_footprint_mask_u8 is not None:
            payload["footprint_mask_u8"] = mpm_uv_despike_footprint_mask_u8.astype(np.uint8, copy=False)
        if mpm_uv_despike_gate_mask_u8 is not None:
            payload["uv_despike_gate_mask_u8"] = mpm_uv_despike_gate_mask_u8.astype(np.uint8, copy=False)
        if mpm_uv_despike_replaced_mask_u8 is not None:
            payload["uv_despike_replaced_mask_u8"] = mpm_uv_despike_replaced_mask_u8.astype(np.uint8, copy=False)
        if mpm_uv_despike_capped_mask_u8 is not None:
            payload["uv_despike_capped_mask_u8"] = mpm_uv_despike_capped_mask_u8.astype(np.uint8, copy=False)
        if fem_depth_mm is not None:
            payload["fem_depth_mm"] = fem_depth_mm.astype(np.float32, copy=False)
        if fem_marker_disp is not None:
            payload["fem_marker_disp"] = fem_marker_disp.astype(np.float32, copy=False)
        if fem_contact_mask_u8 is not None:
            payload["fem_contact_mask_u8"] = fem_contact_mask_u8.astype(np.uint8, copy=False)

        out_path = intermediate_dir / f"frame_{int(frame):04d}.npz"
        try:
            np.savez_compressed(out_path, **payload)
        except Exception as e:
            print(f"Warning: failed to export intermediate frame {frame}: {e}")

    def _export_frame_artifacts(
        self,
        frame_id: int,
        fem_rgb: Optional[np.ndarray],
        mpm_rgb: Optional[np.ndarray],
        mpm_height_field: Optional[np.ndarray],
        mpm_uv_disp: Optional[np.ndarray],
        mpm_uv_cnt_i32: Optional[np.ndarray] = None,
        mpm_uv_nonzero_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_hole_count_i32: Optional[np.ndarray] = None,
        mpm_uv_pseudohole_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_pseudohole_count_i32: Optional[np.ndarray] = None,
        mpm_uv_du_mm: Optional[np.ndarray] = None,
        mpm_uv_du_cnt_i32: Optional[np.ndarray] = None,
        mpm_uv_du_nonzero_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_du_hole_count_i32: Optional[np.ndarray] = None,
        mpm_uv_mask_footprint_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_despike_footprint_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_despike_gate_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_despike_replaced_mask_u8: Optional[np.ndarray] = None,
        mpm_uv_despike_capped_mask_u8: Optional[np.ndarray] = None,
    ) -> None:
        if not self.save_dir:
            return

        # Metrics (one-shot per frame)
        if (
            fem_rgb is not None
            and mpm_rgb is not None
            and frame_id not in self._exported_metrics_frames
        ):
            self._exported_metrics_frames.add(int(frame_id))
            try:
                metrics = _compute_rgb_diff_metrics(fem_rgb, mpm_rgb)
                phase = None
                if self._frame_to_phase is not None and frame_id < len(self._frame_to_phase):
                    phase = self._frame_to_phase[frame_id]
                self._metrics_rows.append(
                    {
                        "frame": int(frame_id),
                        "phase": phase,
                        "mode": str(self.mode),
                        **metrics,
                    }
                )
                self._write_metrics_files()
            except Exception as e:
                print(f"Warning: failed to compute metrics for frame {frame_id}: {e}")

        # Intermediate arrays (one-shot per frame, with --export-intermediate-every)
        if self.export_intermediate and frame_id not in self._exported_intermediate_frames:
            self._exported_intermediate_frames.add(int(frame_id))
            every = int(self.export_intermediate_every) if int(self.export_intermediate_every) > 0 else 1
            if frame_id % every == 0:
                fem_depth_mm = None
                fem_marker_disp = None
                fem_contact_mask_u8 = None
                if self.fem_renderer is not None:
                    try:
                        fem_depth_mm = self.fem_renderer.sensor_sim.get_depth()
                    except Exception:
                        pass
                    try:
                        fem_marker_disp = self.fem_renderer.sensor_sim.get_marker_displacement()
                    except Exception:
                        pass
                    fem_contact_mask_u8 = self._compute_fem_contact_mask_u8()
                self._export_intermediate_frame(
                    frame=frame_id,
                    mpm_height_field_mm=mpm_height_field,
                    mpm_uv_disp_mm=mpm_uv_disp,
                    mpm_uv_cnt_i32=mpm_uv_cnt_i32,
                    mpm_uv_nonzero_mask_u8=mpm_uv_nonzero_mask_u8,
                    mpm_uv_hole_count_i32=mpm_uv_hole_count_i32,
                    mpm_uv_pseudohole_mask_u8=mpm_uv_pseudohole_mask_u8,
                    mpm_uv_pseudohole_count_i32=mpm_uv_pseudohole_count_i32,
                    mpm_uv_du_mm=mpm_uv_du_mm,
                    mpm_uv_du_cnt_i32=mpm_uv_du_cnt_i32,
                    mpm_uv_du_nonzero_mask_u8=mpm_uv_du_nonzero_mask_u8,
                    mpm_uv_du_hole_count_i32=mpm_uv_du_hole_count_i32,
                    mpm_uv_mask_footprint_mask_u8=mpm_uv_mask_footprint_mask_u8,
                    mpm_uv_despike_footprint_mask_u8=mpm_uv_despike_footprint_mask_u8,
                    mpm_uv_despike_gate_mask_u8=mpm_uv_despike_gate_mask_u8,
                    mpm_uv_despike_replaced_mask_u8=mpm_uv_despike_replaced_mask_u8,
                    mpm_uv_despike_capped_mask_u8=mpm_uv_despike_capped_mask_u8,
                    fem_depth_mm=fem_depth_mm,
                    fem_marker_disp=fem_marker_disp,
                    fem_contact_mask_u8=fem_contact_mask_u8,
                )

        # Frame images (avoid overwriting when UI is looping)
        if HAS_CV2 and fem_rgb is not None and frame_id not in self._exported_image_frames:
            self._exported_image_frames.add(int(frame_id))
            try:
                cv2.imwrite(
                    str(self.save_dir / f"fem_{int(frame_id):04d}.png"),
                    cv2.cvtColor(fem_rgb, cv2.COLOR_RGB2BGR),
                )
                if mpm_rgb is not None:
                    cv2.imwrite(
                        str(self.save_dir / f"mpm_{int(frame_id):04d}.png"),
                        cv2.cvtColor(mpm_rgb, cv2.COLOR_RGB2BGR),
                    )
            except Exception as e:
                print(f"Warning: failed to write frame images for {frame_id}: {e}")

    def run_comparison(self, fps: int = 30, record_interval: int = 5, interactive: bool = True):
        """Run side-by-side comparison visualization"""
        if not HAS_EZGL:
            print("ezgl not available, cannot run visualization")
            return

        # Run MPM simulation first to get position history
        if self.mpm_sim:
            positions_history = self.mpm_sim.run_trajectory(record_interval=record_interval)
        else:
            positions_history = []
            print("MPM not available, showing FEM only")

        # Pre-configure MPM sensor frame mapping early so run_manifest can record the exact
        # x_center_mm/y_min_mm used by extraction (avoid heuristic inference downstream).
        if (
            self.mpm_renderer
            and self.mpm_sim
            and (not bool(getattr(self.mpm_renderer, "_is_configured", False)))
            and (getattr(self.mpm_sim, "initial_positions_m", None) is not None)
            and positions_history
        ):
            try:
                self.mpm_renderer.configure_from_initial_positions(
                    self.mpm_sim.initial_positions_m,
                    self.mpm_sim.initial_top_z_m,
                )
            except Exception:
                # Best-effort: manifest should still be generated even if mapping fails.
                pass

        total_frames = len(positions_history) if positions_history else 100
        self._write_run_manifest(record_interval=record_interval, total_frames=total_frames)

        if interactive:
            # Create UI (loops until closed)
            self._create_ui(positions_history, fps)
        else:
            # Headless batch export (run once and exit)
            self._run_batch(positions_history)

    def _run_batch(self, mpm_positions: List[np.ndarray]) -> None:
        """Run a finite headless loop to export frames, metrics and intermediates."""
        if not self.save_dir:
            print("ERROR: batch mode requires --save-dir")
            return

        total_frames = len(mpm_positions) if mpm_positions else 100
        press_steps = SCENE_PARAMS['press_steps']
        slide_steps = SCENE_PARAMS['slide_steps']
        hold_steps = SCENE_PARAMS['hold_steps']
        total_steps = press_steps + slide_steps + hold_steps
        press_end_ratio = press_steps / total_steps if total_steps > 0 else 0.0
        slide_end_ratio = (press_steps + slide_steps) / total_steps if total_steps > 0 else 0.0

        # Pre-configure MPM renderer mapping if we have initial particle positions
        if (
            self.mpm_renderer
            and self.mpm_sim
            and self.mpm_sim.initial_positions_m is not None
            and mpm_positions
        ):
            self.mpm_renderer.configure_from_initial_positions(
                self.mpm_sim.initial_positions_m, self.mpm_sim.initial_top_z_m
            )
            self.mpm_renderer.scene.set_marker_mode(self.mpm_marker_mode)
            self.mpm_renderer.scene.set_depth_tint_config(self.mpm_depth_tint_mode, max_mm=self.mpm_depth_tint_max_mm)
            self.mpm_renderer.scene.set_indenter_overlay(
                self.mpm_show_indenter, square_size_mm=self.indenter_square_size_mm
            )
            self.mpm_renderer.scene.set_debug_overlay(self.mpm_debug_overlay)

        for frame_id in range(int(total_frames)):
            # Prefer recorded MPM control signals (strict frame alignment)
            if self.mpm_sim and self.mpm_sim.frame_controls and frame_id < len(self.mpm_sim.frame_controls):
                press_amount_m, slide_amount_m = self.mpm_sim.frame_controls[frame_id]
                press_y_mm = float(press_amount_m) * 1000.0
                slide_x_mm = float(slide_amount_m) * 1000.0
            else:
                # Fallback: Use consistent phase ratios with MPM trajectory
                t = frame_id / max(total_frames - 1, 1)
                if t < press_end_ratio:
                    phase_t = t / press_end_ratio if press_end_ratio > 0 else 0
                    press_y_mm = float(SCENE_PARAMS['press_depth_mm']) * phase_t
                    slide_x_mm = 0.0
                elif t < slide_end_ratio:
                    phase_t = (t - press_end_ratio) / (slide_end_ratio - press_end_ratio) if (slide_end_ratio - press_end_ratio) > 0 else 0
                    press_y_mm = float(SCENE_PARAMS['press_depth_mm'])
                    slide_x_mm = float(SCENE_PARAMS['slide_distance_mm']) * phase_t
                else:
                    press_y_mm = float(SCENE_PARAMS['press_depth_mm'])
                    slide_x_mm = float(SCENE_PARAMS['slide_distance_mm'])

            if frame_id % max(total_frames // 10, 1) == 0:
                print(f"[BATCH] frame={frame_id}/{total_frames-1} press={press_y_mm:.2f}mm slide={slide_x_mm:.2f}mm")

            fem_rgb = None
            if self.fem_renderer:
                y_pos_mm = -press_y_mm
                self.fem_renderer.set_indenter_pose(slide_x_mm, y_pos_mm, 0.0)
                fem_rgb = self.fem_renderer.step()
                if self.mode == "diff":
                    fem_rgb = self.fem_renderer.get_diff_image()

            mpm_rgb = None
            mpm_height_field = None
            mpm_uv_disp = None
            mpm_uv_cnt = None
            mpm_uv_nonzero_mask = None
            mpm_uv_hole_count = None
            mpm_uv_pseudohole_mask = None
            mpm_uv_pseudohole_count = None
            mpm_uv_du_mm = None
            mpm_uv_du_cnt = None
            mpm_uv_du_nonzero_mask = None
            mpm_uv_du_hole_count = None
            mpm_uv_mask_footprint = None
            mpm_uv_despike_footprint = None
            mpm_uv_despike_gate = None
            mpm_uv_despike_replaced = None
            mpm_uv_despike_capped = None
            if self.mpm_renderer and self.mpm_sim and mpm_positions and frame_id < len(mpm_positions):
                pos = mpm_positions[frame_id]
                indenter_center_m = None
                if (
                    self.mpm_sim
                    and getattr(self.mpm_sim, "frame_indenter_centers_m", None)
                    and frame_id < len(self.mpm_sim.frame_indenter_centers_m)
                ):
                    indenter_center_m = self.mpm_sim.frame_indenter_centers_m[frame_id]
                vel = None
                dt_s = None
                if (
                    getattr(self.mpm_sim, "velocities_history", None) is not None
                    and frame_id < len(getattr(self.mpm_sim, "velocities_history", []))
                ):
                    vel = self.mpm_sim.velocities_history[frame_id]
                    dt_s = getattr(self.mpm_sim, "frame_dt_s", None)

                mpm_height_field, mpm_uv_disp = self.mpm_renderer.extract_surface_fields(
                    pos,
                    self.mpm_sim.initial_top_z_m,
                    indenter_center_m=indenter_center_m,
                    velocities_m=vel,
                    frame_dt_s=dt_s,
                )
                mpm_uv_cnt = getattr(self.mpm_renderer, "_last_uv_cnt_i32", None)
                mpm_uv_nonzero_mask = getattr(self.mpm_renderer, "_last_uv_nonzero_mask_u8", None)
                mpm_uv_hole_count = getattr(self.mpm_renderer, "_last_uv_hole_count_i32", None)
                mpm_uv_pseudohole_mask = getattr(self.mpm_renderer, "_last_uv_pseudohole_mask_u8", None)
                mpm_uv_pseudohole_count = getattr(self.mpm_renderer, "_last_uv_pseudohole_count_i32", None)
                mpm_uv_du_mm = getattr(self.mpm_renderer, "_last_uv_du_mm", None)
                mpm_uv_du_cnt = getattr(self.mpm_renderer, "_last_uv_du_cnt_i32", None)
                mpm_uv_du_nonzero_mask = getattr(self.mpm_renderer, "_last_uv_du_nonzero_mask_u8", None)
                mpm_uv_du_hole_count = getattr(self.mpm_renderer, "_last_uv_du_hole_count_i32", None)
                mpm_uv_mask_footprint = getattr(self.mpm_renderer, "_last_uv_mask_footprint_mask_u8", None)
                mpm_uv_despike_footprint = getattr(self.mpm_renderer, "_last_uv_despike_footprint_mask_u8", None)
                mpm_uv_despike_gate = getattr(self.mpm_renderer, "_last_uv_despike_gate_mask_u8", None)
                mpm_uv_despike_replaced = getattr(self.mpm_renderer, "_last_uv_despike_replaced_mask_u8", None)
                mpm_uv_despike_capped = getattr(self.mpm_renderer, "_last_uv_despike_capped_mask_u8", None)
                self.mpm_renderer.scene.set_uv_displacement(mpm_uv_disp, uv_du_mm=mpm_uv_du_mm)
                if self.mode == 'diff':
                    mpm_rgb = self.mpm_renderer.get_diff_image(mpm_height_field)
                else:
                    mpm_rgb = self.mpm_renderer.render(mpm_height_field)

                if self.mpm_show_indenter and self.mpm_sim and self.mpm_sim.frame_controls and frame_id < len(self.mpm_sim.frame_controls):
                    _, slide_amount_m = self.mpm_sim.frame_controls[frame_id]
                    self.mpm_renderer.scene.set_indenter_center(float(slide_amount_m) * 1000.0, SCENE_PARAMS['gel_size_mm'][1] / 2.0)

            self._export_frame_artifacts(
                frame_id=frame_id,
                fem_rgb=fem_rgb,
                mpm_rgb=mpm_rgb,
                mpm_height_field=mpm_height_field,
                mpm_uv_disp=mpm_uv_disp,
                mpm_uv_cnt_i32=mpm_uv_cnt,
                mpm_uv_nonzero_mask_u8=mpm_uv_nonzero_mask,
                mpm_uv_hole_count_i32=mpm_uv_hole_count,
                mpm_uv_pseudohole_mask_u8=mpm_uv_pseudohole_mask,
                mpm_uv_pseudohole_count_i32=mpm_uv_pseudohole_count,
                mpm_uv_du_mm=mpm_uv_du_mm,
                mpm_uv_du_cnt_i32=mpm_uv_du_cnt,
                mpm_uv_du_nonzero_mask_u8=mpm_uv_du_nonzero_mask,
                mpm_uv_du_hole_count_i32=mpm_uv_du_hole_count,
                mpm_uv_mask_footprint_mask_u8=mpm_uv_mask_footprint,
                mpm_uv_despike_footprint_mask_u8=mpm_uv_despike_footprint,
                mpm_uv_despike_gate_mask_u8=mpm_uv_despike_gate,
                mpm_uv_despike_replaced_mask_u8=mpm_uv_despike_replaced,
                mpm_uv_despike_capped_mask_u8=mpm_uv_despike_capped,
            )

        print(f"[BATCH] done: frames={total_frames}, save_dir={self.save_dir}")

    def _create_ui(self, mpm_positions: List[np.ndarray], fps: int):
        """Create side-by-side image display UI"""
        frame_idx = [0]
        total_frames = len(mpm_positions) if mpm_positions else 100

        # Trajectory phase boundaries (matching MPM trajectory phases)
        press_steps = SCENE_PARAMS['press_steps']
        slide_steps = SCENE_PARAMS['slide_steps']
        hold_steps = SCENE_PARAMS['hold_steps']
        total_steps = press_steps + slide_steps + hold_steps

        # Calculate phase boundaries as ratios
        press_end_ratio = press_steps / total_steps
        slide_end_ratio = (press_steps + slide_steps) / total_steps

        # Storage for image view widgets (set in UI building)
        fem_view = [None]
        mpm_view = [None]

        # Pre-configure MPM renderer mapping if we have initial particle positions
        if self.mpm_renderer and self.mpm_sim and self.mpm_sim.initial_positions_m is not None and mpm_positions:
            self.mpm_renderer.configure_from_initial_positions(
                self.mpm_sim.initial_positions_m, self.mpm_sim.initial_top_z_m
            )
            self.mpm_renderer.scene.set_marker_mode(self.mpm_marker_mode)
            self.mpm_renderer.scene.set_depth_tint_config(self.mpm_depth_tint_mode, max_mm=self.mpm_depth_tint_max_mm)
            self.mpm_renderer.scene.set_indenter_overlay(self.mpm_show_indenter, square_size_mm=self.indenter_square_size_mm)
            self.mpm_renderer.scene.set_debug_overlay(self.mpm_debug_overlay)

        def on_timeout():
            # Get FEM image
            press_depth = SCENE_PARAMS['press_depth_mm']
            slide_dist = SCENE_PARAMS['slide_distance_mm']

            # Prefer recorded MPM control signals (strict frame alignment).
            if self.mpm_sim and self.mpm_sim.frame_controls and frame_idx[0] < len(self.mpm_sim.frame_controls):
                press_amount_m, slide_amount_m = self.mpm_sim.frame_controls[frame_idx[0]]
                press_y_mm = float(press_amount_m) * 1000.0
                slide_x_mm = float(slide_amount_m) * 1000.0
            else:
                # Fallback: Use consistent phase ratios with MPM trajectory
                t = frame_idx[0] / max(total_frames - 1, 1)
                if t < press_end_ratio:
                    phase_t = t / press_end_ratio if press_end_ratio > 0 else 0
                    press_y_mm = press_depth * phase_t
                    slide_x_mm = 0.0
                elif t < slide_end_ratio:
                    phase_t = (t - press_end_ratio) / (slide_end_ratio - press_end_ratio) if (slide_end_ratio - press_end_ratio) > 0 else 0
                    press_y_mm = press_depth
                    slide_x_mm = slide_dist * phase_t
                else:
                    press_y_mm = press_depth
                    slide_x_mm = slide_dist

            # FEM rendering
            # Depth camera: eye=(0,0,0), center=(0,1,0) -> faces +y direction
            # FEM expects depth values close to 0 or negative for contact detection.
            # From debug output: depth_min equals object center y-position (not surface).
            # So we place object center at y = -press to get negative depth when pressing.
            fem_rgb = None
            if self.fem_renderer:
                # Object center at y=0 means just touching surface (depth=0)
                # Object center at y=-press means pressing into surface (depth<0)
                y_pos_mm = -press_y_mm
                print(f"[FEM] Frame {frame_idx[0]}: press={press_y_mm:.2f}mm, slide={slide_x_mm:.2f}mm")
                self.fem_renderer.set_indenter_pose(slide_x_mm, y_pos_mm, 0.0)

                if SCENE_PARAMS.get("debug_verbose", False):
                    mpm_center_mm = None
                    if (
                        self.mpm_sim
                        and getattr(self.mpm_sim, "frame_indenter_centers_m", None)
                        and frame_idx[0] < len(self.mpm_sim.frame_indenter_centers_m)
                    ):
                        cx, cy, cz = self.mpm_sim.frame_indenter_centers_m[frame_idx[0]]
                        mpm_center_mm = [cx * 1000.0, cy * 1000.0, cz * 1000.0]

                    fem_raw_mm = None
                    try:
                        fem_raw_mm = (self.fem_renderer.depth_scene.get_object_pose_raw().xyz * 1000.0).tolist()
                    except Exception:
                        pass

                    if mpm_center_mm is not None or fem_raw_mm is not None:
                        print(f"[POSE] frame={frame_idx[0]} mpm_center_mm={mpm_center_mm} fem_raw_pose_mm={fem_raw_mm}")
                fem_rgb = self.fem_renderer.step()
                if self.mode == 'diff':
                    fem_rgb = self.fem_renderer.get_diff_image()
                if fem_view[0] is not None:
                    fem_view[0].setData(fem_rgb)

            # MPM rendering
            mpm_rgb = None
            mpm_height_field = None
            mpm_uv_disp = None
            mpm_uv_cnt = None
            mpm_uv_nonzero_mask = None
            mpm_uv_hole_count = None
            mpm_uv_pseudohole_mask = None
            mpm_uv_pseudohole_count = None
            mpm_uv_du_mm = None
            mpm_uv_du_cnt = None
            mpm_uv_du_nonzero_mask = None
            mpm_uv_du_hole_count = None
            mpm_uv_mask_footprint = None
            mpm_uv_despike_footprint = None
            mpm_uv_despike_gate = None
            mpm_uv_despike_replaced = None
            mpm_uv_despike_capped = None
            if self.mpm_renderer and mpm_positions and frame_idx[0] < len(mpm_positions):
                pos = mpm_positions[frame_idx[0]]
                indenter_center_m = None
                if (
                    self.mpm_sim
                    and getattr(self.mpm_sim, "frame_indenter_centers_m", None)
                    and frame_idx[0] < len(self.mpm_sim.frame_indenter_centers_m)
                ):
                    indenter_center_m = self.mpm_sim.frame_indenter_centers_m[frame_idx[0]]
                vel = None
                dt_s = None
                if (
                    self.mpm_sim is not None
                    and getattr(self.mpm_sim, "velocities_history", None) is not None
                    and frame_idx[0] < len(getattr(self.mpm_sim, "velocities_history", []))
                ):
                    vel = self.mpm_sim.velocities_history[frame_idx[0]]
                    dt_s = getattr(self.mpm_sim, "frame_dt_s", None)
                mpm_height_field, mpm_uv_disp = self.mpm_renderer.extract_surface_fields(
                    pos,
                    self.mpm_sim.initial_top_z_m,
                    indenter_center_m=indenter_center_m,
                    velocities_m=vel,
                    frame_dt_s=dt_s,
                )
                mpm_uv_cnt = getattr(self.mpm_renderer, "_last_uv_cnt_i32", None)
                mpm_uv_nonzero_mask = getattr(self.mpm_renderer, "_last_uv_nonzero_mask_u8", None)
                mpm_uv_hole_count = getattr(self.mpm_renderer, "_last_uv_hole_count_i32", None)
                mpm_uv_pseudohole_mask = getattr(self.mpm_renderer, "_last_uv_pseudohole_mask_u8", None)
                mpm_uv_pseudohole_count = getattr(self.mpm_renderer, "_last_uv_pseudohole_count_i32", None)
                mpm_uv_du_mm = getattr(self.mpm_renderer, "_last_uv_du_mm", None)
                mpm_uv_du_cnt = getattr(self.mpm_renderer, "_last_uv_du_cnt_i32", None)
                mpm_uv_du_nonzero_mask = getattr(self.mpm_renderer, "_last_uv_du_nonzero_mask_u8", None)
                mpm_uv_du_hole_count = getattr(self.mpm_renderer, "_last_uv_du_hole_count_i32", None)
                mpm_uv_mask_footprint = getattr(self.mpm_renderer, "_last_uv_mask_footprint_mask_u8", None)
                mpm_uv_despike_footprint = getattr(self.mpm_renderer, "_last_uv_despike_footprint_mask_u8", None)
                mpm_uv_despike_gate = getattr(self.mpm_renderer, "_last_uv_despike_gate_mask_u8", None)
                mpm_uv_despike_replaced = getattr(self.mpm_renderer, "_last_uv_despike_replaced_mask_u8", None)
                mpm_uv_despike_capped = getattr(self.mpm_renderer, "_last_uv_despike_capped_mask_u8", None)
                self.mpm_renderer.scene.set_uv_displacement(mpm_uv_disp, uv_du_mm=mpm_uv_du_mm)
                if self.mode == 'diff':
                    mpm_rgb = self.mpm_renderer.get_diff_image(mpm_height_field)
                else:
                    mpm_rgb = self.mpm_renderer.render(mpm_height_field)
                if mpm_view[0] is not None:
                    mpm_view[0].setData(mpm_rgb)

                if self.mpm_show_indenter and self.mpm_sim and self.mpm_sim.frame_controls and frame_idx[0] < len(self.mpm_sim.frame_controls):
                    _, slide_amount_m = self.mpm_sim.frame_controls[frame_idx[0]]
                    # y 方向用胶体中心做可视化对齐（square 压头在 y 方向不移动）
                    self.mpm_renderer.scene.set_indenter_center(float(slide_amount_m) * 1000.0, SCENE_PARAMS['gel_size_mm'][1] / 2.0)

            if self.save_dir:
                frame_id = int(frame_idx[0])
                self._export_frame_artifacts(
                    frame_id=frame_id,
                    fem_rgb=fem_rgb,
                    mpm_rgb=mpm_rgb,
                    mpm_height_field=mpm_height_field,
                    mpm_uv_disp=mpm_uv_disp,
                    mpm_uv_cnt_i32=mpm_uv_cnt,
                    mpm_uv_nonzero_mask_u8=mpm_uv_nonzero_mask,
                    mpm_uv_hole_count_i32=mpm_uv_hole_count,
                    mpm_uv_pseudohole_mask_u8=mpm_uv_pseudohole_mask,
                    mpm_uv_pseudohole_count_i32=mpm_uv_pseudohole_count,
                    mpm_uv_du_mm=mpm_uv_du_mm,
                    mpm_uv_du_cnt_i32=mpm_uv_du_cnt,
                    mpm_uv_du_nonzero_mask_u8=mpm_uv_du_nonzero_mask,
                    mpm_uv_du_hole_count_i32=mpm_uv_du_hole_count,
                    mpm_uv_mask_footprint_mask_u8=mpm_uv_mask_footprint,
                    mpm_uv_despike_footprint_mask_u8=mpm_uv_despike_footprint,
                    mpm_uv_despike_gate_mask_u8=mpm_uv_despike_gate,
                    mpm_uv_despike_replaced_mask_u8=mpm_uv_despike_replaced,
                    mpm_uv_despike_capped_mask_u8=mpm_uv_despike_capped,
                )

            next_idx = (frame_idx[0] + 1) % total_frames
            # UI loops: advect_points is stateful, so reset when wrapping to frame 0.
            if next_idx == 0 and self.mpm_renderer is not None:
                try:
                    self.mpm_renderer.scene.reset_advect_points()
                except Exception:
                    pass
            frame_idx[0] = next_idx

        # Build UI
        with tb.window("MPM vs FEM RGB Compare", None, 10, pos=(100, 100), size=(900, 800)):
            tb.add_text(f"Mode: {self.mode} | FPS: {fps}")
            tb.add_spacer(10)

            with tb.group("images", horizontal=True, show=False):
                # FEM panel
                with tb.group("FEM Sensor", horizontal=False, show=True):
                    fem_view[0] = tb.add_image_view("fem_image", None, img_size=(400, 700), img_format="rgb")

                # MPM panel
                with tb.group("MPM Sensor", horizontal=False, show=True):
                    mpm_view[0] = tb.add_image_view("mpm_image", None, img_size=(400, 700), img_format="rgb")

            tb.add_timer("update_timer", int(1000 / fps), on_timeout)

        tb.exec()


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    help_epilog = (__doc__ or "").rstrip() + "\n\n" + "\n".join(
        [
            "Environment probe:",
            f"  HAS_TAICHI={HAS_TAICHI} (import taichi)",
            f"  HAS_EZGL={HAS_EZGL} (import xensesdk.ezgl)",
            "",
            "If HAS_EZGL is False and you are using the repo-local xensesdk under ./xensesdk/,",
            "ensure PYTHONPATH includes the outer directory, e.g. PowerShell:",
            "  $env:PYTHONPATH=\"$PWD\\xensesdk;$env:PYTHONPATH\"",
            "Or install it editable:",
            "  pip install -e xensesdk",
        ]
    )
    parser = argparse.ArgumentParser(
        description='MPM vs FEM Sensor RGB Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=help_epilog
    )
    parser.add_argument(
        '--fem-file', type=str,
        default=str(PROJ_DIR / "assets/data/fem_data_gel_2035.npz"),
        help='Path to FEM NPZ file'
    )
    parser.add_argument(
        '--object-file', type=str, default=None,
        help='Path to indenter STL file (default: sphere)'
    )
    parser.add_argument(
        '--fem-indenter-face', type=str, choices=['base', 'tip'], default='tip',
        help='FEM indenter STL face selection: base (y_min, may look like square base) or tip (y_max, round tip)'
    )
    parser.add_argument(
        '--fem-indenter-geom', type=str, choices=['auto', 'stl', 'box', 'sphere'], default='auto',
        help=('FEM indenter geometry: auto (stl when --object-file is set or when '
              'MPM indenter-type=cylinder; otherwise match MPM indenter-type), '
              'stl, box, sphere')
    )
    parser.add_argument(
        '--mode', type=str, choices=['raw', 'diff'], default='raw',
        help='Visualization mode: raw (direct RGB) or diff (relative to reference)'
    )
    parser.add_argument(
        '--preset', type=str, choices=['none', 'publish', 'publish_v0', 'publish_v1'], default='none',
        help=('Preset parameter bundles. `publish` applies the current best-known MPM baseline '
              '(dx=0.2mm, gap=0mm, steps=300, record_interval=3, dt=2e-5). '
              '`publish_v0` further locks stable render defaults (marker=advect, uv_mask_footprint=on, depth_tint=off). '
              '`publish_v1` further applies v1 defaults (marker=advect, mu_s/mu_k=0.4, k_t=200) while keeping uv_mask_footprint off. '
              'CLI flags override presets.')
    )
    parser.add_argument(
        '--press-mm', type=float, default=SCENE_PARAMS['press_depth_mm'],
        help='Press depth in mm'
    )
    parser.add_argument(
        '--slide-mm', type=float, default=SCENE_PARAMS['slide_distance_mm'],
        help='Slide distance in mm'
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Total simulation steps (default: press + slide + hold)'    
    )
    parser.add_argument(
        '--record-interval', type=int, default=5,
        help='Record MPM positions every N steps (affects total frames and phase mapping)'
    )
    parser.add_argument(
        '--indenter-type',
        type=str,
        choices=['sphere', 'cylinder', 'box'],
        default=str(SCENE_PARAMS.get('indenter_type', 'cylinder')),
        help=('MPM indenter type: sphere (curved) | cylinder (flat round, matches '
              'circle_r4.STL tip) | box (flat square)')
    )
    parser.add_argument(
        '--mpm-indenter-gap-mm', type=float, default=None,
        help='MPM indenter initial clearance above gel in mm (overrides indenter_start_gap_mm)'
    )
    parser.add_argument(
        '--mpm-grid-dx-mm', type=float, default=None,
        help='MPM grid spacing dx in mm (overrides mpm_grid_dx_mm)'
    )
    parser.add_argument(
        '--mpm-sticky-boundary', type=str, choices=['on', 'off'], default='on',
        help='MPM grid boundary clamp (legacy sticky walls): on|off (default: on)'
    )
    parser.add_argument(
        '--mpm-sticky-boundary-width', type=int, default=int(SCENE_PARAMS.get("mpm_sticky_boundary_width", 3)),
        help='Width (grid cells) for --mpm-sticky-boundary clamp (default: 3)'
    )
    parser.add_argument(
        '--mpm-batch-quality', type=str, choices=['on', 'off'], default='on',
        help=('When running batch mode (--save-dir without --interactive), apply quality defaults '
              '(gap=0mm, dx=0.2mm) unless explicitly overridden. Set off to preserve legacy behavior.')
    )
    parser.add_argument(
        '--fric', type=float, default=None,
        help=('Set friction for both FEM and MPM: FEM fric_coef and MPM mu_s/mu_k '
              '(FEM uses a single coefficient; MPM uses static/kinetic).')
    )
    parser.add_argument(
        '--fem-fric', type=float, default=None,
        help='FEM friction coefficient (overrides fem_fric_coef)'
    )
    parser.add_argument(
        '--mpm-mu-s', type=float, default=None,
        help='MPM static friction coefficient mu_s (overrides --fric for MPM side)'
    )
    parser.add_argument(
        '--mpm-mu-k', type=float, default=None,
        help='MPM kinetic friction coefficient mu_k (overrides --fric for MPM side)'
    )
    parser.add_argument(
        '--mpm-k-normal', type=float, default=None,
        help='MPM contact stiffness (normal) (overrides scene default)'
    )
    parser.add_argument(
        '--mpm-k-tangent', type=float, default=None,
        help='MPM contact stiffness (tangent) (overrides scene default)'
    )
    parser.add_argument(
        '--mpm-dt', type=float, default=None,
        help='MPM time step dt in seconds (overrides scene default)'
    )
    parser.add_argument(
        '--mpm-ogden-mu', type=float, default=None,
        help='MPM Ogden shear modulus mu in Pa (overrides scene default; sets single-term Ogden)'
    )
    parser.add_argument(
        '--mpm-ogden-kappa', type=float, default=None,
        help='MPM Ogden bulk modulus kappa in Pa (overrides scene default)'
    )
    parser.add_argument(
        '--mpm-enable-bulk-viscosity', type=str, choices=['on', 'off'], default=None,
        help='Enable Kelvin-Voigt bulk viscosity for MPM: on|off'
    )
    parser.add_argument(
        '--mpm-bulk-viscosity', type=float, default=None,
        help='MPM bulk viscosity coefficient eta_bulk in Pa*s (enables bulk viscosity if provided)'
    )
    parser.add_argument(
        '--fem-marker', type=str, choices=['on', 'off'], default='on',
        help='FEM marker rendering: on|off (off = white background for shading comparison)'
    )
    parser.add_argument(
        '--mpm-marker',
        type=str,
        choices=['off', 'static', 'warp', 'advect', 'advect_points'],
        default=DEFAULT_MPM_MARKER_MODE,
        help=('MPM marker rendering: off|static|warp|advect|advect_points '
              '(advect_points is an alias of advect; '
              'warp uses texture remap (can smear under shear); '
              'advect integrates dot centers over time and renders ellipses under local shear)')
    )
    parser.add_argument(
        '--marker-appearance-mode',
        type=str,
        choices=['grid', 'random_ellipses'],
        default='grid',
        help=('Marker appearance (initial texture/dot set): grid|random_ellipses. '
              'Default: grid (backward compatible).')
    )
    parser.add_argument(
        '--marker-appearance-seed',
        type=int,
        default=None,
        help=('Marker appearance RNG seed (int). Use -1 for random (resolved seed is recorded in run_manifest). '
              'Only affects appearance initialization; motion remains driven by physical fields.')
    )
    parser.add_argument(
        '--mpm-depth-tint', type=str, choices=['on', 'off'], default='on',
        help='(Legacy) MPM depth tint overlay: on|off. Prefer --mpm-depth-tint-mode.'
    )
    parser.add_argument(
        '--mpm-depth-tint-mode', type=str, choices=['off', 'fixed', 'frame'], default=None,
        help=('MPM depth tint normalization: off|fixed|frame. '
              'Default: derived from --mpm-depth-tint (on->frame, off->off).')
    )
    parser.add_argument(
        '--mpm-depth-tint-max-mm', type=float, default=None,
        help='Max depth (mm) for --mpm-depth-tint-mode fixed (default: --press-mm)'
    )
    parser.add_argument(
        '--mpm-light-profile', type=str, choices=['default', 'publish_v1'], default='default',
        help='MPM lighting profile: default|publish_v1 (publish_v1 is stable/reproducible)'
    )
    parser.add_argument(
        '--mpm-disable-light-file', type=str, choices=['on', 'off'], default='off',
        help='Disable loading repo-local xengym/assets/data/light.txt: on|off (default: off)'
    )
    parser.add_argument(
        '--mpm-render-shadow', type=str, choices=['on', 'off'], default='on',
        help='Enable shadow rendering for MPM LineLight: on|off (default: on; publish_v0 uses off)'
    )
    parser.add_argument(
        '--mpm-render-flip-x', type=str, choices=['on', 'off'], default='off',
        help=('MPM render horizontal flip for height_field/uv/overlay: on|off '
              '(off aligns with FEM baseline; on reproduces legacy output)')
    )
    parser.add_argument(
        '--mpm-zmap-convention', type=str, choices=['sensor_depth', 'indentation'],
        default=str(SCENE_PARAMS.get("mpm_zmap_convention", "sensor_depth")),
        help=('MPM mesh Z convention: sensor_depth keeps negative indentation (legacy SensorScene depth); '
              'indentation converts to positive indentation zmap (xensim-aligned, avoids near/far clipping).')
    )
    parser.add_argument(
        '--mpm-warp-flip-x', type=str, choices=['on', 'off'], default=None,
        help='Marker warp: flip dx in texture space (default: auto, follows --mpm-render-flip-x)'
    )
    parser.add_argument(
        '--mpm-warp-flip-y', type=str, choices=['on', 'off'], default=None,
        help='Marker warp: flip dy in texture space (default: on)'
    )
    parser.add_argument(
        '--mpm-uv-smooth', type=str, choices=['on', 'off'], default='on',
        help='MPM uv_disp smoothing (box blur): on|off (default: on)'
    )
    parser.add_argument(
        '--mpm-uv-smooth-iters', type=int, default=int(SCENE_PARAMS.get("mpm_uv_smooth_iters", 1)),
        help='Iterations for --mpm-uv-smooth (default: 1)'
    )
    parser.add_argument(
        '--mpm-uv-fill-holes', type=str, choices=['on', 'off'], default='on',
        help='MPM uv_disp hole filling (diffusion) for no-coverage cells: on|off (default: on)'
    )
    parser.add_argument(
        '--mpm-uv-fill-holes-iters', type=int, default=int(SCENE_PARAMS.get("mpm_uv_fill_holes_iters", 10)),
        help='Iterations for --mpm-uv-fill-holes (default: 10)'
    )
    parser.add_argument(
        '--mpm-uv-ref-preclamp-height', type=str, choices=['on', 'off'], default='off',
        help=('Use pre-clamp height_field as reference when selecting top-surface particles for uv_disp_mm: '
              'on|off (default: off; publish_v1 uses on)')
    )
    parser.add_argument(
        '--mpm-uv-fill-footprint-holes', type=str, choices=['on', 'off'], default='off',
        help=('Fill uv_disp holes inside indenter footprint using footprint median (reduce trailing-side freeze): '
              'on|off (default: off; publish_v1 uses on)')
    )
    parser.add_argument(
        '--mpm-uv-bin-init-xy', type=str, choices=['on', 'off'], default='off',
        help=('Bin uv_disp_mm by particles initial XY (Lagrangian u(X)) for marker advection: on|off '
              '(default: off; publish_v1 uses on). '
              'Recommended when --mpm-marker=advect/advect_points to avoid the “front moves / back stays” slide artifact.')
    )
    parser.add_argument(
        '--mpm-uv-despike', type=str, choices=['on', 'off'], default='off',
        help='MPM uv_disp despike in footprint boundary band: on|off (default: off)'
    )
    parser.add_argument(
        '--mpm-uv-despike-scope', type=str, choices=['boundary', 'footprint'],
        default=str(SCENE_PARAMS.get("mpm_uv_despike_scope", "boundary")),
        help='Scope for --mpm-uv-despike gate mask: boundary|footprint (default: boundary)'
    )
    parser.add_argument(
        '--mpm-uv-despike-abs-mm', type=float, default=float(SCENE_PARAMS.get("mpm_uv_despike_abs_mm", 0.8)),
        help='UV magnitude threshold (mm) for median replace when --mpm-uv-despike on (default: 0.8)'
    )
    parser.add_argument(
        '--mpm-uv-despike-cap-mm', type=float, default=SCENE_PARAMS.get("mpm_uv_despike_cap_mm", None),
        help='Cap |uv| to <= cap_mm after replace; default: abs_mm-1e-3 when omitted'
    )
    parser.add_argument(
        '--mpm-uv-despike-boundary-iters', type=int, default=int(SCENE_PARAMS.get("mpm_uv_despike_boundary_iters", 2)),
        help='Dilations applied to footprint boundary to form the despike band (default: 2)'
    )
    parser.add_argument(
        '--mpm-uv-mask-footprint', type=str, choices=['on', 'off'], default='off',
        help='Mask uv_disp outside indenter footprint (set to 0) before marker warp: on|off (default: off)'
    )
    parser.add_argument(
        '--mpm-uv-mask-footprint-dilate-iters', type=int, default=int(SCENE_PARAMS.get("mpm_uv_mask_footprint_dilate_iters", 0)),
        help='Dilations applied to footprint before masking uv_disp (default: 8)'
    )
    parser.add_argument(
        '--mpm-uv-mask-footprint-blur-iters', type=int, default=int(SCENE_PARAMS.get("mpm_uv_mask_footprint_blur_iters", 6)),
        help='Extra box blur iterations applied after masking (reduce boundary discontinuity; default: 6)'
    )
    parser.add_argument(
        '--mpm-uv-cap-mm', type=float, default=SCENE_PARAMS.get("mpm_uv_cap_mm", None),
        help='Hard cap |uv| magnitude (mm) after all UV postprocess steps; default: disabled (None)'
    )
    parser.add_argument(
        '--mpm-uv-scale', type=float, default=float(SCENE_PARAMS.get("mpm_uv_scale", 1.0)),
        help='Scale uv_disp_mm after all UV postprocess steps (default: 1.0)'
    )
    parser.add_argument(
        '--mpm-height-fill-holes', type=str, choices=['on', 'off'], default='on',
        help='MPM height_field hole filling (diffusion) before rendering: on|off'
    )
    parser.add_argument(
        '--mpm-height-fill-holes-iters', type=int, default=int(SCENE_PARAMS.get("mpm_height_fill_holes_iters", 10)),
        help='Iterations for --mpm-height-fill-holes (default: 10)'
    )
    parser.add_argument(
        '--mpm-height-smooth', type=str, choices=['on', 'off'], default='on',
        help='MPM height_field smoothing before rendering: on|off'
    )
    parser.add_argument(
        '--mpm-height-smooth-iters', type=int, default=int(SCENE_PARAMS.get("mpm_height_smooth_iters", 2)),
        help='Box blur iterations for --mpm-height-smooth (default: 2)'
    )
    parser.add_argument(
        '--mpm-height-reference-edge', type=str, choices=['on', 'off'], default='on',
        help='MPM height_field baseline reference: edge (on) vs none (off)'
    )
    parser.add_argument(
        '--mpm-height-clamp-indenter', type=str, choices=['on', 'off'], default='on',
        help='Clamp MPM height_field not below indenter surface (suppress penetration artifacts): on|off'
    )
    parser.add_argument(
        '--mpm-height-clip-outliers', type=str, choices=['on', 'off'], default='off',
        help='Clip extreme negative MPM height_field outside indenter footprint: on|off'
    )
    parser.add_argument(
        '--mpm-height-clip-outliers-min-mm', type=float, default=float(SCENE_PARAMS.get("mpm_height_clip_outliers_min_mm", 2.0)),
        help='Negative depth threshold in mm for --mpm-height-clip-outliers (default: 2.0)'
    )
    parser.add_argument(
        '--mpm-show-indenter', action='store_true', default=False,
        help='Overlay MPM indenter projection in the RGB view (2D overlay)'
    )
    parser.add_argument(
        '--mpm-debug-overlay', type=str, choices=['off', 'uv', 'warp'], default='off',
        help='MPM debug overlay mode'
    )
    parser.add_argument(
        '--indenter-size-mm', type=float, default=None,
        help='Square indenter side length in mm (only for box mode). If omitted, try infer from STL name like square_d6.STL.'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Display frame rate'
    )
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='Directory to save frame images'
    )
    parser.add_argument(
        '--interactive', action='store_true', default=False,
        help='Run interactive UI loop (default: headless batch when --save-dir is set)'
    )
    parser.add_argument(
        '--export-intermediate', action='store_true', default=False,
        help='Export intermediate arrays (height_field_mm/uv_disp_mm/contact_mask) to --save-dir/intermediate (npz)'
    )
    parser.add_argument(
        '--export-intermediate-every', type=int, default=1,
        help='Export intermediate every N frames (default: 1); effective only with --export-intermediate'
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Enable verbose per-frame debug logging'
    )

    args = parser.parse_args()

    # Presets: apply before batch defaults, so batch defaults won't overwrite them.
    preset_name = str(getattr(args, "preset", "none")).lower().strip()
    preset_applied: List[str] = []

    def _argv_has_flag(prefixes: Sequence[str]) -> bool:
        argv = list(sys.argv[1:])
        for a in argv:
            for p in prefixes:
                if a == p or a.startswith(p + "="):
                    return True
        return False

    if preset_name in ("publish", "publish_v0", "publish_v1"):
        # Apply only when the corresponding CLI flag is not explicitly provided.
        if not _argv_has_flag(["--mpm-indenter-gap-mm"]) and args.mpm_indenter_gap_mm is None:
            args.mpm_indenter_gap_mm = 0.0
            preset_applied.append("--mpm-indenter-gap-mm=0")
        if not _argv_has_flag(["--mpm-grid-dx-mm"]) and args.mpm_grid_dx_mm is None:
            args.mpm_grid_dx_mm = 0.2
            preset_applied.append("--mpm-grid-dx-mm=0.2")
        if not _argv_has_flag(["--steps"]) and args.steps is None:
            args.steps = 300
            preset_applied.append("--steps=300")
        if not _argv_has_flag(["--record-interval"]):
            args.record_interval = 3
            preset_applied.append("--record-interval=3")
        if not _argv_has_flag(["--mpm-dt"]) and args.mpm_dt is None:
            args.mpm_dt = 2e-5
            preset_applied.append("--mpm-dt=2e-5")
        if preset_name == "publish":
            # publish: default to advect_points (material-point advection); use --mpm-marker warp explicitly for debug.
            if not _argv_has_flag(["--mpm-marker"]):
                args.mpm_marker = "advect_points"
                preset_applied.append("--mpm-marker=advect_points")
        if preset_name == "publish_v0":
            # v0: stabilize marker appearance and UV postprocess by default.
            if not _argv_has_flag(["--mpm-marker"]):
                args.mpm_marker = "advect_points"
                preset_applied.append("--mpm-marker=advect_points")
            if not _argv_has_flag(["--mpm-depth-tint-mode", "--mpm-depth-tint"]):
                args.mpm_depth_tint_mode = "off"
                args.mpm_depth_tint = "off"
                preset_applied.append("--mpm-depth-tint-mode=off")
            if not _argv_has_flag(["--mpm-uv-mask-footprint"]):
                args.mpm_uv_mask_footprint = "on"
                preset_applied.append("--mpm-uv-mask-footprint=on")
            if not _argv_has_flag(["--mpm-light-profile"]):
                args.mpm_light_profile = "publish_v1"
                preset_applied.append("--mpm-light-profile=publish_v1")
            if not _argv_has_flag(["--mpm-disable-light-file"]):
                args.mpm_disable_light_file = "on"
                preset_applied.append("--mpm-disable-light-file=on")
            if not _argv_has_flag(["--mpm-render-shadow"]):
                args.mpm_render_shadow = "off"
                preset_applied.append("--mpm-render-shadow=off")

        if preset_name == "publish_v1":
            # v1: keep marker output stable/reproducible for writing; use --mpm-marker warp explicitly for debug.
            if not _argv_has_flag(["--mpm-marker"]):
                args.mpm_marker = "advect_points"
                preset_applied.append("--mpm-marker=advect_points")
            # v1: align mesh zmap convention with xensim (positive indentation) to avoid camera clipping.
            if not _argv_has_flag(["--mpm-zmap-convention"]):
                args.mpm_zmap_convention = "indentation"
                preset_applied.append("--mpm-zmap-convention=indentation")
            # v1: lock stable lighting/shadow defaults (avoid halo/breathing and light.txt drift).
            if not _argv_has_flag(["--mpm-depth-tint-mode", "--mpm-depth-tint"]):
                args.mpm_depth_tint_mode = "off"
                args.mpm_depth_tint = "off"
                preset_applied.append("--mpm-depth-tint-mode=off")
            if not _argv_has_flag(["--mpm-light-profile"]):
                args.mpm_light_profile = "publish_v1"
                preset_applied.append("--mpm-light-profile=publish_v1")
            if not _argv_has_flag(["--mpm-disable-light-file"]):
                args.mpm_disable_light_file = "on"
                preset_applied.append("--mpm-disable-light-file=on")
            if not _argv_has_flag(["--mpm-render-shadow"]):
                args.mpm_render_shadow = "off"
                preset_applied.append("--mpm-render-shadow=off")
            # v1: tune contact parameters to better match FEM displacement scale (see docs/contact_sweep_summary.md).
            # Keep uv_mask_footprint OFF by default so out_u_p50_px remains meaningful for v1.
            if not _argv_has_flag(["--fric", "--mpm-mu-s"]) and args.mpm_mu_s is None:
                args.mpm_mu_s = 0.4
                preset_applied.append("--mpm-mu-s=0.4")
            if not _argv_has_flag(["--fric", "--mpm-mu-k"]) and args.mpm_mu_k is None:
                args.mpm_mu_k = 0.4
                preset_applied.append("--mpm-mu-k=0.4")
            if not _argv_has_flag(["--mpm-k-tangent"]) and args.mpm_k_tangent is None:
                args.mpm_k_tangent = 200.0
                preset_applied.append("--mpm-k-tangent=200")
            # Reduce rare UV spikes / local tearing risk without zeroing outside-footprint UV
            # (zeroing would collapse out_u_p50_px in marker_uv_compare).
            if not _argv_has_flag(["--mpm-uv-despike"]):
                args.mpm_uv_despike = "on"
                preset_applied.append("--mpm-uv-despike=on")
            # NOTE: The default abs_mm=0.8 is too aggressive for v1 (it can clamp normal slide motion and
            # increase gradients). Use a higher threshold so despike only catches true outliers.
            if not _argv_has_flag(["--mpm-uv-despike-abs-mm"]):
                args.mpm_uv_despike_abs_mm = 3.0
                preset_applied.append("--mpm-uv-despike-abs-mm=3")
            if not _argv_has_flag(["--mpm-uv-smooth-iters"]):
                args.mpm_uv_smooth_iters = 2
                preset_applied.append("--mpm-uv-smooth-iters=2")
            # v1: use pre-clamp height reference to improve UV coverage under flat indenters (more FEM-like continuity).
            if not _argv_has_flag(["--mpm-uv-ref-preclamp-height"]):
                args.mpm_uv_ref_preclamp_height = "on"
                preset_applied.append("--mpm-uv-ref-preclamp-height=on")
            # Stabilize under-indenter tangential motion when height_field is clamped but UV has local holes.
            if not _argv_has_flag(["--mpm-uv-fill-footprint-holes"]):
                args.mpm_uv_fill_footprint_holes = "on"
                preset_applied.append("--mpm-uv-fill-footprint-holes=on")
            # v1: marker 默认使用 advect（点中心平移）。为避免“前动后不动”，UV 需按初始 XY 组装为 u(X)。
            if not _argv_has_flag(["--mpm-uv-bin-init-xy"]):
                args.mpm_uv_bin_init_xy = "on"
                preset_applied.append("--mpm-uv-bin-init-xy=on")
            # 对齐 MPM ↔ FEM 的切向位移量级：降低 uv_disp_mm 幅值以满足 fem_ratio gate。
            # 注意：out_u_p50_px 以像素空间位移衡量（见 publish pipeline），应同时确保“整体拖拽”仍可见。
            if not _argv_has_flag(["--mpm-uv-scale"]):
                args.mpm_uv_scale = 0.62
                preset_applied.append("--mpm-uv-scale=0.62")

        if preset_applied:
            print(f"[Preset {preset_name}] applied: {', '.join(preset_applied)} (override explicitly via CLI)")

    # Batch-mode defaults: prioritize auditability/realism over speed unless explicitly overridden.
    if (
        str(args.mpm_batch_quality).lower().strip() != "off"
        and args.save_dir
        and not bool(args.interactive)
    ):
        applied = []
        if args.mpm_indenter_gap_mm is None:
            args.mpm_indenter_gap_mm = 0.0
            applied.append("--mpm-indenter-gap-mm=0")
        if args.mpm_grid_dx_mm is None:
            args.mpm_grid_dx_mm = 0.2
            applied.append("--mpm-grid-dx-mm=0.2")
        if applied:
            print(f"[Batch defaults] applied: {', '.join(applied)} (override explicitly or use --mpm-batch-quality off)")

    def _validate_nonneg(name: str, value: Optional[float]) -> bool:
        if value is None:
            return True
        if float(value) < 0.0:
            print(f"ERROR: {name} must be >= 0")
            return False
        return True
    def _validate_pos(name: str, value: Optional[float]) -> bool:
        if value is None:
            return True
        if float(value) <= 0.0:
            print(f"ERROR: {name} must be > 0")
            return False
        return True

    if not (
        _validate_nonneg("--fric", args.fric)
        and _validate_nonneg("--fem-fric", args.fem_fric)
        and _validate_nonneg("--mpm-indenter-gap-mm", args.mpm_indenter_gap_mm)
        and _validate_pos("--mpm-grid-dx-mm", args.mpm_grid_dx_mm)
        and _validate_nonneg("--mpm-sticky-boundary-width", args.mpm_sticky_boundary_width)
        and _validate_nonneg("--mpm-mu-s", args.mpm_mu_s)
        and _validate_nonneg("--mpm-mu-k", args.mpm_mu_k)
        and _validate_nonneg("--mpm-k-normal", args.mpm_k_normal)
        and _validate_nonneg("--mpm-k-tangent", args.mpm_k_tangent)
        and _validate_nonneg("--mpm-uv-despike-abs-mm", args.mpm_uv_despike_abs_mm)
        and _validate_nonneg("--mpm-uv-despike-cap-mm", args.mpm_uv_despike_cap_mm)
        and _validate_nonneg("--mpm-uv-cap-mm", args.mpm_uv_cap_mm)
        and _validate_nonneg("--mpm-uv-scale", args.mpm_uv_scale)
        and _validate_pos("--mpm-dt", args.mpm_dt)
        and _validate_pos("--mpm-ogden-mu", args.mpm_ogden_mu)
        and _validate_pos("--mpm-ogden-kappa", args.mpm_ogden_kappa)
        and _validate_nonneg("--mpm-bulk-viscosity", args.mpm_bulk_viscosity)
    ):
        return 1

    if int(args.export_intermediate_every) <= 0:
        print("ERROR: --export-intermediate-every must be a positive integer")
        return 1
    if int(args.mpm_height_fill_holes_iters) < 0:
        print("ERROR: --mpm-height-fill-holes-iters must be >= 0")
        return 1
    if int(args.mpm_height_smooth_iters) < 0:
        print("ERROR: --mpm-height-smooth-iters must be >= 0")
        return 1
    if int(args.mpm_uv_despike_boundary_iters) < 0:
        print("ERROR: --mpm-uv-despike-boundary-iters must be >= 0")
        return 1
    if int(args.mpm_uv_smooth_iters) < 0:
        print("ERROR: --mpm-uv-smooth-iters must be >= 0")
        return 1
    if int(args.mpm_uv_fill_holes_iters) < 0:
        print("ERROR: --mpm-uv-fill-holes-iters must be >= 0")
        return 1
    if int(args.mpm_uv_mask_footprint_dilate_iters) < 0:
        print("ERROR: --mpm-uv-mask-footprint-dilate-iters must be >= 0")
        return 1
    if int(args.mpm_uv_mask_footprint_blur_iters) < 0:
        print("ERROR: --mpm-uv-mask-footprint-blur-iters must be >= 0")
        return 1
    if str(args.mpm_height_clip_outliers).lower().strip() == "on" and float(args.mpm_height_clip_outliers_min_mm) <= 0.0:
        print("ERROR: --mpm-height-clip-outliers-min-mm must be > 0 when --mpm-height-clip-outliers on")
        return 1

    # Update scene params from args
    SCENE_PARAMS['press_depth_mm'] = args.press_mm
    SCENE_PARAMS['slide_distance_mm'] = args.slide_mm
    SCENE_PARAMS['indenter_type'] = args.indenter_type
    SCENE_PARAMS['debug_verbose'] = args.debug
    SCENE_PARAMS["mpm_light_profile"] = str(args.mpm_light_profile).lower().strip()
    SCENE_PARAMS["mpm_disable_light_file"] = (str(args.mpm_disable_light_file).lower().strip() == "on")
    SCENE_PARAMS["mpm_render_shadow"] = (str(args.mpm_render_shadow).lower().strip() == "on")
    if args.mpm_indenter_gap_mm is not None:
        gap_mm = float(args.mpm_indenter_gap_mm)
        SCENE_PARAMS['indenter_start_gap_mm'] = gap_mm
        SCENE_PARAMS['mpm_indenter_gap_mm'] = gap_mm
    else:
        # Keep alias in sync for audit (manifest/tuning_notes).
        SCENE_PARAMS['mpm_indenter_gap_mm'] = float(SCENE_PARAMS.get('indenter_start_gap_mm', 0.0))
    if args.mpm_grid_dx_mm is not None:
        SCENE_PARAMS['mpm_grid_dx_mm'] = float(args.mpm_grid_dx_mm)
    SCENE_PARAMS["mpm_sticky_boundary"] = (str(args.mpm_sticky_boundary).lower().strip() == "on")
    SCENE_PARAMS["mpm_sticky_boundary_width"] = int(args.mpm_sticky_boundary_width)
    if args.mpm_dt is not None:
        SCENE_PARAMS['mpm_dt'] = float(args.mpm_dt)
    if args.mpm_ogden_mu is not None:
        SCENE_PARAMS['ogden_mu'] = [float(args.mpm_ogden_mu)]
        # 保持与单项 mu 对齐，避免长度不一致导致本构计算混淆
        if isinstance(SCENE_PARAMS.get('ogden_alpha'), list) and len(SCENE_PARAMS['ogden_alpha']) != 1:
            SCENE_PARAMS['ogden_alpha'] = [float(SCENE_PARAMS['ogden_alpha'][0])]
    if args.mpm_ogden_kappa is not None:
        SCENE_PARAMS['ogden_kappa'] = float(args.mpm_ogden_kappa)
    if args.mpm_enable_bulk_viscosity is not None:
        SCENE_PARAMS["mpm_enable_bulk_viscosity"] = (str(args.mpm_enable_bulk_viscosity).lower().strip() == "on")
    if args.mpm_bulk_viscosity is not None:
        SCENE_PARAMS["mpm_bulk_viscosity"] = float(args.mpm_bulk_viscosity)
        # 用户显式提供 eta_bulk 时，默认启用体粘性（除非显式 --mpm-enable-bulk-viscosity off）
        if str(args.mpm_enable_bulk_viscosity).lower().strip() != "off":
            SCENE_PARAMS["mpm_enable_bulk_viscosity"] = True
    SCENE_PARAMS["mpm_height_fill_holes"] = (str(args.mpm_height_fill_holes).lower().strip() == "on")
    SCENE_PARAMS["mpm_height_fill_holes_iters"] = int(args.mpm_height_fill_holes_iters)
    SCENE_PARAMS["mpm_height_smooth"] = (str(args.mpm_height_smooth).lower().strip() != "off")
    SCENE_PARAMS["mpm_height_smooth_iters"] = int(args.mpm_height_smooth_iters)
    SCENE_PARAMS["mpm_height_reference_edge"] = (str(args.mpm_height_reference_edge).lower().strip() != "off")
    SCENE_PARAMS["mpm_height_clamp_indenter"] = (str(args.mpm_height_clamp_indenter).lower().strip() != "off")
    SCENE_PARAMS["mpm_height_clip_outliers"] = (str(args.mpm_height_clip_outliers).lower().strip() == "on")
    SCENE_PARAMS["mpm_height_clip_outliers_min_mm"] = float(args.mpm_height_clip_outliers_min_mm)
    SCENE_PARAMS["mpm_render_flip_x"] = (str(args.mpm_render_flip_x).lower().strip() == "on")
    SCENE_PARAMS["mpm_zmap_convention"] = str(args.mpm_zmap_convention).lower().strip()
    if args.mpm_warp_flip_x is None:
        SCENE_PARAMS["mpm_warp_flip_x"] = bool(SCENE_PARAMS.get("mpm_render_flip_x", False))
    else:
        SCENE_PARAMS["mpm_warp_flip_x"] = (str(args.mpm_warp_flip_x).lower().strip() == "on")
    if args.mpm_warp_flip_y is None:
        SCENE_PARAMS["mpm_warp_flip_y"] = True
    else:
        SCENE_PARAMS["mpm_warp_flip_y"] = (str(args.mpm_warp_flip_y).lower().strip() == "on")

    SCENE_PARAMS["mpm_uv_smooth"] = (str(args.mpm_uv_smooth).lower().strip() == "on")
    SCENE_PARAMS["mpm_uv_smooth_iters"] = int(args.mpm_uv_smooth_iters)
    SCENE_PARAMS["mpm_uv_fill_holes"] = (str(args.mpm_uv_fill_holes).lower().strip() == "on")
    SCENE_PARAMS["mpm_uv_fill_holes_iters"] = int(args.mpm_uv_fill_holes_iters)
    SCENE_PARAMS["mpm_uv_ref_preclamp_height"] = (str(args.mpm_uv_ref_preclamp_height).lower().strip() == "on")
    SCENE_PARAMS["mpm_uv_fill_footprint_holes"] = (str(args.mpm_uv_fill_footprint_holes).lower().strip() == "on")
    SCENE_PARAMS["mpm_uv_bin_init_xy"] = (str(args.mpm_uv_bin_init_xy).lower().strip() == "on")
    SCENE_PARAMS["mpm_uv_despike"] = (str(args.mpm_uv_despike).lower().strip() == "on")
    SCENE_PARAMS["mpm_uv_despike_scope"] = str(args.mpm_uv_despike_scope).lower().strip()
    SCENE_PARAMS["mpm_uv_despike_abs_mm"] = float(args.mpm_uv_despike_abs_mm)
    SCENE_PARAMS["mpm_uv_despike_cap_mm"] = None if args.mpm_uv_despike_cap_mm is None else float(args.mpm_uv_despike_cap_mm)
    SCENE_PARAMS["mpm_uv_despike_boundary_iters"] = int(args.mpm_uv_despike_boundary_iters)
    SCENE_PARAMS["mpm_uv_mask_footprint"] = (str(args.mpm_uv_mask_footprint).lower().strip() == "on")
    SCENE_PARAMS["mpm_uv_mask_footprint_dilate_iters"] = int(args.mpm_uv_mask_footprint_dilate_iters)
    SCENE_PARAMS["mpm_uv_mask_footprint_blur_iters"] = int(args.mpm_uv_mask_footprint_blur_iters)
    SCENE_PARAMS["mpm_uv_cap_mm"] = None if args.mpm_uv_cap_mm is None else float(args.mpm_uv_cap_mm)
    SCENE_PARAMS["mpm_uv_scale"] = float(args.mpm_uv_scale)

    # Resolve depth_tint config for auditability/repro (avoid frame-wise normalization "breathing").
    mpm_depth_tint_mode = str(args.mpm_depth_tint_mode).lower().strip() if args.mpm_depth_tint_mode is not None else ""
    if not mpm_depth_tint_mode:
        # Backward compatible default: --mpm-depth-tint on->frame, off->off
        mpm_depth_tint_mode = "off" if str(args.mpm_depth_tint).lower().strip() == "off" else "frame"
    if mpm_depth_tint_mode not in ("off", "fixed", "frame"):
        print(f"ERROR: invalid --mpm-depth-tint-mode: {mpm_depth_tint_mode}")
        return 1
    if mpm_depth_tint_mode == "fixed":
        max_mm = float(args.mpm_depth_tint_max_mm) if args.mpm_depth_tint_max_mm is not None else float(args.press_mm)
        if max_mm < 0.0:
            print("ERROR: --mpm-depth-tint-max-mm must be >= 0")
            return 1
        mpm_depth_tint_max_mm = float(max_mm)
    else:
        mpm_depth_tint_max_mm = None

    if args.fric is not None:
        fric = float(args.fric)
        SCENE_PARAMS["fem_fric_coef"] = fric
        SCENE_PARAMS["mpm_mu_s"] = fric
        SCENE_PARAMS["mpm_mu_k"] = fric
    if args.fem_fric is not None:
        SCENE_PARAMS["fem_fric_coef"] = float(args.fem_fric)
    if args.mpm_mu_s is not None:
        SCENE_PARAMS["mpm_mu_s"] = float(args.mpm_mu_s)
    if args.mpm_mu_k is not None:
        SCENE_PARAMS["mpm_mu_k"] = float(args.mpm_mu_k)
    if args.mpm_k_normal is not None:
        SCENE_PARAMS["mpm_contact_stiffness_normal"] = float(args.mpm_k_normal)
    if args.mpm_k_tangent is not None:
        SCENE_PARAMS["mpm_contact_stiffness_tangent"] = float(args.mpm_k_tangent)

    if args.indenter_size_mm is not None:
        square_d_mm = float(args.indenter_size_mm)
    else:
        square_d_mm = _infer_square_size_mm_from_stl_path(args.object_file)
    if square_d_mm is not None and args.indenter_type == "box":
        half = square_d_mm / 2.0
        SCENE_PARAMS['indenter_half_extents_mm'] = (half, half, half)

    # Handle --steps: distribute among press/slide/hold phases
    if args.steps is not None:
        # Distribute: 30% press, 55% slide, 15% hold
        SCENE_PARAMS['press_steps'] = int(args.steps * 0.30)
        SCENE_PARAMS['slide_steps'] = int(args.steps * 0.55)
        SCENE_PARAMS['hold_steps'] = args.steps - SCENE_PARAMS['press_steps'] - SCENE_PARAMS['slide_steps']

    if args.record_interval <= 0:
        print("ERROR: --record-interval must be a positive integer")
        return 1

    print("=" * 60)
    print("MPM vs FEM Sensor RGB Comparison")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Press depth: {args.press_mm} mm")
    print(f"Slide distance: {args.slide_mm} mm")
    print(f"MPM indenter type: {args.indenter_type}")
    print(f"MPM render flip_x: {'on' if bool(SCENE_PARAMS.get('mpm_render_flip_x', False)) else 'off'}")
    print(
        "MPM marker warp flip: "
        f"x={'on' if bool(SCENE_PARAMS.get('mpm_warp_flip_x', False)) else 'off'}, "
        f"y={'on' if bool(SCENE_PARAMS.get('mpm_warp_flip_y', True)) else 'off'}"
    )
    fem_fric = float(SCENE_PARAMS.get("fem_fric_coef", 0.4))
    mpm_mu_s = float(SCENE_PARAMS.get("mpm_mu_s", 2.0))
    mpm_mu_k = float(SCENE_PARAMS.get("mpm_mu_k", 1.5))
    aligned_fric = (abs(fem_fric - mpm_mu_s) < 1e-9) and (abs(fem_fric - mpm_mu_k) < 1e-9)
    print(f"Friction: FEM fric_coef={fem_fric:.4g}, MPM mu_s={mpm_mu_s:.4g}, mu_k={mpm_mu_k:.4g}, aligned={aligned_fric}")
    k_n = float(SCENE_PARAMS.get("mpm_contact_stiffness_normal", 0.0))
    k_t = float(SCENE_PARAMS.get("mpm_contact_stiffness_tangent", 0.0))
    print(f"MPM contact stiffness: k_n={k_n:.4g}, k_t={k_t:.4g}")
    print(
        f"MPM material: ogden_mu={SCENE_PARAMS.get('ogden_mu')}, "
        f"ogden_alpha={SCENE_PARAMS.get('ogden_alpha')}, "
        f"ogden_kappa={SCENE_PARAMS.get('ogden_kappa')}, "
        f"bulk_viscosity={'on' if bool(SCENE_PARAMS.get('mpm_enable_bulk_viscosity', False)) else 'off'} "
        f"(eta={float(SCENE_PARAMS.get('mpm_bulk_viscosity', 0.0)):.4g})"
    )

    gel_w_mm, gel_h_mm = [float(v) for v in SCENE_PARAMS.get("gel_size_mm", (0.0, 0.0))]
    cam_w_mm = float(SCENE_PARAMS.get("cam_view_width_m", 0.0)) * 1000.0
    cam_h_mm = float(SCENE_PARAMS.get("cam_view_height_m", 0.0)) * 1000.0
    tol_mm = 0.2
    dw_mm = cam_w_mm - gel_w_mm
    dh_mm = cam_h_mm - gel_h_mm
    scale_consistent = (abs(dw_mm) <= tol_mm) and (abs(dh_mm) <= tol_mm)
    print(
        f"Scale: gel_size_mm=({gel_w_mm:.2f}, {gel_h_mm:.2f}), "
        f"cam_view_mm=({cam_w_mm:.2f}, {cam_h_mm:.2f}), "
        f"delta_mm=({dw_mm:+.2f}, {dh_mm:+.2f}), consistent={scale_consistent}"
    )
    if not scale_consistent:
        print("Note: gel_size_mm follows VecTouchSim defaults; cam_view_* follows demo_simple_sensor camera calibration.")

    fem_indenter_geom = args.fem_indenter_geom
    if fem_indenter_geom == "auto":
        if args.object_file:
            fem_indenter_geom = "stl"
        elif str(args.indenter_type).lower().strip() == "cylinder":
            # FEM 侧没有 cylinder primitive，使用 circle_r4.STL (tip) 作为圆柱压头基线。
            fem_indenter_geom = "stl"
        else:
            fem_indenter_geom = args.indenter_type
    print(f"FEM indenter geom: {fem_indenter_geom}")
    if args.object_file and fem_indenter_geom != "stl":
        print(f"Note: --object-file is ignored because --fem-indenter-geom={fem_indenter_geom}")
    if fem_indenter_geom == "stl":
        default_stl = _PROJECT_ROOT / "xengym" / "assets" / "obj" / "circle_r4.STL"
        stl_path_display = args.object_file if args.object_file else str(default_stl)
        print(f"FEM indenter STL: {stl_path_display}")
        print(f"FEM indenter face: {args.fem_indenter_face}")

    # Print effective indenter size (MPM vs FEM) for auditability.
    mpm_indenter_size: Optional[Dict[str, object]] = None
    if args.indenter_type == "box":
        half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
        if half_extents_mm is None:
            r_mm = float(SCENE_PARAMS["indenter_radius_mm"])
            half_extents_mm = (r_mm, r_mm, r_mm)
        hx_mm, hy_mm, hz_mm = [float(v) for v in half_extents_mm]
        mpm_indenter_size = {
            "half_extents_mm": [float(hx_mm), float(hy_mm), float(hz_mm)],
            "full_extents_mm": [float(hx_mm * 2.0), float(hy_mm * 2.0), float(hz_mm * 2.0)],
        }
        print(f"Indenter size (box, mm): half_extents=({hx_mm:.2f}, {hy_mm:.2f}, {hz_mm:.2f}), "
              f"full=({hx_mm*2:.2f}, {hy_mm*2:.2f}, {hz_mm*2:.2f})")
    elif args.indenter_type == "cylinder":
        r_mm = float(SCENE_PARAMS["indenter_radius_mm"])
        half_h_mm = SCENE_PARAMS.get("indenter_cylinder_half_height_mm", None)
        if half_h_mm is None:
            half_h_mm = r_mm
        half_h_mm = float(half_h_mm)
        mpm_indenter_size = {
            "radius_mm": float(r_mm),
            "diameter_mm": float(r_mm * 2.0),
            "half_height_mm": float(half_h_mm),
            "height_mm": float(half_h_mm * 2.0),
        }
        print(
            f"Indenter size (cylinder, mm): radius={r_mm:.2f}, diameter={r_mm*2:.2f}, "
            f"height={half_h_mm*2:.2f}"
        )
    else:
        r_mm = float(SCENE_PARAMS["indenter_radius_mm"])
        mpm_indenter_size = {
            "radius_mm": float(r_mm),
            "diameter_mm": float(r_mm * 2.0),
        }
        print(f"Indenter size (sphere, mm): radius={r_mm:.2f}, diameter={r_mm*2:.2f}")

    stl_stats = None
    fem_contact_face_key = None
    fem_contact_face_size_mm = None
    if fem_indenter_geom == "stl":
        stl_path = Path(args.object_file) if args.object_file else (_PROJECT_ROOT / "xengym" / "assets" / "obj" / "circle_r4.STL")
        stl_stats = _analyze_binary_stl_endfaces_mm(stl_path) if stl_path.exists() else None
        if stl_stats is not None:
            try:
                ymin = stl_stats["endfaces_mm"]["y_min"]
                ymax = stl_stats["endfaces_mm"]["y_max"]
                print(
                    "Indenter STL endfaces (mm): "
                    f"y_min size≈{ymin['size_x_mm']:.1f}x{ymin['size_z_mm']:.1f}, "
                    f"y_max size≈{ymax['size_x_mm']:.1f}x{ymax['size_z_mm']:.1f}, "
                    f"height≈{float(stl_stats['height_mm']):.1f}"
                )
            except Exception:
                pass
            try:
                fem_contact_face_key = "y_max" if str(args.fem_indenter_face).lower().strip() == "tip" else "y_min"
                endfaces = stl_stats.get("endfaces_mm") if isinstance(stl_stats, dict) else None
                endface = endfaces.get(fem_contact_face_key) if isinstance(endfaces, dict) else None
                if isinstance(endface, dict):
                    fem_contact_face_size_mm = {
                        "size_x_mm": float(endface.get("size_x_mm", 0.0)),
                        "size_z_mm": float(endface.get("size_z_mm", 0.0)),
                    }
                    print(
                        f"FEM contact face ({args.fem_indenter_face}/{fem_contact_face_key}) "
                        f"size≈{fem_contact_face_size_mm['size_x_mm']:.1f}x{fem_contact_face_size_mm['size_z_mm']:.1f}mm"
                    )
            except Exception:
                pass
    print(f"MPM marker: {args.mpm_marker}")
    print(f"FEM marker: {args.fem_marker}")
    if mpm_depth_tint_mode == "fixed":
        print(f"MPM depth tint: mode=fixed (max_mm={float(mpm_depth_tint_max_mm):.3g})")
    else:
        print(f"MPM depth tint: mode={mpm_depth_tint_mode}")
    print(
        "MPM height_field: "
        f"fill_holes={bool(SCENE_PARAMS.get('mpm_height_fill_holes', False))} "
        f"(iters={int(SCENE_PARAMS.get('mpm_height_fill_holes_iters', 0))}), "
        f"smooth={bool(SCENE_PARAMS.get('mpm_height_smooth', True))} "
        f"(iters={int(SCENE_PARAMS.get('mpm_height_smooth_iters', 0))}), "
        f"ref_edge={bool(SCENE_PARAMS.get('mpm_height_reference_edge', True))}, "
        f"clamp_indenter={bool(SCENE_PARAMS.get('mpm_height_clamp_indenter', True))}, "
        f"clip_outliers={bool(SCENE_PARAMS.get('mpm_height_clip_outliers', False))} "
        f"(min_mm={float(SCENE_PARAMS.get('mpm_height_clip_outliers_min_mm', 0.0)):.3f})"
    )
    if args.mpm_show_indenter:
        print("MPM indenter overlay: enabled")
    if args.steps:
        print(f"Total steps: {args.steps}")
    press_steps = int(SCENE_PARAMS["press_steps"])
    slide_steps = int(SCENE_PARAMS["slide_steps"])
    hold_steps = int(SCENE_PARAMS["hold_steps"])
    total_steps = press_steps + slide_steps + hold_steps
    expected_frames = (total_steps + int(args.record_interval) - 1) // int(args.record_interval)
    print(f"Record interval: {args.record_interval} (expected frames: {expected_frames})")
    print(f"Phase steps: press={press_steps}, slide={slide_steps}, hold={hold_steps} (total={total_steps})")
    print(f"FPS: {args.fps}")
    if args.save_dir:
        print(f"Saving frames to: {args.save_dir}")
        print("Run manifest: run_manifest.json (params + frame→phase mapping)")
        print("Metrics: metrics.csv / metrics.json")
        if args.export_intermediate:
            print(f"Intermediate: enabled (every={int(args.export_intermediate_every)}) -> intermediate/frame_XXXX.npz")
    print()

    marker_appearance_cfg = resolve_marker_appearance_config(
        mode=getattr(args, "marker_appearance_mode", None),
        seed=getattr(args, "marker_appearance_seed", None),
    )
    # Make appearance config visible to renderers that only see SCENE_PARAMS.
    # NOTE: This affects only marker initialization; motion is still driven by physical fields.
    SCENE_PARAMS["marker_appearance_mode"] = str(marker_appearance_cfg.mode)
    SCENE_PARAMS["marker_appearance_seed"] = marker_appearance_cfg.seed
    marker_appearance_manifest = {
        "mode": marker_appearance_cfg.mode,
        "seed": marker_appearance_cfg.seed,
        "config": {k: v for k, v in marker_appearance_cfg.to_manifest().items() if k not in {"mode", "seed"}},
    }

    run_context = {
        "args": vars(args),
        "resolved": {
            "preset": {"name": str(preset_name), "applied": list(preset_applied)},
            "square_indenter_size_mm": float(square_d_mm) if square_d_mm is not None else None,
            "indenter_stl": stl_stats,
            "fem_indenter_geom": fem_indenter_geom,
            "indenter": {
                "mpm": {
                    "type": str(args.indenter_type),
                    "size_mm": mpm_indenter_size,
                    "gap_mm": float(SCENE_PARAMS.get("mpm_indenter_gap_mm", 0.0)),
                },
                "fem": {
                    "geom": str(fem_indenter_geom),
                    "face": str(args.fem_indenter_face),
                    "contact_face_key": fem_contact_face_key,
                    "contact_face_size_mm": fem_contact_face_size_mm,
                },
            },
            "conventions": {
                "mpm_height_field_flip_x": bool(SCENE_PARAMS.get("mpm_render_flip_x", False)),
                "mpm_uv_disp_flip_x": bool(SCENE_PARAMS.get("mpm_render_flip_x", False)),
                "mpm_uv_disp_u_negate": False,
                "mpm_warp_flip_x": bool(SCENE_PARAMS.get("mpm_warp_flip_x", False)),
                "mpm_warp_flip_y": bool(SCENE_PARAMS.get("mpm_warp_flip_y", True)),
                "mpm_overlay_flip_x_mm": bool(SCENE_PARAMS.get("mpm_render_flip_x", False)),
                "mpm_zmap_convention": str(SCENE_PARAMS.get("mpm_zmap_convention", "sensor_depth")),
                # Intermediate exports are written in solver/grid orientation (before render-layer flips).
                "mpm_intermediate_height_field_flip_x": False,
                "mpm_intermediate_uv_disp_flip_x": False,
                # Explicit axis conventions for audit/debug.
                "uv_disp_mm_axes": {"u": "+x (right)", "v": "+y (up)"},
                "image_axes": {"x": "+col (right)", "y": "+row (down)"},
                "mpm_depth_sign": "indentation_negative_mm (height_field_mm<=0)",
                "mpm_height_fill_holes": bool(SCENE_PARAMS.get("mpm_height_fill_holes", False)),
                "mpm_height_fill_holes_iters": int(SCENE_PARAMS.get("mpm_height_fill_holes_iters", 0)),
                "mpm_height_smooth": bool(SCENE_PARAMS.get("mpm_height_smooth", True)),
                "mpm_height_smooth_iters": int(SCENE_PARAMS.get("mpm_height_smooth_iters", 0)),
                "mpm_height_reference_edge": bool(SCENE_PARAMS.get("mpm_height_reference_edge", True)),
                "mpm_height_clamp_indenter": bool(SCENE_PARAMS.get("mpm_height_clamp_indenter", True)),
                "mpm_height_clip_outliers": bool(SCENE_PARAMS.get("mpm_height_clip_outliers", False)),
                "mpm_height_clip_outliers_min_mm": float(SCENE_PARAMS.get("mpm_height_clip_outliers_min_mm", 0.0)),
            },
            "friction": {
                "fem_fric_coef": float(fem_fric),
                "mpm_mu_s": float(mpm_mu_s),
                "mpm_mu_k": float(mpm_mu_k),
                "aligned": bool(aligned_fric),
            },
            "contact": {
                "mpm_contact_stiffness_normal": float(k_n),
                "mpm_contact_stiffness_tangent": float(k_t),
            },
            "render": {
                "mpm_marker": str(args.mpm_marker),
                "fem_marker": str(args.fem_marker),
                "mpm_depth_tint": bool(mpm_depth_tint_mode != "off"),
                "mpm_depth_tint_mode": str(mpm_depth_tint_mode),
                "mpm_depth_tint_max_mm": float(mpm_depth_tint_max_mm) if mpm_depth_tint_mode == "fixed" else None,
            },
            "lighting": {
                # For auditability: record whether repo-local light.txt is present (it affects shadows/highlights).
                "light_file": str((ASSET_DIR / "data/light.txt").as_posix()),
                "light_file_exists": bool((ASSET_DIR / "data/light.txt").exists()),
                "mpm_light_profile": str(SCENE_PARAMS.get("mpm_light_profile", "default")),
                "mpm_disable_light_file": bool(SCENE_PARAMS.get("mpm_disable_light_file", False)),
                "mpm_render_shadow": bool(SCENE_PARAMS.get("mpm_render_shadow", True)),
                "light_file_loaded": bool(
                    bool((ASSET_DIR / "data/light.txt").exists())
                    and (not bool(SCENE_PARAMS.get("mpm_disable_light_file", False)))
                ),
            },
            "marker_grid": {
                "rows": int(SCENE_PARAMS.get("marker_grid_rows", 0)),
                "cols": int(SCENE_PARAMS.get("marker_grid_cols", 0)),
                "dx_mm": float(SCENE_PARAMS.get("marker_dx_mm", 0.0)),
                "dy_mm": float(SCENE_PARAMS.get("marker_dy_mm", 0.0)),
                "radius_px": int(SCENE_PARAMS.get("marker_radius_px", 0)),
                "tex_size_wh": [
                    int((SCENE_PARAMS.get("marker_tex_size_wh", (320, 560)) or (320, 560))[0]),
                    int((SCENE_PARAMS.get("marker_tex_size_wh", (320, 560)) or (320, 560))[1]),
                ],
            },
            "marker_appearance": marker_appearance_manifest,
            "scale": {
                "gel_size_mm": [float(gel_w_mm), float(gel_h_mm)],
                "cam_view_mm": [float(cam_w_mm), float(cam_h_mm)],
                "delta_mm": [float(dw_mm), float(dh_mm)],
                "consistent": bool(scale_consistent),
                "tolerance_mm": float(tol_mm),
                "mpm_grid_dx_mm": float(SCENE_PARAMS.get("mpm_grid_dx_mm", 0.0)),
                # Explicit mm→px mapping for auditability (marker texture + RGB image).
                "mpm_marker_tex_px_per_mm": [
                    float(int((SCENE_PARAMS.get("marker_tex_size_wh", (320, 560)) or (320, 560))[0])) / max(float(gel_w_mm), 1e-6),
                    float(int((SCENE_PARAMS.get("marker_tex_size_wh", (320, 560)) or (320, 560))[1])) / max(float(gel_h_mm), 1e-6),
                ],
                "mpm_rgb_camera_size_wh": [400, 700],
                "mpm_rgb_px_per_mm": [400.0 / max(float(gel_w_mm), 1e-6), 700.0 / max(float(gel_h_mm), 1e-6)],
                "mpm_marker_mm_to_px_source": "gel_size_mm",
                "mpm_rgb_mm_to_px_source": "gel_size_mm",
            },
            "export": {
                "intermediate": bool(args.export_intermediate),
                "intermediate_every": int(args.export_intermediate_every),
            },
            "time": {
                "mpm_dt": float(SCENE_PARAMS.get("mpm_dt", 0.0)),
                "steps_cli": int(args.steps) if args.steps is not None else None,
                "press_steps": int(press_steps),
                "slide_steps": int(slide_steps),
                "hold_steps": int(hold_steps),
                "total_steps": int(total_steps),
                "record_interval": int(args.record_interval),
                "expected_frames": int(expected_frames),
            },
            "uv": {
                # Keep these explicit for audit/repro; some are currently hard-coded in extraction.
                "fill_holes": bool(SCENE_PARAMS.get("mpm_uv_fill_holes", True)),
                "fill_holes_iters": int(SCENE_PARAMS.get("mpm_uv_fill_holes_iters", 0)),
                "mpm_uv_ref_preclamp_height": bool(SCENE_PARAMS.get("mpm_uv_ref_preclamp_height", False)),
                "mpm_uv_fill_footprint_holes": bool(SCENE_PARAMS.get("mpm_uv_fill_footprint_holes", False)),
                "mpm_uv_bin_init_xy": bool(SCENE_PARAMS.get("mpm_uv_bin_init_xy", False)),
                "smooth": bool(SCENE_PARAMS.get("mpm_uv_smooth", True)),
                "smooth_iters": (
                    0
                    if not bool(SCENE_PARAMS.get("mpm_uv_smooth", True))
                    else (
                        max(
                            int(SCENE_PARAMS.get("mpm_uv_smooth_iters", 1)),
                            int(SCENE_PARAMS.get("mpm_uv_mask_footprint_blur_iters", 1)),
                        )
                        if bool(SCENE_PARAMS.get("mpm_uv_mask_footprint", False))
                        else int(SCENE_PARAMS.get("mpm_uv_smooth_iters", 1))
                    )
                ),
                "mpm_uv_despike": bool(SCENE_PARAMS.get("mpm_uv_despike", False)),
                "mpm_uv_despike_scope": str(SCENE_PARAMS.get("mpm_uv_despike_scope", "boundary")),
                "mpm_uv_despike_abs_mm": float(SCENE_PARAMS.get("mpm_uv_despike_abs_mm", 0.0)),
                "mpm_uv_despike_cap_mm": SCENE_PARAMS.get("mpm_uv_despike_cap_mm", None),
                "mpm_uv_despike_boundary_iters": int(SCENE_PARAMS.get("mpm_uv_despike_boundary_iters", 0)),
                "mpm_uv_mask_footprint": bool(SCENE_PARAMS.get("mpm_uv_mask_footprint", False)),
                "mpm_uv_mask_footprint_dilate_iters": int(SCENE_PARAMS.get("mpm_uv_mask_footprint_dilate_iters", 0)),
                "mpm_uv_mask_footprint_blur_iters": int(SCENE_PARAMS.get("mpm_uv_mask_footprint_blur_iters", 0)),
                "mpm_uv_cap_mm": SCENE_PARAMS.get("mpm_uv_cap_mm", None),
                "mpm_uv_scale": float(SCENE_PARAMS.get("mpm_uv_scale", 1.0)),
            },
        },
    }

    if args.save_dir:
        preflight_reason = None
        if not HAS_EZGL:
            preflight_reason = "ezgl not available"
        elif not HAS_TAICHI:
            preflight_reason = "taichi not available (mpm disabled)"
        _write_preflight_run_manifest(
            Path(args.save_dir),
            record_interval=int(args.record_interval),
            total_frames=int(expected_frames),
            run_context=run_context,
            reason=preflight_reason,
        )

    # Check dependencies
    if not HAS_EZGL:
        print("ERROR: ezgl not available, cannot run visualization")
        return 1

    if not HAS_TAICHI:
        print("WARNING: Taichi not available, MPM will be disabled")

    interactive = bool(args.interactive) or not bool(args.save_dir)

    # Run comparison
    engine = RGBComparisonEngine(
        fem_file=args.fem_file,
        object_file=args.object_file,
        mode=args.mode,
        visible=True,
        save_dir=args.save_dir,
        fem_indenter_face=args.fem_indenter_face,
        fem_indenter_geom=fem_indenter_geom,
    )
    engine.mpm_marker_mode = args.mpm_marker
    engine.mpm_depth_tint_mode = str(mpm_depth_tint_mode)
    engine.mpm_depth_tint_max_mm = mpm_depth_tint_max_mm
    engine.mpm_depth_tint = bool(mpm_depth_tint_mode != "off")
    engine.set_fem_show_marker(str(args.fem_marker).lower().strip() != "off")
    engine.mpm_show_indenter = args.mpm_show_indenter
    engine.mpm_debug_overlay = args.mpm_debug_overlay
    engine.export_intermediate = bool(args.export_intermediate)
    engine.export_intermediate_every = int(args.export_intermediate_every)
    if square_d_mm is not None:
        engine.indenter_square_size_mm = float(square_d_mm)
    engine.run_context = run_context
    engine.run_comparison(
        fps=args.fps,
        record_interval=int(args.record_interval),
        interactive=interactive,
    )

    return 0


if __name__ == '__main__':
    exit(main())
