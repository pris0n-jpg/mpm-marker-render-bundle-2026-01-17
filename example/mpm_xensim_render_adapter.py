"""
MPM → xensim render adapter (mesh-driven marker + SimMeshItem RGB).

This script builds a minimal offscreen rendering pipeline that reuses xensim's:
- MarkerTextureCamera (mesh + texcoords → marker_texture)
- SimMeshItem (depth zmap → RGB via calibration table)

The adapter consumes per-frame MPM-exported surface mesh vertices/normals (mm).
For convenience (and to keep the demo reproducible), it can also run a short
MPM press+slide trajectory (steps<=10) and export the mesh on the fly.

Run (PowerShell)
---------------
  conda run -n xengym python example/mpm_xensim_render_adapter.py --calibrate-file xensim/examples/calib_table.npz

Outputs are written under output/ and are intentionally not tracked by git.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_XENSESDK_PATH = _REPO_ROOT / "xensesdk"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _XENSESDK_PATH.exists() and str(_XENSESDK_PATH) not in sys.path:
    sys.path.insert(0, str(_XENSESDK_PATH))

from xengym.marker_appearance import (  # noqa: E402
    generate_random_ellipses_attenuation_texture_u8,
    resolve_marker_appearance_config,
)


def _load_py_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_xensim_core_module(core_filename: str, module_name: str):
    module_path = _REPO_ROOT / "xensim" / "xensim" / "core" / core_filename
    if not module_path.exists():
        raise FileNotFoundError(f"xensim core module not found: {module_path}")
    return _load_py_module(module_path, module_name)


def _alias_ezgl_to_xensesdk_ezgl() -> str:
    """
    xensim/xensim/core/sim_mesh_item.py 依赖顶层包名 `ezgl`。

    本仓库运行时使用 `xensesdk.ezgl`；为避免同一份 ezgl 以两种包名导入导致 Matrix4x4 类型不一致，
    这里把 `ezgl.*` 显式 alias 到 `xensesdk.ezgl.*`。
    """
    # NOTE: xensesdk.ezgl.__init__ 会把 GLGraphicsItem 作为“类”导出为同名属性，
    # 这会导致 `import xensesdk.ezgl.GLGraphicsItem as m` 绑定到类而不是子模块。
    # 因此这里用 importlib.import_module 强制拿到“模块对象”。
    import importlib  # noqa: WPS433

    xezgl = importlib.import_module("xensesdk.ezgl")
    xitems = importlib.import_module("xensesdk.ezgl.items")
    xexp = importlib.import_module("xensesdk.ezgl.experimental")
    xell = importlib.import_module("xensesdk.ezgl.experimental.GLEllipseItem")
    xgi = importlib.import_module("xensesdk.ezgl.GLGraphicsItem")

    for name in list(sys.modules.keys()):
        if name == "ezgl" or name.startswith("ezgl."):
            del sys.modules[name]

    if "xensesdk" not in sys.modules:
        pkg = ModuleType("xensesdk")
        pkg.__path__ = []
        sys.modules["xensesdk"] = pkg

    sys.modules["ezgl"] = xezgl
    sys.modules["ezgl.GLGraphicsItem"] = xgi
    sys.modules["ezgl.items"] = xitems
    sys.modules["ezgl.experimental"] = xexp
    sys.modules["ezgl.experimental.GLEllipseItem"] = xell

    ezgl_file = getattr(xezgl, "__file__", "<no __file__>")
    xgi_file = getattr(xgi, "__file__", "<no __file__>")
    return f"ezgl.__file__={ezgl_file} | ezgl.GLGraphicsItem.__file__={xgi_file}"


def _import_marker_texture_camera():
    try:
        from xensim.core.utils import MarkerTextureCamera, gen_texcoords  # type: ignore

        return MarkerTextureCamera, gen_texcoords, "import:xensim.core.utils"
    except Exception as exc:
        utils_mod = _load_xensim_core_module("utils.py", "xensim_core_utils_fallback")
        return (
            utils_mod.MarkerTextureCamera,
            utils_mod.gen_texcoords,
            f"fallback:file:xensim/xensim/core/utils.py ({type(exc).__name__}: {exc})",
        )


def _import_sim_mesh_item():
    try:
        from xensim.core.sim_mesh_item import SimMeshItem  # type: ignore

        return SimMeshItem, "import:xensim.core.sim_mesh_item"
    except Exception as exc:
        ezgl_origin = _alias_ezgl_to_xensesdk_ezgl()
        sim_mesh_mod = _load_xensim_core_module("sim_mesh_item.py", "xensim_core_sim_mesh_item_fallback")
        return (
            sim_mesh_mod.SimMeshItem,
            f"fallback:file:xensim/xensim/core/sim_mesh_item.py ({type(exc).__name__}: {exc}) ({ezgl_origin})",
        )


def _ensure_taichi():
    try:
        import taichi as ti  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Taichi is required for the --demo-mpm mode") from e
    return ti


def _make_marker_texture(tex_size: Tuple[int, int], rows: int, cols: int, radius_px: int) -> np.ndarray:
    w, h = int(tex_size[0]), int(tex_size[1])
    tex = np.full((h, w, 3), 255, dtype=np.uint8)

    # Keep dots away from boundaries to avoid clipping/edge artifacts after AA/blur.
    # +4: keep enough margin for AA halo + optional Gaussian blur (ksize=5 -> radius=2).
    pad = max(int(radius_px), 0) + 4
    x_min, x_max = float(pad), float(w - 1 - pad)
    y_min, y_max = float(pad), float(h - 1 - pad)
    if x_max < x_min:
        x_min, x_max = 0.0, float(w - 1)
    if y_max < y_min:
        y_min, y_max = 0.0, float(h - 1)

    xs = np.linspace(x_min, x_max, cols, dtype=np.float32)
    ys = np.linspace(y_min, y_max, rows, dtype=np.float32)
    if cv2 is not None:
        for y in ys:
            for x in xs:
                cv2.circle(tex, (int(round(x)), int(round(y))), int(radius_px), (0, 0, 0), -1, cv2.LINE_AA)
        tex = cv2.GaussianBlur(tex, (5, 5), 0, borderType=cv2.BORDER_REPLICATE)
        return tex

    # Numpy fallback: draw square dots (no AA).
    r = max(int(radius_px), 1)
    for y in ys.astype(np.int32):
        for x in xs.astype(np.int32):
            tex[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)] = 0
    return tex


def _apply_indenter_overlay(
    *,
    rgb_u8: np.ndarray,
    zmap_mm: np.ndarray,
    gel_w_mm: float,
    gel_h_mm: float,
    indenter_radius_mm: Optional[float] = None,
) -> np.ndarray:
    rgb = np.asarray(rgb_u8)
    if rgb.ndim != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0 or rgb.shape[2] < 3:
        return rgb_u8
    z = np.asarray(zmap_mm, dtype=np.float32)
    if z.ndim != 2 or z.size == 0:
        return rgb_u8

    z_max = float(np.nanmax(z)) if z.size else 0.0
    if not math.isfinite(z_max):
        return rgb_u8
    # Be robust to height/depth sign conventions: accept either +indentation or -indentation.
    if z_max <= 0.0:
        z_min = float(np.nanmin(z)) if z.size else 0.0
        if math.isfinite(z_min) and z_min < 0.0:
            z = -z
            z_max = float(-z_min)
        else:
            return rgb_u8

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    zh, zw = int(z.shape[0]), int(z.shape[1])
    if zh <= 0 or zw <= 0:
        return rgb_u8

    # Center from deepest point in zmap (robust even when the rest of the surface has small offsets).
    idx = int(np.nanargmax(z))
    cy_m, cx_m = divmod(idx, zw)
    cx = int(round(float(cx_m) * float(w - 1) / max(float(zw - 1), 1.0)))
    cy = int(round(float(cy_m) * float(h - 1) / max(float(zh - 1), 1.0)))
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    # Radius in pixel space (ellipse if pixel aspect differs in mm).
    if indenter_radius_mm is not None and float(indenter_radius_mm) > 0.0:
        rx = float(indenter_radius_mm) / max(float(gel_w_mm), 1e-6) * float(w)
        ry = float(indenter_radius_mm) / max(float(gel_h_mm), 1e-6) * float(h)
    else:
        # Fallback: estimate from the top-quantile region of zmap.
        try:
            thr = float(np.nanpercentile(z, 99))
        except Exception:
            thr = 0.2 * z_max
        if not math.isfinite(thr) or thr <= 0.0:
            thr = 0.2 * z_max
        m = z >= thr
        if bool(m.any()):
            ys, xs = np.nonzero(m)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            rx = (float(x1 - x0 + 1) * 0.5) * float(w) / max(float(zw), 1.0)
            ry = (float(y1 - y0 + 1) * 0.5) * float(h) / max(float(zh), 1.0)
        else:
            rx = ry = 0.2 * float(min(w, h))

    rx = float(np.clip(rx, 2.0, max(float(w) * 0.5, 2.0)))
    ry = float(np.clip(ry, 2.0, max(float(h) * 0.5, 2.0)))

    out = rgb.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)

    # Draw ellipse outline + small center cross (no external deps).
    n = int(max(64.0, 2.0 * math.pi * max(rx, ry)))
    for k in range(n):
        t = (2.0 * math.pi * float(k)) / float(n)
        x = int(round(float(cx) + rx * math.cos(t)))
        y = int(round(float(cy) + ry * math.sin(t)))
        if 0 <= x < w and 0 <= y < h:
            out[y, x, :3] = color
            # thicken by 4-neighborhood
            if x + 1 < w:
                out[y, x + 1, :3] = color
            if x - 1 >= 0:
                out[y, x - 1, :3] = color
            if y + 1 < h:
                out[y + 1, x, :3] = color
            if y - 1 >= 0:
                out[y - 1, x, :3] = color

    cross = 4
    for dx in range(-cross, cross + 1):
        x = cx + dx
        if 0 <= x < w:
            out[cy, x, :3] = color
    for dy in range(-cross, cross + 1):
        y = cy + dy
        if 0 <= y < h:
            out[y, cx, :3] = color

    return out


def _apply_post_gamma_u8(rgb_u8: np.ndarray, post_gamma: float) -> np.ndarray:
    """
    Apply a simple post gamma correction on uint8 RGB(A) output.

    Why:
    - xensim vs FEM background/lighting may differ in transfer curve; this is a cheap ablation knob.
    - Keep it CPU-side to avoid touching shader code / xensim internals.
    """
    try:
        g = float(post_gamma)
    except Exception:
        return rgb_u8
    if not math.isfinite(g) or g <= 0.0 or abs(g - 1.0) < 1e-6:
        return rgb_u8
    rgb = np.asarray(rgb_u8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb_u8

    # Apply only to RGB channels; preserve alpha if present.
    rgb_f = rgb[..., :3].astype(np.float32) / 255.0
    rgb_f = np.clip(rgb_f, 0.0, 1.0)
    rgb_f = np.power(rgb_f, 1.0 / g)
    out = rgb.copy()
    out[..., :3] = np.clip(rgb_f * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return out


@dataclass(frozen=True)
class RenderParams:
    calibrate_file: str
    xensim_bg_mode: str
    out_dir: str
    gel_w_mm: float
    gel_h_mm: float
    mesh_nrow: int
    mesh_ncol: int
    marker_tex_w: int
    marker_tex_h: int
    marker_rows: int
    marker_cols: int
    marker_radius_px: int
    marker_appearance_mode: str
    marker_appearance_seed: Optional[int]
    rgb_w: int
    rgb_h: int
    steps: int


class MPMXensimRenderer:
    def __init__(
        self,
        *,
        calibrate_file: str,
        gel_w_mm: float,
        gel_h_mm: float,
        mesh_shape: Tuple[int, int],
        marker_tex_size: Tuple[int, int],
        rgb_size: Tuple[int, int],
        marker_grid: Tuple[int, int],
        marker_radius_px: int,
        marker_mode: str = "mesh_warp",
        marker_enabled: bool = True,
        marker_appearance_mode: str = "grid",
        marker_appearance_seed: Optional[int] = None,
        warp_flip_x: bool = False,
        warp_flip_y: bool = True,
        xensim_bg_mode: str = "ref",
        xensim_color_scale: float = 1.0,
        xensim_post_gamma: float = 1.0,
        indenter_overlay: bool = False,
        indenter_radius_mm: Optional[float] = None,
        visible: bool = False,
    ) -> None:
        from xensesdk.ezgl.experimental.GLEllipseItem import surface_indices  # noqa: WPS433
        from xensesdk.ezgl.items import GLMeshItem, RGBCamera, Texture2D, gl  # noqa: WPS433
        from xensesdk.ezgl.items.scene import Scene  # noqa: WPS433

        self._gl = gl
        self._Texture2D = Texture2D
        self._GLMeshItem = GLMeshItem
        self._RGBCamera = RGBCamera
        self._surface_indices = surface_indices

        self.gel_w_mm = float(gel_w_mm)
        self.gel_h_mm = float(gel_h_mm)
        self.mesh_nrow = int(mesh_shape[0])
        self.mesh_ncol = int(mesh_shape[1])
        self.marker_tex_w = int(marker_tex_size[0])
        self.marker_tex_h = int(marker_tex_size[1])
        self.rgb_w = int(rgb_size[0])
        self.rgb_h = int(rgb_size[1])

        self.marker_rows = int(marker_grid[0])
        self.marker_cols = int(marker_grid[1])
        self.marker_radius_px = int(marker_radius_px)
        self.marker_mode = str(marker_mode).strip().lower()
        if self.marker_mode not in {"mesh_warp", "advect"}:
            raise ValueError("--marker-mode must be one of: mesh_warp, advect")
        self.marker_enabled = bool(marker_enabled)
        self.marker_appearance = resolve_marker_appearance_config(
            mode=str(marker_appearance_mode) if marker_appearance_mode is not None else None,
            seed=marker_appearance_seed,
        )
        self._warp_flip_x = bool(warp_flip_x)
        self._warp_flip_y = bool(warp_flip_y)
        self.xensim_bg_mode = str(xensim_bg_mode).strip().lower()
        if self.xensim_bg_mode not in {"ref", "flat"}:
            raise ValueError("--xensim-bg-mode must be one of: ref, flat")

        self.xensim_color_scale = float(xensim_color_scale)
        if not math.isfinite(self.xensim_color_scale) or self.xensim_color_scale < 0.0:
            raise ValueError("--xensim-color-scale must be a finite number >= 0")
        self.xensim_post_gamma = float(xensim_post_gamma)
        if not math.isfinite(self.xensim_post_gamma) or self.xensim_post_gamma <= 0.0:
            raise ValueError("--xensim-post-gamma must be a finite number > 0")

        self.indenter_overlay = bool(indenter_overlay)
        self.indenter_radius_mm = None if indenter_radius_mm is None else float(indenter_radius_mm)
        if self.indenter_radius_mm is not None and self.indenter_radius_mm <= 0.0:
            self.indenter_radius_mm = None

        MarkerTextureCamera, gen_texcoords, marker_import = _import_marker_texture_camera()
        SimMeshItem, sim_mesh_import = _import_sim_mesh_item()
        self.imports = {
            "MarkerTextureCamera": marker_import,
            "SimMeshItem": sim_mesh_import,
        }

        self.scene = Scene(win_width=1, win_height=1, visible=bool(visible), title="mpm_xensim_render_adapter")
        self.scene.cameraLookAt([0, 0, 36], [0, 0, 0], [0, 1, 0])
        with self.scene:
            self._gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.ortho = (-self.gel_w_mm / 2, self.gel_w_mm / 2, -self.gel_h_mm / 2, self.gel_h_mm / 2, 0, 20)

        # Precompute marker dot centers in texture pixel coordinates (matching legacy grid init).
        pad = max(int(self.marker_radius_px), 0) + 4
        x_min, x_max = float(pad), float(self.marker_tex_w - 1 - pad)
        y_min, y_max = float(pad), float(self.marker_tex_h - 1 - pad)
        if x_max < x_min:
            x_min, x_max = 0.0, float(self.marker_tex_w - 1)
        if y_max < y_min:
            y_min, y_max = 0.0, float(self.marker_tex_h - 1)
        cols = max(1, int(self.marker_cols))
        rows = max(1, int(self.marker_rows))
        xs = np.linspace(x_min, x_max, cols, dtype=np.float32)
        ys = np.linspace(y_min, y_max, rows, dtype=np.float32)
        self._marker_centers_px = [(float(x), float(y)) for y in ys.tolist() for x in xs.tolist()]

        # Base marker appearance (initial texture only). Motion remains driven by uv_disp_mm.
        marker_tex_np = _make_marker_texture(
            (self.marker_tex_w, self.marker_tex_h),
            rows=self.marker_rows,
            cols=self.marker_cols,
            radius_px=self.marker_radius_px,
        )
        if self.marker_appearance.mode == "random_ellipses":
            centers_np = np.asarray(self._marker_centers_px, dtype=np.float32)
            marker_tex_np = generate_random_ellipses_attenuation_texture_u8(
                tex_size_wh=(self.marker_tex_w, self.marker_tex_h),
                centers_xy=centers_np,
                cfg=self.marker_appearance,
            )

        # Keep a CPU copy (used by mesh_warp base + as a reference background color).
        self._base_marker_tex_np = marker_tex_np.astype(np.uint8, copy=True)
        self._marker_bg_u8 = self._base_marker_tex_np[0, 0].astype(np.uint8, copy=True)

        # Disk mask used by subpixel splatting (reduces temporal jitter vs int rounding).
        self._marker_r = max(int(self.marker_radius_px), 0)
        if self._marker_r > 0:
            yy, xx = np.mgrid[-self._marker_r : self._marker_r + 1, -self._marker_r : self._marker_r + 1]
            self._marker_disk = (xx * xx + yy * yy) <= (self._marker_r * self._marker_r)
        else:
            self._marker_disk = np.zeros((1, 1), dtype=np.bool_)

        self.marker_mesh = None
        self.marker_cam = None
        self.marker_tex = None
        self._marker_tex_adv = None

        if self.marker_mode == "mesh_warp":
            # Legacy xensim behavior: render marker_texture by projecting base marker texture onto the deformed mesh.
            self.marker_tex = Texture2D(marker_tex_np)

            indices = surface_indices(self.mesh_nrow, self.mesh_ncol)
            texcoords = gen_texcoords(self.mesh_nrow, self.mesh_ncol)
            self.marker_mesh = GLMeshItem(indices=indices, texcoords=texcoords, calc_normals=False, usage=gl.GL_DYNAMIC_DRAW)
            self.marker_mesh.setVisible(False)
            self.scene.addItem(self.marker_mesh)

            self.marker_cam = MarkerTextureCamera(
                self.scene,
                img_size=(self.marker_tex_w, self.marker_tex_h),
                eye=(0, 0, 10),
                up=(0, 1, 0),
                ortho_space=self.ortho,
                frustum_visible=False,
            )
            self.marker_cam.init(self.marker_mesh, self.marker_tex)
        else:
            # Advect mode: keep marker dots circular by translating dot centers using uv_disp_mm.
            init_marker = np.empty((self.marker_tex_h, self.marker_tex_w, 3), dtype=np.uint8)
            init_marker[...] = self._marker_bg_u8
            self._marker_tex_adv = Texture2D(
                init_marker,
                min_filter=gl.GL_LINEAR,
                mag_filter=gl.GL_LINEAR,
                wrap_s=gl.GL_CLAMP_TO_EDGE,
                wrap_t=gl.GL_CLAMP_TO_EDGE,
            )

        calibrate_path = Path(calibrate_file)
        if not calibrate_path.exists():
            raise FileNotFoundError(f"calibrate_file not found: {calibrate_file}")

        self.sim_mesh = SimMeshItem(
            shape=(self.mesh_nrow, self.mesh_ncol),
            x_range=(-self.gel_w_mm / 2, self.gel_w_mm / 2),
            y_range=(-self.gel_h_mm / 2, self.gel_h_mm / 2),
            table_path=str(calibrate_path),
            zmap=np.zeros((20, 20), dtype=np.float32),
        )
        if self.marker_mode == "mesh_warp":
            if self.marker_cam is None:
                raise RuntimeError("marker_cam missing for mesh_warp mode")
            self.sim_mesh.marker_texture = self.marker_cam.texture
        else:
            self.sim_mesh.marker_texture = self._marker_tex_adv
        self.sim_mesh.marker_flag = bool(self.marker_enabled)
        # `ref` => use calib_table.npz ref texture (legacy xensim behavior)
        # `flat` => use constant gray background (often closer to legacy SensorScene brightness)
        self.sim_mesh.background_flag = self.xensim_bg_mode == "ref"
        self.sim_mesh.color_scale = float(self.xensim_color_scale)
        self.scene.addItem(self.sim_mesh)

        self.rgb_cam = RGBCamera(
            self.scene,
            img_size=(self.rgb_w, self.rgb_h),
            eye=(0, 0, 10),
            up=(0, 1, 0),
            ortho_space=self.ortho,
            frustum_visible=False,
        )

    def _bilinear_sample_uv_mm(self, uv_disp_mm: np.ndarray, x_px: float, y_px: float) -> Tuple[float, float]:
        """Sample uv_disp_mm (H,W,2) at a marker texture pixel coordinate (x_px,y_px)."""
        h, w = int(uv_disp_mm.shape[0]), int(uv_disp_mm.shape[1])
        if h <= 0 or w <= 0:
            return 0.0, 0.0
        if self.marker_tex_w <= 1 or self.marker_tex_h <= 1:
            return float(uv_disp_mm[0, 0, 0]), float(uv_disp_mm[0, 0, 1])

        # Map marker texture pixel -> uv grid index space (treat uv field as image-aligned like cv2.resize).
        col_f = (float(x_px) / float(self.marker_tex_w - 1)) * float(w - 1)
        row_f = (float(y_px) / float(self.marker_tex_h - 1)) * float(h - 1)

        c0 = int(math.floor(col_f))
        r0 = int(math.floor(row_f))
        c1 = min(c0 + 1, w - 1)
        r1 = min(r0 + 1, h - 1)
        c0 = max(c0, 0)
        r0 = max(r0, 0)

        wc = float(col_f - float(c0))
        wr = float(row_f - float(r0))
        wc = 0.0 if not math.isfinite(wc) else min(max(wc, 0.0), 1.0)
        wr = 0.0 if not math.isfinite(wr) else min(max(wr, 0.0), 1.0)

        uv00 = uv_disp_mm[r0, c0].astype(np.float32, copy=False)
        uv10 = uv_disp_mm[r0, c1].astype(np.float32, copy=False)
        uv01 = uv_disp_mm[r1, c0].astype(np.float32, copy=False)
        uv11 = uv_disp_mm[r1, c1].astype(np.float32, copy=False)

        u0 = float(uv00[0]) * (1.0 - wc) + float(uv10[0]) * wc
        v0 = float(uv00[1]) * (1.0 - wc) + float(uv10[1]) * wc
        u1 = float(uv01[0]) * (1.0 - wc) + float(uv11[0]) * wc
        v1 = float(uv01[1]) * (1.0 - wc) + float(uv11[1]) * wc

        u = u0 * (1.0 - wr) + u1 * wr
        v = v0 * (1.0 - wr) + v1 * wr
        if not (math.isfinite(u) and math.isfinite(v)):
            return 0.0, 0.0
        return u, v

    def _advect_marker_texture_u8(self, uv_disp_mm: np.ndarray) -> np.ndarray:
        """
        Render marker texture by translating dot centers using uv_disp_mm (translation-only, dots stay circular).

        Why: improves temporal continuity and avoids mesh-warp induced smear/ellipses.
        """
        tex_h, tex_w = int(self.marker_tex_h), int(self.marker_tex_w)
        bg = self._marker_bg_u8
        out = np.empty((tex_h, tex_w, 3), dtype=np.uint8)
        out[...] = bg
        if not self._marker_centers_px:
            return out
        if self.marker_appearance.mode != "random_ellipses" and self._marker_r <= 0:
            return out

        uv = np.asarray(uv_disp_mm, dtype=np.float32)
        if uv.ndim != 3 or uv.shape[2] != 2:
            return out

        if self.marker_appearance.mode == "random_ellipses":
            # Draw per-dot ellipses at advected centers using a fixed seed (appearance is deterministic).
            # Note: We keep the dot ordering stable to keep per-dot RNG mapping stable even if some dots go OOB.
            centers = np.full((len(self._marker_centers_px), 2), np.nan, dtype=np.float32)
            r = max(int(self._marker_r), 0)
            for i, (x0, y0) in enumerate(self._marker_centers_px):
                u_mm, v_mm = self._bilinear_sample_uv_mm(uv, x0, y0)
                dx_px = (u_mm / max(self.gel_w_mm, 1e-6)) * float(tex_w)
                dy_px = (v_mm / max(self.gel_h_mm, 1e-6)) * float(tex_h)
                if self._warp_flip_x:
                    dx_px = -dx_px
                if self._warp_flip_y:
                    dy_px = -dy_px

                x1 = float(x0) + float(dx_px)
                y1 = float(y0) + float(dy_px)
                if not (math.isfinite(x1) and math.isfinite(y1)):
                    continue
                if x1 < -float(r) or x1 > float(tex_w - 1 + r) or y1 < -float(r) or y1 > float(tex_h - 1 + r):
                    continue
                centers[i, 0] = np.float32(x1)
                centers[i, 1] = np.float32(y1)
            return generate_random_ellipses_attenuation_texture_u8(
                tex_size_wh=(tex_w, tex_h),
                centers_xy=centers,
                cfg=self.marker_appearance,
            )

        alpha = np.zeros((tex_h, tex_w), dtype=np.float32)
        disk = self._marker_disk
        r = int(self._marker_r)

        for (x0, y0) in self._marker_centers_px:
            u_mm, v_mm = self._bilinear_sample_uv_mm(uv, x0, y0)
            dx_px = (u_mm / max(self.gel_w_mm, 1e-6)) * float(tex_w)
            dy_px = (v_mm / max(self.gel_h_mm, 1e-6)) * float(tex_h)
            if self._warp_flip_x:
                dx_px = -dx_px
            if self._warp_flip_y:
                dy_px = -dy_px

            x1 = float(x0) + float(dx_px)
            y1 = float(y0) + float(dy_px)
            if not (math.isfinite(x1) and math.isfinite(y1)):
                continue
            if x1 < -float(r) or x1 > float(tex_w - 1 + r) or y1 < -float(r) or y1 > float(tex_h - 1 + r):
                continue

            ix = int(math.floor(x1))
            iy = int(math.floor(y1))
            fx = float(x1 - float(ix))
            fy = float(y1 - float(iy))

            # Bilinear weights onto 4 nearest pixel centers.
            for ox in (0, 1):
                wx = (1.0 - fx) if ox == 0 else fx
                if wx <= 0.0:
                    continue
                cx = ix + ox
                if cx < 0 or cx >= tex_w:
                    continue
                for oy in (0, 1):
                    wy = (1.0 - fy) if oy == 0 else fy
                    w = float(wx * wy)
                    if w <= 0.0:
                        continue
                    cy = iy + oy
                    if cy < 0 or cy >= tex_h:
                        continue

                    x_lo = max(0, cx - r)
                    x_hi = min(tex_w, cx + r + 1)
                    y_lo = max(0, cy - r)
                    y_hi = min(tex_h, cy + r + 1)
                    if x_hi <= x_lo or y_hi <= y_lo:
                        continue
                    d_x0 = x_lo - (cx - r)
                    d_y0 = y_lo - (cy - r)
                    d_x1 = d_x0 + (x_hi - x_lo)
                    d_y1 = d_y0 + (y_hi - y_lo)
                    mask = disk[d_y0:d_y1, d_x0:d_x1]
                    if mask.size == 0:
                        continue
                    patch = alpha[y_lo:y_hi, x_lo:x_hi]
                    # 加法 alpha 比 max() 更稳定：亚像素移动时亮度更连续，降低“闪烁/抖动”观感。
                    patch[mask] = np.clip(patch[mask] + np.float32(w), 0.0, 1.0)

        if float(alpha.max()) > 0.0:
            a = np.clip(alpha, 0.0, 1.0)[..., None]
            out = (out.astype(np.float32) * (1.0 - a)).clip(0, 255).astype(np.uint8)
        return out

    def render_frame(
        self,
        vertices_mm_hw3: np.ndarray,
        uv_disp_mm: Optional[np.ndarray] = None,
        uv_du_mm: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render one frame.

        Returns:
            rgb_u8: (H,W,3/4) uint8
            marker_u8: (H,W,3) uint8
        """
        v = np.asarray(vertices_mm_hw3, dtype=np.float32)
        if v.ndim != 3 or v.shape[-1] != 3:
            raise ValueError("vertices_mm_hw3 must be (H,W,3)")
        if v.shape[0] != self.mesh_nrow or v.shape[1] != self.mesh_ncol:
            raise ValueError("vertices shape must match mesh_shape")

        marker_u8: np.ndarray
        if not self.marker_enabled:
            marker_u8 = np.empty((self.marker_tex_h, self.marker_tex_w, 3), dtype=np.uint8)
            marker_u8[...] = self._marker_bg_u8
        elif self.marker_mode == "mesh_warp":
            if self.marker_mesh is None or self.marker_cam is None:
                raise RuntimeError("mesh_warp mode requires marker_mesh/marker_cam")

            vf = v.reshape(-1, 3).copy()
            vf[:, 1] -= np.float32(self.gel_h_mm / 2.0)  # y: [0..h] -> [-h/2..h/2]
            vf[:, 2] = 0.0  # marker texture camera只需 x/y；避免 z 超出 near/far 被裁剪

            normals = np.zeros_like(vf, dtype=np.float32)
            normals[:, 2] = 1.0
            self.marker_mesh.setData(vertexes=vf, normals=normals)
            self.marker_cam.render()
            marker_u8 = (self.marker_cam.get_texture_np() * 255).astype(np.uint8)
        else:
            # Translation-only marker dots (cpu), uploaded to GPU for SimMeshItem composition.
            # NOTE: advect mode currently uses uv_disp_mm (cumulative displacement).
            # uv_du_mm is accepted for API consistency with MPMSensorSceneRenderer, but not required here.
            _ = uv_du_mm
            uv = uv_disp_mm
            if uv is None:
                uv = np.zeros((self.mesh_nrow, self.mesh_ncol, 2), dtype=np.float32)
            marker_u8 = self._advect_marker_texture_u8(np.asarray(uv, dtype=np.float32))
            if self._marker_tex_adv is not None:
                self._marker_tex_adv.setTexture(marker_u8)

        # SimMeshItem expects a depth-like zmap; use positive intrusion for stability (>=0).
        height_mm = v[..., 2].astype(np.float32, copy=False)
        zmap = np.clip(-height_mm, 0.0, None)  # mm, >=0
        # Keep these as per-frame knobs so ablations don't require recreating the renderer.
        self.sim_mesh.background_flag = self.xensim_bg_mode == "ref"
        self.sim_mesh.marker_flag = bool(self.marker_enabled)
        self.sim_mesh.color_scale = float(self.xensim_color_scale)
        self.sim_mesh.setData(zmap)

        rgb = (self.rgb_cam.render() * 255).astype(np.uint8)
        rgb = _apply_post_gamma_u8(rgb, self.xensim_post_gamma)
        if self.indenter_overlay:
            rgb = _apply_indenter_overlay(
                rgb_u8=rgb,
                zmap_mm=zmap,
                gel_w_mm=float(self.gel_w_mm),
                gel_h_mm=float(self.gel_h_mm),
                indenter_radius_mm=self.indenter_radius_mm,
            )
        return rgb, marker_u8

    def _apply_indenter_overlay(self, rgb_u8: np.ndarray, zmap_mm: np.ndarray) -> np.ndarray:
        """
        Draw a lightweight debug overlay to make the indenter/contact region easy to see.

        Why:
        - C(MPM→xensim) may look “flat” / “no indenter” depending on bg_mode/light calibration.
        - Overlay is purely a visualization aid and must not affect the underlying zmap/rendering.
        """
        rgb = np.asarray(rgb_u8)
        if rgb.ndim != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0 or rgb.shape[2] < 3:
            return rgb_u8
        z = np.asarray(zmap_mm, dtype=np.float32)
        if z.ndim != 2 or z.size == 0:
            return rgb_u8

        z_max = float(np.nanmax(z)) if z.size else 0.0
        if not math.isfinite(z_max) or z_max <= 0.0:
            return rgb_u8

        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        zh, zw = int(z.shape[0]), int(z.shape[1])
        if zh <= 0 or zw <= 0:
            return rgb_u8

        # Center from deepest point in zmap (robust even when the rest of the surface has small offsets).
        idx = int(np.nanargmax(z))
        cy_m, cx_m = divmod(idx, zw)
        cx = int(round(float(cx_m) * float(w - 1) / max(float(zw - 1), 1.0)))
        cy = int(round(float(cy_m) * float(h - 1) / max(float(zh - 1), 1.0)))
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))

        # Radius in pixel space (ellipse if pixel aspect differs in mm).
        if self.indenter_radius_mm is not None:
            rx = float(self.indenter_radius_mm) / max(self.gel_w_mm, 1e-6) * float(w)
            ry = float(self.indenter_radius_mm) / max(self.gel_h_mm, 1e-6) * float(h)
        else:
            # Fallback: estimate from the top-quantile region of zmap.
            try:
                thr = float(np.nanpercentile(z, 99))
            except Exception:
                thr = 0.2 * z_max
            if not math.isfinite(thr) or thr <= 0.0:
                thr = 0.2 * z_max
            m = z >= thr
            if bool(m.any()):
                ys, xs = np.nonzero(m)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                rx = (float(x1 - x0 + 1) * 0.5) * float(w) / max(float(zw), 1.0)
                ry = (float(y1 - y0 + 1) * 0.5) * float(h) / max(float(zh), 1.0)
            else:
                rx = ry = 0.2 * float(min(w, h))

        rx = float(np.clip(rx, 2.0, max(float(w) * 0.5, 2.0)))
        ry = float(np.clip(ry, 2.0, max(float(h) * 0.5, 2.0)))

        out = rgb.copy()
        color = np.array([255, 0, 0], dtype=np.uint8)

        # Draw ellipse outline + small center cross (no external deps).
        n = int(max(64.0, 2.0 * math.pi * max(rx, ry)))
        for k in range(n):
            t = (2.0 * math.pi * float(k)) / float(n)
            x = int(round(float(cx) + rx * math.cos(t)))
            y = int(round(float(cy) + ry * math.sin(t)))
            if 0 <= x < w and 0 <= y < h:
                out[y, x, :3] = color
                # thicken by 4-neighborhood
                if x + 1 < w:
                    out[y, x + 1, :3] = color
                if x - 1 >= 0:
                    out[y, x - 1, :3] = color
                if y + 1 < h:
                    out[y + 1, x, :3] = color
                if y - 1 >= 0:
                    out[y - 1, x, :3] = color

        cross = 4
        for dx in range(-cross, cross + 1):
            x = cx + dx
            if 0 <= x < w:
                out[cy, x, :3] = color
        for dy in range(-cross, cross + 1):
            y = cy + dy
            if 0 <= y < h:
                out[y, cx, :3] = color

        return out


class MPMSensorSceneRenderer:
    """
    Renderer that reuses the GLSurfMeshItem/SensorScene-style shading from `example/mpm_fem_rgb_compare.py`.

    Intended use:
    - As an optional C renderer mode in `mpm_xensim_triplet_runner.py` to reduce B(FEM) vs C RGB diff.
    - Consumes only intermediate fields: height_field_mm + uv_disp_mm (mm).
    """

    def __init__(
        self,
        *,
        gel_w_mm: float,
        gel_h_mm: float,
        mesh_shape: Tuple[int, int],
        marker_tex_size: Tuple[int, int],
        rgb_size: Tuple[int, int],
        marker_grid: Tuple[int, int],
        marker_radius_px: int,
        marker_mode: str = "advect",
        marker_enabled: bool = True,
        warp_flip_x: bool = False,
        warp_flip_y: bool = True,
        xensim_post_gamma: float = 1.0,
        indenter_overlay: bool = False,
        indenter_radius_mm: Optional[float] = None,
        visible: bool = False,
    ) -> None:
        self.gel_w_mm = float(gel_w_mm)
        self.gel_h_mm = float(gel_h_mm)
        self.mesh_nrow = int(mesh_shape[0])
        self.mesh_ncol = int(mesh_shape[1])
        self.marker_tex_w = int(marker_tex_size[0])
        self.marker_tex_h = int(marker_tex_size[1])
        self.rgb_w = int(rgb_size[0])
        self.rgb_h = int(rgb_size[1])

        self.marker_rows = int(marker_grid[0])
        self.marker_cols = int(marker_grid[1])
        self.marker_radius_px = int(marker_radius_px)
        self.marker_mode = str(marker_mode).strip().lower()
        if self.marker_mode not in {"mesh_warp", "advect"}:
            raise ValueError("--marker-mode must be one of: mesh_warp, advect")
        self.marker_enabled = bool(marker_enabled)
        self._warp_flip_x = bool(warp_flip_x)
        self._warp_flip_y = bool(warp_flip_y)

        self.xensim_post_gamma = float(xensim_post_gamma)
        if not math.isfinite(self.xensim_post_gamma) or self.xensim_post_gamma <= 0.0:
            raise ValueError("--xensim-post-gamma must be a finite number > 0")

        self.indenter_overlay = bool(indenter_overlay)
        self.indenter_radius_mm = None if indenter_radius_mm is None else float(indenter_radius_mm)
        if self.indenter_radius_mm is not None and self.indenter_radius_mm <= 0.0:
            self.indenter_radius_mm = None

        # NOTE: MPMSensorScene currently hardcodes RGBCamera img_size=(400,700).
        # Keep a guard to avoid silent resizes that would inflate diff metrics.
        if (self.rgb_w, self.rgb_h) != (400, 700):
            raise ValueError(f"sensor_scene expects rgb_size=(400,700), got {(self.rgb_w, self.rgb_h)}")

        mpm_mod_path = _REPO_ROOT / "example" / "mpm_fem_rgb_compare.py"
        mpm_mod = _load_py_module(mpm_mod_path, "mpm_fem_rgb_compare_sensor_scene_adapter")
        if not hasattr(mpm_mod, "MPMSensorScene"):
            raise RuntimeError("MPMSensorScene not found in example/mpm_fem_rgb_compare.py")
        if not hasattr(mpm_mod, "SCENE_PARAMS"):
            raise RuntimeError("SCENE_PARAMS not found in example/mpm_fem_rgb_compare.py")

        # Clone defaults and override only what this adapter needs.
        sp = dict(getattr(mpm_mod, "SCENE_PARAMS"))
        sp["gel_size_mm"] = (self.gel_w_mm, self.gel_h_mm)
        sp["marker_tex_size_wh"] = (self.marker_tex_w, self.marker_tex_h)
        sp["marker_grid_rows"] = int(self.marker_rows)
        sp["marker_grid_cols"] = int(self.marker_cols)
        sp["marker_radius_px"] = int(self.marker_radius_px)
        sp["mpm_warp_flip_x"] = bool(self._warp_flip_x)
        sp["mpm_warp_flip_y"] = bool(self._warp_flip_y)

        # Match publish_v1 defaults to align with the FEM baseline (stable lighting, no synthetic tint).
        sp["mpm_zmap_convention"] = "indentation"
        sp["mpm_depth_tint_mode"] = "off"
        sp["mpm_light_profile"] = "publish_v1"
        sp["mpm_disable_light_file"] = True
        sp["mpm_render_shadow"] = False

        setattr(mpm_mod, "SCENE_PARAMS", sp)
        self.imports = {"MPMSensorScene": f"file:{mpm_mod_path.as_posix()}"}

        SceneCls = getattr(mpm_mod, "MPMSensorScene")
        self._scene = SceneCls(
            gel_size_mm=(self.gel_w_mm, self.gel_h_mm),
            grid_shape=(self.mesh_nrow, self.mesh_ncol),
            visible=bool(visible),
        )

        if not self.marker_enabled:
            self._scene.set_marker_mode("off")
        else:
            # Map xensim marker mode -> MPMSensorScene marker mode.
            # - mesh_warp => warp
            # - advect    => advect
            self._scene.set_show_marker(True)
            self._scene.set_marker_mode("warp" if self.marker_mode == "mesh_warp" else "advect")
        try:
            self._scene.set_depth_tint_config("off")
        except Exception:
            pass

    def _marker_texture_u8(self) -> np.ndarray:
        scene = self._scene
        if not getattr(scene, "_show_marker", True) or str(getattr(scene, "_marker_mode", "off")) == "off":
            tex = getattr(scene, "white_tex_np", None)
            if isinstance(tex, np.ndarray):
                return tex.astype(np.uint8, copy=False)
            return np.full((self.marker_tex_h, self.marker_tex_w, 3), 255, dtype=np.uint8)

        if str(getattr(scene, "_marker_mode", "static")) == "static" or getattr(scene, "_uv_disp_mm", None) is None:
            tex = getattr(scene, "marker_tex_np", None)
            if isinstance(tex, np.ndarray):
                return tex.astype(np.uint8, copy=False)
        cached = getattr(scene, "_cached_warped_tex", None)
        if isinstance(cached, np.ndarray):
            return cached.astype(np.uint8, copy=False)
        tex = getattr(scene, "marker_tex_np", None)
        if isinstance(tex, np.ndarray):
            return tex.astype(np.uint8, copy=False)
        return np.full((self.marker_tex_h, self.marker_tex_w, 3), 255, dtype=np.uint8)

    def render_frame(
        self,
        *,
        height_field_mm: np.ndarray,
        uv_disp_mm: Optional[np.ndarray] = None,
        uv_du_mm: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        height = np.asarray(height_field_mm, dtype=np.float32)
        if height.ndim != 2 or height.shape[0] != self.mesh_nrow or height.shape[1] != self.mesh_ncol:
            raise ValueError("height_field_mm must be (H,W) and match mesh_shape")
        uv = None if uv_disp_mm is None else np.asarray(uv_disp_mm, dtype=np.float32)
        if uv is not None and (uv.ndim != 3 or uv.shape[0] != self.mesh_nrow or uv.shape[1] != self.mesh_ncol or uv.shape[2] != 2):
            raise ValueError("uv_disp_mm must be (H,W,2) and match mesh_shape")
        du = None if uv_du_mm is None else np.asarray(uv_du_mm, dtype=np.float32)
        if du is not None and (du.ndim != 3 or du.shape[0] != self.mesh_nrow or du.shape[1] != self.mesh_ncol or du.shape[2] != 2):
            raise ValueError("uv_du_mm must be (H,W,2) and match mesh_shape")

        # Update marker first (warps/advects texture); update height/depth for shading second.
        try:
            self._scene.set_uv_displacement(uv, uv_du_mm=du)
        except Exception:
            pass
        self._scene.set_height_field(height, smooth=True)

        rgb = self._scene.get_image()
        rgb = _apply_post_gamma_u8(rgb, self.xensim_post_gamma)
        marker_u8 = self._marker_texture_u8()

        if self.indenter_overlay:
            zmap = np.clip(-height, 0.0, None)  # mm, >=0
            rgb = _apply_indenter_overlay(
                rgb_u8=rgb,
                zmap_mm=zmap,
                gel_w_mm=float(self.gel_w_mm),
                gel_h_mm=float(self.gel_h_mm),
                indenter_radius_mm=self.indenter_radius_mm,
            )

        return rgb, marker_u8


def _create_demo_mpm_mesh_sequence(
    *,
    steps: int,
    mesh_shape: Tuple[int, int],
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """
    Create a short MPM press+slide trajectory and export vertices per step.

    Returns:
        frames: list of vertices_mm (H,W,3)
        meta: dict (mpm params + export stats)
    """
    ti = _ensure_taichi()
    ti.init(arch=ti.cpu)

    # Keep the demo small & deterministic.
    gel_w_mm, gel_h_mm, gel_t_mm = 30.0, 30.0, 12.0
    dx_mm = 1.0
    particles_per_cell = 2.0
    dt = 2e-5

    press_steps = min(5, steps)
    slide_steps = max(steps - press_steps, 0)
    press_depth_mm = 2.0
    slide_dist_mm = 2.0

    indenter_radius_mm = 4.0
    indenter_gap_mm = 0.0

    pad_xy = 6
    pad_z_bottom = 6
    pad_z_top = 16

    gel_w_m, gel_h_m, gel_t_m = gel_w_mm * 1e-3, gel_h_mm * 1e-3, gel_t_mm * 1e-3
    dx_m = dx_mm * 1e-3

    def create_block_particles() -> np.ndarray:
        spacing = float(dx_m) / max(float(particles_per_cell), 1.0)
        nx = int(np.ceil(gel_w_m / spacing))
        ny = int(np.ceil(gel_h_m / spacing))
        nz = int(np.ceil(gel_t_m / spacing))
        x = np.linspace(-gel_w_m / 2.0, gel_w_m / 2.0, nx, dtype=np.float32)
        y = np.linspace(0.0, gel_h_m, ny, dtype=np.float32)
        z = np.linspace(0.0, gel_t_m, nz, dtype=np.float32)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32, copy=False)
        padding_vec = np.array([pad_xy * dx_m, pad_xy * dx_m, pad_z_bottom * dx_m], dtype=np.float32)
        positions += (padding_vec - positions.min(axis=0))
        return positions

    def grid_extent() -> Tuple[int, int, int]:
        return (
            int(np.ceil(gel_w_m / dx_m)) + pad_xy * 2,
            int(np.ceil(gel_h_m / dx_m)) + pad_xy * 2,
            int(np.ceil(gel_t_m / dx_m)) + pad_z_bottom + pad_z_top,
        )

    positions0 = create_block_particles()
    n_particles = int(positions0.shape[0])
    initial_top_z_m = float(np.max(positions0[:, 2]))

    z_min = float(positions0[:, 2].min())
    base_mask = positions0[:, 2] <= (z_min + 2.0 * dx_m)
    base_indices = np.nonzero(base_mask)[0].astype(np.int32, copy=False)
    base_init_pos = positions0[base_indices].copy()

    from xengym.mpm import (
        MPMConfig,
        GridConfig,
        TimeConfig,
        OgdenConfig,
        MaterialConfig,
        ContactConfig,
        OutputConfig,
        MPMSolver,
        SDFConfig,
    )
    from xengym.mpm.surface_mesh import TopSurfaceMeshExporter

    indenter_half_height_m = float(indenter_radius_mm * 1e-3)
    center0 = np.array(
        [
            float(positions0[:, 0].mean()),
            float(positions0[:, 1].mean()),
            float(initial_top_z_m + indenter_gap_mm * 1e-3 + indenter_half_height_m),
        ],
        dtype=np.float32,
    )
    obstacles = [
        SDFConfig(sdf_type="plane", center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
        SDFConfig(
            sdf_type="cylinder",
            center=tuple(center0.tolist()),
            half_extents=(float(indenter_radius_mm * 1e-3), float(indenter_radius_mm * 1e-3), float(indenter_half_height_m)),
        ),
    ]

    config = MPMConfig(
        grid=GridConfig(
            grid_size=grid_extent(),
            dx=dx_m,
            sticky_boundary=True,
            sticky_boundary_width=3,
        ),
        time=TimeConfig(dt=dt, num_steps=1),
        material=MaterialConfig(
            density=1000.0,
            ogden=OgdenConfig(mu=[500.0], alpha=[2.0], kappa=10000.0),
            maxwell_branches=[],
            enable_bulk_viscosity=False,
            bulk_viscosity=0.0,
        ),
        contact=ContactConfig(
            enable_contact=True,
            contact_stiffness_normal=8e2,
            contact_stiffness_tangent=4e2,
            mu_s=2.0,
            mu_k=1.5,
            obstacles=obstacles,
        ),
        output=OutputConfig(),
    )

    solver = MPMSolver(config, n_particles)
    try:
        solver.gravity = ti.Vector([0.0, 0.0, 0.0])
    except Exception:
        pass

    solver.initialize_particles(positions0, np.zeros_like(positions0, dtype=np.float32))

    base_indices_ti = ti.field(dtype=ti.i32, shape=int(base_indices.size))
    base_init_pos_ti = ti.Vector.field(3, dtype=ti.f32, shape=int(base_indices.size))
    base_indices_ti.from_numpy(base_indices)
    base_init_pos_ti.from_numpy(base_init_pos)

    @ti.kernel
    def fix_base():
        for k in range(base_indices_ti.shape[0]):
            p = base_indices_ti[k]
            solver.fields.x[p] = base_init_pos_ti[k]
            solver.fields.v[p] = ti.Vector([0.0, 0.0, 0.0])
            solver.fields.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            solver.fields.F[p] = ti.Matrix.identity(ti.f32, 3)

    exporter = TopSurfaceMeshExporter(
        grid_shape=mesh_shape,
        initial_positions_m=positions0,
        dx_m=dx_m,
        particles_per_cell=particles_per_cell,
        initial_top_z_m=initial_top_z_m,
        reference_edge=True,
        fill_holes=True,
        fill_iters=10,
        smooth_iters=1,
    )

    frames: List[np.ndarray] = []
    stats: List[Dict[str, float]] = []
    v0 = exporter.extract(solver.fields.x.to_numpy()).vertices_mm
    for step in range(steps):
        if step < press_steps:
            t = step / max(press_steps - 1, 1)
            dz_mm = press_depth_mm * t
            dx_slide_mm = 0.0
        else:
            t = (step - press_steps) / max(slide_steps - 1, 1) if slide_steps > 0 else 0.0
            dz_mm = press_depth_mm
            dx_slide_mm = slide_dist_mm * t

        center = center0.copy()
        center[0] += np.float32(dx_slide_mm * 1e-3)
        center[2] -= np.float32(dz_mm * 1e-3)
        centers_np = solver.obstacle_centers.to_numpy()
        centers_np[1] = center
        solver.obstacle_centers.from_numpy(centers_np)

        solver.step()
        fix_base()

        fr = exporter.extract(solver.fields.x.to_numpy())
        v = fr.vertices_mm
        frames.append(v)

        dxy = (v[..., :2] - v0[..., :2]).astype(np.float32, copy=False)
        mag = np.sqrt(dxy[..., 0] ** 2 + dxy[..., 1] ** 2)
        stats.append(
            {
                "step": float(step),
                "dx_slide_mm": float(dx_slide_mm),
                "dz_mm": float(dz_mm),
                "height_min_mm": float(np.min(v[..., 2])),
                "height_max_mm": float(np.max(v[..., 2])),
                "tangent_max_mm": float(np.max(mag)),
            }
        )

    meta = {
        "mpm": {
            "gel_size_mm": [gel_w_mm, gel_h_mm, gel_t_mm],
            "dx_mm": dx_mm,
            "particles_per_cell": particles_per_cell,
            "dt": dt,
            "press_depth_mm": press_depth_mm,
            "slide_dist_mm": slide_dist_mm,
            "press_steps": int(press_steps),
            "slide_steps": int(slide_steps),
            "indenter_radius_mm": indenter_radius_mm,
        },
        "mesh_shape": [int(mesh_shape[0]), int(mesh_shape[1])],
        "sequence_stats": stats,
    }
    return frames, meta


def main() -> int:
    default_calib = _REPO_ROOT / "xensim" / "examples" / "calib_table.npz"
    parser = argparse.ArgumentParser(description="MPM→xensim render adapter (mesh-driven marker + SimMeshItem RGB)")
    parser.add_argument("--calibrate-file", type=str, default=str(default_calib), help="Calibration npz used by SimMeshItem")
    parser.add_argument("--out-root", type=str, default="output/mpm_xensim_render_adapter", help="Output root directory")
    parser.add_argument("--steps", type=int, default=6, help="Demo steps (<=10 recommended)")
    parser.add_argument("--mesh-h", type=int, default=175, help="Mesh rows (nrow)")
    parser.add_argument("--mesh-w", type=int, default=100, help="Mesh cols (ncol)")
    parser.add_argument("--marker-w", type=int, default=320, help="Marker texture width")
    parser.add_argument("--marker-h", type=int, default=560, help="Marker texture height")
    parser.add_argument("--marker-rows", type=int, default=20, help="Marker grid rows")
    parser.add_argument("--marker-cols", type=int, default=11, help="Marker grid cols")
    parser.add_argument("--marker-radius-px", type=int, default=2, help="Marker dot radius in pixels")
    parser.add_argument(
        "--marker-appearance-mode",
        choices=["grid", "random_ellipses"],
        default="grid",
        help="Marker appearance (initial texture/dot set): grid|random_ellipses (opt-in). Default: grid.",
    )
    parser.add_argument(
        "--marker-appearance-seed",
        type=int,
        default=None,
        help="Marker appearance seed (int). Use -1 for random (resolved seed is recorded in manifest).",
    )
    parser.add_argument("--rgb-w", type=int, default=400, help="RGB output width")
    parser.add_argument("--rgb-h", type=int, default=700, help="RGB output height")
    parser.add_argument(
        "--xensim-bg-mode",
        choices=["ref", "flat"],
        default="ref",
        help="SimMeshItem background mode: `ref` uses calib_table ref texture; `flat` uses constant gray (reduces shadow).",
    )
    args = parser.parse_args()

    steps = int(args.steps)
    if steps <= 0 or steps > 10:
        raise ValueError("--steps must be in [1,10] for the demo")

    out_root = Path(args.out_root)
    run_dir = out_root / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mesh_shape = (int(args.mesh_h), int(args.mesh_w))
    frames, meta = _create_demo_mpm_mesh_sequence(steps=steps, mesh_shape=mesh_shape)

    # Use gel size from demo meta (keeps renderer consistent with mesh exporter frame).
    gel_w_mm, gel_h_mm, _ = meta["mpm"]["gel_size_mm"]

    renderer = MPMXensimRenderer(
        calibrate_file=str(args.calibrate_file),
        gel_w_mm=float(gel_w_mm),
        gel_h_mm=float(gel_h_mm),
        mesh_shape=mesh_shape,
        marker_tex_size=(int(args.marker_w), int(args.marker_h)),
        rgb_size=(int(args.rgb_w), int(args.rgb_h)),
        marker_grid=(int(args.marker_rows), int(args.marker_cols)),
        marker_radius_px=int(args.marker_radius_px),
        marker_appearance_mode=str(args.marker_appearance_mode),
        marker_appearance_seed=(int(args.marker_appearance_seed) if args.marker_appearance_seed is not None else None),
        xensim_bg_mode=str(args.xensim_bg_mode),
        visible=False,
    )

    manifest = RenderParams(
        calibrate_file=str(args.calibrate_file),
        xensim_bg_mode=str(args.xensim_bg_mode),
        out_dir=str(run_dir).replace("\\", "/"),
        gel_w_mm=float(gel_w_mm),
        gel_h_mm=float(gel_h_mm),
        mesh_nrow=int(mesh_shape[0]),
        mesh_ncol=int(mesh_shape[1]),
        marker_tex_w=int(args.marker_w),
        marker_tex_h=int(args.marker_h),
        marker_rows=int(args.marker_rows),
        marker_cols=int(args.marker_cols),
        marker_radius_px=int(args.marker_radius_px),
        marker_appearance_mode=str(args.marker_appearance_mode),
        marker_appearance_seed=(int(args.marker_appearance_seed) if args.marker_appearance_seed is not None else None),
        rgb_w=int(args.rgb_w),
        rgb_h=int(args.rgb_h),
        steps=int(steps),
    )

    run_manifest = {
        "params": asdict(manifest),
        "marker_appearance": renderer.marker_appearance.to_manifest(),
        "imports": renderer.imports,
        "mpm_meta": meta,
        "frames": [],
    }

    for i, v in enumerate(frames):
        rgb_u8, marker_u8 = renderer.render_frame(v)
        frame_rec: Dict[str, object] = {
            "i": int(i),
            "vertices_shape": list(v.shape),
            "vertices_dtype": str(v.dtype),
            "rgb_shape": list(rgb_u8.shape),
            "rgb_dtype": str(rgb_u8.dtype),
            "marker_shape": list(marker_u8.shape),
            "marker_dtype": str(marker_u8.dtype),
        }

        if cv2 is not None:
            cv2.imwrite(str(run_dir / f"rgb_{i:03d}.png"), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(run_dir / f"marker_{i:03d}.png"), marker_u8)
        run_manifest["frames"].append(frame_rec)

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    out_dir_str = str(run_dir).replace("\\", "/")
    print(f"[mpm_xensim_render_adapter] out_dir={out_dir_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
