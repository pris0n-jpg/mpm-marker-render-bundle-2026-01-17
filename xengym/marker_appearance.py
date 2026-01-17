"""Marker appearance configuration (initial texture/dot set only).

This module is intentionally dependency-light: it MUST NOT import xengym.render (xensesdk/ezgl)
at import time, and it MUST NOT require cv2 unless the caller explicitly enables a cv2-only mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


_SUPPORTED_MODES = ("grid", "random_ellipses")


def _normalize_mode(mode: Optional[str]) -> str:
    m = str(mode).strip().lower() if mode is not None else ""
    if not m:
        return "grid"
    if m not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported marker_appearance.mode={m!r}. Supported: {', '.join(_SUPPORTED_MODES)}")
    return m


def _resolve_seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    s = int(seed)
    if s == -1:
        # 32-bit seed is compatible with numpy RNGs and easy to record in manifests.
        return int(secrets.randbits(32))
    return s


@dataclass(frozen=True)
class MarkerAppearanceConfig:
    """Resolved marker appearance config (deterministic in a pinned runtime).

    Notes:
    - This config MUST ONLY affect the *initial* marker appearance.
    - Marker motion over time MUST still be driven by physical motion fields (uv_disp_mm/uv_du_mm/v_surface, etc.).
    """

    mode: str = "grid"
    seed: Optional[int] = None

    # Optional parameters mainly used by random_ellipses (kept for manifest stability / future extension).
    density: Optional[float] = None
    size_px: Optional[Tuple[float, float]] = None
    color_rgb: Optional[Tuple[int, int, int]] = None
    blur: Optional[int] = None
    jitter_px: Optional[float] = None
    texture_dtype: str = "uint8"

    def to_manifest(self) -> Dict[str, Any]:
        def _maybe_seq(v: Any) -> Any:
            if isinstance(v, tuple):
                return list(v)
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray, list, dict)):
                return list(v)
            return v

        # Texture contract:
        # - marker_texture composes via multiplication in shader space (0..1): `bg_color *= marker_texture`.
        # - Prefer uint8 0..255 on CPU so the GPU samples it as normalized 0..1.
        dtype = str(self.texture_dtype)
        texture_value_range: Any = [0, 255] if dtype == "uint8" else [0.0, 1.0]
        out: Dict[str, Any] = {
            "mode": str(self.mode),
            "seed": int(self.seed) if self.seed is not None else None,
            "texture_dtype": dtype,
            "texture_semantics": "attenuation_mul_bg",
            "texture_value_range": texture_value_range,
            "shader_sample_range": [0.0, 1.0],
        }
        if self.density is not None:
            out["density"] = float(self.density)
        if self.size_px is not None:
            out["size_px"] = _maybe_seq(self.size_px)
        if self.color_rgb is not None:
            out["color_rgb"] = _maybe_seq(self.color_rgb)
        if self.blur is not None:
            out["blur"] = int(self.blur)
        if self.jitter_px is not None:
            out["jitter_px"] = float(self.jitter_px)
        return out


def resolve_marker_appearance_config(
    *,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> MarkerAppearanceConfig:
    """Resolve user inputs into a concrete MarkerAppearanceConfig.

    - mode defaults to "grid" (backward compatible).
    - seed=None keeps seed unset for grid; for random_ellipses we default to seed=0 to be deterministic.
    - seed=-1 is allowed and will be resolved to a concrete 32-bit seed (record it in manifests).
    """

    resolved_mode = _normalize_mode(mode)
    resolved_seed = _resolve_seed(seed)
    if resolved_seed is None and resolved_mode == "random_ellipses":
        resolved_seed = 0
    return MarkerAppearanceConfig(mode=resolved_mode, seed=resolved_seed)


def generate_random_ellipses_attenuation_texture_u8(
    *,
    tex_size_wh: Tuple[int, int],
    centers_xy: np.ndarray,
    cfg: MarkerAppearanceConfig,
) -> np.ndarray:
    """Generate an attenuation marker texture (u8 0..255) using random ellipses.

    This is the "env-style" appearance generator:
    - Draw ellipses into an overlayer (0..255).
    - Convert to attenuation texture: attenuation = 255 - overlayer.
      In shader space this composes as `bg_color *= marker_texture` (0..1),
      equivalent to the env pipeline: `img -= overlayer * img / 255`.

    Requirements:
    - cv2 is required for this mode and MUST fail fast with an actionable message when unavailable.
    - Output MUST be deterministic for the same (mode, seed, cfg) in the same runtime environment.
    """

    if str(cfg.mode).strip().lower() != "random_ellipses":
        raise ValueError("generate_random_ellipses_attenuation_texture_u8 requires cfg.mode='random_ellipses'")

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "marker_appearance.mode=random_ellipses requires cv2. "
            "Install opencv-python, or switch to marker_appearance.mode=grid."
        ) from exc

    w, h = int(tex_size_wh[0]), int(tex_size_wh[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid tex_size_wh={tex_size_wh!r}")

    pts = np.asarray(centers_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"centers_xy must have shape (N,2); got {pts.shape}")

    seed = int(cfg.seed) if cfg.seed is not None else 0
    rng = np.random.default_rng(seed)

    over = np.zeros((h, w, 3), dtype=np.float32)

    base_color = np.array(cfg.color_rgb if cfg.color_rgb is not None else (100, 140, 140), dtype=np.float32)
    size_default = cfg.size_px is None
    jitter_sigma = float(cfg.jitter_px) if cfg.jitter_px is not None else 1.0

    # Pad to avoid boundary clipping/blur artifacts.
    if size_default:
        max_axis = 11
    else:
        max_axis = int(max(float(cfg.size_px[0]), float(cfg.size_px[1]), 1.0))
    if cfg.blur is None:
        blur_radius = 9  # max for env-style random kernel up to 19
    else:
        k = int(cfg.blur)
        if k <= 0:
            blur_radius = 0
        else:
            if k % 2 == 0:
                k += 1
            blur_radius = k // 2
    pad = int(max_axis + blur_radius + 2)
    x_min, x_max = float(pad), float(w - 1 - pad)
    y_min, y_max = float(pad), float(h - 1 - pad)
    if x_max < x_min:
        x_min, x_max = 0.0, float(w - 1)
    if y_max < y_min:
        y_min, y_max = 0.0, float(h - 1)

    oob_pad = float(pad)
    for x0, y0 in pts:
        # Sample per-dot randomness in a stable order (important for determinism when some dots go OOB).
        if jitter_sigma > 0:
            jx = float(rng.normal(0.0, jitter_sigma))
            jy = float(rng.normal(0.0, jitter_sigma))
        else:
            jx, jy = 0.0, 0.0

        if size_default:
            a = int(np.clip(rng.normal(7.0, 1.0), 1.0, 11.0))
            b = int(np.clip(rng.normal(6.0, 1.0), 1.0, 9.0))
        else:
            a = int(max(1.0, float(cfg.size_px[0])))
            b = int(max(1.0, float(cfg.size_px[1])))

        angle = float(rng.uniform(0.0, 360.0))
        color = float(rng.uniform(0.6, 1.4)) * base_color
        color = np.clip(color, 0.0, 255.0)

        x = float(x0) + jx
        y = float(y0) + jy
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if x < -oob_pad or x > float(w - 1 + oob_pad) or y < -oob_pad or y > float(h - 1 + oob_pad):
            continue
        x = float(np.clip(x, x_min, x_max))
        y = float(np.clip(y, y_min, y_max))

        cv2.ellipse(
            over,
            (int(round(x)), int(round(y))),
            (int(a), int(b)),
            angle,
            0,
            360,
            (float(color[0]), float(color[1]), float(color[2])),
            -1,
            lineType=cv2.LINE_AA,
        )

    if cfg.blur is None:
        k = int(rng.integers(4, 10)) * 2 + 1  # 9..19 odd
    else:
        k = int(cfg.blur)
        if k < 0:
            k = 0
        if k % 2 == 0 and k > 0:
            k += 1
    if k > 1:
        over = cv2.GaussianBlur(over, (k, k), 0, borderType=cv2.BORDER_REPLICATE)

    over = np.clip(over, 0.0, 255.0)
    atten = 255.0 - over
    return np.clip(atten, 0.0, 255.0).astype(np.uint8)
