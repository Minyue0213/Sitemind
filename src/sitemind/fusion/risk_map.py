"""Small, inspectable semantic-geometric fusion functions."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _as_risk(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    finite = array[np.isfinite(array)]
    if finite.size and (finite.min() < 0 or finite.max() > 1):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return np.nan_to_num(array, nan=1.0, posinf=1.0, neginf=1.0)


def fuse_risk_maps(
    semantic_risk: np.ndarray,
    geometry_risk: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    uncertainty_weight: float = 0.0,
) -> np.ndarray:
    """Fuse risk conservatively so one high-risk modality is not averaged away."""

    semantic = _as_risk("semantic_risk", semantic_risk)
    geometry = _as_risk("geometry_risk", geometry_risk)
    if semantic.shape != geometry.shape:
        raise ValueError("semantic_risk and geometry_risk must have the same shape")
    fused = np.maximum(semantic, geometry)
    if uncertainty is not None:
        uncertain = _as_risk("uncertainty", uncertainty)
        if uncertain.shape != fused.shape:
            raise ValueError("uncertainty must have the same shape as risk maps")
        if not 0 <= uncertainty_weight <= 1:
            raise ValueError("uncertainty_weight must lie in [0, 1]")
        fused = np.maximum(fused, uncertainty_weight * uncertain)
    return np.clip(fused, 0.0, 1.0)
