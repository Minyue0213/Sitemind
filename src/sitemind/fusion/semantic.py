"""Convert fine-grained semantic IDs to a compact SiteMind risk layer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import yaml


def load_risk_values(path: Path | str) -> Dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or "risk_groups" not in document:
        raise ValueError("risk mapping YAML must contain risk_groups")
    values: Dict[str, float] = {}
    for group in document["risk_groups"]:
        risk = float(group["risk"])
        if not 0 <= risk <= 1:
            raise ValueError("risk values must lie in [0, 1]")
        for class_name in group["classes"]:
            normalized = str(class_name).strip().lower()
            if normalized in values:
                raise ValueError(f"semantic class appears twice: {normalized}")
            values[normalized] = risk
    return values


def semantic_ids_to_risk(
    label_ids: np.ndarray,
    id_to_name: Dict[int, str],
    class_risk: Dict[str, float],
    *,
    unknown_risk: float = 1.0,
) -> np.ndarray:
    labels = np.asarray(label_ids)
    if labels.ndim == 0:
        raise ValueError("label_ids must contain at least one dimension")
    result = np.full(labels.shape, unknown_risk, dtype=np.float32)
    for label_id in np.unique(labels):
        class_name = id_to_name.get(int(label_id), "").strip().lower()
        result[labels == label_id] = class_risk.get(class_name, unknown_risk)
    return result
