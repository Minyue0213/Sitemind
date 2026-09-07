import numpy as np

from sitemind.fusion import fuse_risk_maps


def test_conservative_fusion_keeps_highest_risk():
    semantic = np.array([[0.1, 0.8], [0.2, np.nan]], dtype=np.float32)
    geometry = np.array([[0.4, 0.2], [0.9, 0.1]], dtype=np.float32)
    fused = fuse_risk_maps(semantic, geometry)
    np.testing.assert_allclose(fused, [[0.4, 0.8], [0.9, 1.0]])


def test_uncertainty_can_raise_but_not_lower_risk():
    semantic = np.array([[0.1, 0.8]], dtype=np.float32)
    geometry = np.array([[0.2, 0.3]], dtype=np.float32)
    uncertainty = np.array([[0.9, 0.1]], dtype=np.float32)
    fused = fuse_risk_maps(
        semantic,
        geometry,
        uncertainty,
        uncertainty_weight=0.5,
    )
    np.testing.assert_allclose(fused, [[0.45, 0.8]])
