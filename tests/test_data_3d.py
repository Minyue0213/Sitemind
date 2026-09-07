from pathlib import Path

import numpy as np

from sitemind.data import (
    discover_point_cloud_pairs,
    load_point_cloud,
    load_point_labels,
)


def test_point_cloud_pair_and_semantic_instance_unpacking(tmp_path: Path):
    scan_dir = tmp_path / "lidar" / "val" / "sequence_a"
    label_dir = tmp_path / "labels" / "val" / "sequence_a"
    scan_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    stem = "scene_0001_1725353422795"
    scan_path = scan_dir / f"{stem}_vls128.bin"
    label_path = label_dir / f"{stem}_goose.label"
    points = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], dtype=np.float32)
    packed = np.array([(3 << 16) | 7, (9 << 16) | 12], dtype=np.uint32)
    points.tofile(scan_path)
    packed.tofile(label_path)

    pairs = discover_point_cloud_pairs(tmp_path)
    assert len(pairs) == 1
    np.testing.assert_allclose(load_point_cloud(scan_path), points)
    semantic, instance = load_point_labels(label_path, expected_points=2)
    np.testing.assert_array_equal(semantic, [7, 12])
    np.testing.assert_array_equal(instance, [3, 9])


def test_point_cloud_label_length_mismatch_is_rejected(tmp_path: Path):
    label_path = tmp_path / "bad.label"
    np.array([1], dtype=np.uint32).tofile(label_path)
    try:
        load_point_labels(label_path, expected_points=2)
    except ValueError as error:
        assert "length mismatch" in str(error)
    else:
        raise AssertionError("mismatched scan and label lengths should fail")
