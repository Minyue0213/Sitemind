from pathlib import Path

import numpy as np
from PIL import Image

from sitemind.data import discover_frame_pairs, load_label_mapping


def test_mapping_and_timestamp_pairing(tmp_path: Path):
    mapping_file = tmp_path / "goose_label_mapping.csv"
    mapping_file.write_text("class_name,label_key\nundefined,0\nsoil,4\n", encoding="utf-8")
    assert load_label_mapping(mapping_file) == {0: "undefined", 4: "soil"}

    image_dir = tmp_path / "images" / "val" / "sequence_a"
    label_dir = tmp_path / "labels" / "val" / "sequence_a"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
        image_dir / "scene_0001_1725353422795_camera_left.png"
    )
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(
        label_dir / "scene_0001_1725353422795_labelids.png"
    )
    pairs = discover_frame_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].timestamp == "1725353422795"


def test_incomplete_pairing_is_rejected(tmp_path: Path):
    image_dir = tmp_path / "images" / "val" / "sequence_a"
    label_dir = tmp_path / "labels" / "val" / "sequence_a"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
        image_dir / "scene_1725353422795_camera_left.png"
    )
    try:
        discover_frame_pairs(tmp_path)
    except ValueError as error:
        assert "1 images lack labels" in str(error)
    else:
        raise AssertionError("incomplete data should not be accepted")
