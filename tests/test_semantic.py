from pathlib import Path

import numpy as np

from sitemind.fusion import load_risk_values, semantic_ids_to_risk


def test_semantic_name_mapping_and_unknown_default(tmp_path: Path):
    config = tmp_path / "risk.yaml"
    config.write_text(
        "risk_groups:\n"
        "  - name: low\n"
        "    risk: 0.1\n"
        "    classes: [soil]\n"
        "  - name: blocked\n"
        "    risk: 1.0\n"
        "    classes: [person]\n",
        encoding="utf-8",
    )
    risks = load_risk_values(config)
    labels = np.array([[1, 2, 99]])
    result = semantic_ids_to_risk(labels, {1: "soil", 2: "person"}, risks)
    np.testing.assert_allclose(result, [[0.1, 1.0, 1.0]])
    point_result = semantic_ids_to_risk(np.array([1, 2, 99]), {1: "soil", 2: "person"}, risks)
    np.testing.assert_allclose(point_result, [0.1, 1.0, 1.0])
