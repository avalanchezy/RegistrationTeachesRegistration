import numpy as np

from scripts.pseudolabels_to_candidate_run import candidate_from_label


def test_candidate_from_label_maps_full_centroid() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [3.0, -2.0, 1.0]
    row = candidate_from_label(
        {
            "transform": transform.tolist(),
            "predicted_tre_mm": 0.7,
            "teacher": "geometry_self_teacher",
            "source_teacher_case_id": "101",
        },
        np.asarray([1.0, 2.0, 3.0]),
    )
    assert row["selection_score_mm"] == 0.7
    assert row["chirality"] == 1
    assert np.allclose(row["predicted_full_centroid"], [4.0, 0.0, 4.0])
