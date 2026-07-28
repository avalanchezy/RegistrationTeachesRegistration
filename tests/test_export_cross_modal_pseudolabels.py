import numpy as np

from scripts.export_cross_modal_pseudolabels import (
    agreement_confidence,
    selected_transform,
)


def test_agreement_confidence_is_bounded_and_monotonic() -> None:
    assert agreement_confidence(0.0, 0.5) == 1.0
    assert agreement_confidence(0.25, 0.5) > agreement_confidence(0.5, 0.5)
    assert 0.0 < agreement_confidence(0.5, 0.5) < 1.0


def test_selected_transform_uses_requested_teacher() -> None:
    threshold = {"transform": np.eye(4)}
    toothseg_transform = np.eye(4)
    toothseg_transform[:3, 3] = [2.0, 0.0, 0.0]
    toothseg = {"transform": toothseg_transform}
    center = np.zeros(3)
    assert np.allclose(
        selected_transform("threshold", threshold, toothseg, center), np.eye(4)
    )
    assert np.allclose(
        selected_transform("toothseg", threshold, toothseg, center),
        toothseg_transform,
    )
    assert np.allclose(
        selected_transform("midpoint", threshold, toothseg, center)[:3, 3],
        [1.0, 0.0, 0.0],
    )
