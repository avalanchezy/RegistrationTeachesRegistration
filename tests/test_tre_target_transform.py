from __future__ import annotations

import numpy as np

from scripts.train_semisupervised_candidate_reranker import (
    inverse_tre_target,
    transform_tre_target,
)


def test_tre_target_transforms_round_trip() -> None:
    values = np.asarray([0.0, 0.25, 1.0, 5.0, 12.0])
    for mode in ("log1p", "sqrt", "identity"):
        recovered = inverse_tre_target(transform_tre_target(values, mode), mode)
        assert np.allclose(recovered, values)


def test_non_log_predictions_are_nonnegative() -> None:
    predictions = np.asarray([-2.0, 0.5, 2.0])
    assert np.all(inverse_tre_target(predictions, "identity") >= 0.0)
    assert np.all(inverse_tre_target(predictions, "sqrt") >= 0.0)
