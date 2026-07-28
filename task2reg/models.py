from __future__ import annotations

import numpy as np


class MeanRegressor:
    """Fit several regressors and average their scalar predictions."""

    def __init__(self, estimators: list[object]):
        if not estimators:
            raise ValueError("MeanRegressor needs at least one estimator")
        self.estimators = estimators

    def fit(self, x, y, sample_weight=None):
        for estimator in self.estimators:
            estimator.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x) -> np.ndarray:
        predictions = np.stack([estimator.predict(x) for estimator in self.estimators])
        return predictions.mean(axis=0)
