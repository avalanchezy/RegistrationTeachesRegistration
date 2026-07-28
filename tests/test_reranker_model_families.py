from types import SimpleNamespace

import numpy as np

from scripts.sweep_pairwise_multimodal_reranker import build_classifier
from scripts.train_semisupervised_candidate_reranker import build_estimator


def regression_args(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_type=model_type,
        cv_trees=12,
        tree_criterion="squared_error",
        tree_max_depth=0,
        min_samples_leaf=2,
        max_features=0.5,
        n_jobs=1,
        hgb_ensemble_leaf_nodes=(),
        hgb_max_leaf_nodes=7,
        hgb_learning_rate=0.1,
        hgb_l2=1.0,
        hgb_early_stopping=False,
    )


def test_all_regression_families_fit_and_predict() -> None:
    rng = np.random.default_rng(20260715)
    features = rng.normal(size=(48, 9))
    target = np.square(features[:, 0]) + 0.25 * features[:, 1]

    for model_type in ("extra_trees", "random_forest", "hist_gradient_boosting"):
        estimator = build_estimator(regression_args(model_type), seed=17)
        estimator.fit(features, target)
        prediction = np.asarray(estimator.predict(features[:8]), dtype=np.float64)
        assert prediction.shape == (8,)
        assert np.isfinite(prediction).all()


def test_all_pairwise_families_fit_and_predict_probabilities() -> None:
    rng = np.random.default_rng(20260715)
    features = rng.normal(size=(48, 9))
    labels = np.asarray([0, 1] * 24, dtype=np.int64)

    for model_type in ("extra_trees", "random_forest"):
        classifier = build_classifier(
            {
                "model_type": model_type,
                "criterion": "gini",
                "min_samples_leaf": 2,
                "max_features": 0.5,
            },
            trees=12,
            n_jobs=1,
            seed=17,
        )
        classifier.fit(features, labels)
        probability = np.asarray(classifier.predict_proba(features[:8]))
        assert probability.shape == (8, 2)
        assert np.isfinite(probability).all()
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
