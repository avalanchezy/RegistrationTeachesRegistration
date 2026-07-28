from __future__ import annotations

from types import SimpleNamespace

from scripts.fit_final_multiseed_ensemble import resolved_pairwise_selection


def test_pairwise_selection_uses_its_tuned_configuration() -> None:
    args = SimpleNamespace(
        top_unsupervised=36,
        top_oracle=4,
        model_scope="jaw",
        pairwise_top_unsupervised=60,
        pairwise_top_oracle=8,
        pairwise_model_scope="shared",
    )

    assert resolved_pairwise_selection(args) == {
        "top_unsupervised": 60,
        "top_oracle": 8,
        "model_scope": "shared",
    }


def test_pairwise_selection_preserves_legacy_fallbacks() -> None:
    args = SimpleNamespace(
        top_unsupervised=20,
        top_oracle=8,
        model_scope="jaw",
        pairwise_top_unsupervised=0,
        pairwise_top_oracle=0,
        pairwise_model_scope=None,
    )

    assert resolved_pairwise_selection(args) == {
        "top_unsupervised": 20,
        "top_oracle": 8,
        "model_scope": "jaw",
    }
