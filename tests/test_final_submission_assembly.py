from __future__ import annotations

import pytest

from scripts.assemble_final_submission import (
    SOURCE_PROVENANCE_ASSETS,
    crown_branch_weights,
)


def test_final_bundle_keeps_strict_selection_provenance() -> None:
    assert "selection_provenance.json" in SOURCE_PROVENANCE_ASSETS
    assert "strict_fair_selection.json" in SOURCE_PROVENANCE_ASSETS
    assert "strict_official_mode_selection.json" in SOURCE_PROVENANCE_ASSETS


def test_crown_branch_weights_build_complementary_class_weights() -> None:
    supervised, semisupervised = crown_branch_weights(
        {
            "upper_supervised_weight": 0.6,
            "lower_supervised_weight": 0.4,
        },
        supervised_weight=0.5,
    )

    assert supervised == {"background": 0.5, "upper": 0.6, "lower": 0.4}
    assert semisupervised == {"background": 0.5, "upper": 0.4, "lower": 0.6}


def test_crown_branch_weights_require_both_jaws() -> None:
    with pytest.raises(ValueError, match="Both upper and lower"):
        crown_branch_weights(
            {"upper_supervised_weight": 0.6}, supervised_weight=0.5
        )
