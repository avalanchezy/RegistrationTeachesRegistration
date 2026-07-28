from __future__ import annotations

import pytest

from scripts.run_submission_inference import (
    enhance_candidates_with_crown,
    global_crown_refinement_enabled,
    global_geometry_candidate_budget,
)


def test_global_geometry_budget_defaults_to_strict_training_budget() -> None:
    assert global_geometry_candidate_budget({}) == 30


def test_global_geometry_budget_reads_deployment_policy() -> None:
    assert global_geometry_candidate_budget({"global_geometry_candidate_budget": 48}) == 48


@pytest.mark.parametrize("value", [0, -1, 10001])
def test_global_geometry_budget_rejects_unsafe_values(value: int) -> None:
    with pytest.raises(ValueError, match="Invalid global geometry candidate budget"):
        global_geometry_candidate_budget({"global_geometry_candidate_budget": value})


def test_global_crown_refinement_preserves_legacy_default() -> None:
    assert global_crown_refinement_enabled({}) is True
    assert (
        global_crown_refinement_enabled({"global_include_crown_refinement": False})
        is False
    )


def test_global_crown_refinement_rejects_non_boolean_policy() -> None:
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        global_crown_refinement_enabled({"global_include_crown_refinement": 0})


def test_crown_enhancement_can_skip_unselected_refinement(tmp_path) -> None:
    candidate_run = tmp_path / "candidate_run"
    candidate_run.mkdir()
    work_dir = tmp_path / "work"
    consistency_dir = work_dir / "crown_consistency"
    consistency_dir.mkdir(parents=True)
    (consistency_dir / "summary.json").write_text("{}", encoding="utf-8")
    augmented_run = work_dir / "crown_augmented" / candidate_run.name
    augmented_run.mkdir(parents=True)

    result = enhance_candidates_with_crown(
        tmp_path / "manifest.csv",
        "001",
        [candidate_run],
        tmp_path / "crown_masks",
        work_dir,
        refine_consistency_outputs=True,
        include_refinement=False,
    )

    assert result == [augmented_run]
    assert not (work_dir / "crown_refinement").exists()
