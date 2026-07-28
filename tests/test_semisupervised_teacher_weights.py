from scripts.train_semisupervised_candidate_reranker import (
    pseudo_teacher_multiplier,
)


def test_geometry_teacher_has_an_independent_weight() -> None:
    assert pseudo_teacher_multiplier("exact_ios_template_transfer", 4.0, 0.5, 0.0) == 4.0
    assert pseudo_teacher_multiplier("geometry_self_teacher", 4.0, 0.5, 0.0) == 0.5
    assert pseudo_teacher_multiplier("learned_threshold_teacher", 4.0, 0.5, 0.0) == 0.0
    assert (
        pseudo_teacher_multiplier(
            "cross_modal_consensus_teacher", 4.0, 0.5, 0.0, 0.75
        )
        == 0.75
    )
