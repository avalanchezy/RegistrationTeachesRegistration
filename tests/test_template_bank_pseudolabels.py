from scripts.build_template_bank_pseudolabels import training_teacher_kind


def test_training_teacher_kind_preserves_geometry_teacher() -> None:
    assert training_teacher_kind("labeled") == "exact_ios_template_transfer"
    assert (
        training_teacher_kind("exact_ios_template_transfer")
        == "exact_ios_template_transfer"
    )
    assert training_teacher_kind("geometry_self_teacher") == "geometry_self_teacher"
    assert training_teacher_kind("learned_threshold_teacher") == "learned_threshold_teacher"
