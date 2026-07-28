from scripts.build_surface_template_bank import (
    eligible_base_entry,
    is_geometry_teacher_method as is_surface_geometry_teacher_method,
)
from scripts.extend_template_bank_from_inference import (
    is_geometry_teacher_method as is_exact_geometry_teacher_method,
)


def test_base_roi_teacher_requires_explicit_opt_in() -> None:
    entry = {"predicted_tre_mm": 0.9, "roi_used": True}

    assert not eligible_base_entry(entry, 1.5, include_roi_teachers=False)
    assert eligible_base_entry(entry, 1.5, include_roi_teachers=True)


def test_base_teacher_still_respects_predicted_tre_gate() -> None:
    entry = {"predicted_tre_mm": 1.6, "roi_used": False}

    assert not eligible_base_entry(entry, 1.5, include_roi_teachers=False)


def test_base_roi_teacher_can_use_a_stricter_gate() -> None:
    entry = {"predicted_tre_mm": 1.3, "roi_used": True}

    assert eligible_base_entry(entry, 1.5, True)
    assert not eligible_base_entry(entry, 1.5, True, max_roi_predicted_tre_mm=1.2)


def test_enhanced_geometry_audit_rows_are_valid_teachers() -> None:
    for method in ("geometry", "geometry_crown_ensemble"):
        assert is_surface_geometry_teacher_method(method)
        assert is_exact_geometry_teacher_method(method)
    for method in ("emergency", "exact_template", "surface_template", None):
        assert not is_surface_geometry_teacher_method(method)
        assert not is_exact_geometry_teacher_method(method)
