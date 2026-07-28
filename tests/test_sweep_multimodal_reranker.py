from scripts.sweep_multimodal_reranker import exact_fallback_errors


def test_exact_fallback_replaces_only_matched_jaws() -> None:
    rows = [
        {"case_id": "a", "jaw": "upper", "mean_tre_mm": 4.0},
        {"case_id": "b", "jaw": "lower", "mean_tre_mm": 2.0},
    ]
    values = exact_fallback_errors(rows, {("a", "upper"): 0.01})
    assert values == [0.01, 2.0]
