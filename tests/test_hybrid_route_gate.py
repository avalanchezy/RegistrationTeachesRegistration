from scripts.tune_hybrid_route_gate import selected_error, threshold_selected


def test_route_gate_requires_confident_threshold_advantage() -> None:
    row = {
        "toothseg_predicted_tre_mm": 4.0,
        "threshold_predicted_tre_mm": 2.0,
        "toothseg_tre_mm": 5.0,
        "threshold_tre_mm": 1.0,
    }
    assert threshold_selected(row, cap=3.0, advantage=1.0)
    assert selected_error(row, cap=3.0, advantage=1.0) == 1.0
    assert not threshold_selected(row, cap=1.5, advantage=1.0)
    assert not threshold_selected(row, cap=3.0, advantage=2.5)
