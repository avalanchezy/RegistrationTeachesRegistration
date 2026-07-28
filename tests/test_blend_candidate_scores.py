from scripts.blend_candidate_scores import blend_tables


def test_blend_tables_combines_within_group_ranks() -> None:
    first = {
        ("reg", "001", "upper"): {
            0: {"ensemble_score": "1", "mean_tre_mm": "2", "candidate_run": "a"},
            1: {"ensemble_score": "2", "mean_tre_mm": "1", "candidate_run": "b"},
        }
    }
    second = {
        ("pair", "001", "upper"): {
            0: {"ensemble_score": "2", "mean_tre_mm": "2", "candidate_run": "a"},
            1: {"ensemble_score": "1", "mean_tre_mm": "1", "candidate_run": "b"},
        }
    }
    rows = blend_tables(first, second, ["reg"], ["pair"], [0.0, 0.5, 1.0])
    by_method = {}
    for row in rows:
        by_method.setdefault(row["ensemble_method"], {})[row["candidate_index"]] = row[
            "ensemble_score"
        ]
    assert by_method["reg__pair__a0p000"][1] < by_method["reg__pair__a0p000"][0]
    assert by_method["reg__pair__a1p000"][0] < by_method["reg__pair__a1p000"][1]
    assert by_method["reg__pair__a0p500"][0] == by_method["reg__pair__a0p500"][1]
