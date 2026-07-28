from scripts.tune_candidate_reranker import stratified_case_folds


def _rows(sign: int) -> list[dict]:
    return [{"ground_truth_chirality": sign}]


def test_duplicate_cbct_cases_stay_in_same_fold() -> None:
    groups = {
        ("001", "upper"): _rows(1),
        ("001", "lower"): _rows(1),
        ("101", "upper"): _rows(1),
        ("101", "lower"): _rows(1),
        ("002", "upper"): _rows(-1),
        ("003", "upper"): _rows(-1),
    }
    case_groups = {
        "001": "shared-payload",
        "101": "shared-payload",
        "002": "payload-2",
        "003": "payload-3",
    }

    folds = stratified_case_folds(groups, folds=3, seed=7, group_by_case=case_groups)

    duplicate_folds = [index for index, fold in enumerate(folds) if "001" in fold]
    assert len(duplicate_folds) == 1
    assert "101" in folds[duplicate_folds[0]]
    assert set().union(*folds) == {"001", "101", "002", "003"}
