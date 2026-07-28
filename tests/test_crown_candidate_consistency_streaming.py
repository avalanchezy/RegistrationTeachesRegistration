from __future__ import annotations

import json

from scripts.evaluate_crown_candidate_consistency import (
    _load_candidate_group,
    _write_augmented_group,
)


def test_streamed_candidate_group_preserves_run_order_and_writes_by_source(tmp_path):
    runs = [tmp_path / "run_a", tmp_path / "run_b"]
    group_name = "001_upper"
    for index, run_dir in enumerate(runs):
        group_dir = run_dir / group_name
        group_dir.mkdir(parents=True)
        (group_dir / "candidates.json").write_text(
            json.dumps(
                [
                    {
                        "candidate_id": index,
                        "crown_symmetric_trim20_mm": 0.5 + index,
                    }
                ]
            ),
            encoding="utf-8",
        )
    (runs[0] / group_name / "result.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )

    candidates = _load_candidate_group(runs, ("001", "upper"))

    assert [candidate["candidate_id"] for candidate in candidates] == [0, 1]
    assert [candidate["candidate_run"] for candidate in candidates] == [
        str(runs[0]),
        str(runs[1]),
    ]
    assert all(candidate["candidate_jaw"] == "upper" for candidate in candidates)

    output_root = tmp_path / "augmented"
    written = _write_augmented_group(
        candidates, ("001", "upper"), runs, output_root
    )

    assert written == {str(runs[0]): 1, str(runs[1]): 1}
    for index, run_dir in enumerate(runs):
        output = json.loads(
            (output_root / run_dir.name / group_name / "candidates.json").read_text(
                encoding="utf-8"
            )
        )
        assert output[0]["candidate_id"] == index
        assert "candidate_run" not in output[0]
        assert "candidate_jaw" not in output[0]
    assert (output_root / "run_a" / group_name / "result.json").exists()
    assert not (output_root / "run_b" / group_name / "result.json").exists()
