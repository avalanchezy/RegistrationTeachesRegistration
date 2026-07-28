from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_nested_crown_fusion_selection import cbct_groups


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("split", "case_id", "jaw", "cbct_path")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_cbct_groups_keep_duplicate_payload_cases_together(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    cache = tmp_path / "hashes.json"
    rows = []
    for case_id, cbct_path in (("001", "a.nii.gz"), ("002", "b.nii.gz")):
        for jaw in ("upper", "lower"):
            rows.append(
                {
                    "split": "Train-Labeled",
                    "case_id": case_id,
                    "jaw": jaw,
                    "cbct_path": cbct_path,
                }
            )
    write_manifest(manifest, rows)
    cache.write_text(
        json.dumps(
            {
                "a.nii.gz": {"sha256": "ABC"},
                "b.nii.gz": {"sha256": "abc"},
            }
        ),
        encoding="utf-8",
    )

    assert cbct_groups(manifest, cache, ["001", "002"]) == {
        "abc": ["001", "002"]
    }


def test_cbct_groups_reject_missing_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    cache = tmp_path / "hashes.json"
    write_manifest(
        manifest,
        [
            {
                "split": "Train-Labeled",
                "case_id": "001",
                "jaw": "upper",
                "cbct_path": "missing.nii.gz",
            }
        ],
    )
    cache.write_text("{}", encoding="utf-8")

    with pytest.raises(KeyError, match="Missing CBCT hash"):
        cbct_groups(manifest, cache, ["001"])
