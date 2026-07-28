import gzip
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from scripts.train_semisupervised_candidate_reranker import leakage_safe_pseudo_keys
from scripts.train_semisupervised_candidate_reranker import cbct_groups_by_case


def test_cbct_payload_grouping_ignores_gzip_repacking_and_caches() -> None:
    payload = b"synthetic-nifti-payload" * 128
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        first = root / "first.nii.gz"
        second = root / "second.nii.gz"
        with gzip.GzipFile(first, "wb", mtime=1) as stream:
            stream.write(payload)
        with gzip.GzipFile(second, "wb", mtime=2) as stream:
            stream.write(payload)
        cache_path = root / "hashes.json"
        records = [
            SimpleNamespace(case_id="a", cbct_path=str(first)),
            SimpleNamespace(case_id="b", cbct_path=str(second)),
        ]
        groups = cbct_groups_by_case(records, {"a", "b"}, cache_path)
        cached = json.loads(cache_path.read_text(encoding="utf-8"))

    assert groups["a"] == groups["b"]
    assert len(cached) == 2


def test_pseudo_target_with_duplicate_cbct_is_excluded() -> None:
    pseudo = {
        ("u1", "upper"): {
            "teacher": "geometry_self_teacher",
            "source_labeled_case_id": "",
        },
        ("u2", "upper"): {
            "teacher": "geometry_self_teacher",
            "source_labeled_case_id": "",
        },
    }
    groups = {"val": "scan-a", "u1": "scan-a", "u2": "scan-b"}
    allowed = leakage_safe_pseudo_keys(
        pseudo, {"val"}, groups, include_threshold=False, include_geometry=True
    )
    assert ("u1", "upper") not in allowed
    assert ("u2", "upper") in allowed


def test_exact_teacher_with_duplicate_cbct_is_excluded() -> None:
    pseudo = {
        ("u1", "lower"): {
            "teacher": "exact_ios_template_transfer",
            "source_labeled_case_id": "teacher-copy",
        },
        ("u2", "lower"): {
            "teacher": "exact_ios_template_transfer",
            "source_labeled_case_id": "teacher-safe",
        },
    }
    groups = {
        "val": "scan-a",
        "u1": "scan-b",
        "u2": "scan-c",
        "teacher-copy": "scan-a",
        "teacher-safe": "scan-d",
    }
    allowed = leakage_safe_pseudo_keys(
        pseudo, {"val"}, groups, include_threshold=False, include_geometry=False
    )
    assert ("u1", "lower") not in allowed
    assert ("u2", "lower") in allowed


def test_cross_modal_teacher_requires_explicit_cv_opt_in() -> None:
    pseudo = {
        ("u1", "upper"): {
            "teacher": "cross_modal_consensus_teacher",
            "source_labeled_case_id": "",
        }
    }
    groups = {"val": "scan-a", "u1": "scan-b"}
    assert not leakage_safe_pseudo_keys(
        pseudo,
        {"val"},
        groups,
        include_threshold=False,
        include_geometry=False,
    )
    assert leakage_safe_pseudo_keys(
        pseudo,
        {"val"},
        groups,
        include_threshold=False,
        include_geometry=False,
        include_cross_modal=True,
    ) == {("u1", "upper")}
