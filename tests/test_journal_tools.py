import json
from pathlib import Path
import subprocess
import sys

from test_journal_engine import synthetic_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_merge_rebases_paths_and_preflight_checks_payloads(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    output = tmp_path / "merged" / "manifest.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/merge_journal_manifests.py"),
                             "--inputs", str(manifest), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = json.loads(output.read_text())["records"]
    assert all((output.parent / r["npz_path"]).exists() for r in rows)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/journal_preflight.py"),
                             "--manifest", str(output), "--check-data", "--device", "cpu"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["checked_records"] == 3


def test_merge_rejects_duplicate_experiments(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/merge_journal_manifests.py"),
                             "--inputs", str(manifest), str(manifest), "--output", str(tmp_path / "bad.json")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Duplicate" in result.stderr
