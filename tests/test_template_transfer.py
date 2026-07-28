from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from task2reg.template_transfer import match_template, transform_points


rng = np.random.default_rng(17)
query = rng.normal(size=(256, 3))
query_to_reference = np.eye(4)
query_to_reference[:3, :3] = Rotation.from_euler("xyz", (13, -21, 8), degrees=True).as_matrix()
query_to_reference[:3, 3] = (4.0, -7.0, 2.5)
reference = transform_points(query, query_to_reference)
reference_to_cbct = np.eye(4)
reference_to_cbct[:3, :3] = Rotation.from_euler("xyz", (-9, 3, 27), degrees=True).as_matrix()
reference_to_cbct[:3, 3] = (-11.0, 5.0, 32.0)
entry = {
    "case_id": "reference",
    "jaw": "upper",
    "cbct_sha256": "abc",
    "cbct_payload_sha256": "payload-abc",
    "vertex_count": len(query),
    "indices": np.arange(len(query), dtype=np.int32),
    "reference_points": reference.astype(np.float32),
    "reference_transform": reference_to_cbct,
}

match = match_template(query, "upper", "abc", [entry])
assert match is not None
expected = reference_to_cbct @ query_to_reference
assert np.allclose(match.transform, expected, atol=1e-6)
assert match.rms_mm < 1e-5
assert match.template_kind == "labeled"
assert match.full_p90_mm is None
assert match.cbct_match_kind == "raw"

payload_match = match_template(
    query,
    "upper",
    "recompressed-file-hash",
    [entry],
    allow_topology_fallback=False,
    cbct_payload_hash="payload-abc",
)
assert payload_match is not None
assert payload_match.reference_case_id == "reference"
assert payload_match.cbct_hash_match
assert payload_match.cbct_match_kind == "payload"

noisy_pseudo = {
    **entry,
    "case_id": "pseudo",
    "template_kind": "learned_threshold_teacher",
    "confidence": 0.95,
    "predicted_tre_mm": 0.5,
    "reference_transform": np.eye(4),
}
quality_match = match_template(query, "upper", "abc", [noisy_pseudo, entry])
assert quality_match is not None
assert quality_match.reference_case_id == "reference"

raw_pseudo = {
    **noisy_pseudo,
    "cbct_sha256": "raw-pseudo",
}
payload_quality_match = match_template(
    query,
    "upper",
    "raw-pseudo",
    [raw_pseudo, entry],
    allow_topology_fallback=False,
    cbct_payload_hash="payload-abc",
)
assert payload_quality_match is not None
assert payload_quality_match.reference_case_id == "reference"
assert payload_quality_match.cbct_match_kind == "payload"

unrelated = rng.normal(size=query.shape)
assert match_template(unrelated, "upper", "abc", [entry]) is None
assert match_template(query, "lower", "abc", [entry]) is None

print("template transfer tests passed")
