# KBS registration field implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or
> superpowers:executing-plans for the independent tasks below.

**Goal:** Deliver a tested, portable research branch for the KBS extension.

**Architecture:** Add an isolated journal package and command-line pipeline.
Reuse existing volume preprocessing and backbone without changing frozen
challenge runtime files. Train and evaluate against explicit patient splits.

**Tech Stack:** Python 3.10+, PyTorch 2.5–2.8, NumPy, SciPy, existing NIfTI/mesh stack.

**Spec:** `docs/superpowers/specs/2026-10-02-kbs-registration-field-design.md`.

## Global constraints

- No real data or long training on this machine; synthetic verification only.
- Preserve source hashes of the submitted runtime and its APIs.
- World coordinates are millimetres; full affines and improper transforms work.
- Reject patient/source leakage; silver reference is not manual ground truth.
- Real improvements and journal readiness require subsequent experiments.
- New manifest schema and NPZ keys are as specified in the design.

## Review focus

- Oblique affine and tensor axis reversal: analytic interpolation tests, Task 1.
- Higher-order gradients at zero twist: finite double-backward tests, Task 1.
- Patient duplicate across source names: explicit leakage rejection, Task 2.
- Duplicated best basin hides ambiguity: rejection test, Task 3.
- Resume after relocation/config changes: contract and smoke checks, Task 4.

### Task 1: Continuous field and optimization primitives

Files: `task2reg/journal/field.py`, `tests/test_registration_field.py`.
Interfaces: `RegistrationField(base_channels=8, mode='implicit', truncation_mm=8)`;
`encode(image)` returns feature context including `logits`; `query(context,
points_world[B,N,3], affine[B,4,4], jaw[B]) -> distances[B,N]` (jaw 0/1).
Functions for physical ROI distance, robust energy, weighted Huber, pair rank
loss, centered SE(3), pose displacement, and differentiable refinement.

- [x] Write failing tests for affine values/gradients, ranking, parity, refinement.
- [x] Implement dense/implicit heads, interpolation, losses and refinement.
- [x] Verify first/second derivatives and unchanged old checkpoint compatibility.

### Task 2: Data adapters and strict research protocol

Files: `task2reg/journal/data.py`, `scripts/prepare_registration_field_data.py`,
`tests/test_journal_data.py`.
Interfaces: `load_journal_manifest(path, data_root=None)` returns validated records;
`validate_protocol(records)`; `load_case(record)` returns checked NPZ arrays;
`sample_queries(case, count, rng, truncation_mm=8)` returns points/targets/weights.

- [x] Write failing tests for group isolation, relocation, sampling and old import.
- [x] Implement adapter from existing CSV + prepared NPZ + metadata/candidates.
- [x] Test manual/pseudo/reference roles, OOF provenance, and deterministic queries.

### Task 3: Verified pseudo supervision and evaluation

Files: `task2reg/journal/verification.py`, `task2reg/journal/evaluation.py`,
`scripts/evaluate_journal_registration.py`, corresponding tests.
Interfaces: pure NumPy medoid/uncertainty/verification gates and diagnostics.
Gate inputs include transforms, points, energies, heldout distances, sector IDs,
outside masks and distinct basin candidates. Outputs reasons and point weights.

- [x] Write failing tests for consensus, holdout/sector rejection and ambiguity.
- [x] Implement auditable fail-closed gates and grouped evaluation/reporting.
- [x] Verify bootstrap patient unit and missing/nonfinite data rejection.

### Task 4: Train, infer, pseudo export, reproducibility and documentation

Files: `task2reg/journal/engine.py`, training/inference/pseudo/smoke scripts,
`configs/journal/`, `docs/JOURNAL_*.md`, README links.
Consumes Tasks 1–3 through their explicit interfaces.

- [x] Add failing integration tests for synthetic training/checkpoint resume.
- [x] Implement config-driven stages, complete checkpoints, inference and export.
- [x] Provide synthetic smoke, six-group configs and portable environment setup.
- [x] Document current-interface conversion and full remote training sequence.
- [x] Run focused and full tests, release audit, independent branch review.
- [ ] Commit and push `research/kbs-registration-field`; verify remote SHA.

## Execution decisions

The user explicitly requested implementation, documentation and a branch push;
proceed within that authorization. Use parallel workers for independent modules
and one integration owner. Formal dataset and GPU choices remain configurable.
