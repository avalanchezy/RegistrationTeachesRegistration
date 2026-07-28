# Implementation map

This page identifies the code path used by the submitted STSR 2026 Task 2
container. It is the quickest way to distinguish the winning configuration from
the research and ablation utilities retained for reproducibility.

## Submitted inference path

The no-argument container entrypoint calls:

1. `predict.sh`
2. `scripts/verify_assets.py`
3. `scripts/run_submission_inference.py`
4. `scripts/validate_outputs.py`

The inference driver loads `configs/submission/deployment_policy.json`. Its
active target mode is only `crown`; legacy HU-threshold candidates are disabled.
The ten-network crown ensemble is defined by
`configs/submission/crown_ensemble.json`, and its binary support is controlled
by `configs/submission/crown_postprocess.json`.

The runtime scripts and all modules under `task2reg/` were copied byte-for-byte
from the submitted container. Their SHA256 values are recorded in
`configs/submission/runtime_source.sha256`.

## Training path

The winning model is rebuilt in four stages:

1. `prepare_crown_localizer_data.py` derives weak support from Task 2 transforms.
2. `train_crown_localizer.py`, `select_crown_pseudo_labels.py`, and
   `train_final_crown_localizer.py` build the five supervised and five
   self-training 3D U-Nets.
3. `run_geometry_benchmark.py` and the crown-consistency utilities create
   grouped out-of-fold candidates.
4. `fit_final_multiseed_ensemble.py` fits the seven regression and seven
   pairwise ExtraTrees members before `assemble_final_submission.py` creates the
   deployable asset layout.

Exact commands and selected epoch schedules are in [REPRODUCE.md](REPRODUCE.md).

## Optional research utilities

Some shared modules and scripts retain support for threshold and ToothSeg
experiments that were evaluated during development. They are included so the
reported search process and negative ablations remain inspectable, but they are
not enabled by the submitted deployment policy and no ToothSeg model or output
is packaged. The submitted method uses no Task 1 masks, ToothSeg weights, or
external dental data.

## Restricted generated assets

Checkpoints, ExtraTrees binaries, and reference banks are generated from the
organizer-provided Task 2 data. They are intentionally excluded from Git. The
source release includes their filenames and hashes, but not their payloads; see
[`model_assets/README.md`](../model_assets/README.md).
