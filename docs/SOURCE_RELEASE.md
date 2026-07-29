# Source-release checklist

This repository is the code release associated with the `avalanchezy` STSR
2026 Task 2 submission.

## Included

- training and inference source
- exact final hyperparameters and deployment policies
- preprocessing and postprocessing
- grouped-validation and official-metric evaluation code
- Docker definition and output validation
- 126 regression tests
- model-asset and runtime-source SHA256 manifests

## Intentionally excluded

- organizer-provided CBCT and IOS data
- ground-truth and predicted transforms
- generated weak labels and pseudo labels
- neural checkpoints and tree-model binaries
- template banks containing sampled organizer geometry
- Docker image archives and internal evaluation logs

## Before a release tag

```bash
python -m pytest -q
python scripts/audit_source_release.py
python scripts/package_source_release.py \
  --output ../RegistrationTeachesRegistration-source-v1.0.3.zip
```

Confirm that the archive expands to one top-level project directory and that
its SHA256 matches the value reported by the packaging script.
