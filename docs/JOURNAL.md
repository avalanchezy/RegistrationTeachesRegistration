# Journal research branch

This branch extends RTR with an executable research pipeline, not new validated
registration results. It preserves the historical challenge runtime source
hashes and adds independent modules in `task2reg/journal`.

Implemented: dense and multi-scale implicit unsigned fields, physical-coordinate
querying with higher-order gradients, candidate discrimination, centered
parity-preserving refinement, frozen-teacher spatial verification/pseudo targets,
legacy support-pseudo controls, portable manifests, patient/source exclusion,
resumable training and patient-level evaluation.

Start with the [training guide](JOURNAL_TRAINING_zh-CN.md),
[data/interface guide](JOURNAL_DATA_zh-CN.md), and
[research protocol](JOURNAL_RESEARCH_PLAN_zh-CN.md). These detailed guides are in
Chinese, as requested by the collaborating research team.

```bash
python -m pip install -r requirements.journal.txt
python -m pip install -e . --no-deps
python -m pytest -q
python scripts/smoke_registration_field.py --output-dir /tmp/rtr-smoke --device cpu
```

Choose a suitable PyTorch 2.6.0 CPU/CUDA wheel before installing requirements.
Use `journal_preflight.py` on the training machine. No private data, trained
weights or reference banks are distributed. Keep generated artifacts outside
this repository so the source-release audit remains meaningful.

The six core configurations separate rank supervision from no SSL, legacy
support SSL and verified field SSL. Optional Eikonal/two-step pose training is
a separate configuration. The fixed checkpoint criterion is patient-mean
selected fixed-anchor displacement D on validation candidates; it does not
change with rank warmup. D is not independent-landmark TRE. Experimental
benefits, external generalization, clinical references and KBS publication
readiness still require real-data studies.
