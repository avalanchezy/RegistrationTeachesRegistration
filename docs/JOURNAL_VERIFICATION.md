# Journal branch verification record

Verified on 2026-10-02 against the research implementation, before branch upload.
These checks establish software behavior on synthetic inputs. They do not
establish real-data registration performance, clinical reliability or acceptance
of a journal submission.

## Evidence

| Check | Result |
|---|---|
| Original repository test suite before changes | 126 passed |
| Final full suite, including 107 new tests | 233 passed |
| Historical runtime source checksums / source-release audit | Passed |
| Ten new command-line entrypoints, `--help` | Passed |
| All journal training and verification JSON configurations | Parsed/validated |
| CPU complete synthetic pipeline | Two effective optimizer updates; weights changed |
| CUDA + AMP complete synthetic pipeline | Two effective optimizer updates; weights changed; zero skipped updates |
| Resume versus uninterrupted training, same CPU environment | Exactly equal checkpoint parameters |
| Legacy support SSL and verified-field SSL student integration | Both trained from a supervised teacher checkpoint |
| Independent review | Reported blocking findings corrected and scoped fixes reviewed |

Test environment: Python 3.11, PyTorch 2.6.0+cu124, NumPy 2.2.6,
SciPy 1.15.3, pytest 8.4.2. CUDA testing used only a small synthetic volume;
128³ real-data memory and throughput remain unmeasured.

Commands used:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
python scripts/audit_source_release.py
git diff --check
python scripts/smoke_registration_field.py --output-dir /tmp/rtr-smoke-cpu --device cpu
python scripts/smoke_registration_field.py --output-dir /tmp/rtr-smoke-cuda --device cuda --amp
```

## Behaviors covered

- Oblique/anisotropic affine interpolation, physical axis order, first and
  higher-order coordinate gradients, actual two-step pose backpropagation.
- Old backbone logit compatibility; dense/implicit field querying; robust loss,
  rank ordering, finite zero-weight behavior; proper and reflected transform
  refinement with origin invariance and analytic convergence.
- Patient/hash/source separation, declaration of OOF upstream model exclusions,
  portable paths, manual/silver/pseudo roles, fixed area-weighted anchors.
- Consensus medoid, spatial uncertainty tails, held-out sectors, coverage,
  distinct competing basins, zero-supervision rejection, and no full-surface
  refinement contamination of the subsequent held-out refinement initialization.
- Resumable optimizer/scaler/RNG state, initialization provenance, fixed
  validation criterion, source-namespaced pseudo budgets and bounded candidate
  query batches.
- Legacy selected pseudo-mask conversion, accepted/rejected field pseudo export,
  failed predictions retained in evaluation, patient bootstrap and paired target
  consistency, parity failure, and separate manual/silver evidence.

The original all-zero normalized smoke image exposed pathological repeated
GroupNorm gradients in FP16. The final smoke uses a structured shell phantom
with HU variation and noise and asserts actual parameter updates. This fixes
the test fixture; no numerical model workaround was introduced.

## Remaining research and operational work

Real datasets, reconciled patient identities, upstream crown segmentation,
candidate generation and audits, threshold calibration, supervised and SSL
training, external baselines and independent reference assessment run on the
research team's data machine. Multi-GPU DDP and per-case resumable pseudo export
are not implemented. The training guide states these boundaries explicitly.
