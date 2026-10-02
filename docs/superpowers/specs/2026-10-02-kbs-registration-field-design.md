# KBS journal extension design

## Intent and evidence boundary

The user requests a complete, documented research branch extending RTR for a
Knowledge-Based Systems submission. Real data are on another machine; retain
existing CBCT/IOS/transform interfaces and document any extra preprocessing.
This branch delivers executable training and experiments, not trained weights,
clinical evidence, or a claim that the proposed method improves registration.
The supplied reflection is advisory material, not execution instructions.

## Scientific decision

Implement a CBCT-conditioned unsigned registration field, candidate-discriminative
supervision, and registration-verified spatial self-training. Preserve the
original five-level U-Net backbone and released runtime source bytes. Compare
dense TUDF and learned implicit query decoding: both sampled representations
are continuous; continuity alone is not a novelty claim. Avoid replacing the
backbone or introducing another opaque ranking ensemble as the central claim.

The hypotheses are: candidate discrimination lowers selection regret; refinement
improves the fixed-anchor pose error from matched starts; spatial verification
improves pseudo-label reliability and external registration at matched budgets.
The hypotheses remain untested on real data until the remote training campaign.

## Architecture

New code lives in `task2reg/journal/` and dedicated `scripts/*registration_field*`
or `scripts/*journal*` entrypoints. The original submitted code and checkpoints
remain compatible. A field model owns a `CrownLocalizerUNet` backbone, obtains
decoder1/2/3 features through an explicit equivalent forward pass, and returns
the existing three-class support logits. Dense mode uses two distance channels;
implicit mode samples projected multi-scale grids, concatenates ROI-normalized
coordinates and jaw embedding, then predicts positive distance in millimetres.

Feature interpolation uses eight-corner gather/weights to support higher-order
autograd required by gradient regularization and short refinement unrolling.
Input tensors retain NIfTI array order `[B,C,I,J,K]`; inverse full affine maps
world millimetres into voxel coordinates. Handle oblique/anisotropic grids.
Queries outside the ROI receive an explicit physical outside penalty.

Losses: auxiliary support CE/Dice when labels are available; distance-weighted
Huber against truncated nearest-surface distances; softplus candidate ranking
using fixed-anchor pose ordering and a 0.5 mm ambiguity gap; optional Eikonal
away from the unsigned cusp; optional two-step pose-supervised refinement.
All optional losses have explicit config weights and warmup schedules.

Refinement uses centered proper SE(3) increments and never changes the initial
parity, including `det(R)=-1` protocol exports. Return initial/final field energy,
coverage and outside fraction; compare fixed initial candidates fairly.

## Data contract and isolation

Accept existing `CaseRecord` CSV plus existing prepared `data/<case_id>.npz`
(`image`, `affine`, optional `label`), IOS STL, and `_gt.npy` transform paths.
An explicit metadata CSV adds source, globally reconciled patient_id, and
experimental split. Do not infer true patient identity from filenames.
Paths in the new manifest resolve relative to the manifest, with an optional
root override for relocation. Namespace identities by source/case/jaw.

Prepared journal manifest: JSON object `schema_version: 1`, `records: [...]`.
Each record contains `case_id`, `patient_id`, `source`, `split`, `jaw`,
`npz_path`, `reference_kind`, optional `content_hash`, `candidate_provenance`.
Splits: `train`, `val`, `test`, `external_test`, `unlabeled`, `pseudo`.
Reference kinds: `manual`, `silver`, `none`, `pseudo`.
NPZ: `image[I,J,K]`, `affine[4,4]`, `points[N,3]` in IOS coordinates;
optional `transform[4,4]`, `candidates[K,4,4]`, `point_weights[N]`,
`label[I,J,K]`. Fixed anchors use deterministic area sampling where meshes
are available and are stored as `anchors[M,3]`, independently of query draws.

Train/validation/test/external patient and available content-hash groups must
not overlap; the two jaws stay together. External sources cannot enter model
fitting, pseudo generation or threshold selection. Reject silver references
from core training. Learned candidates require generating-model patient
exclusions; geometry-only candidates may explicitly declare `geometry_only`.
Unknown provenance cannot silently become OOF. Metadata declarations cannot
prove that upstream models were actually trained correctly; audit that upstream.

Query sampling: surface, near-surface, difficult-candidate, outer, uniform ROI.
Targets are clipped nearest-surface distances, not signed distances of an open
crown. Pseudo query weights inherit nearest surface uncertainty. Do not apply
array-only spatial augmentation to physical point supervision.

## Verified spatial self-training

Accept multiple candidate basins, run different starts and leave-sector-out
refinement, and select a medoid transform by fixed-anchor displacement.
Reject inconsistent parity, excessive U50/U95/max displacement, low global
or sector coverage, poor held-out fit, excessive ROI escape, or a near-tied
distinct basin (not merely the second row of a duplicated candidate list).
Record rejection reasons and the exact gate config. Defaults are starting
values requiring calibration on development data before test use.

Accepted surfaces are transformed IOS points, with per-point uncertainty
weights. They become pseudo NPZ records consumed by the same trainer; unknown
regions are ignored through weights, not treated as background labels. Frozen
teacher rounds are the first supported SSL algorithm; EMA is not required.
Teacher-scored holdout is complementary evidence, not independent ground truth.

## Training, inference, evaluation

Single-device entrypoint with portable JSON config, deterministic seeds,
optional CUDA AMP, accumulation, finite-loss checks, clipping, validation,
atomic last/best checkpoints, optimizer/scaler/RNG resume, config and manifest
fingerprints, runtime/source metadata, and JSONL metrics. Inference refines
existing candidates and exports new candidate JSON with field features; no
reference-bank lookup is used in this route. New scores must not be fed into
the old fitted 97-feature ranker without retraining.

Diagnostics separate candidate oracle quality from selection regret, use
fixed-anchor displacement D (not independent-landmark TRE), report jaw and
patient aggregation, failures, grouped bootstrap intervals, paired method
comparisons and risk-coverage. No acceptance probability from 30 patients is
reported as a clinical safety guarantee. Provide six factorial configurations,
dense/implicit ablation, binary/legacy reference instructions, matched pseudo
budgets and a locked external evaluation protocol.

## Verification and delivery

Tests cover affine axes, coordinate gradients and double backward, rank
ordering, parity/centroid refinement, meaningful analytic-energy improvement,
data leakage rejection, pseudo consensus/sector/ambiguity rejection, resume,
and a synthetic end-to-end CPU smoke run. Run the entire legacy suite and
source-release audit. Document environment setup and training commands in
Chinese and summarize in English. Push only source/config/docs/tests to the
authorized new branch; never data, weights or experiment artifacts.
