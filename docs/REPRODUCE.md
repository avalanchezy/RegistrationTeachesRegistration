# Reproducing the Submitted Method

This document separates four reproducibility levels:

1. **Source verification**: no challenge data or GPU required.
2. **Crown-support training**: reproduces the ten-network ensemble.
3. **Geometric selector training**: reproduces candidate generation and ranking.
4. **Submission assembly**: builds the exact model-asset layout expected by Docker.

The commands below use Bash syntax and paths relative to the repository root.
PowerShell users can replace line-continuation backslashes with backticks.

## 1. Environment

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/audit_source_release.py
```

Training was performed with PyTorch 2.8.0 and CUDA 12.8. Deployment is pinned
to PyTorch 2.5.1 and CUDA 12.1 in `Dockerfile`; the serialized networks use
standard PyTorch layers and were tested in that deployment environment.

## 2. Manifest and content groups

```bash
export DATA_ROOT=/path/to/MICCAI-Chllenge-STS26-Task2

mkdir -p manifests runs work
python scripts/build_manifest.py \
  --data-root "$DATA_ROOT" \
  --output manifests/task2.csv

python scripts/build_cbct_hash_cache.py \
  --manifest manifests/task2.csv \
  --output runs/cbct_payload_hash_cache.json
```

The hash cache uses decompressed NIfTI payload hashes. Grouped folds therefore
keep byte-equivalent volume content together even when gzip headers differ.

## 3. Affine-preserving dental ROIs

```bash
python scripts/prepare_threshold_dental_roi.py \
  --manifest manifests/task2.csv \
  --split Train-Labeled \
  --dataset-name LabeledDentalROI \
  --work-root work \
  --threshold 1600 \
  --margin-mm 20

python scripts/prepare_threshold_dental_roi.py \
  --manifest manifests/task2.csv \
  --split Train-Unlabeled \
  --dataset-name UnlabeledDentalROI \
  --work-root work \
  --threshold 1600 \
  --margin-mm 20
```

No learned segmentation is used in this stage.

## 4. Registration-derived labeled targets

The final checkpoints use a thin `0.7 mm` surface neighborhood and no HU gate:

```bash
python scripts/prepare_crown_localizer_data.py \
  --manifest manifests/task2.csv \
  --roi-dir work/inputs/LabeledDentalROI/imagesTs \
  --output-dir runs/crown_labeled \
  --split Train-Labeled \
  --grid-size 128 \
  --spacing-mm 1.25 \
  --surface-radius-mm 0.7 \
  --minimum-hu -1000 \
  --crown-fraction 0.35 \
  --ios-points 120000

python scripts/prepare_crown_localizer_inference_data.py \
  --manifest manifests/task2.csv \
  --roi-dir work/inputs/UnlabeledDentalROI/imagesTs \
  --output-dir runs/crown_unlabeled \
  --split Train-Unlabeled \
  --grid-size 128 \
  --spacing-mm 1.25
```

## 5. Grouped supervised teachers

```bash
python scripts/train_crown_localizer.py \
  --data-dir runs/crown_labeled/data \
  --manifest manifests/task2.csv \
  --cbct-hash-cache runs/cbct_payload_hash_cache.json \
  --output-dir runs/crown_supervised_oof \
  --folds 5 \
  --epochs 100 \
  --patience 22 \
  --base-channels 24 \
  --batch-size 1 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --background-weight 0.10 \
  --surface-tolerance-voxels 0 \
  --seed 20260715 \
  --device cuda \
  --cache-data
```

For each fold `F = 0..4`, predict all complete unlabeled inputs with that fold
teacher. The explicit fold flag keeps the five pseudo-label sources separate:

```bash
python scripts/predict_crown_localizer.py \
  --data-dir runs/crown_unlabeled/data \
  --model-dir runs/crown_supervised_oof \
  --output-dir runs/unlabeled_teacher_fold_F \
  --minimum-probability 0.5 \
  --minimum-component-voxels 12 \
  --maximum-components 0 \
  --minimum-hu -1000 \
  --fold-indices F \
  --ensemble-all-folds \
  --device cuda \
  --save-label-arrays
```

Select fold-specific pseudo labels while enforcing CBCT-content diversity:

```bash
python scripts/select_crown_pseudo_labels.py \
  --prediction-dirs \
    runs/unlabeled_teacher_fold_0 \
    runs/unlabeled_teacher_fold_1 \
    runs/unlabeled_teacher_fold_2 \
    runs/unlabeled_teacher_fold_3 \
    runs/unlabeled_teacher_fold_4 \
  --output-root runs/crown_pseudo \
  --manifest manifests/task2.csv \
  --cbct-hash-cache runs/cbct_payload_hash_cache.json \
  --max-per-cbct-group 1 \
  --top-counts 20 40 80 \
  --minimum-jaw-voxels 500 \
  --maximum-jaw-voxels 6000 \
  --minimum-confidence 0.45 \
  --maximum-entropy 1.0
```

## 6. Final ten U-Nets

The selected grouped-validation epoch schedules are replayed on all 30 labeled
cases. They are fixed in `configs/training/final_method.json`.

```bash
python scripts/train_final_crown_localizer.py \
  --data-dir runs/crown_labeled/data \
  --output-dir runs/final_crown_supervised \
  --seeds 20260721 20260722 20260723 20260724 20260725 \
  --epochs 1 \
  --epochs-per-member 81 68 40 53 66 \
  --base-channels 24 \
  --batch-size 1 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --background-weight 0.10 \
  --surface-tolerance-voxels 0 \
  --device cuda \
  --cache-data

python scripts/train_final_crown_localizer.py \
  --data-dir runs/crown_labeled/data \
  --unlabeled-data-dir runs/crown_unlabeled/data \
  --pseudo-label-root runs/crown_pseudo/top_80 \
  --output-dir runs/final_crown_semisupervised \
  --seeds 20260731 20260732 20260733 20260734 20260735 \
  --epochs 1 \
  --epochs-per-member 45 76 27 48 69 \
  --base-channels 24 \
  --batch-size 1 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --background-weight 0.10 \
  --surface-tolerance-voxels 0 \
  --pseudo-weight 0.20 \
  --max-pseudo-cases 80 \
  --device cuda \
  --cache-data
```

Deployment averages the five supervised and five self-training members with
branch weights `0.5/0.5`. The live ensemble definition is
`configs/submission/crown_ensemble.json`.

## 7. Grouped OOF masks and candidate generation

Candidate rankers must be trained from grouped OOF support predictions, not
predictions made by a network trained on the same case. Generate OOF
probabilities for the supervised and self-training ensembles, blend them with
`scripts/blend_crown_probabilities.py`, and materialize masks with:

```bash
python scripts/materialize_crown_probabilities.py \
  --data-dir runs/crown_labeled/data \
  --probability-dir runs/crown_oof_fused_probabilities \
  --output-dir runs/crown_oof_fused \
  --postprocess-config configs/submission/crown_postprocess.json \
  --save-label-arrays
```

Generate reflection-aware candidates for multiple deterministic seeds. The
submitted geometry settings are:

```bash
python scripts/run_geometry_benchmark.py \
  --manifest manifests/task2.csv \
  --output-dir runs/candidates_seed_20260711 \
  --split Train-Labeled \
  --target-mode crown \
  --crown-mask-dir runs/crown_oof_fused/registration_labels \
  --tracked-jaw-mode both \
  --methods pca \
  --ios-source-mode sides \
  --ios-crop-fractions 0.25 0.35 0.45 \
  --max-target-candidates 1 \
  --pca-refine-top-k 24 \
  --basin-refine-top-k 8 \
  --basin-samples 384 \
  --basin-selection source-target-diverse \
  --leave-one-cbct-group-out-prior \
  --cbct-hash-cache runs/cbct_payload_hash_cache.json \
  --chirality-mode metadata \
  --prior-max-angle-deg 90 \
  --seed 20260711 \
  --stable-record-seeds \
  --resume-completed-records \
  --no-visualizations
```

Repeat for seeds `20260712` through `20260717`. Then attach full-IOS and crown
consistency descriptors with `augment_candidate_geometry.py`,
`evaluate_crown_candidate_consistency.py`, and
`refine_candidates_with_crown.py`. These scripts update candidate artifacts but
never use ground truth to choose an inference candidate.

## 8. Final candidate rankers

Fit all seven regression and pairwise members on every labeled candidate row:

```bash
python scripts/fit_final_multiseed_ensemble.py \
  --manifest manifests/task2.csv \
  --labeled-runs runs/candidates_seed_20260711 \
                 runs/candidates_seed_20260712 \
                 runs/candidates_seed_20260713 \
                 runs/candidates_seed_20260714 \
                 runs/candidates_seed_20260715 \
                 runs/candidates_seed_20260716 \
                 runs/candidates_seed_20260717 \
  --output-dir runs/final_rerankers \
  --seeds 20260711 20260712 20260713 20260714 20260715 20260716 20260717 \
  --top-unsupervised 20 \
  --top-oracle 8 \
  --eval-top-candidates 20 \
  --min-samples-leaf 2 \
  --max-features 0.2 \
  --model-scope jaw \
  --model-type extra_trees \
  --target-transform log1p \
  --tree-criterion squared_error \
  --regression-trees 400 \
  --fit-pairwise \
  --pairwise-min-samples-leaf 4 \
  --pairwise-max-features 0.2 \
  --pairwise-model-type extra_trees \
  --pairwise-criterion gini \
  --pairwise-min-log-tre-gap 0.05 \
  --pairwise-trees 250 \
  --max-pairs-per-group 600 \
  --eval-opponents 30 \
  --group-context-features \
  --roi-view-feature \
  --balance-candidate-runs \
  --exclude-upper-opposite-axial \
  --regression-aggregation vote \
  --pairwise-aggregation median \
  --blend-alpha 0.575 \
  --joint-pair-top-k 8 \
  --joint-angle-weight 0.01 \
  --joint-translation-weight 0.075 \
  --global-crown-modes crown \
  --exclude-legacy-threshold-candidates \
  --global-include-crown-refinement \
  --crown-tta-mode none
```

The submitted fit contains 4,929 candidate rows from 60 jaws. The script writes
`regression_ensemble.joblib`, `pairwise_ensemble.joblib`, the rotation prior,
the deployment policy, and a machine-readable summary.

## 9. Reference banks

Build the labeled bank and admit only transformations that pass fixed gates:

```bash
python scripts/build_template_bank.py \
  --manifest manifests/task2.csv \
  --output runs/template_assets/labeled_templates.joblib \
  --split Train-Labeled \
  --sample-points 8192

python scripts/build_exact_transfer_pseudolabels.py \
  --manifest manifests/task2.csv \
  --template-bank runs/template_assets/labeled_templates.joblib \
  --output-dir runs/exact_transfer_pseudo \
  --split Train-Unlabeled \
  --max-rms-mm 0.02 \
  --max-p95-mm 0.05
```

`extend_template_bank_from_inference.py` and
`build_surface_template_bank.py` add only quality-gated pseudo transforms from
unlabeled geometry runs. The final bank composition is recorded in
`configs/training/final_method.json`: 94 paired-vertex entries and 97 surface
entries. Use `enrich_template_bank_payload_hashes.py` last so repeated CBCTs are
matched by NIfTI content, not gzip bytes.

Template banks contain recoverable sampled organizer geometry. Keep them
private unless the organizer explicitly authorizes redistribution.

## 10. Asset assembly

Place the trained fallback/ROI rerankers, final tree ensembles, transform prior,
and template banks in staging directories, then run:

```bash
python scripts/assemble_final_submission.py \
  --destination dist/docker_context \
  --legacy-assets runs/runtime_rerankers \
  --template-assets runs/template_assets \
  --enhanced-assets runs/final_rerankers \
  --crown-supervised runs/final_crown_supervised \
  --crown-semisupervised runs/final_crown_semisupervised \
  --fusion-summary runs/crown_fusion_selection/summary.json \
  --supervised-weight 0.5 \
  --fusion-mode arithmetic \
  --crown-tta-mode none \
  --postprocess-config configs/submission/crown_postprocess.json
```

The assembly is atomic. It validates member counts, confirms that every final
network saw all 30 labeled cases, writes SHA256 manifests, synchronizes the
runtime closure, and runs the asset verifier before replacing a prior bundle.

## 11. Reproducibility boundary

The source, algorithms, hyperparameters, and tests are public. The organizer's
data and data-derived template geometry are not. Exact numerical reproduction
therefore requires authorized access to the same Task 2 release. This boundary
prevents an open-source release from becoming an unauthorized copy of the
challenge dataset.
