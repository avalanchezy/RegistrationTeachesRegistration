# Method

## 1. Problem

For each case, the input is a CBCT volume and upper/lower IOS meshes. The output
is one homogeneous matrix per jaw:

```text
T = [[R, t],
     [0, 1]]
```

The matrix maps IOS vertices into CBCT physical coordinates. The implementation
supports both proper and improper orthogonal matrices because the released
labels include acquisition/export protocols with `det(R) = -1`.

## 2. Transform-derived weak crown support

No voxelwise tooth labels are used. For each labeled jaw:

1. Preserve the NIfTI affine and crop a non-learned dental ROI.
2. Resample the ROI to `128 x 128 x 128` at `1.25 mm` isotropic spacing.
3. Sample the lower and upper 35% tails along the smallest-variance IOS PCA axis.
4. Transform both alternatives with the supplied registration matrix.
5. Keep the alternative with stronger CBCT-density evidence.
6. Rasterize a `0.7 mm` neighborhood of that transformed surface.

The submitted checkpoints were trained with `minimum_hu = -1000`, so the final
weak target is the thin geometric neighborhood itself rather than an
HU-thresholded tooth segmentation. These labels represent registration support,
not full anatomical teeth.

## 3. Crown-support network

The predictor is a five-level 3D U-Net:

- input: `B x 1 x 128 x 128 x 128`
- widths: `24, 48, 96, 192, 384`
- two `3 x 3 x 3` convolutions, GroupNorm, and SiLU per block
- max-pooling encoder and transposed-convolution decoder
- output: `B x 3 x 128 x 128 x 128`
- classes: background, upper support, lower support
- parameters per network: 12,701,691

The loss is weighted cross entropy plus foreground soft Dice. Training uses
AdamW, batch size 1, learning rate `3e-4`, weight decay `1e-4`, mixed precision,
and geometric/intensity augmentation.

Five supervised members are trained with CBCT-content-grouped validation. Five
self-training members additionally use the top 80 pseudo-labeled complete
unlabeled cases per fold at effective weight 0.2. Pseudo labels must satisfy:

- 500 to 6000 voxels per jaw
- foreground confidence at least 0.45
- foreground entropy at most 1.0
- at most one sample per identical-CBCT hash group

All ten final members are refitted on the 30 labeled cases. Their probabilities
are averaged with supervised/self-training weights `0.5/0.5`. Inference keeps
probabilities at least 0.25 and removes components smaller than four voxels.

## 4. Evidence-first correspondence routes

The system first checks whether the query CBCT content matches a fixed reference
bank built before test inference.

### Paired-vertex route

A reference transform is reused only when same-index IOS correspondence has
RMS at most `0.02 mm` and P95 at most `0.05 mm`.

### Surface-correspondence route

For a differently tessellated IOS on the same CBCT, a rigid surface mapping is
composed with the reference transform only when all deployment gates pass,
including predicted teacher TRE at most `1.5 mm`, bidirectional score at most
`0.8 mm`, median at most `1 mm`, P90 at most `2 mm`, and at least 95% coverage
within `2 mm` in both directions.

The reference banks are generated from organizer-provided Task 2 data. They are
not included in the public repository because they contain sampled geometry.

## 5. Reflection-aware geometric search

If neither correspondence route is accepted, the neural-geometric route runs:

1. Construct six IOS crown-side surfaces from low/high PCA tails at 25%, 35%,
   and 45%.
2. Select the expected transform parity from acquisition metadata.
3. Enumerate signed PCA-axis permutations consistent with that parity.
4. Add jaw-specific transform-prior initializations fitted on labeled cases.
5. Refine with trimmed ICP at 8, 5, 3, 2, and 1.25 mm correspondence scales,
   eight iterations per scale, retaining the best 70% of correspondences.
6. Keep up to eight source/target-diverse basins.
7. Run 384 perturbations per stage at `(12 deg, 8 mm)`, `(6 deg, 4 mm)`, and
   `(3 deg, 2 mm)`, with ICP refinement.
8. Append crown-overlap features and refine the strongest candidates once more.

The deployed search uses the postprocessed binary crown-support mask. Legacy
HU-threshold candidates, probability targets, and D4 test-time augmentation are
disabled in `configs/submission/deployment_policy.json`.

## 6. Candidate ranking and jaw coupling

Each candidate is represented by 97 scalar descriptors covering trimmed fit,
coverage, normals, full-IOS consistency, target geometry, transform parity,
source view, preliminary ranks, and within-jaw context.

- Seven 400-tree ExtraTrees regressors predict `log(1 + TRE)`.
- Seven 250-tree pairwise ExtraTrees classifiers compare candidate pairs.
- Regression votes and median pairwise costs are converted to fractional ranks.
- The ranks are fused with `alpha = 0.575`.
- At most 20 tree-ranked candidates per jaw enter selection.
- The top eight per jaw enter joint upper/lower selection.

Joint selection penalizes disagreement with the learned relative jaw pose using
`0.01 mm/deg` for rotation and `0.075` for translation. Pairs with different
parity are forbidden.

## 7. Output validation

Before exit, every output is checked for:

- shape `(4,4)` and dtype `float64`
- finite entries
- homogeneous last row
- orthogonal `R`
- determinant magnitude near one
- exactly one upper and one lower matrix for every discovered case

Any contract violation causes a nonzero container exit instead of silently
emitting malformed predictions.
