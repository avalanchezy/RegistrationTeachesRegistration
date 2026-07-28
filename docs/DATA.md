# Data

## Official Task 2 layout

The code accepts a manifest generated from the organizer's directory tree. A
typical case is:

```text
<data-root>/
  Train-Labeled/
    003/
      CBCT.nii.gz
      upper.stl
      lower.stl
      upper_gt.npy
      lower_gt.npy
  Train-Unlabeled/
    <case_id>/
      CBCT.nii.gz
      upper.stl
      lower.stl
  Validation/
    <case_id>/
      CBCT.nii.gz
      upper.stl
      lower.stl
```

Ground-truth `.npy` files are homogeneous `float64 (4,4)` registration
matrices. They are not segmentation masks.

Build the portable manifest:

```bash
python scripts/build_manifest.py \
  --data-root /path/to/MICCAI-Chllenge-STS26-Task2 \
  --output manifests/task2.csv
```

## Release audit used by the method

- 30 labeled cases / 60 labeled jaws
- 300 unlabeled cases
- 289 unlabeled cases with a complete usable upper/lower IOS pair
- 50 public validation cases / 100 jaws

CBCT array shapes, voxel spacings, and affines vary. All processing therefore
uses NIfTI physical coordinates rather than assuming a fixed index-space
orientation or spacing.

## Data policy

The repository does not distribute organizer data or recoverable derivatives.
In particular, do not commit:

- `CBCT.nii.gz`
- IOS meshes
- ground-truth transforms
- generated weak-label volumes
- pseudo-label volumes
- template banks containing sampled IOS geometry
- validation or hidden-test predictions

The source-release audit fails when these extensions or naming patterns are
found outside explicitly documented synthetic fixtures.
