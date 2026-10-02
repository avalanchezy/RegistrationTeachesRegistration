# Data

## Official Task 2 layout

The code accepts a manifest generated from the organizer's directory tree. A
typical case is:

```text
<data-root>/
  Train-Labeled/
    images/
      003/
        CBCT.nii.gz
        upper.stl
        lower.stl
    labels/
      003/
        upper_gt.npy
        lower_gt.npy
  Train-Unlabeled/
    images/
      <case_id>/
        CBCT.nii.gz
        upper.stl
        lower.stl
  Validation/
    images/
      <case_id>/
        CBCT.nii.gz
        upper.stl
        lower.stl
```

Ground-truth `.npy` files are homogeneous `float64 (4,4)` registration
matrices. They are not segmentation masks.

The importer also accepts `Images`/`Labels`. Build the legacy manifest (its paths
are absolute, so rebuild it after moving the dataset to a different machine):

```bash
python scripts/build_manifest.py \
  --data-root /path/to/MICCAI-Chllenge-STS26-Task2 \
  --output manifests/task2.csv
```

The journal branch adds explicit patient/source/split metadata and relative-path
manifests. See [the journal data guide](JOURNAL_DATA_zh-CN.md) for conversion from
this existing interface; the original CSV schema remains unchanged.

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
