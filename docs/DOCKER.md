# Docker

## Prerequisite

Populate `model_assets/` with the reproduced assets listed in
`configs/submission/model_assets.sha256`. `scripts/verify_assets.py` must pass
before building.

The Dockerfile intentionally validates assets during the image build. This
prevents an apparently successful image from missing a checkpoint or ranker.

## Build

```bash
python scripts/verify_assets.py
docker build -t registration-teaches-registration:latest .
```

The deployment base is pinned to PyTorch 2.5.1, CUDA 12.1, and cuDNN 9. The
container runs as an unprivileged `algorithm` user and writes only to
`/outputs`.

## Official-style run

```bash
mkdir -p outputs
chmod a+rwx outputs

docker container run \
  --gpus all \
  -m 8G \
  --name registration-teaches-registration \
  --rm \
  -v "$PWD/test_case_data:/inputs:ro" \
  -v "$PWD/outputs:/outputs:rw" \
  registration-teaches-registration:latest
```

The entrypoint requires no arguments. It verifies the GPU, model-asset hashes,
runtime source closure, inference outputs, and rigid-matrix contract.

## Expected output

```text
/outputs/<case_id>/upper_gt.npy
/outputs/<case_id>/lower_gt.npy
```

Run the host-side validator when developing locally:

```bash
python scripts/validate_outputs.py \
  --input-dir test_case_data \
  --output-dir outputs
```

## Difference from the submitted archive

The public Dockerfile changes only OCI labels and the startup banner to remove
internal release naming. The algorithm modules, inference script, dependency
versions, model contract, and deployment configuration are the submitted
implementation. Their hashes are recorded under `configs/submission/`.
