#!/usr/bin/env bash
set -euo pipefail

echo "[RegistrationTeachesRegistration] STSR 2026 Task 2 inference"
echo "[STS26 Task2] input=${INPUT_DIR:-/inputs} output=${OUTPUT_DIR:-/outputs}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is false")
print(
    "[STS26 Task2] "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)} architectures={torch.cuda.get_arch_list()}"
)
PY

python /opt/algorithm/scripts/verify_assets.py
python /opt/algorithm/scripts/run_submission_inference.py
python /opt/algorithm/scripts/validate_outputs.py \
    --input-dir "${INPUT_DIR:-/inputs}" \
    --output-dir "${OUTPUT_DIR:-/outputs}"

echo "[STS26 Task2] inference and output validation completed"
