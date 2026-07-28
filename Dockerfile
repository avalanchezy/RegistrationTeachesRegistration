ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime@sha256:831247999fbf7e08f61b3e39f6d77ee434f38f6f07f769d00db451e853878067
FROM ${BASE_IMAGE}

USER root

LABEL org.opencontainers.image.title="STSR2026 Task 2 CBCT-IOS Registration"
LABEL org.opencontainers.image.description="STSR 2026 Task 2 winning semi-supervised CBCT-IOS registration method"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/avalanchezy/RegistrationTeachesRegistration"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/algorithm \
    HOME=/home/algorithm \
    INPUT_DIR=/inputs \
    OUTPUT_DIR=/outputs \
    MODEL_DIR=/opt/algorithm/model_assets \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.cuda121.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN if ! getent group algorithm >/dev/null; then groupadd --gid 999 algorithm; fi \
    && if ! id algorithm >/dev/null 2>&1; then useradd --uid 999 --gid algorithm --create-home --shell /bin/bash algorithm; fi \
    && mkdir -p /inputs /outputs \
    && chown algorithm:algorithm /inputs /outputs

WORKDIR /opt/algorithm
COPY --chown=algorithm:algorithm task2reg/ /opt/algorithm/task2reg/
COPY --chown=algorithm:algorithm scripts/ /opt/algorithm/scripts/
COPY --chown=algorithm:algorithm model_assets/ /opt/algorithm/model_assets/
COPY --chown=algorithm:algorithm predict.sh /opt/algorithm/predict.sh

RUN chmod 755 /opt/algorithm/predict.sh \
    && python /opt/algorithm/scripts/verify_assets.py \
    && python -m py_compile /opt/algorithm/scripts/*.py /opt/algorithm/task2reg/*.py

USER algorithm
ENTRYPOINT ["/opt/algorithm/predict.sh"]
