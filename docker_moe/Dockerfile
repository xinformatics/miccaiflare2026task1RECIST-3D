FROM --platform=linux/amd64 python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    OPENBLAS_NUM_THREADS=12 \
    MEDSAM2_REPO=/workspace/baseline \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY baseline /workspace/baseline
COPY docker_moe/legacy_effmedsam2_infer.py /workspace/baseline/legacy_effmedsam2_infer.py
RUN pip install --no-cache-dir \
      numpy==2.1.3 scipy==1.14.1 pandas==2.2.3 Pillow==11.0.0 \
      matplotlib==3.9.2 SimpleITK==2.4.0 hydra-core==1.3.2 \
      iopath==0.1.10 tqdm==4.67.0 && \
    pip install --no-cache-dir --no-build-isolation --no-deps /workspace/baseline

COPY roi3d /workspace/roi3d
RUN mkdir -p /workspace/docker_moe
COPY docker_moe/predict.sh docker_moe/run_legacy_sam.py docker_moe/route_cpu_cases.py \
     /workspace/docker_moe/
COPY docker_moe/predict.sh /workspace/predict.sh
RUN mkdir -p /workspace/models
COPY docker_moe/models/effmedsam2_small.pt docker_moe/models/effmedsam2_tiny.pt \
     docker_moe/models/organ_moe_epoch65.pt \
     /workspace/models/
RUN chmod +x /workspace/predict.sh /workspace/docker_moe/predict.sh

CMD ["bash", "/workspace/predict.sh"]
