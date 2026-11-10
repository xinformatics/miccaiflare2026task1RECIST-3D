FROM --platform=linux/amd64 python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    SAM2_BUILD_CUDA=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY baseline /workspace/baseline
RUN pip install --no-cache-dir \
      numpy==2.1.3 scipy==1.14.1 pandas==2.2.3 Pillow==11.0.0 \
      matplotlib==3.9.2 SimpleITK==2.4.0 hydra-core==1.3.2 \
      iopath==0.1.10 tqdm==4.67.0 && \
    pip install --no-cache-dir --no-deps /workspace/baseline

COPY checkpoints/eff_medsam2_tiny_FLARE25_RECIST_baseline.pt \
     /workspace/checkpoints/eff_medsam2_tiny_FLARE25_RECIST_baseline.pt
COPY solution /workspace/solution
COPY predict.sh /workspace/predict.sh
RUN chmod +x /workspace/predict.sh /workspace/solution/predict.sh

CMD ["/bin/bash"]

