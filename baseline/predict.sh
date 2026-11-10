#!/bin/bash
set -e

# ==========================================================
# FOR EFFICIENT-MEDSAM2 (Small)
#
# --weights avoids the HuggingFace download at run time: the container has no
#   network and does not ship huggingface_hub. Run the matching
#   download_EfficientMedSAM2_*.sh BEFORE `docker build` so the checkpoint is
#   copied into the image.
# --model-size must match the weights; it selects the EfficientTAM config.
# --device cpu is required: the Dockerfile sets CUDA_VISIBLE_DEVICES="".

# python3 Efficient_MedSAM2_infer_CT_lesion_npz_recist.py \
#     --weights checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt \
#     --model-size small \
#     --device cpu \
#     --prompt box \
#     -i /workspace/inputs \
#     -o /workspace/outputs
# ==========================================================


# ==========================================================
# FOR EFFICIENT-MEDSAM2 (Tiny)

python3 Efficient_MedSAM2_infer_CT_lesion_npz_recist.py \
    --weights checkpoints/eff_medsam2_tiny_FLARE25_RECIST_baseline.pt \
    --model-size tiny \
    --device cpu \
    --prompt box \
    -i /workspace/inputs \
    -o /workspace/outputs
# ==========================================================


# ==========================================================
# FOR MEDSAM2

# python3 MedSAM2_infer_CT_lesion_npz_recist.py \
#     --imgs_path /workspace/inputs \
#     --pred_save_dir /workspace/outputs
# ==========================================================
