# MedSAM2-RECIST

Inference pipeline for MICCAI FLARE Challenge [Pan-Cancer segmentation task](https://www.codabench.org/competitions/7149/). 
This repository supports these baselines: MedSAM2 and Efficient MedSAM2 Small/Tiny.

## Installation

Python **3.12** and PyTorch **2.5.1** are required (torch 2.5.1 publishes no wheels for Python 3.13+).
[`uv`](https://docs.astral.sh/uv/) is the recommended installer — it downloads Python 3.12 itself if
the system does not provide it.

```bash
git clone https://github.com/medfm-flare/MedSAM2-RECIST && cd MedSAM2-RECIST

# 1. Create the virtual environment
uv venv --python 3.12

# 2. Install PyTorch first, from the CUDA 12.4 wheel index (Linux + CUDA 12.4).
#    On a CPU-only machine, swap the index for https://download.pytorch.org/whl/cpu
uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install this repository in editable mode, with the dev extras
SAM2_BUILD_CUDA=0 uv pip install --no-build-isolation -e '.[dev]'
```

Activate the environment before running anything: `source .venv/bin/activate`.

`huggingface_hub` is not installed by the commands above. It is needed only for `download_dockers.py`
and for the HuggingFace checkpoint auto-download that happens when the Efficient MedSAM2 script is
run *without* `--weights`:

```bash
uv pip install huggingface_hub
```

### Verifying the installation

```bash
python -c "import torch, sam2, efficient_track_anything; print(torch.__version__, torch.cuda.is_available())"
```

## Local inference

Download a checkpoint first (the `download_*.sh` scripts used in the Docker sections below put them
in `checkpoints/`), then run either baseline directly:

```bash
# Efficient MedSAM2 (Tiny) — device is selectable
python Efficient_MedSAM2_infer_CT_lesion_npz_recist.py \
    --weights checkpoint/checkpoint.pt \
    --model-size tiny \
    --device cuda \
    --prompt box \
    -i data/validation_public_npz \
    -o data/out

# MedSAM2 — this script forces CPU internally
python MedSAM2_infer_CT_lesion_npz_recist.py \
    -i data/validation_public_npz \
    -o data/out
```

`--model-size` must match the checkpoint, otherwise `load_state_dict` fails. Each run writes one NPZ
per case (containing the predicted `segs`) into the output directory, plus an `inference_times.csv`
timing summary.

## Building Docker

### For MedSAM2

- Download MedSAM2 RECIST checkpoint
```bash
sh download_MedSAM2.sh
```
- Navigate to `predict.sh` and please make sure the command is as follows
```bash
python3 MedSAM2_infer_CT_lesion_npz_recist.py \
    --imgs_path /workspace/inputs \
    --pred_save_dir /workspace/outputs
```
- Build docker image
```bash
docker build --no-cache -t medsam2:latest -f Dockerfile .
```
- Run on example case
```bash
docker container run -m 8G --rm \
  -v $PWD/data/validation_public_npz:/workspace/inputs/ \
  -v $PWD/data/out:/workspace/outputs/ \
  medsam2:latest \
  /bin/bash -c "sh predict.sh" 
```
- Save docker image
```
docker save medsam2:latest | gzip > medsam2.tar.gz 
```

### For Efficient MedSAM2 Small

- Download Efficient MedSAM2 small checkpoint
```bash
sh download_EfficientMedSAM2_Small.sh
```
- Navigate to `predict.sh` and please make sure the command is as follows
```bash
python3 Efficient_MedSAM2_infer_CT_lesion_npz_recist.py \
    --weights checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt \
    --model-size small \
    --device cpu \
    --prompt box \
    -i /workspace/inputs \
    -o /workspace/outputs
```
- Build docker image
```bash
docker build --no-cache -t efficientmedsam2small:latest -f Dockerfile .
```
- Run on example case
```bash
docker container run -m 8G --rm \
  -v $PWD/data/validation_public_npz:/workspace/inputs/ \
  -v $PWD/data/out:/workspace/outputs/ \
  efficientmedsam2small:latest \
  /bin/bash -c "sh predict.sh" 
```
- Save docker image
```
docker save efficientmedsam2small:latest | gzip > efficientmedsam2small.tar.gz
```

### For Efficient MedSAM2 Tiny

- Download Efficient MedSAM2 Tiny checkpoint
```bash
sh download_EfficientMedSAM2_Tiny.sh
```
- Navigate to `predict.sh` and please make sure the command is as follows
```bash
python3 Efficient_MedSAM2_infer_CT_lesion_npz_recist.py \
    --weights checkpoints/eff_medsam2_tiny_FLARE25_RECIST_baseline.pt \
    --model-size tiny \
    --device cpu \
    --prompt box \
    -i /workspace/inputs \
    -o /workspace/outputs
```
- Build docker image
```bash
docker build --no-cache -t efficientmedsam2tiny:latest -f Dockerfile .
```
- Run on example case
```bash
docker container run -m 8G --rm \
  -v $PWD/data/validation_public_npz:/workspace/inputs/ \
  -v $PWD/data/out:/workspace/outputs/ \
  efficientmedsam2tiny:latest \
  /bin/bash -c "sh predict.sh" 
```
- Save docker image
```
docker save efficientmedsam2tiny:latest | gzip > efficientmedsam2tiny.tar.gz
```

## Download baseline dockers directly
- MedSAM2 RECIST, Efficient MedSAM2 Small/Tiny RECIST dockers can be downloaded [here](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task1-PancancerRECIST-to-3D-Dockers).
To download all dockers, please run
```bash
python download_dockers.py
```

## Acknowledgements

- We highly appreciate all the challenge organizers and dataset owners for providing the public datasets to the community.
- We thank Meta AI for making the source code of [SAM2](https://github.com/facebookresearch/sam2) and [EfficientTAM](https://github.com/yformer/EfficientTAM) publicly available. Please also cite these papers when using MedSAM2. 


## Bibtex

```bash
@article{MedSAM2,
    title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2504.03600},
    year={2025}
}
```
Please also cite SAM2
```
@inproceedings{SAM2,
title={{SAM} 2: Segment Anything in Images and Videos},
    author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman R{\"a}dle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollar and Christoph Feichtenhofer},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
```

and EfficientTAM

```
@article{xiong2024efficienttam,
    title={Efficient Track Anything},
    author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra},
    journal={preprint arXiv:2411.18933},
    year={2024}
}
```

