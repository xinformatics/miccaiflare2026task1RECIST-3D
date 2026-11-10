# RECIST-guided 3-D lesion segmentation — FLARE Task 2

Team: aidaraomayoclinic.

Code: https://github.com/xinformatics/miccaiflare2026task1RECIST-3D

| System | Hidden DSC | Hidden NSD |
|---|---:|---:|
| Six-stream U-Net–MedSAM research ensemble | 0.8114 | 0.6811 |
| Organ-MoE research extension | 0.8115 | 0.6813 |
| Submitted adaptive CPU Docker | 0.8001 | 0.6654 |

The research ensemble and submitted Docker use different prediction graphs.
The Docker contains organ-MoE, EfficientMedSAM2 Small, and EfficientMedSAM2 Tiny checkpoints. Cases with at most 40 bounded SAM propagation slices use four streams; 41–120 use MoE plus Tiny; larger cases use MoE alone.

## Run the submitted Docker

Download `aidaraomayoclinic-rich-cap120.tar.gz` from the companion model deposit, then:

```bash
docker load -i aidaraomayoclinic-rich-cap120.tar.gz
mkdir -p outputs
docker run --rm -m 8G -v "$PWD/inputs:/workspace/inputs" -v "$PWD/outputs:/workspace/outputs" aidaraomayoclinic:latest /bin/bash -c "sh predict.sh"
```

Inputs are NPZ files containing `imgs`, `recist`, and `spacing`. Each output contains a same-shaped uint8 `segs` array preserving the RECIST lesion IDs.

## Build from source

Extract `miccaiflare2026task1RECIST-3D-zenodo.zip` at the repository root; it restores each checkpoint to the path expected by its implementation. Then:

```bash
bash solution/build_moe_docker.sh artifacts/aidaraomayoclinic-rebuilt.tar.gz
```

The archived image is the exact submitted artifact. A fresh build can differ because operating-system packages and some transitive dependencies are not fully pinned.

## Research reproduction

`winning.md`, section 9, documents the final six-stream graph and epoch-195 training recipe. `REPRODUCE.md` contains historical training details. `REBOOT_HANDOFF_2026-10-01.md` records the final verified method, score distinctions, and CPU policy. Older subdirectory documentation describes earlier experiments; this README and the handoff take precedence for final results.

The released A2 checkpoint uses `othermachine/release/src`; the independently trained epoch-195 checkpoint uses `roi3d`. They have different state-dict layouts. `docker_moe/fuse_final.py` implements the final six-stream fusion.

Weights and Docker are distributed separately from GitHub. No patient scans, masks, prediction archives, credentials, or training caches are included in this code release. See `THIRD_PARTY.md` for provenance.
