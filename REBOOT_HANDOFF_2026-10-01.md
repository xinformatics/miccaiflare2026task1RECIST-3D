# FLARE Task 2 reboot and presentation handoff

Saved: 2026-10-01

Workspace: `/mnt/nvme0/shashank/challenge2`

Team: `aidaraomayoclinic`

This file is the durable context needed to resume work after a machine reboot. It
records the verified development history, terminology, model graph, scores, Docker
contents, and presentation wording. Do not conflate the unrestricted research
ensemble with the CPU Docker deployment.

## 1. Task

FLARE/PanCancerSeg Task 2 is RECIST-to-3-D lesion segmentation. Each input NPZ
contains:

- `imgs`: a CT volume, already represented as uint8 intensities;
- `recist`: one labelled 2-D RECIST marker per lesion on one axial slice;
- `spacing`: physical voxel spacing;
- outputs must be same-shaped uint8 `segs` masks preserving each RECIST lesion ID.

Evaluation is per lesion using DSC and normalized surface Dice (NSD) at a 1 mm
tolerance.

## 2. Verified score trajectory

| Development stage | Codabench DSC | Codabench NSD |
|---|---:|---:|
| Official MedSAM2-RECIST baseline | 0.7722 | 0.6291 |
| `small_hybrid25-50_z14` | 0.7773 | 0.6338 |
| Glance-and-Focus-inspired fine-tuning plus fusion | 0.7869 | 0.6443 |
| Early four-way ensemble | 0.7931 | 0.6517 |
| `submission_4way_a2_long300_e35.zip` | 0.7965 | 0.6603 |
| Tuned prompt/long-trained ensemble | 0.8023 | 0.6669 |
| Intermediate refined ensemble | 0.8037 | 0.6704 |
| U-Net-anchor-to-SAM2 / released five-stream system | 0.8065 | 0.6743 |
| Final U-Net-MedSAM ensemble (`submission_6way_e195_w40.zip`) | **0.8114** | **0.6811** |
| Organ-MoE extension | 0.8115 | 0.6813 |
| Final rich CPU Docker policy | **0.8001** | **0.6654** |

The organ-MoE gain over the pre-MoE ensemble was only +0.0001 DSC and +0.0002
NSD. For a clean research presentation, use the U-Net-MedSAM system at
0.8114/0.6811 as the primary method. If describing the submitted Docker, the MoE
must still be disclosed because it is physically present in that container.

## 3. Development story

### 3.1 Official baseline

The official MedSAM2-RECIST baseline converted the RECIST marker into a spatial
prompt, segmented the anchor slice with EfficientMedSAM2, and propagated that mask
through adjacent CT slices. Its result was 0.7722 DSC / 0.6291 NSD. Its principal
weaknesses were anchor-mask errors, propagation drift, limited volumetric context,
and sensitivity to RECIST-to-prompt geometry.

### 3.2 Early experiments

Restricted propagation, Glance-and-Focus-inspired training, and simple multi-model
fusion gave gradual gains. These experiments established that 3-D models and SAM2
make complementary errors: the 3-D U-Net supplies volumetric coherence, while
EfficientMedSAM2 can improve local boundary placement and propagation.

### 3.3 RECIST geometry fixes

The RECIST line is a thick raster, so taking the first and last foreground pixels
does not recover geometric endpoints. The final implementation:

1. collects all line pixels;
2. estimates the principal line axis by SVD;
3. projects the pixels onto that axis;
4. recovers the geometric endpoints;
5. subtracts 2 pixels of raster end-cap overmeasurement.

This corrected midpoint and diameter are used for ROI construction, spatial prompts,
and axial-slab estimation.

### 3.4 EfficientMedSAM2 prompt variants

A prompt is a spatial cue analogous to interactive clicking or drawing a box. It is
not a text prompt and EfficientMedSAM2 is not a vision-language model.

- Box prompt: an axis-aligned square centered at the RECIST midpoint, with side
  length `0.9 * corrected RECIST diameter`.
- Three-point prompt: positive foreground points at 25%, 50%, and 75% along the
  corrected RECIST axis (passed center-first).
- Mask prompt: the 3-D U-Net anchor-slice prediction supplied to SAM2 for hybrid
  propagation.

The box is axis-aligned to the image, not rotated with the RECIST line. Use
"RECIST-derived box prompt" rather than "RECIST-aligned box".

Real example from
`dataset/validation_npz/validation_public_npz/CT_Lesion_FLARE23Ts_0051.npz`,
lesion 1:

- anchor slice: z = 59;
- corrected endpoints: (436.90, 179.07) and (329.10, 238.93);
- corrected diameter: 123.30 pixels;
- midpoint: (383, 209);
- box-0.9 prompt: `[328, 154, 438, 264]`;
- box token coordinates: `[[328,154],[438,264]]`, labels `[2,3]`;
- point coordinates: `[[383.00,209.00],[409.95,194.04],[356.05,223.96]]`,
  labels `[1,1,1]`.

In EfficientMedSAM2, box labels 2 and 3 identify box corners, while point label 1
means a positive foreground click. The anchor mask is propagated superiorly and
inferiorly through SAM2 memory within a physically bounded z slab.

### 3.5 RECIST-guided scale-normalized 3-D U-Net

Preferred method name:

> RECIST-Guided Scale-Normalized 3-D U-Net

Avoid calling it simply a "prompt-conditioned U-Net," because that can imply a
SAM-style prompt encoder. The U-Net receives RECIST-derived information as ordinary
input channels.

Each lesion is sampled into a fixed 112 x 176 x 176 ROI. The isotropic spacing is:

```text
s(D) = max(clip(2.9*D/192, 0.45, 1.10), 1.6*D/192) mm
```

The four input channels are:

1. normalized CT intensity;
2. rasterized RECIST line;
3. `exp(-distance_mm / (0.4*D))` distance-to-line prior;
4. constant `log(D/30)/1.5` size channel.

Probabilities are resampled to the native CT grid before thresholding.

Architecture:

- five resolution levels with channels 32/64/128/256/320;
- four downsampling stages and one bottleneck;
- four decoder/upsampling stages, 320 -> 256 -> 128 -> 64 -> 32;
- about 16.5 million parameters;
- InstanceNorm, LeakyReLU, and deep supervision.

The standard description is "a five-level 3-D U-Net." Counting every encoder,
bottleneck, and decoder block separately gives nine processing stages.

Training used AdamW, BF16 autocast, cosine decay, affine/flip/intensity augmentation,
simulated slice thickness, domain-balanced sampling, and lesion-size reweighting.

The 3-D U-Net did not use external pretraining. The final A2 stage was warm-started
from an earlier checkpoint of our own task-trained U-Net. The second epoch-195 model
was also initialized from an earlier local checkpoint and trained independently.
Presentation wording:

> The 3-D U-Net was trained on the challenge training set. Its final
> scale-normalized training stage was initialized from an earlier task-specific
> checkpoint; no external pretraining was used.

EfficientMedSAM2 Small and Tiny, by contrast, used the organizers' task-specific
FLARE checkpoints.

### 3.6 Hybrid propagation

The hybrid uses the 3-D U-Net's segmentation on the RECIST slice as a SAM2 mask
prompt. SAM2 then propagates it through adjacent slices. This reduces dependence on
an anchor mask inferred only from a point or box. The hybrid/released five-stream
system reached 0.8065 DSC / 0.6743 NSD.

### 3.7 Final U-Net-MedSAM ensemble

The final research system uses two model families and six prediction streams:

1. primary RECIST-guided 3-D U-Net probability;
2. independently trained epoch-195 3-D U-Net;
3. EfficientMedSAM2 Small box-0.9 mask;
4. EfficientMedSAM2 Tiny box-0.9 mask;
5. EfficientMedSAM2 Small three-point mask;
6. U-Net-anchor-to-SAM2 hybrid mask.

"Member" or "prediction stream" does not mean a unique architecture. Several streams
share an architecture but differ in checkpoint, model size, prompt, or anchor.

Let `P3D_1` and `P3D_2` be the two U-Net probabilities, and let `B_Sb`, `B_Tb`,
and `B_Sp` be the three direct EfficientMedSAM2 masks. Define:

```text
B_SAM_mean = (B_Sb + B_Tb + B_Sp) / 3
B_aux = [0.46*P3D_2 + 0.54*B_SAM_mean > 0.5]
M = [0.40*P3D_1 + 0.12*(B_hybrid + B_Sb + B_Tb + B_Sp + B_aux) > 0.5]
```

This is a two-stage weighted fusion. The exact result without organ MoE is 0.8114
DSC / 0.6811 NSD. Do not claim 0.8115/0.6813 for this simplified graph.

Postprocessing is lesion-specific:

- retain the connected component intersecting the corresponding RECIST raster;
- fall back to the largest component if none intersects;
- resolve lesion overlaps using the higher fusion score;
- preserve the RECIST lesion IDs.

Generic connected-component removal and boundary expansion did not improve the
Codabench result.

## 4. Terminology and manuscript wording

Recommended section sequence:

1. RECIST-Derived Spatial Guidance
2. RECIST-Guided Scale-Normalized 3-D U-Net
3. EfficientMedSAM2 Prompting and 3-D Propagation
4. Multi-Model Ensemble Fusion

Prompt explanation:

> A prompt is a spatial cue that identifies the lesion of interest, analogous to
> a user clicking inside a lesion or drawing a bounding box in an interactive
> segmentation system; it is not a text instruction. We automatically derived all
> prompts from the RECIST annotation, eliminating manual interaction. For the 3-D
> U-Net, RECIST guidance was encoded as the rasterized line, a distance-to-line map,
> and the measured lesion diameter. For EfficientMedSAM2, the same annotation was
> converted into positive point coordinates or an axis-aligned box centered at the
> RECIST midpoint.

Sparse-ensemble justification:

> We term the approach a sparse-guidance ensemble because each volumetric prediction
> is conditioned on only a single RECIST diameter annotated on one axial slice—a
> low-dimensional spatial cue compared with a dense 3-D segmentation—while
> complementary 3-D U-Net and EfficientMedSAM2 predictions are combined during
> fusion.

Use "sparse-guidance ensemble" or "sparse-prompt ensemble," not merely "sparse
ensemble." Sparse refers to the input guidance, not sparse parameters or selective
expert activation.

## 5. Filename-derived anatomy

Filename-derived anatomy was used only by the organ-MoE experiments and the final
Docker, not by the simplified 0.8114 U-Net-MedSAM research graph.

`roi3d/predict.py::filename_domain()` searches case names for liver, pancreas,
kidney/renal, lung, or colon. For the MoE, that index selects a residual adapter in
the last two decoder stages. Unrecognized names fall back to the learned router.

During generalist training, source categories were also used for balanced sampling,
but were not passed to the simplified U-Net at inference.

Challenge limitation wording:

> A limitation of the benchmark is that case identifiers often contain
> organ-specific prefixes. These identifiers can provide privileged anatomical
> information and enable metadata-based expert routing without requiring the model
> to infer the organ from the image. Such routing may not generalize to anonymized
> clinical datasets or differently named cases.

Method limitation wording:

> The method depends on a correctly placed RECIST annotation on a representative
> axial slice; errors in this sparse cue can affect both anchor-slice segmentation
> and subsequent 3-D propagation.

## 6. Exact CPU Docker deployment

The final submitted rich Docker is:

```text
artifacts/aidaraomayoclinic-rich-cap120.tar.gz
image tag: aidaraomayoclinic:latest
archive size: approximately 686 MiB
SHA-256: 6ff79ea30a4e60dbc8627d45a99acb192bfab7967f1f000914aa25d9214f922e
Codabench: 0.8001 DSC / 0.6654 NSD
```

It did not package the six-stream research ensemble. The Dockerfile copies exactly
three checkpoints:

1. `organ_moe_epoch65.pt` — one 3-D U-Net with organ-specific residual adapters;
2. `effmedsam2_small.pt` — used for Small box and Small three-point predictions;
3. `effmedsam2_tiny.pt` — used for the Tiny box prediction.

`ours_3d_A2.pt`, the independent epoch-195 U-Net, and the U-Net-anchor hybrid are not
copied into the Docker image.

The Docker uses adaptive CPU inference based on total bounded SAM propagation slices:

### Cost <= 40 slices

Runs four streams:

```text
M = [0.46*P_MoE + 0.18*B_SmallBox + 0.18*B_TinyBox
     + 0.18*B_Small3Points > 0.5]
```

Small/Tiny box use box scale 0.9 and slab scale 1.6. Small three-points uses bounded
propagation and slab scale 1.4.

### Cost 41--120 slices

Runs two streams:

```text
M = [0.57*P_MoE + 0.43*B_TinyBox > 0.5]
```

### Cost > 120 slices

Runs the organ-MoE 3-D model alone.

The 217 hidden cases routed 110/77/30 into the low/medium/high tiers. The exact
organizer-style full CPU run completed in about 1 hour 20 minutes under `-m 8G`.
Stress tests remained below 60 seconds per case. Outputs are flat NPZ files containing
only same-shaped uint8 `segs` arrays with preserved lesion IDs.

Presentation wording:

> The CPU Docker contained one organ-adapted 3-D U-Net checkpoint and two
> EfficientMedSAM2 checkpoints. A runtime-aware policy used four streams for
> inexpensive cases, two streams for intermediate cases, and only the 3-D model for
> expensive cases.

Do not say that six variants were packaged. The organ-MoE contains six internal
adapters but is one checkpoint and one prediction stream.

## 7. Research versus Docker results

| System | DSC | NSD | Scope |
|---|---:|---:|---|
| Final U-Net-MedSAM research ensemble | 0.8114 | 0.6811 | Unrestricted multi-stream predictions |
| Organ-MoE research extension | 0.8115 | 0.6813 | Marginal +0.0001/+0.0002 gain |
| Submitted rich CPU Docker | 0.8001 | 0.6654 | CPU-only, 8 GB, adaptive 1/2/4 streams |

The Docker lost 0.0113 DSC and 0.0157 NSD relative to the clean 0.8114 research
ensemble, in exchange for satisfying the deployment budget.

## 8. Presentation summary

> We progressed from direct EfficientMedSAM2 propagation to a sparse-guidance
> ensemble combining scale-normalized 3-D segmentation, calibrated RECIST-derived
> prompts, and SAM2 propagation. The main gains came from corrected RECIST geometry,
> higher-resolution scale-normalized ROIs, physically bounded propagation, use of a
> 3-D U-Net prediction as a SAM2 anchor, and fusion of independently trained U-Nets
> with complementary box-, point-, and mask-prompted EfficientMedSAM2 streams. The
> research system improved Codabench performance from 0.7722/0.6291 to
> 0.8114/0.6811 DSC/NSD. A separate runtime-aware Docker achieved 0.8001/0.6654 under
> the CPU-only deployment constraints.

## 9. Important files

- `winning.md`: exact pre-MoE winning recipe and history.
- `CLAUDE_HANDOFF.md`: experiment and MoE handoff.
- `REPRODUCE.md`: training and reproduction details.
- `artifacts/subtask2_manuscript_handoff/SUBTASK2_MANUSCRIPT_HANDOFF.md`: manuscript handoff.
- `artifacts/subtask2_manuscript_handoff/RESULTS_LEDGER.csv`: verified score ledger.
- `docker_moe/Dockerfile`: exact files copied into the CPU image.
- `docker_moe/predict.sh`: exact adaptive Docker inference graph.
- `artifacts/rich_candidate_full_cpu_20260902/README.md`: final rich Docker verification.
- `reports/recist_to_3d_case_visualization_vertical_spaced.png`: final vertical case visualization.
- `reports/recist_3d_pipeline_asset_vertical_white.png`: white-background vertical pipeline asset.

