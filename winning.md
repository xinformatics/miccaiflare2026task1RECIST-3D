# Winning recipe — FLARE Pan-Cancer RECIST-to-3D (Task 2)

**Current best: DSC 0.8114 / NSD 0.6811** on the 217-case hidden validation set.
Organizers' provided baseline: 0.7722 / 0.6291. Submission:
`submissions/submission_6way_e195_w40.zip`. Sections 1--8 document the historical
0.8023 recipe on which the current result was built; Section 9 is the complete
epoch-195/six-member handoff for reproducing the current best on another machine.

| submission | DSC | NSD | what changed |
|---|---|---|---|
| our 3D model alone (early ckpt) | 0.7667 | 0.6061 | finer ROI grid (v2) |
| organizers' baseline | 0.7722 | 0.6291 | *the reference to beat* |
| 2-member ensemble | 0.7869 | 0.6443 | + baseline as a member |
| 4-member ensemble | 0.7931 | 0.6517 | + two more SAM2 variants |
| 3-member, tuned prompt (historical) | 0.8023 | 0.6669 | BOX_SCALE 1.0 -> 0.9 |
| released five-member recipe | 0.8065 | 0.6743 | + 3D-anchored SAM2 hybrid |
| **six-member epoch-195 recipe (current)** | **0.8114** | **0.6811** | **+ independent local vote** |

The current recipe is ahead of the organizers' baseline by **+0.0392 DSC / +0.0520 NSD**.

---

## 1. The recipe in one equation

A weighted vote over four members, one soft and three binary:

```
score = 0.46*P_ours + 0.18*B_box09 + 0.18*B_tinybox09 + 0.18*B_3pts     threshold 0.5
```

with **`BOX_SCALE = 0.9`** on every box-prompted member (§1a — this is the change that
crossed 0.80).

| member | definition | public-val DSC / NSD |
|---|---|---|
| `P_ours` | our 3D prompt-conditioned U-Net — **probability**, §2 | 0.8262 / 0.6745 |
| `B_box09` | EfficientMedSAM2-**Small**, `box`, BOX_SCALE 0.9 | **0.8331 / 0.6913** |
| `B_tinybox09` | EfficientMedSAM2-**Tiny**, `box`, BOX_SCALE 0.9 | **0.8279 / 0.6743** |
| `B_3pts` | EfficientMedSAM2-**Small**, `3points` | 0.7933 / 0.6365 |

All three primary members are now comparable in strength, which is the ideal condition for
ensembling. Equal weights among the binary members: a 300-vector search over per-member
weights (with a held-out half of public val) improved on this by **+0.0008**, i.e. nothing —
with binary votes the decision depends on *how many* members agree, not their exact weights.

Post-processing, per lesion:
1. threshold the combined score at 0.5;
2. keep the connected component seeded by the RECIST line — **fall back to the largest
   component**, because the drawn stroke lies outside its own lesion ~10 % of the time
   (concave lesions plus `cv2.line` end-caps);
3. resolve overlaps between lesions in the same case by higher score.

Only our member emits probabilities, so with binary votes the rule behaves as an **adaptive
threshold**: where all three external members vote yes, our model needs only `p > 0.06`;
where all vote no, it needs `p > 0.83`. **The weight on the soft member must exceed 0.5 in the
two-member case**, or the rule collapses to the binary member exactly.

### Why it wins
Not because the 3D model is excellent — it is 0.8262 against box's 0.8214, effectively a tie.
It wins because the two families **agree at only 0.765 Dice**. Ensembling comparable models
with decorrelated errors gave **+0.0147 DSC / +0.0152 NSD** over the better member, while every
single-model lever tried returned ~0.001 (§5).

---

## 2. Our member: prompt-conditioned 3D U-Net on a scale-normalized ROI

Each lesion is segmented independently from a crop keyed to its **own RECIST diameter `D`**,
so a 12 mm nodule and a 150 mm mass are presented at the same relative scale and at constant
compute cost.

**ROI geometry** (`src/roi_geom.py`, version `v2`):
```
patch (Z,Y,X) = (112, 176, 176)
s(D) = max( clip(2.9*D/192, 0.45, 1.10), 1.6*D/192 )   mm, isotropic
```
- The **1.10 mm cap** exists because NSD's tolerance is 1.0 mm: the earlier 2.2 mm grid made
  measured NSD collapse from 0.896 to 0.359 on large lesions **while DSC stayed flat** — the
  signature of a resolution-limited metric. 29 % of test lesions exceed 50 mm, where the old
  grid was coarser than the tolerance. Worth **+0.0098 NSD real**.
- The **`1.6*D/192` floor** keeps the field of view at least 1.6x the diameter, so the finer
  grid does not truncate very large lesions (that cost −0.035 NSD above 110 mm before it was
  added).

**Network** (`src/model.py`): 5-level 3D U-Net, base 32 (16.5 M params), InstanceNorm +
LeakyReLU, deep supervision on the 3 finest scales. Four input channels:

| ch | content | why |
|---|---|---|
| 0 | image / 255 | |
| 1 | rasterized RECIST line | the prompt |
| 2 | `exp(-dist_mm / (0.4*D))` | soft distance-to-line prior |
| 3 | `log(D/30) / 1.5` (constant plane) | absolute size, which scale-normalization erases |

**Inference order matters**: predict probabilities on the ROI grid, resample to the native
grid, **then** threshold. Never threshold first. `geom.sample_roi` / `geom.scatter_roi` are
exact inverses (round-trip DSC 1.000).

---

## 3. Training recipe

```bash
bash src/build_geo_all.sh        # 28,068 lesion ROIs, ~45 min, 147 GB
bash src/keeper.sh ./run_A2.sh logs/train_A2.log logs/A2.pid 299
```
```
--base 32 --bs 2 --lr 5e-4 --epochs 300 --steps-per-epoch 250
--cache-root cache_geo --p-cache 128,192,192 --p-train 112,176,176
--rot-z 0.35 --rot-xy 0.10 --scale-aug 0.15 --shift-aug 6
--band-w 3.0 --band-k 2 --size-alpha 0.75
```

Loss: Dice + BCE, with BCE up-weighted 4x within 2 voxels of the target surface.

**Two sampling corrections, both from measured distribution shifts:**

- **Domain-balanced** (`TARGET_W` in `src/train.py`) — Liver/Pancreas/Kidney 0.20 each,
  Lung 0.18, Colon 0.08, remaining five datasets 0.14 combined. Without it Liver alone
  (9,157 of 28,068 lesions) dominates.
- **Size-stratified** (`--size-alpha 0.75`) — training lesions are far smaller than test
  lesions (**57 % of training lesions are under 15 mm against 7 % of test lesions**; liver
  median 12.7 mm vs 37.8 mm). Below ~25 mm the ROI spacing clips at its floor, so those
  lesions are not scale-normalized the way larger ones are. alpha=0.75 is a deliberate
  partial correction: exact matching would cut effective unique lesions from 7,205 to 2,963,
  and 84 % of the gap closes by 0.75.

**Augmentation** (`src/augment.py`, all on GPU): affine rotation/scale/shift/flip sampled out
of a larger cached ROI with the RECIST endpoints carried through the same transform; intensity
gamma/brightness/noise; and `simulate_slice_thickness`, which blurs and re-discretizes along z
because the corpus is thinner-sliced than the test set (median 2.50 mm vs 3.74 mm; **zero**
held-out lung case exceeds 3.5 mm while 45 % of test lung does). Image only — the metric
resamples our prediction onto the native grid, so the fine mask remains the correct target.

---

## 4. Generating the external members

```bash
cd ref/MedSAM2-RECIST
bash download_EfficientMedSAM2_Small.sh
bash download_EfficientMedSAM2_Tiny.sh

python3 Efficient_MedSAM2_infer_CT_lesion_npz_recist.py \
    --weights checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt \
    --model-size small --device cpu --prompt box \
    --gts-path <dir of zero-filled gts> -i <in> -o <out>
```

### Three traps
1. **`--prompt box`, not `3points`.** The organizers' zip is named `..._3points.zip` but was
   generated with **box** — our box run matches their masks at **0.960** Dice versus 0.880 for
   3points. Running 3points scores **0.7218 vs 0.7722**, a 0.05 DSC loss, silently.
2. **The script demands a `gts` key even for inference.** It only saves and overlays it, never
   uses it to segment, so pass `--gts-path` at a directory of zero-filled `gts` arrays (the
   hidden set has none).
3. `SLAB_FORMULA` / `Z_SLAB_SCALE` near the top of the file changed after the published
   baseline was made. Measured: they move the score by <=0.003, so this is **not** worth
   chasing — noted only because the comments invite it.

---

## 5. What did NOT work — do not re-run these

| lever | measured effect |
|---|---|
| test-time augmentation (4-flip) | **+0.001** — the model is already flip-invariant from training augmentation |
| threshold tuning | ±0.0002 — probabilities are saturated |
| morphological post-processing | <=0.001 (opening: **−0.16**) |
| RECIST-diameter threshold calibration | ~0 |
| logit-space interpolation | +0.0000 |
| boundary-weighted loss (as a *change*) | +0.003 local over 77 epochs |
| **Glance-and-Focus 2D boundary refinement** | **−0.003** |
| more training past ~ep115 | ~+0.001 |

**Fine-tuning EfficientMedSAM2 did not work** for a structural reason worth remembering: the
released checkpoint had **already been fine-tuned by the organizers on this exact data**, so
12 epochs on 2,298 z-cropped cases produced a model worse than its starting point. Fine-tuning
helps when adapting a *generic* model to a task; that was not the situation.

**Glance-and-Focus deserves its own note** because the idea is compelling and the oracle was
seductive. An oracle said perfecting the in-plane contour on the slices already found would
recover **88 % of NSD loss (+0.125)**. A real refiner trained on 19,175 boundary patches
delivered **−0.0032**. The oracle asked *"what if the contour were correct?"* — but for small
lesions stage 1 already runs at 0.45 mm, finer than the 0.78 mm acquisition, so there is no
resolution to recover and refinement only injects noise. **The binding constraint is
information in the image, not grid resolution.**

An oracle picking the better of our two strongest members per lesion scores only **+0.011**
over the weighted vote — so no cleverer *combination* of existing members gets much further.
Gains must come from new, genuinely decorrelated members.

---

## 6. Evaluation discipline

**Use `validation_public_npz` (49 cases, ships GT). Never tune on held-out training cases.**

A held-out split of the training data is badly misleading: the baseline scores **0.634 DSC
there but 0.772 on the hidden set**, and is worse than our model in *every* lesion-size bin,
because it shares scanners and annotation protocols with our training data. Reweighting by
lesion size does not fix it. Public val reproduces the hidden set's ordering and its absolute
NSD (~0.67 vs the hidden 0.61–0.64), where the training split read an inflated 0.86.

**Public-val gains translate to roughly 0.3–0.6x on the hidden set** — treat them as an upper
bound. (Predicted +0.011 DSC / +0.024 NSD for the 4-way; reality gave +0.0062 / +0.0074.)

**Prefer paired A/B on identical lesions.** Paired tests predicted reality well (+0.012
predicted vs +0.0098 actual); global reweighted estimates overstated by 4x.

```bash
python3 src/score_dir_pub.py <pred_dir> ...                        # score vs public val
python3 src/paired_dirs.py --a <A> --b <B> --ref <ref>             # paired A/B
python3 src/multi_ens.py --ours <pkl> --dirs <m1> <m2> ... --ref <ref>   # member sweep
python3 src/make_submission.py --out <zip> --verify-only           # 6 format checks
```

---

## 7. Reproducing the submission

```bash
# external members on the hidden set (see §4 for the gts-path trick)
#   -> work/msam_hidden_box, work/msam_hidden, work/msam_hidden_tiny

python3 src/make_ensemble_submission.py \
    --ckpt models/A2_final.pt \
    --member-dirs work/hid_box09_small work/hid_box09_tiny work/msam_hidden \
    --weights 0.46,0.18,0.18,0.18 \
    --out subs/submission_box09.zip --preds preds/hidden_box09 --geo v2 --device cuda
```
Verification is automatic: flat archive, exact filenames, `segs` key, uint8, shapes matching
each `imgs` volume, and no lesion id outside the prompts. Any one of those failing scores 0.

---

## 8. Inference budget (the September Docker must fit 60 s/case, CPU-only, 8 GB)

Measured, 8 CPU threads:

| component | cost |
|---|---|
| our 3D model, base 32 | 4.25 s/lesion -> 21.2 s on a 5-lesion case |
| our 3D model, base 40 | 6.51 s/lesion -> 32.5 s |
| our 3D model, base 48 | 7.92 s/lesion -> 39.6 s |
| EfficientMedSAM2-Small | 3.4–10.6 s/case |
| EfficientMedSAM2-Tiny | ~half of Small |

base 32 + two SAM2 members ~ **36 s**; base 40 + two ~ **47 s**; base 48 leaves only ~5 s of
margin on a host whose speed is unknown, so **base 40 is the largest defensible 3D member**
*for the container*. The limit does **not** apply to validation-leaderboard submissions (results
zip), so the leaderboard entry and the container may legitimately differ in capacity.

**The 60 s CPU limit is not achievable by any SAM2-based method on large multi-lesion cases.**
The organizers' own unmodified script takes **145 s** on the worst hidden case (Liver_038,
5 lesions, 274 MB) on CPU; our unified pipeline takes 95 s on the same case. Their published
timings (mean 3.6-5.0 s, max 52 s, none over 60) were measured on **GPU**. The container
therefore needs per-case adaptive compute that drops ensemble members rather than overrunning
— implemented in `docker2/src/predict_ensemble.py`, and a necessity rather than a nicety.
About 7% of cases (5 lesions) are affected.

ONNX gives **no** speedup at the v2 patch (4.06 s either way); an earlier apparent 2x came from
comparing different patch sizes, not from ONNX.
`src/predict_onnx.py` already implements per-case adaptive compute: members are dropped, most
uncertain patches first, if the measured per-view cost implies an overrun — it self-calibrates
from observed timings rather than trusting a hardcoded constant.

---

## 9. Current best: independent epoch-195 model as a sixth vote

This section is intentionally self-contained so it can be copied to another machine. The
epoch-195 checkpoint is an **independently trained local reconstruction**, not the released
`ours_3d_A2.pt`. Its gain comes from decorrelated errors. Keep the released five-member core
and add the epoch-195 model's fused binary output as a sixth vote; do not replace the released
soft model with it.

### 9.1 Measured result

| recipe | public lesion DSC / NSD | hidden DSC / NSD |
|---|---:|---:|
| released five-member core | 0.850001 / 0.656186 | 0.8065 / 0.6743 |
| + independent epoch-195 binary vote | **0.851507 / 0.658056** | **0.8114 / 0.6811** |

The real gain is **+0.0049 DSC / +0.0068 NSD**. Freeze 0.8114 / 0.6811 as the
hidden baseline for subsequent experiments.

### 9.2 Files and integrity

- Checkpoint: `artifacts/roi3d-a2-long300/epoch_195.pt`
- Checkpoint SHA-256:
  `b9630a53243dc699ae6d098062bc377ac1b33a83402620ad467f245b17cdb047`
- Initializer: `artifacts/roi3d-a2-corrected/epoch_3.pt`
- Initializer SHA-256:
  `3ef789c914e5c2475f9bd80b7035f5eab97941557b039843e8343dfa0d4ce3fe`
- Required implementation: `roi3d/model.py`, `roi3d/geometry.py`, `roi3d/train.py`,
  `roi3d/predict.py`, and `roi3d/build_cache.py`
- Exact launcher: `solution/train_roi3d_corrected.sh`

The local checkpoint is not state-dict compatible with
`othermachine/release/models/ours_3d_A2.pt`: the two implementations use different module
names. Transfer the local `roi3d` source with the checkpoint.

### 9.3 Architecture and inputs

- Prompt-conditioned five-level 3D U-Net, 16,546,883 parameters, base width 32.
- Encoder widths: 32, 64, 128, 256, 320.
- Two 3x3x3 convolutions per block, InstanceNorm with affine parameters, and LeakyReLU 0.01.
- Transposed-convolution decoder and deep supervision at the three finest output scales.
- Four input channels:
  1. image divided by 255;
  2. rasterized RECIST line;
  3. `exp(-distance_mm / (0.4 * corrected_diameter_mm))`;
  4. constant `log(corrected_diameter_mm / 30) / 1.5`.

### 9.4 ROI geometry

- Cache ROI: `(128, 192, 192)` in Z,Y,X order.
- Training and inference patch: `(112, 176, 176)`.
- Estimate the line direction by the principal axis of the RECIST raster.
- Correct the thickness-2 raster's end caps by subtracting **1.75 pixels** from its extent.
- Use isotropic ROI spacing:

```python
s = max(
    clip(2.9 * corrected_diameter_mm / 192, 0.45, 1.10),
    1.6 * corrected_diameter_mm / 192,
)
```

At inference, predict a probability on the ROI grid, resample that probability to the native
volume with trilinear interpolation, and only then threshold or fuse. Thresholding before
native-grid resampling changes the boundary and must not be used.

### 9.5 Initialization and exact training schedule

The corrected epoch-3 initializer fixed the source alias so
`Dataset006_PancreaticCancer` maps to `pancreas`, not the 14% `other` pool. Load only its
model weights, then start a **fresh optimizer and fresh 300-epoch scheduler**. Thus epoch 195
means 195 epochs of the new long run, not epoch 198 of one continuous optimizer trajectory.

```text
seed                 2026
batch size           18
steps per epoch      250
selected epoch       195 (48,750 long-run optimizer steps)
planned epochs       300
initial LR           5e-4
final LR             5e-6
optimizer            fused AdamW, weight_decay=1e-5
scheduler            cosine over 75,000 planned steps
precision            BF16 autocast
gradient clipping    norm 12
memory format        channels_last_3d
compile              torch.compile(mode="reduce-overhead")
workers              4
save interval         5 epochs
```

The exact command is:

```bash
bash solution/train_roi3d_corrected.sh
```

or equivalently:

```bash
python -m roi3d.train \
  --cache-root cache_geo_v2 \
  --hidden dataset/validation_npz/validation_hidden_npz \
  --output artifacts/roi3d-a2-long300 \
  --init artifacts/roi3d-a2-corrected/epoch_3.pt \
  --base 32 --batch-size 18 --lr 5e-4 --epochs 300 \
  --steps-per-epoch 250 --workers 4 --size-alpha 0.75 \
  --rot-z 0.35 --rot-xy 0.10 --scale-aug 0.15 --shift-aug 6 \
  --band-w 3.0 --band-k 2 --save-every 5 --compile
```

### 9.6 Sampling and augmentation

The completed cache contains 29,754 lesion ROIs. Use a weighted sampler with replacement and
the following target domain mass:

```text
liver      0.20
pancreas   0.20
kidney     0.20
lung       0.18
colon      0.08
other      0.14 combined
```

Apply size-stratified sampling with alpha 0.75 and diameter bins
`[0,10,15,20,30,40,60,90,130,inf]` mm. The target histogram is computed from the hidden
inputs' RECIST diameters only; hidden labels are never used.

Augmentation:

- Z rotation +/-0.35 radians and X/Y rotation +/-0.10 radians.
- Isotropic scale +/-15%, translation +/-6 voxels, and independent axis flips at p=0.5.
- Slice-thickness simulation at p=0.5 with factor 2 or 3: average-pool and nearest-neighbour
  re-discretize the image along Z while retaining the fine segmentation target.
- Carry the physical RECIST endpoints through the sampled affine before rebuilding the line
  and distance channels.

### 9.7 Loss

At each deep-supervision scale use boundary-weighted BCE-with-logits plus soft Dice loss.
Deep-supervision weights are 1.0, 0.5, and 0.25. The boundary band has radius two voxels and
the BCE weight is `1 + 3 * boundary_band`.

### 9.8 Why epoch 195, not the final checkpoint

| local four-way member | public lesion DSC | public lesion NSD |
|---|---:|---:|
| epoch 195 | **0.849817** | 0.653710 |
| epoch 250 | 0.849030 | **0.654420** |
| epoch 275 | 0.847560 | 0.652456 |
| epoch 300 | 0.847999 | 0.652963 |

Later training reduced validation generalization despite lowering training loss. Use epoch
195 unless a new paired evaluation demonstrates a better independently trained member.

### 9.9 First-level epoch-195 fusion

The sixth vote is not the raw epoch-195 mask. First create a four-member fusion:

```text
score_e195 =
    0.46 * P_epoch195
  + 0.18 * B_small_box09
  + 0.18 * B_tiny_box09
  + 0.18 * B_small_true_3points

B_e195 = score_e195 > 0.5
```

Per lesion, retain the connected component touched by the RECIST line, falling back to the
largest component when the raster does not touch a prediction. Resolve overlaps between
lesions by retaining the label with the higher fusion score.

Generate it with:

```bash
python -m roi3d.predict \
  -i <input_npz_dir> \
  -o <epoch195_fourway_dir> \
  --checkpoint artifacts/roi3d-a2-long300/epoch_195.pt \
  --member-dirs <small_box09_dir> <tiny_box09_dir> <small_true_3points_dir> \
  --ensemble-weights 0.46,0.18,0.18,0.18 \
  --device cuda:0
```

### 9.10 Final six-member recipe

Preserve the exact released five-member core and add `B_e195`:

```text
score_final =
    0.40 * P_released_epoch275
  + 0.12 * B_released_hybrid_anchor
  + 0.12 * B_small_box09
  + 0.12 * B_tiny_box09
  + 0.12 * B_small_true_3points
  + 0.12 * B_e195

final = score_final > 0.5
```

Again apply RECIST-seeded connected-component selection with largest-component fallback and
resolve overlaps using the higher final score. The validated hidden archive is
`submissions/submission_6way_e195_w40.zip`, SHA-256
`4f446e676e7ec8c7e80fb11f1e72b7aed0f16a3c6259f836fcfa10e39ee5e7bc`.

Do not submit any archive whose name begins with `public-`; those contain only the 49 public
validation cases and score zero on the 217-case hidden evaluation.
