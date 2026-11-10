# FLARE Pan-Cancer RECIST-to-3D (Task 2) — full reproduction guide

Everything needed to rebuild, retrain and extend the current best submission on another
machine. Written for a reader with the repo and a GPU but none of this session's context.

**Current best: `subs/submission_ens058.zip` — DSC 0.7869 / NSD 0.6443** on the 217-case
hidden validation set, versus the organizers' provided baseline at 0.7722 / 0.6291.

---

## 1. The task in one paragraph

Given a CT volume (`imgs`, uint8, already windowed to [0,255]) and a 2D RECIST diameter line
per lesion (`recist`, drawn on exactly one axial slice, valued with the lesion id), predict a
3D mask (`segs`, uint8, same shape, ids matching the prompt). Scored per lesion by DSC and
NSD at **1.0 mm tolerance**, then averaged per case. Final delivery is a **CPU-only Docker
container, < 60 s per case, 8 GB RAM**.

---

## 2. Hardware used, and what a bigger machine changes

| | this run | effect of more |
|---|---|---|
| GPU | 1x RTX 4060 Ti, 16 GB | **batch size is the binding constraint** — the v2 patch (112,176,176) only fits at bs=2. 24-48 GB would allow bs=4-8, roughly 2-4x faster convergence |
| Disk | 396 GB raw + 226 GB caches | needs ~650 GB total |
| RAM | 124 GB | comfortable; caches are memmapped |
| CPU | 16 cores | cache building is IO/CPU bound, ~45 min per full pass |

The single highest-value upgrade is **GPU memory**: at bs=2 the model saw ~85 k samples in
~24 h. On a 48 GB card at bs=8 the same schedule is roughly 4x cheaper.

---

## 3. Data

```bash
pip install huggingface_hub hf_transfer
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('FLARE-MedFM/PancancerRECIST-to-3D', repo_type='dataset',
                  local_dir='data', max_workers=8)
PY
```
424 GB, 9,225 training cases across 10 cancer-type datasets, plus `validation_hidden_npz`
(217 cases, the submission target, **no GT**) and `validation_public_npz` (49 cases, **has
GT** — see §7, this is the only trustworthy local proxy).

---

## 4. Method — prompt-conditioned 3D U-Net on a scale-normalized ROI

Each lesion is segmented independently from a crop keyed to its own RECIST diameter `D`, so a
12 mm nodule and a 150 mm mass are presented at the same relative scale and at constant cost.

**ROI geometry (`src/roi_geom.py`)** — two versions exist; **v2 is the one to use**:

| | patch (Z,Y,X) | spacing s(D) |
|---|---|---|
| v1 | (112,128,128) | `clip(2.9*D/144, 0.5, 2.2)` mm |
| **v2** | **(112,176,176)** | **`max(clip(2.9*D/192, 0.45, 1.10), 1.6*D/192)`** |

v2 exists because **NSD is resolution-limited**: with a 1.0 mm tolerance, v1's spacing exceeds
it above ~50 mm diameter (29 % of test lesions), and measured NSD there collapses from 0.896
to 0.359 while DSC stays flat. The `1.6*D/192` floor prevents the finer grid from shrinking
the field of view and truncating very large lesions (that cost -0.035 NSD above 110 mm before
it was added). **v1 -> v2 was worth +0.0098 NSD on the real leaderboard.**

**Network** (`src/model.py`): 5-level 3D U-Net, base 32, InstanceNorm + LeakyReLU, deep
supervision on the 3 finest scales, 16.5 M params. Four input channels:
1. image / 255
2. rasterized RECIST line
3. `exp(-dist_mm / (0.4*D))` — soft distance-to-line prior
4. constant `log(D/30)/1.5` — absolute size, which scale normalization would otherwise destroy

**Inference**: predict probabilities on the ROI grid, resample to the native grid, *then*
threshold — never the reverse. `geom.sample_roi` / `geom.scatter_roi` are exact inverses
(verified round-trip DSC 1.000).

---

## 5. Build the cache and train

```bash
# ROI cache (v2 geometry) — ~45 min, 147 GB, 28,068 lesions
bash src/build_geo_all.sh

# train (resumes automatically; keeper restarts on transient CUDA faults)
bash src/keeper.sh ./run_A2.sh logs/train_A2.log logs/A2.pid 299
```

`run_A2.sh` is the exact recipe:
```
--base 32 --bs 2 --lr 5e-4 --epochs 300 --steps-per-epoch 250
--cache-root cache_geo --p-cache 128,192,192 --p-train 112,176,176
--rot-z 0.35 --rot-xy 0.10 --scale-aug 0.15 --shift-aug 6
--band-w 3.0 --band-k 2 --init models/main32/last.pt
```
On a larger GPU, raise `--bs` and scale `--lr` with it.

**Two non-obvious pieces of the recipe, both measured:**

- **Domain-balanced sampling** (`TARGET_W` in `src/train.py`) matches the hidden set's mix:
  Liver/Pancreas/Kidney 0.20 each, Lung 0.18, Colon 0.08, the rest 0.14 combined. Without it
  Liver (9,157 of 28,068 lesions) dominates.
- **Size-stratified sampling** (`--size-alpha 0.75`). Training lesions are far smaller than
  test lesions (liver median 12.7 mm vs 37.8 mm; **57 % of training lesions are under 15 mm
  against 7 % of test lesions**), and below ~25 mm the ROI spacing clips at its floor so those
  lesions are not scale-normalized the way larger ones are. alpha=0.75 is a deliberate partial
  correction: exact matching cuts effective unique lesions from 7,205 to 2,963, and 84 % of
  the distribution gap closes by 0.75.

**GPU augmentation** (`src/augment.py`): affine (rotation/scale/shift/flip) sampled out of a
larger cached ROI, with the RECIST endpoints carried through the same transform; plus
`simulate_slice_thickness`, which blurs and re-discretizes along z because the corpus is
thinner-sliced than the test set (median 2.50 mm vs 3.74 mm; *zero* held-out lung case exceeds
3.5 mm while 45 % of test lung does). Image only — the metric resamples our prediction onto
the native grid, so the fine mask stays the correct target.

---

## 6. The ensemble — where the win actually came from

Our model alone reached 0.7667 / 0.6061. The organizers' baseline reaches 0.7722 / 0.6291 by
a completely different mechanism (per-slice mask propagation at native 512x512 from a SAM2
foundation model) and **agrees with ours at only 0.765 Dice**. Averaging two comparable models
with decorrelated errors beat both:

```bash
python3 src/make_ensemble_submission.py --ckpt models/A2/last.pt \
    --baseline-zip subs/baseline_efficientmedsam2small_3points.zip \
    --out subs/submission_ens.zip --preds preds/hidden_ens --w 0.58 --geo v2
```

The baseline ships **masks, not probabilities**, so the rule is `w*p + (1-w)*b > 0.5`, which
with binary `b` acts as an adaptive threshold: where the baseline votes yes we need
`p > 0.167`, where it votes no we need `p > 0.833`. **`w` must exceed 0.5** or the rule
collapses to the baseline exactly. w=0.58 gave **+0.0147 DSC / +0.0152 NSD over the better
member**.

### CRITICAL: regenerating the baseline masks yourself

For the September Docker the container must produce these masks itself. `medfm-flare/
MedSAM2-RECIST` **has been updated since the published baseline was generated**, and two
constants near the top of `Efficient_MedSAM2_infer_CT_lesion_npz_recist.py` control it:

```python
Z_SLAB_SCALE = 1.6      # was 1.0
SLAB_FORMULA = "correct"     # alternatives: "repo", "docker"
```

Their own comment says `docker` is *"the variant baked into the benchmark docker images, i.e.
the one that produced every historical result."* These are **not CLI flags** — edit the file.
Running with the current defaults scored **0.7218 / 0.5667**, i.e. 0.05 DSC *worse* than the
published 0.7722, because `correct` bounds z-propagation to roughly half the extent and
truncates lesions. Command otherwise:

```bash
bash download_EfficientMedSAM2_Small.sh
python3 Efficient_MedSAM2_infer_CT_lesion_npz_recist.py \
  --weights checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt \
  --model-size small --device cpu --prompt 3points -i <in> -o <out>
```
The script demands a `gts` key even for inference (it only saves/overlays it, never uses it to
segment) — pass `--gts-path` at a directory of zero-filled `gts` arrays for the hidden set.

---

## 7. Evaluation — read this before trusting any local number

**Use `validation_public_npz` (49 cases, has GT). Do not tune on held-out training cases.**

A held-out split of the *training* data (`data/local_val`) is badly misleading: the baseline
scores **0.634 DSC there but 0.772 on the hidden set**, and is worse than our model in every
lesion-size bin, because local_val shares scanners and annotation protocols with our training
data. Reweighting it by lesion size does not fix this. The public validation set reproduces
the hidden set's ordering (baseline competitive with ours, ensemble beating both) and its
absolute NSD (~0.67 vs the hidden 0.61-0.64), where local_val read an inflated 0.86.

**Always prefer paired A/B on identical lesions.** Paired tests predicted reality well
(+0.012 predicted vs +0.0098 actual); global reweighted estimates overstated by 4x.

```bash
python3 src/score_dir_pub.py <pred_dir> ...            # score against public val
python3 src/paired_dirs.py --a <dirA> --b <dirB> --ref <ref_dir>   # paired A/B
python3 src/make_submission.py --out <zip> --verify-only           # 6 format checks
```

---

## 8. Measured results — what worked and what did not

| lever | effect | verdict |
|---|---|---|
| finer ROI grid v1 -> v2 | **+0.0098 NSD (real)** | ship |
| ensemble with the baseline | **+0.0147 DSC / +0.0152 NSD (real)** | ship |
| size-stratified + domain-balanced sampling | in the recipe | ship |
| slice-thickness augmentation | in the recipe | ship |
| more training past ~ep115 | ~+0.001 | exhausted |
| test-time augmentation (4-flip) | +0.001 | dead — model already flip-invariant from training aug |
| threshold tuning | +/-0.0002 | dead — probabilities are saturated |
| morphological post-processing | <=0.001 (opening -0.16) | dead |
| RECIST-diameter threshold calibration | ~0 | dead |
| logit-space interpolation | +0.0000 | dead |
| boundary-weighted loss | +0.003 local over 77 epochs | not the lever |
| **Glance-and-Focus 2D refinement** | **-0.003** | dead — see below |

**Why Glance-and-Focus failed, because it is a tempting idea.** An oracle said perfecting the
in-plane contour on the slices already found would recover **88 % of NSD loss (+0.125)**. A
real refiner trained on 19,175 boundary patches delivered **-0.0032**. The oracle asked *"what
if the contour were correct?"* — but for small lesions stage 1 already runs at 0.45 mm, finer
than the 0.78 mm acquisition, so there is no resolution to recover and refinement only adds
noise. **The binding constraint is information in the image, not grid resolution.** Only large
lesions improved (+0.0057), and gating to them recovers just +0.0011.

**An oracle that picks the better of our two models per lesion scores only +0.011 over the
w=0.58 ensemble** — so no smarter combination of *these two* models gets much further. Beyond
that requires genuinely better or more diverse members.

---

## 9. Data quirks that cost real time to find

- The RECIST stroke is drawn with `cv2.line(thickness=2)`, whose end-caps extend past the true
  endpoints. A naive farthest-pair length **overshoots the GT diameter by 6.7 %** (13 % on
  small lesions). Principal-axis extent minus **1.75 px** recovers it to 0.1 mm MAE
  (`geom.LINE_CAP_PX`). `diam_mm` (raw) and `diam_true_mm` (de-biased) are distinct.
- The prompt always sits on the lesion's largest-area axial slice, on exactly one z slice.
- The stroke lies **outside** its own lesion ~10 % of the time (concave lesions plus the
  caps), so any seed-based method needs a fallback.
- Kidney/Colon/Pancreas volumes have direction cosines `(-1,-1,1)`; both validation sets are
  identity — the anatomy is mirrored. Random flip augmentation neutralizes it.
- Some training npz (e.g. `Colon-MSD_005`) have no `recist` key; regenerate from `gts`.
- **Thicker slices slightly *raise* NSD** (+0.012 on a controlled same-case test), because a
  correctly placed thick slab contributes a large face entirely inside the 1 mm tolerance.
  Thickness costs DSC (-0.022), not NSD. The intuition runs the other way.

---

## 10. Where to take it next

1. **Retune the EfficientMedSAM2 slab settings.** `SLAB_FORMULA`/`Z_SLAB_SCALE` were never
   tuned — the published baseline just used whatever the old docker had. Sweeping them on the
   public validation set should give a member *stronger* than 0.7722, lifting the ensemble.
   (`src/slab_sweep.sh`)
2. **More diverse members**: EfficientMedSAM2-Tiny, and the Small model with a `box` prompt —
   same weights, different error pattern. MedSAM2-full currently fails to load (Hydra config
   error in the repo).
3. **Fine-tune EfficientMedSAM2 on the full 424 GB.** The organizers fine-tuned it on some
   subset; we have everything. This is the most promising unexplored direction and wants a
   big GPU.
4. **Bigger batch / longer schedule for A2** on a larger card.
5. **September Docker**: must run both models CPU-only under 60 s/case. Measured so far —
   EfficientMedSAM2-Small 3.4–10.6 s/case on CPU, our 3D model ~2 s/lesion. `src/predict_onnx.py`
   already has adaptive per-case compute budgeting and self-calibrating timing for an unknown
   host.
