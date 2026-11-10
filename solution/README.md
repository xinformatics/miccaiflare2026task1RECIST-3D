# Competition solution

The persistent rationale and benchmark record is maintained in
[`../DESIGN_DECISIONS.md`](../DESIGN_DECISIONS.md).

This is a high-performance, rules-compliant starting point, not a promise of first
place. A hidden leaderboard cannot be guaranteed. The implementation is designed
against the published Task 2 constraints: CPU-only inference, 8 GB RAM, at most 60
seconds per case, flat NPZ output, and per-lesion DSC/NSD scoring.

## Why this design

The first model track fine-tunes the official EfficientMedSAM2-Small RECIST
checkpoint because the supplied Small/3-point submission scores 0.7722 DSC and
0.6291 NSD on the hidden validation leaderboard. Tiny remains the fallback if
Small cannot meet the CPU limit. The RECIST marker is both a model prompt and a
hard instance anchor.

The inference fork adds several correctness and robustness changes:

1. RECIST endpoints are recovered with a principal-axis fit rather than raster
   scan order. The line-cap over-measure is removed before deriving the box.
2. In-plane diameter is converted to slice extent with the correct `(x,y,z)`
   spacing convention.
3. Forward and reverse propagation start from the same anchor mask.
4. Disconnected 3D false positives are removed by retaining components touching
   the RECIST marker. Predicted marker pixels retain their own lesion ID; the raw
   raster is not forced into the mask because rounded line caps can sit outside GT.
5. Overlapping instances cannot overwrite an instance assigned earlier.
6. Hidden inputs do not need a `gts` key, nested inputs are discovered, and final
   NPZs contain only `segs` unless `--debug-output` is requested.
7. Zero-prompt healthy cases naturally produce an all-zero mask.

## Data protocol

The official public validation set must not be used for training. Use it only once
for final model selection. For iterative development, `make_split.py` creates a
deterministic 90/10 split stratified by source folder and groups obvious BL/FU or
series variants so related scans cannot cross the split.

```bash
python solution/audit_dataset.py dataset/RECIST_train_npz \
  --output artifacts/audit.json
python solution/make_split.py dataset/RECIST_train_npz --output-dir splits
```

## Fine-tuning

The official trainer has been made path-configurable and can initialize from the
task-specific Small checkpoint. Training now starts each one-lesion clip on its
actual RECIST slice, samples either propagation direction, supplies three
positive principal-diameter points for Small or a diameter-sized RECIST square box for
Tiny, and source-balances the highly skewed cancer datasets. These changes mirror
test-time interaction and add no inference cost.
The default run is the CPU-oriented Tiny student: 384-pixel inputs (24x24
stride-16 features), 12-frame one-direction clips, and batch size 8 per GPU.
Small remains at 512 with 8-frame clips. Set `RESOLUTION=512` for the Tiny
accuracy-reference ablation; lower `TRAIN_BATCH_SIZE` if the first run exhausts
GPU memory. `TRAIN_NUM_WORKERS` is per GPU and defaults to 6 on this 12-core,
two-GPU workstation.
On the two local GPUs:

```bash
bash solution/train.sh
```

Useful overrides are environment variables:

```bash
GPUS=2 EPOCHS=6 MODEL_SIZE=small TRAIN_LIST=$PWD/splits/train.txt \
INIT=$PWD/checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt \
bash solution/train.sh dataset/RECIST_train_npz artifacts/finetune
```

The primary CPU-student run is:

```bash
GPUS=2 EPOCHS=6 MODEL_SIZE=tiny RESOLUTION=384 NUM_FRAMES=12 \
bash solution/train.sh dataset/RECIST_train_npz artifacts/finetune-tiny-384
```

Select a checkpoint by mean per-lesion DSC and NSD on `splits/holdout.txt`, with
runtime as a hard feasibility constraint. Do not select by aggregate foreground
Dice; that hides failures on small lesions and multi-lesion cases.

Recommended ablations, in order:

- baseline checkpoint versus fine-tuned epochs 4, 8, and 12;
- box prompts versus 3 interior points;
- prompt-anchored component filtering on/off;
- physical z-slab multipliers 1.2, 1.4, 1.6;
- Tiny versus Small only if Small remains below 60 seconds on the organizer-like CPU.

Change one factor at a time and lock it on the private training holdout before the
single public-validation comparison. The public validation set is not training
data and should not drive repeated tuning.

## Local validation

Predictions can include debug keys locally:

```bash
python baseline/Efficient_MedSAM2_infer_CT_lesion_npz_recist.py \
  --weights checkpoints/eff_medsam2_tiny_FLARE25_RECIST_baseline.pt \
  --model-size tiny --device cuda:0 --no-compile --prompt box --debug-output \
  -i dataset/validation_npz/validation_public_npz \
  -o artifacts/public-validation

python solution/evaluate.py \
  dataset/validation_npz/validation_public_npz artifacts/public-validation \
  --csv artifacts/public-validation-lesions.csv
```

The local NSD implementation uses physical spacing and a 1.0 mm tolerance. Exact
tie behavior can differ slightly from the private scorer, so Codabench remains the
authority.

## Submission and Docker

Raw-result submission:

```bash
python solution/validate_submission.py INPUTS OUTPUTS
python solution/package_submission.py OUTPUTS submission.zip
```

The ZIP is deliberately flat. Instance IDs are preserved and arrays are `uint8`
with the same `(D,H,W)` shape as `imgs`.

Build and smoke-test the CPU image:

```bash
bash solution/build_docker.sh
bash solution/smoke_docker.sh
```

The smoke test enforces `--cpus 12 -m 8g`, mounts inputs read-only, runs the exact
`sh predict.sh` entry point, and validates every output. Benchmark on a machine
close to the published Xeon environment; GPU timing is not evidence of compliance.

## Before the final submission

- Register with real identity/affiliation and comply with the team-size rule.
- Check that any external public model/dataset was disclosed by the organizer's
  deadline. This implementation only relies on the official published baseline.
- Send the Docker archive and sanity-test recording as specified by organizers.
- Produce the required reproducibility paper; the code alone is not a complete
  challenge submission.
- Use the hidden test opportunity only after the container passes the 8 GB and
  60-second tests across the largest validation cases.
