# Validation & Inference Guide

How to validate a trained classification experiment, run inference (HDF5 or
raw event directories), and use the Python API.

**Related:** [Training guide](training.md) · [Configuration guide](configuration.md) · [Model building & loading guide](model_loading.md)

---

## Validation

Edit `run_validation_v2.sh` to set `CHECKPOINT_DIR` and `DATA_PATH`, then run:

```bash
bash run_validation_v2.sh
```

Or call the CLI directly:

```bash
bioairmet-validate \
    --experiment_dir /path/to/cls_experiment_dir \
    --data_path /path/to/validation_data.h5 \
    --checkpoint_type best \
    --save_path ./validation_results \
    --gpu_id 0
```

Outputs saved to `--save_path`:

| File | Description |
|---|---|
| `validation_log.txt` | Top-1 accuracy, loss, and ACE calibration score |
| `absolute_values_absolute.png` | Absolute confusion matrix |
| `normalized_values_normalized.png` | Row-normalised confusion matrix |

### CLI reference: `bioairmet-validate`

```
bioairmet-validate --experiment_dir DIR --data_path PATH [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--experiment_dir` | Yes | — | Experiment directory (`--checkpoint_dir` is a deprecated alias) |
| `--data_path` | Yes | — | HDF5 validation file |
| `--checkpoint_type` | No | `best` | `best` or `last` |
| `--save_path` | No | `./validation_results` | Output directory (`--plot_path` is a deprecated alias) |
| `--gpu_id` | No | `0` | GPU index, `-1` for CPU |
| `--batch_size` | No | *(from config)* | Override batch size |
---

## Inference

Edit `run_inference.sh` to set `EXPERIMENT_DIR`, `DATA_PATH` and `SAVE_PATH`, then run:

```bash
bash run_inference.sh
```

Or call the CLI directly:

```bash
bioairmet-inference \
    --experiment_dir /path/to/cls_experiment_dir \
    --data_path /path/to/data.h5 \
    --save_path ./inference_results \
    --confidence_threshold 0.0 \
    --checkpoint_type best \
    --gpu_id 0
```

**For unlabeled raw directory mode**, per-event **quality checks** filter out
events whose spectra/holograms fail basic validity gates (only used when
`--data_path` points to a raw event directory; ignored for HDF5 input —
that data was already cleaned during preparation):

| Parameter | Default | Description |
|---|---|---|
| `--min_area` | `500` | Minimum region area (pixels) |
| `--min_solidity` | `0.7` | Minimum region solidity |
| `--min_intensity_delta` | `0.1` | Minimum max–min intensity difference |

By default, events that fail these checks are **dropped** from the dataset
(the run log reports how many were skipped). If you want to keep *every*
event with usable image files — including ones that fail the checks — pass
`--skip_validation`; the gates are then bypassed entirely.

`img_reader_type`, `batch_size`, and `num_workers` are read automatically
from `experiment_dir/config.yaml` — no need to specify them on the command
line. `num_workers` is capped at the machine's CPU count automatically
(a warning is logged when it is reduced), so inference always runs, even on
small machines.

**Output directory.** For labeled HDF5 input, all outputs are written into a
subdirectory of `--save_path` named after the HDF5 file without its
extension — e.g. `./validation_results/dataxyz/matrix_normalized.png` — so
several files evaluated against the same model don't mix up. Unlabeled
raw-directory input writes directly into `--save_path`.

Outputs (written to that output directory):

| File | Description |
|---|---|
| `predictions_<timestamp>.csv` | Per-sample predictions. `--output_mode compact` (default): `image_path, predicted_class, probability` (+ `true_class, correct` for labeled data). `--output_mode detailed`: additionally `predicted_class_id`, top-2/top-3 alternatives and the full per-class probability distribution (`prob_<class>` columns). |
| `inference_summary_<timestamp>.json` | Machine-readable run summary (accuracy, confidence statistics, per-class counts). |
| `inference_<timestamp>.log` | Detailed run log. |
| `absolute_values_absolute.png` / `normalized_values_normalized.png` | Confusion matrices (labeled data only). |

### CLI reference: `bioairmet-inference`

```
bioairmet-inference --experiment_dir DIR --data_path PATH --save_path DIR [options]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--experiment_dir` | Yes | — | Experiment directory |
| `--data_path` | Yes | — | HDF5 data file or raw event directory |
| `--save_path` | Yes | — | Base output directory; labeled HDF5 input is written into a subdirectory named after the file (see *Output directory*) |
| `--checkpoint_type` | No | `best` | `best` or `last` |
| `--confidence_threshold` | No | `0.0` | Filter low-confidence predictions |
| `--min_area` | No | `500` | Min contour area (unlabeled mode only) |
| `--min_solidity` | No | `0.7` | Min contour solidity (unlabeled mode only) |
| `--min_intensity_delta` | No | `0.1` | Min intensity delta (unlabeled mode only) |
| `--skip_validation` | No | *(off)* | Skip the per-event quality checks and keep every event with usable images (unlabeled mode only) |
| `--gpu_id` | No | `0` | GPU index, `-1` for CPU |
| `--batch_size` | No | *(from config)* | Override batch size from config.yaml |

---

## Using the Python API

You can also call the inference function directly from Python code:

```python
from bioairmet.training.inference import run_inference

result = run_inference(
    data_path="/path/to/raw_event_directory",   # or HDF5 file path
    checkpoint_path="/path/to/model/best_classification_model.pth",
    save_path="./inference_results",    # labeled HDF5 input → ./inference_results/<hdf5_stem>/
    config_path="/path/to/model/config.yaml",
    gpu_id=0,                                     # -1 for CPU
    batch_size=64,
    num_workers=6,
    img_reader_type="minmax",                     # or "legacy"
    confidence_threshold=0.0,
    min_area=300,                                 # unlabeled mode only
    min_solidity=0.7,                             # unlabeled mode only
    min_intensity_delta=0.1,                      # unlabeled mode only
    skip_validation=False,                        # True = bypass the quality checks (unlabeled mode only)
    output_mode="compact",                        # or "detailed" (full per-class probabilities)
)

result["csv_path"]          # → path to predictions_<timestamp>.csv
result["summary_json_path"] # → path to inference_summary_<timestamp>.json
result["predictions"]       # → pandas DataFrame with the per-sample predictions
result["summary"]           # → dict: confidence stats, per-class counts, top-1 accuracy (labeled)
```

### Example: batch inference over multiple directories

```python
import os
from bioairmet.training.inference import run_inference

experiment_dir = "/path/to/cls_experiment"
for data_dir in ["/data/batch_1", "/data/batch_2", "/data/batch_3"]:
    run_inference(
        data_path=data_dir,
        checkpoint_path=f"{experiment_dir}/best_classification_model.pth",
        save_path=f"./results/{os.path.basename(data_dir)}",
        config_path=f"{experiment_dir}/config.yaml",
        gpu_id=0,
        batch_size=64,
    )
```

