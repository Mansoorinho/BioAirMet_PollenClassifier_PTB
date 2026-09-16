# Data Preparation Guide

This guide explains what raw data BioAirMet expects, the HDF5 schema the
trainers read, and how to build both the **unlabeled** (Stage 1, SSL) and
**labeled** (Stage 2, classification) datasets.

```
Raw events (per-event directory: JSON + 1-2 holographic images)
      │
      ├──► notebooks/data.ipynb ──► unlabeled_data.h5  ──► Stage 1 (SSL)
      │        (clean_unlabeled_data.py helpers)
      │
      └──► notebooks/data.ipynb ──► train_data.h5 / test_data.h5 ──► Stage 2
               (+ labels from metadata)
```

## 1. Expected raw data layout

BioAirMet works with *events* — one airborne particle passing the instrument.
Each event is represented by:

| Item | Description |
|---|---|
| Event JSON file | Contains computed holographic properties (area, solidity, intensity delta) and, under `computed_data.fluorescence.processed_data.spectra.relative_spectra`, the relative fluorescence spectra as a (3, 5) array. |
| 1–2 holographic images | Grayscale **uint16** raw holograms of the particle (two views are typical; the pipeline also supports one). |
| `data_clean_ids/` (labeled only) | Directory of event IDs that passed the cleaning stage. Events not listed here are discarded. |
| Metadata spreadsheet (labeled only) | `.xlsx` file with the per-event class labels (e.g. pollen species). |

The raw events can live in a flat directory or nested per-date subdirectories;
the helpers in `src/bioairmet/data/clean_unlabeled_data.py`
(`get_directories_and_files` → `get_event_name_fast` → `get_df`) scan it
recursively, pair each JSON event file with its images, and drop events that
do not have the expected image count or valid holographic properties.

**Quality filters.** An event is kept only if its spectra check passes
(`check_eventNgetspectra`):

| Property | Default minimum | Meaning |
|---|---|---|
| `min_area` | `500` | Region area in pixels |
| `min_solidity` | `0.7` | Region pixels / convex-hull pixels |
| `min_intensity_delta` | `0.1` | Max–min intensity difference |

The same thresholds are available as CLI flags for raw-directory inference
(`bioairmet-inference --min_area ... --min_solidity ... --min_intensity_delta ...`).

## 2. HDF5 schema (what the trainers read)

Both stages read **HDF5 files with one dataset per DataFrame column**
(`h5py.File(path, 'r')`), written by `save_df_to_hdf5`.
The file attributes contain `attrs['columns']` — a JSON list of the column
names.

### Labeled file (Stage 2 — `train_data.h5` / `test_data.h5`)

| Column | Type | Description |
|---|---|---|
| `images` | variable-length object | List of 1–2 **absolute** image file paths (uint16 grayscale holograms), one per row. |
| `relative_spectra` | float array, shape `(N, 3, 5)` | Relative fluorescence spectra per event (zeros if unavailable). |
| `category_num` | int | Integer class label (see [class mapping](#6-class-mapping-and-cats_dicttxt)). |
| `category` | string | Class name (e.g. `Betula`). |
| `event` | string | Event path (provenance; not consumed by the model). |

### Unlabeled file (Stage 1 — `unlabeled_data.h5`)

Same layout **without** the label columns: `event`, `images`,
`relative_spectra`.

### Column names are configurable

The trainers read the column names from the config instead of hard-coding
them, so alternatively named files also work:

| Config key (`data:` section) | Default |
|---|---|
| `image_column_name` | `images` |
| `fluorescence_spectra_column_name` | `relative_spectra` |
| `category_number_column_name` | `category_num` |
| `category_string_column_name` | `category` |
| `img_count_per_sample` | `1` (set `2` when `images` holds a pair per row) |

> **Consistency across stages matters more than the names**: Stage 2 is locked
> to the architecture the SSL weights were trained with (the SSL bundle's
> `architecture.yaml` is merged in at startup), and image reading settings
> (`img_reader_type`, `image_size`, `stitch_images`) **must match** between
> Stage 1 and Stage 2.
## 3. Building the labeled dataset (`train_data.h5` / `test_data.h5`)

Use the provided notebook **[`notebooks/data.ipynb`](../notebooks/data.ipynb)**.
Set the three inputs at the top:

| Variable | Description |
|---|---|
| `root_path` | Directory containing the raw events (per-event subdirectories / JSON + images). |
| `data_clean_ids` | Directory with the event IDs that passed cleaning (keeps only valid events). |
| `meta_data_path` | Metadata spreadsheet (`.xlsx`) with the per-event class labels. |

The notebook then:

1. **Scans** `root_path` and pairs event JSON files with their image paths
   (`get_directories_and_files` → `get_event_name_fast` → `get_df`, invalid
   entries filtered out).
2. **Joins labels** from the metadata + clean-ID list
   (`get_final_df_with_labels`).
3. **Extracts the relative fluorescence spectra** per event — a float32 tensor
   of shape `(3, 5)` (zeros if the spectra are missing).
4. **Maps category names → integer IDs** from the categories present in the
   data; `"Garbage"` is always assigned the **last** ID. The mapping is saved
   as a text file for validation / inference (see
   [class mapping](#6-class-mapping-and-cats_dicttxt)).
5. **Splits stratified per category** into train/test (75/25,
   `random_state=42`) so every class appears in both splits.
6. **Saves** `train_data.h5` and `test_data.h5` via `save_df_to_hdf5`
   (columns: `event`, `images`, `relative_spectra`, `category`,
   `category_num`).

Then point the classification config at the outputs
(`src/bioairmet/config/classification/config_general.yaml`):

```yaml
data:
  train_data_path:  /path/to/your/train_data.h5
  validation_path:  /path/to/your/test_data.h5
```

## 4. Building the unlabeled dataset (`unlabeled_data.h5`)

Stage 1 needs **no labels**. The same helpers from
`src/bioairmet/data/clean_unlabeled_data.py` are used — either through the
unlabeled section of `notebooks/data.ipynb`, or by running the module directly
(its `__main__` block shows the scan → filter → spectra-extraction flow with a
hardcoded `path` you can edit):

```python
from bioairmet.data.clean_unlabeled_data import (
    get_directories_and_files, get_event_name_fast, get_df,
    process_events_parallel, save_df_to_hdf5)

data      = get_directories_and_files(path)
data_fast = get_event_name_fast(data, path)
df, _     = get_df(data_fast)

df["relative_spectra"] = process_events_parallel(df["event"].tolist())
df = df[df["relative_spectra"].apply(lambda x: not isinstance(x, int))]

save_df_to_hdf5(df, "unlabeled_data.h5")
```

`process_events_parallel` reads the spectra with the quality filters from
[§1](#1-expected-raw-data-layout) and drops events whose spectra are invalid.

## 5. Image loading & normalization

Images are uint16 grayscale holograms. The reader/normalization is configured
under `data:` and **must be identical in both stages**:

| Config key | Default | Notes |
|---|---|---|
| `img_reader_type` | `legacy` | Options: `legacy`, `minmax`, `global` (`none`/`raw`/`no_norm` are aliases of `global`). See below. |
| `image_size` | `200` | Square resize applied at load time; also the model input size. |
| `stitch_images` | `False` | If `True`, the two images are stitched into one. **Must match the SSL stage.** |

> **Augmentation** (image + fluorescence) is configured separately under
> `data.augmentation` and differs per stage — see
> [Data augmentation](configuration.md#data-augmentation).

## 6. Class mapping and `cats_dict.txt`

The integer → class-name mapping produced during data preparation is what
validation and inference use to turn logits back into names
(`src/bioairmet/training/inference.py` reads the `cats_dict` mapping). Keep
this file next to your experiment (it is typically saved as
`cats_dict.txt`). Rules:

- `"Garbage"` is always the **last** class ID.
- The mapping is derived from the categories **present in your data**, so a
  new dataset yields a new mapping — do not mix checkpoints trained with
  different mappings.
- Validation / inference read the mapping from the experiment bundle, so as
  long as you validate/infer with the same experiment directory that produced
  the weights, the names stay consistent automatically.

## 7. Quick sanity check

Verify a produced file before training:

```python
import h5py, json
import pandas as pd

with h5py.File("train_data.h5", "r") as f:
    print("columns:", json.loads(f.attrs.get("columns", "[]")))
    for k in f.keys():
        print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")

# or round-trip back to a DataFrame:
from bioairmet.data.clean_unlabeled_data import load_df_from_hdf5
df = load_df_from_hdf5("train_data.h5")
print(df.shape, list(df.columns))
print(df["category_num"].value_counts().sort_index())
```

Expected: an `images` column whose rows are lists of absolute paths, a
`relative_spectra` array of shape `(N, 3, 5)`, and a `category_num` column
with one contiguous range of integer labels per class.

