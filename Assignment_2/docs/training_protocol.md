# Assignment 2 — Training Protocol

Shared preprocessing and metrics so every model is compared the same way. The
only thing that should differ between teammates is the model architecture.

Implementation: [`../protocol_utils.py`](../protocol_utils.py).
You only need one function — `prepare_protocol_data` — plus the bundle it
returns.

## Model contract

Your model must map:

```text
input:  (n_windows, n_sensors, window_size)   float32
output: (n_windows, 4)                         logits
```

Class order is fixed: `0 rest`, `1 task_motor`, `2 task_story_math`,
`3 task_working_memory`.

## Quick start (intra-subject)

```python
import sys; sys.path.append("..")
from protocol_utils import (
    DatasetPaths,
    list_files,
    prepare_protocol_data,
    split_files_train_val,
)

paths = DatasetPaths(data_root="../data")

# One call does it all: load -> z-score -> downsample -> window -> train/val split
data = prepare_protocol_data(paths.intra_train, paths.intra_test)

# ... train your model on data.X_train / data.y_train,
#     monitoring data.X_val / data.y_val ...

def predict_logits(windows):
    # windows: (n_windows, n_sensors, window_size)  ->  (n_windows, 4)
    ...

metrics = data.evaluate(predict_logits)
print(metrics["window_acc"], metrics["file_acc"])
print(metrics["file_confusion_matrix"])

# For hyperparameter selection by file-level validation accuracy, split the file
# list first so the validation files can be passed through evaluate().
train_files, val_files = split_files_train_val(
    list_files(paths.intra_train), val_frac=0.2, seed=0
)
fold = prepare_protocol_data(train_files, test_files=val_files, val_frac=0)
val_metrics = fold.evaluate(predict_logits)
```

`evaluate` runs your model on the test files and returns `window_acc`,
`file_acc`, `file_y_true`, `file_y_pred`, and `file_confusion_matrix`.

## Cross-subject

Tune with leave-one-subject-out over the two training subjects, then train the
final model on both and test once.

```python
from protocol_utils import group_files_by_subject, list_files

subjects = sorted(group_files_by_subject(list_files(paths.cross_train)))

for val_subject in subjects:                       # two folds
    by_subject = group_files_by_subject(list_files(paths.cross_train))
    fold = prepare_protocol_data(
        paths.cross_train,
        test_files=by_subject[val_subject],
        val_subject=val_subject,
    )
    # train on fold.X_train / fold.y_train
    val_file_acc = fold.evaluate(predict_logits)["file_acc"]

# final model: train on all cross-train subjects, no validation split
final = prepare_protocol_data(paths.cross_train, val_frac=0)
# ... train final model on final.X_train / final.y_train ...

for test_dir in (paths.cross_test1, paths.cross_test2, paths.cross_test3):
    acc = final.evaluate(predict_logits, test_files=test_dir)["file_acc"]
    print(test_dir, acc)
```

Pick hyperparameters from the **mean file-level validation accuracy across the
two folds**. The cross test sets are used only for the final numbers — never for
training, validation, early stopping, or normalization choices. Report the three
test accuracies and their unweighted mean.

## Fixed vs. tunable

Keep the left column fixed for a fair comparison. The right column may vary by
architecture — just report the values you used.

| Fixed (don't change) | Tunable per architecture (report it) |
|---|---|
| Labels from filename prefix; class order above | `downsample_factor` (default 4) |
| Files sorted before processing | `window_size` (default 512) |
| Group-aware split by source file (no window leakage) | `stride` / overlap (default 256 / 50%) |
| Intra val: `val_frac=0.2`, `seed=0` | anti-aliased resampling / extra filtering |
| Cross val: leave-one-subject-out over the 2 train subjects | crop / pad policy for short recordings |
| Test files never used for any fitting decision | data augmentation |
| Metrics: window acc, file acc, confusion matrix | window- vs file-level val monitoring |

To change a tunable input, pass a config:

```python
from protocol_utils import PreprocessConfig

cfg = PreprocessConfig(downsample_factor=4, window_size=512, stride=256)
data = prepare_protocol_data(paths.intra_train, paths.intra_test, config=cfg)
```

Changing anything in the left column is fine as research — but report it as a
separate, clearly labeled experiment, not a direct architecture comparison.

## Metrics

- **Window accuracy** — per-window `argmax` vs label. Diagnostic only;
  overlapping windows from one recording are correlated.
- **File accuracy** — average a file's window logits, then `argmax`. This is the
  **primary** score: the recording is the real evaluation unit.
- **Confusion matrix** — built from file-level predictions, class order above.

## Reporting checklist

Per result, report: architecture; intra or cross setting; train / val / test
folders; the tunable values above; window accuracy; file accuracy; file-level
confusion matrix.
