"""Shared preprocessing and metric helpers for Assignment 2 MEG experiments.

The fixed protocol pieces are:

- labels are inferred from file name prefixes;
- train/validation splitting is group-aware by source file;
- default validation split uses seed 0;
- default normalization is per-file, per-sensor z-score over time;
- file-level metrics aggregate window logits with a mean before argmax.

Typical notebook usage from ``Assignment_2/notebooks``:

    import sys
    sys.path.append("..")
    from protocol_utils import (
        PreprocessConfig, list_files, make_windows, split_train_val
    )

The defaults are:
``downsample_factor=4``, ``window_size=512``, ``stride=256``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import h5py
import numpy as np


TASK_LABELS: dict[str, int] = {
    "rest": 0,
    "task_motor": 1,
    "task_story_math": 2,
    "task_working_memory": 3,
}
LABEL_NAMES: dict[int, str] = {value: key for key, value in TASK_LABELS.items()}
CLASS_NAMES: list[str] = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
SAMPLE_RATE_HZ = 2034


@dataclass(frozen=True)
class PreprocessConfig:
    """Configurable preprocessing parameters.

    These defaults define the shared protocol used by the Mamba notebooks.

    Attributes:
        downsample_factor: Keep every Nth time sample. The baseline uses simple
            decimation, not anti-aliased resampling.
        window_size: Number of samples per window after downsampling.
        stride: Window hop in samples after downsampling.
        normalize: Whether to apply per-file, per-sensor z-score normalization.
        flat_sensor_std: Sensors with std below this threshold are treated as
            flat and divided by 1.0 to avoid numerical issues.
    """

    downsample_factor: int = 4
    window_size: int = 512
    stride: int = 256
    normalize: bool = True
    flat_sensor_std: float = 1e-20

    def __post_init__(self) -> None:
        if self.downsample_factor < 1:
            raise ValueError(
                f"downsample_factor must be >= 1, got {self.downsample_factor}"
            )
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")

    @property
    def effective_sample_rate_hz(self) -> float:
        """Sampling rate after downsampling."""

        return SAMPLE_RATE_HZ / self.downsample_factor


@dataclass(frozen=True)
class DatasetPaths:
    """Convenience paths for the assignment dataset layout."""

    data_root: str | Path = Path("data")

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))

    @property
    def intra_train(self) -> Path:
        return self.data_root / "Intra" / "train"

    @property
    def intra_test(self) -> Path:
        return self.data_root / "Intra" / "test"

    @property
    def cross_train(self) -> Path:
        return self.data_root / "Cross" / "train"

    @property
    def cross_test1(self) -> Path:
        return self.data_root / "Cross" / "test1"

    @property
    def cross_test2(self) -> Path:
        return self.data_root / "Cross" / "test2"

    @property
    def cross_test3(self) -> Path:
        return self.data_root / "Cross" / "test3"


def get_dataset_name(filepath: str | Path) -> str:
    """Return the HDF5 dataset name used inside an assignment ``.h5`` file."""

    name = Path(filepath).stem
    return "_".join(name.split("_")[:-1])


def load_h5(filepath: str | Path) -> np.ndarray:
    """Load one MEG recording as a ``(n_sensors, n_timepoints)`` array."""

    path = Path(filepath)
    with h5py.File(path, "r") as h5_file:
        return h5_file[get_dataset_name(path)][()]


def get_label(filepath: str | Path, labels: Mapping[str, int] = TASK_LABELS) -> int:
    """Infer the class label from a file name prefix."""

    name = Path(filepath).name
    for task_name, label in labels.items():
        if name.startswith(task_name):
            return label
    raise ValueError(f"Unknown task prefix in filename: {filepath}")


def get_subject_id(filepath: str | Path) -> str:
    """Infer the subject ID from an assignment file name.

    Expected file names look like ``task_motor_113922_5.h5`` or
    ``task_working_memory_164636_7.h5``. The subject ID is the second-last
    underscore-separated field.
    """

    parts = Path(filepath).stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer subject ID from filename: {filepath}")
    return parts[-2]


def list_files(folder: str | Path) -> list[Path]:
    """Return sorted ``.h5`` files in a dataset folder."""

    return sorted(Path(folder).glob("*.h5"))


def group_files_by_subject(files: Iterable[str | Path]) -> dict[str, list[Path]]:
    """Group files by subject ID, preserving sorted file order within subjects."""

    by_subject: dict[str, list[Path]] = {}
    for filepath in sorted(Path(f) for f in files):
        by_subject.setdefault(get_subject_id(filepath), []).append(filepath)
    return by_subject


def make_leave_one_subject_out_folds(
    files: Sequence[str | Path],
) -> list[tuple[str, list[Path], list[Path]]]:
    """Create subject-held-out folds for cross-subject validation.

    Returns a list of ``(validation_subject, train_files, validation_files)``.
    With the assignment cross-train set, this gives two folds: train on one
    subject and validate on the other.
    """

    by_subject = group_files_by_subject(files)
    if len(by_subject) < 2:
        raise ValueError("Need at least two subjects for leave-one-subject-out folds")

    folds: list[tuple[str, list[Path], list[Path]]] = []
    for validation_subject in sorted(by_subject):
        validation_files = by_subject[validation_subject]
        train_files = [
            filepath
            for subject, subject_files in sorted(by_subject.items())
            if subject != validation_subject
            for filepath in subject_files
        ]
        folds.append((validation_subject, train_files, validation_files))
    return folds


def zscore_normalize(matrix: np.ndarray, flat_sensor_std: float = 1e-20) -> np.ndarray:
    """Per-file, per-sensor z-score normalization over time.

    ``matrix`` is expected to have shape ``(n_sensors, n_timepoints)``. The
    normalization is computed independently for each file, using each sensor's
    time axis. Very small standard deviations are replaced with 1.0 so flat
    sensors do not produce unstable values.
    """

    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std = np.where(std < flat_sensor_std, 1.0, std)
    return (matrix - mean) / std


def downsample(matrix: np.ndarray, factor: int) -> np.ndarray:
    """Simple decimation by keeping every ``factor``-th time sample.

    This intentionally matches the notebook baseline: ``matrix[:, ::factor]``.
    If a model uses anti-aliased resampling instead, report that as a separate
    preprocessing choice.
    """

    if factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {factor}")
    return matrix[:, ::factor]


def preprocess_file(filepath: str | Path, config: PreprocessConfig | None = None) -> np.ndarray:
    """Load and preprocess one file, returning ``float32`` data."""

    cfg = config or PreprocessConfig()
    matrix = load_h5(filepath)
    if cfg.normalize:
        matrix = zscore_normalize(matrix, flat_sensor_std=cfg.flat_sensor_std)
    matrix = downsample(matrix, cfg.downsample_factor)
    return matrix.astype(np.float32, copy=False)


def make_windows_from_matrix(
    matrix: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Split one preprocessed matrix into overlapping windows.

    Returns an array with shape ``(n_windows, n_sensors, window_size)``.
    """

    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    total_time = matrix.shape[1]
    starts = range(0, total_time - window_size + 1, stride)
    windows = [matrix[:, start : start + window_size] for start in starts]
    if not windows:
        raise ValueError(
            f"Recording has {total_time} samples after preprocessing, "
            f"which is shorter than window_size={window_size}"
        )
    return np.stack(windows).astype(np.float32, copy=False)


def make_windows(
    files: Sequence[str | Path],
    config: PreprocessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create window tensors, labels, and source-file groups.

    Args:
        files: Recording files. Each file becomes one group.
        config: Preprocessing and windowing config.

    Returns:
        ``X``: ``float32`` array with shape ``(n_windows, n_sensors, window_size)``.
        ``y``: ``int64`` array with one label per window.
        ``groups``: ``int64`` array where each value is the source-file index.
    """

    cfg = config or PreprocessConfig()
    all_windows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []

    for group_id, filepath in enumerate(files):
        matrix = preprocess_file(filepath, cfg)
        windows = make_windows_from_matrix(matrix, cfg.window_size, cfg.stride)
        label = get_label(filepath)

        all_windows.append(windows)
        labels.extend([label] * len(windows))
        groups.extend([group_id] * len(windows))

    if not all_windows:
        raise ValueError("No files were provided")

    X = np.concatenate(all_windows, axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=np.int64)
    return X, y, group_array


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group-aware train/validation split.

    Whole source files are assigned to either train or validation, preventing
    leakage from overlapping windows of the same recording. The default seed
    matches the Mamba notebook baseline.
    """

    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    n_val = max(1, int(round(val_frac * len(unique_groups))))
    val_groups = set(unique_groups[:n_val].tolist())
    val_mask = np.isin(groups, list(val_groups))
    return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]


def split_files_train_val(
    files: Sequence[str | Path],
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[Path], list[Path]]:
    """Group-aware train/validation split at file-list level.

    This mirrors :func:`split_train_val` for the common case where each file is
    one group. It is useful when validation must be evaluated at file level
    because the caller needs the actual validation file list.
    """

    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")

    file_list = sorted(Path(f) for f in files)
    if not file_list:
        raise ValueError("No files were provided")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(file_list))
    rng.shuffle(indices)

    n_val = max(1, int(round(val_frac * len(file_list))))
    val_indices = set(indices[:n_val].tolist())
    train_files = [filepath for idx, filepath in enumerate(file_list) if idx not in val_indices]
    val_files = [filepath for idx, filepath in enumerate(file_list) if idx in val_indices]
    return train_files, val_files


def take_per_class(files: Iterable[str | Path], k: int) -> list[Path]:
    """Pick up to ``k`` sorted files per class for smoke tests or quick runs."""

    by_class: dict[int, list[Path]] = {}
    for filepath in files:
        by_class.setdefault(get_label(filepath), []).append(Path(filepath))

    selected: list[Path] = []
    for label in sorted(by_class):
        selected.extend(sorted(by_class[label])[:k])
    return sorted(selected)


def iter_file_windows(
    files: Sequence[str | Path],
    config: PreprocessConfig | None = None,
) -> Iterator[tuple[Path, int, np.ndarray]]:
    """Yield ``(filepath, label, windows)`` for file-level model evaluation."""

    cfg = config or PreprocessConfig()
    for filepath in files:
        path = Path(filepath)
        matrix = preprocess_file(path, cfg)
        windows = make_windows_from_matrix(matrix, cfg.window_size, cfg.stride)
        yield path, get_label(path), windows


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy as a Python float."""

    if len(y_true) == 0:
        raise ValueError("Cannot compute accuracy for empty arrays")
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def window_accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Window-level accuracy from logits shaped ``(n_windows, n_classes)``."""

    return accuracy(np.asarray(y_true), np.asarray(logits).argmax(axis=1))


def aggregate_logits_by_group(
    logits: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate window logits into one prediction per source file.

    Logits are averaged within each group and then converted to a class with
    ``argmax``. This is the file-level metric protocol used in the notebook.

    Returns:
        ``group_ids``: sorted group IDs.
        ``file_y_true``: one ground-truth label per group.
        ``file_y_pred``: one predicted label per group.
    """

    logits = np.asarray(logits)
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)
    if not (len(logits) == len(y_true) == len(groups)):
        raise ValueError("logits, y_true, and groups must have matching lengths")

    group_ids = np.unique(groups)
    file_y_true: list[int] = []
    file_y_pred: list[int] = []
    for group_id in group_ids:
        mask = groups == group_id
        labels = np.unique(y_true[mask])
        if len(labels) != 1:
            raise ValueError(f"Group {group_id} has mixed labels: {labels.tolist()}")
        file_y_true.append(int(labels[0]))
        file_y_pred.append(int(logits[mask].mean(axis=0).argmax()))

    return (
        group_ids,
        np.asarray(file_y_true, dtype=np.int64),
        np.asarray(file_y_pred, dtype=np.int64),
    )


def file_accuracy_from_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
) -> float:
    """File-level accuracy after mean-logit aggregation by source file."""

    _, file_y_true, file_y_pred = aggregate_logits_by_group(logits, y_true, groups)
    return accuracy(file_y_true, file_y_pred)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] = tuple(range(len(CLASS_NAMES))),
) -> np.ndarray:
    """Compute a confusion matrix without requiring scikit-learn."""

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, pred_label in zip(np.asarray(y_true), np.asarray(y_pred), strict=True):
        matrix[label_to_index[int(true_label)], label_to_index[int(pred_label)]] += 1
    return matrix


def summarize_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    """Return standard baseline metrics for window logits.

    The returned dict contains ``window_acc``, ``file_acc``, ``file_y_true``,
    ``file_y_pred``, and ``file_confusion_matrix``.
    """

    _, file_y_true, file_y_pred = aggregate_logits_by_group(logits, y_true, groups)
    return {
        "window_acc": window_accuracy_from_logits(logits, y_true),
        "file_acc": accuracy(file_y_true, file_y_pred),
        "file_y_true": file_y_true,
        "file_y_pred": file_y_pred,
        "file_confusion_matrix": confusion_matrix(file_y_true, file_y_pred),
    }


# ---------------------------------------------------------------------------
# One-call entry point
#
# Everything above is the toolbox. For a standard baseline run you only need
# ``prepare_protocol_data`` plus the returned bundle's ``evaluate`` method.
# ---------------------------------------------------------------------------

PredictLogits = Callable[[np.ndarray], np.ndarray]


def _resolve_files(files: str | Path | Iterable[str | Path]) -> list[Path]:
    """Accept a dataset folder or an explicit file list and return a file list."""

    if isinstance(files, (str, Path)):
        return list_files(files)
    return [Path(f) for f in files]


@dataclass
class ProtocolData:
    """Everything a model needs for one baseline run.

    Produced by :func:`prepare_protocol_data`. The ``X_train``/``y_train`` and
    ``X_val``/``y_val`` arrays are ready to feed into a model. Call
    :meth:`evaluate` at test time to get the standard metrics.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    test_files: list[Path]
    config: PreprocessConfig

    def evaluate(
        self,
        predict_logits: PredictLogits,
        test_files: str | Path | Iterable[str | Path] | None = None,
    ) -> dict[str, object]:
        """Run window- and file-level metrics on the test files.

        ``predict_logits`` maps a window batch shaped
        ``(n_windows, n_sensors, window_size)`` to logits shaped
        ``(n_windows, n_classes)``. Pass ``test_files`` to evaluate a different
        set than the one stored on this bundle (e.g. the three cross test sets).
        """

        files = self.test_files if test_files is None else _resolve_files(test_files)
        if not files:
            raise ValueError("No test files to evaluate")

        X_test, y_test, groups = make_windows(files, self.config)
        logits = np.asarray(predict_logits(X_test))
        return summarize_logits(logits, y_test, groups)


def prepare_protocol_data(
    train_files: str | Path | Iterable[str | Path],
    test_files: str | Path | Iterable[str | Path] | None = None,
    config: PreprocessConfig | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
    val_subject: str | None = None,
) -> ProtocolData:
    """Do all preprocessing for one baseline run in a single call.

    Loads every training file, applies per-sensor z-score normalization,
    downsamples, windows, and makes a group-aware train/validation split. The
    returned :class:`ProtocolData` carries ready-to-train arrays and an
    ``evaluate`` method for file-level test metrics.

    Args:
        train_files: A dataset folder (path) or an explicit list of files.
        test_files: Folder or file list for final evaluation. May be ``None``
            and supplied later to :meth:`ProtocolData.evaluate`.
        config: Preprocessing/windowing config. Defaults reproduce the Mamba
            baseline.
        val_frac: Fraction of training files held out for validation when
            ``val_subject`` is not set. Use ``0`` to skip the split (e.g. the
            final cross-subject model trained on all subjects).
        seed: Seed for the random group-aware split.
        val_subject: For cross-subject tuning, hold out this subject's files as
            validation instead of taking a random group split.
    """

    cfg = config or PreprocessConfig()
    train_list = _resolve_files(train_files)
    if not train_list:
        raise ValueError("No training files found")

    if val_subject is not None:
        by_subject = group_files_by_subject(train_list)
        if val_subject not in by_subject:
            raise ValueError(
                f"val_subject {val_subject!r} is not among training subjects "
                f"{sorted(by_subject)}"
            )
        val_list = by_subject[val_subject]
        val_set = set(val_list)
        fit_list = [f for f in train_list if f not in val_set]
        X_train, y_train, _ = make_windows(fit_list, cfg)
        X_val, y_val, _ = make_windows(val_list, cfg)
    elif val_frac <= 0:
        X_train, y_train, _ = make_windows(train_list, cfg)
        X_val = np.empty((0, *X_train.shape[1:]), dtype=X_train.dtype)
        y_val = np.empty((0,), dtype=y_train.dtype)
    else:
        X, y, groups = make_windows(train_list, cfg)
        X_train, y_train, X_val, y_val = split_train_val(
            X, y, groups, val_frac=val_frac, seed=seed
        )

    test_list = _resolve_files(test_files) if test_files is not None else []
    return ProtocolData(X_train, y_train, X_val, y_val, test_list, cfg)
