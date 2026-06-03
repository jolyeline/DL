# Assignment 2 — Mamba Results

This report covers the Mamba model only. The task is 4-class MEG brain-state decoding:
`rest`, `task_motor`, `task_story_math`, and `task_working_memory`. Chance level is `0.25`.

All final Mamba runs use the shared protocol in
[`training_protocol.md`](training_protocol.md) and
[`../protocol_utils.py`](../protocol_utils.py): per-file sensor z-score,
downsampling by 4, `512`-sample windows, `256` stride, group-aware validation, and file-level
test metrics from mean-logit aggregation over windows.

## Why Mamba

MEG recordings are high-dimensional temporal signals: each recording has 248 sensors and a
long time axis. A useful model therefore needs to handle both:

- **Temporal structure**: brain states are not single-sample events. The class signal is spread
  over time, and the model needs enough context to distinguish rest, motor, story/math, and
  working-memory activity.
- **Spatial structure**: the 248 magnetometers measure different parts of the head. Class
  information is partly in which sensors change together, not only in the waveform of one
  sensor.

The model uses a small convolutional stem followed by bidirectional Mamba blocks:

1. A `Conv1d` stem mixes the 248 sensor channels and patches the 512-sample window into a
   shorter token sequence. This gives the model a local spatio-temporal representation before
   the sequence model sees the data.
2. Bidirectional Mamba blocks scan the token sequence forward and backward. This is a good fit
   for window classification because the whole window is available at once, so future context is
   allowed.
3. Mean pooling and a linear head produce the 4-class logits.

Mamba is attractive here because it models long sequences with linear-time state-space scans,
while a Transformer-style attention model would be more expensive for long MEG windows. The
main risk is overfitting: the number of independent recordings is small, even though overlapping
windows create many training examples.

## Final Results

### Intra-Subject

Notebook: `notebooks/mamba_train_intra.ipynb`

The protocol-tuned intra run selected the `lower_capacity` candidate:

| Candidate | val window-acc | val file-acc | final epochs |
|---|---:|---:|---:|
| lower_capacity | 0.682 | 0.833 | 21 |
| stronger_regularisation | 0.616 | 0.833 | 11 |
| original_regularised | 0.606 | 0.833 | 12 |
| gentler_augmentation | 0.601 | 0.667 | 17 |

Final intra test performance:

| Setting | window-acc | file-acc |
|---|---:|---:|
| Intra test | 0.633 | **0.875** |

The original Mamba run in the old notebook had `window-acc=0.659`, `file-acc=0.875`.
The protocol-tuned run preserves the file-level result but has slightly lower window accuracy.

### Cross-Subject

Notebook: `notebooks/mamba_train_cross.ipynb`

Cross-subject hyperparameters were selected with leave-one-subject-out validation over the two
training subjects. The selected candidate was `gentler_augmentation`:

| Candidate | LOSO window-acc | LOSO file-acc | final epochs |
|---|---:|---:|---:|
| gentler_augmentation | 0.516 | 0.734 | 14 |
| original_regularised | 0.520 | 0.719 | 6 |
| stronger_regularisation | 0.510 | 0.688 | 19 |
| lower_capacity | 0.485 | 0.672 | 18 |

Final cross test performance:

| Setting | window-acc | file-acc |
|---|---:|---:|
| Cross test1 | 0.589 | 0.812 |
| Cross test2 | 0.545 | 0.688 |
| Cross test3 | 0.646 | 0.812 |
| **Cross mean** | **0.593** | **0.771** |

The original Mamba run had cross mean `window-acc=0.576`, `file-acc=0.771`. The protocol-tuned
run improves window accuracy slightly while preserving file accuracy.

## Interpretation

The strongest result is the file-level accuracy: `0.875` intra and `0.771` cross. File-level
accuracy is the primary metric because the recording file is the independent evaluation unit.
Window-level accuracy is useful diagnostically, but overlapping windows from the same recording
are highly correlated.

Intra-subject performance is strong because training and test data come from the same subject.
The model can rely on stable subject-specific sensor geometry and amplitude patterns. Cross-subject
performance is lower because the test subjects are unseen. MEG sensor measurements depend on head
position, anatomy, and sensor/source geometry, so the same cognitive state does not necessarily
produce the same sensor-space pattern in another person.

The model still memorises quickly. In many runs, training accuracy approaches 1.0 after a small
number of epochs. This is expected because there are only tens of independent files, while the
window count is inflated by overlap. The regularised Mamba setup controls this enough to preserve
good file-level accuracy, but it does not fully solve cross-subject distribution shift.

The protocol-tuned notebooks are more defensible than the old single-split baseline because model
choices are made from validation data only:

- intra uses a group-aware file split;
- cross uses leave-one-subject-out validation;
- final test sets are evaluated once after selecting hyperparameters.

The tuning did not improve the primary file-level metric, but it did improve cross window accuracy
from `0.576` to `0.593` without reducing cross file accuracy.

## Why We Got This Performance

Several factors likely explain the observed results:

- **Small independent sample size**: overlapping windows do not create independent recordings.
  The effective dataset size is closer to the number of files than the number of windows.
- **Useful temporal context**: 512 samples after downsampling is about one second, which gives
  enough temporal context for the sequence model without making the Mamba scan too slow.
- **Spatial mixing helps**: the convolutional stem can combine the 248 sensors before temporal
  modelling, which is important because MEG class information is distributed over sensor groups.
- **File aggregation is stabilising**: averaging logits across all windows in a recording removes
  some window-level noise, which is why file accuracy is much higher than window accuracy.
- **Cross-subject shift remains hard**: test2 is consistently lower than test1/test3, suggesting
  some subjects are genuinely harder to transfer to.

## What Could Improve It

The most promising improvements target cross-subject transfer and spatial structure:

- **Subject-invariant normalisation**: per-subject or alignment-based normalisation could reduce
  subject-specific amplitude and covariance shifts. A simple per-subject experiment improved some
  window-level results but was unstable at file level, so this needs careful validation.
- **Spatially informed layers**: the current sensor mixing treats sensors as channels without
  explicit sensor geometry. Using sensor positions, graph convolutions, spatial attention, or
  covariance alignment could help the model learn more transferable spatial patterns.
- **More robust validation**: the cross protocol already uses LOSO, but intra could benefit from
  repeated file-level splits or k-fold validation to reduce noise from a single 6-file validation
  split.
- **Calibration of file aggregation**: mean-logit aggregation is simple. Majority vote,
  confidence-weighted aggregation, or temperature calibration may improve file-level decisions.
- **Frequency-aware inputs with caution**: spectral features are physiologically plausible, but
  the earlier band-power attempt reduced performance. A better version would combine raw and
  spectral features instead of replacing the raw signal.
- **More data or pretraining**: the main limitation is the small number of independent recordings.
  Self-supervised pretraining on unlabeled MEG windows or training across more subjects would likely
  help more than small hyperparameter changes.

## Summary

Mamba is a reasonable architecture for this problem because it combines spatial sensor mixing with
efficient temporal sequence modelling. The final protocol-based Mamba achieves strong file-level
accuracy: `0.875` intra-subject and `0.771` cross-subject mean. Hyperparameter tuning improved
cross window accuracy slightly but did not improve the primary file-level metric. The remaining
gap is likely driven less by sequence modelling capacity and more by limited independent data and
cross-subject sensor-space distribution shift.
