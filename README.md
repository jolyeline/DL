# Time Series Forecasting with RNN and LSTM

Deep learning assignment (Group 14) — Utrecht University, 2026.

Comparing SimpleRNN and LSTM architectures for 200-step recursive forecasting on a laser measurement signal.

**Authors:** Keren Better, Joly-Eline Himpers, Floris Kappen, Marieke Morganwiese

---

## Task

Given 1000 laser measurement samples, predict the next 200 values. Both one-step-ahead (using ground truth input) and recursive 200-step forecasting (feeding predictions back as input) are evaluated.

---

## Repository Structure

```
DL/
├── data/
│   ├── Xtrain.mat              # 1000 training samples
│   └── Xtest.mat               # 200 test samples (held out)
├── src/
│   ├── model_RNN.ipynb         # SimpleRNN exploration, grid search, evaluation
│   └── model_lstm.py           # LSTM training and evaluation script
├── models/
│   └── best_lstm_model.keras   # Final LSTM model (window=20, 100 units)
├── output/                     # Training curves, forecast plots, comparisons
├── docs/
│   ├── report.tex              # IEEE conference paper (LaTeX)
│   ├── DL_assignment.pdf       # Assignment specification
│   └── INSTRUCTIONS.md        # Grading criteria
└── requirements.txt
```

Trained RNN models (`best_rnn_ws2.keras`, `best_rnn_ws10.keras`) are saved in `src/`.

---

## Models

### SimpleRNN
- Architecture: `SimpleRNN(units, tanh)` → `Dense(1)`
- Grid search over window sizes [1, 2, 5, 10], units [20, 50, 100], learning rates [0.001, 0.01], batch sizes [32, 64]
- Best config: window size 2, selected for peak test performance

### LSTM
- Architecture: `LSTM(units, tanh)` → `Dense(25, relu)` → `Dense(1)`
- Grid search over window sizes [10, 20, 30, 40, 50], units [32, 50, 100], batch sizes [8, 16, 32]
- Best config: window size 20, 100 units

Both models use Min-Max scaling, Adam optimizer, MSE loss, and early stopping on validation loss.

---

## Results

| Model | One-step MSE | One-step MAE | Recursive MSE | Recursive MAE |
|-------|-------------|-------------|--------------|--------------|
| LSTM (ws=20) | **8.60** | **1.85** | 5036.62 | 52.88 |
| SimpleRNN (ws=2) | 93.22 | 5.97 | 1906.81 | 28.92 |
| Naive baseline (train mean) | — | — | 1672.22 | 27.73 |

LSTM is substantially better at one-step prediction. Both models fail to outperform the naive baseline at the 200-step recursive horizon due to error accumulation — a consequence of teacher-forced training applied at inference time.

---

## Setup

```bash
pip install -r requirements.txt
```

Run the LSTM script:
```bash
python src/model_lstm.py
```

Open the RNN notebook:
```bash
jupyter notebook src/model_RNN.ipynb
```

**Requirements:** Python 3.10+, TensorFlow 2.16, NumPy, SciPy, scikit-learn, Matplotlib, Pandas.
