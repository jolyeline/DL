import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import random
import tensorflow as tf  

# constants
EPOCHS = 100
PREDICTIONS = 200
RANDOM_SEED = 42
VAL_SPLIT = 0.2

# hyperparameters for grid  search
WINDOW_SIZES = [10, 20, 30, 40, 50]
LSTM_UNITS = [32, 50, 100]
BATCH_SIZES = [8, 16, 32]

# setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

data = scipy.io.loadmat("data/Xtrain.mat")
X_raw = data["Xtrain"].flatten().reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_raw).flatten()

best_score = float("inf")
best_params = None
best_model = None
best_history = None
best_X = None
best_Y = None


def create_sequences(data, window):
    ''' Creates sequences of data with specified window size for LSTM training.'''
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        Y.append(data[i + window])
    return np.array(X), np.array(Y)

def build_lstm(window_size, units):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(units, activation="tanh"),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# Grid search over hyperparameters
for window, units, batch in product(WINDOW_SIZES, LSTM_UNITS, BATCH_SIZES):
    print(f"\nTesting: window={window}, units={units}, batch={batch}")

    X, Y = create_sequences(X_scaled, window)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * (1 - VAL_SPLIT))

    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    model = build_lstm(window, units)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=batch,
        verbose=0
    )

    val_preds = model.predict(X_val, verbose=0)
    mse = mean_squared_error(Y_val, val_preds)

    print("MSE:", mse)

    if mse < best_score:
        best_score = mse
        best_params = (window, units, batch)
        best_model = model
        best_history = history
        best_X = X_train
        best_Y = Y_train

print("\nBEST PARAMS (window, units, batch):", best_params)
print("BEST MSE:", best_score)

best_model.save("models/best_lstm_model.keras")

# recursive forecast
WINDOW_SIZE = best_params[0]

last_window = X_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
recursive_preds_scaled = []

current_batch = last_window
for _ in range(PREDICTIONS):
    pred = best_model.predict(current_batch, verbose=0)
    recursive_preds_scaled.append(pred[0, 0])
    current_batch = np.append(current_batch[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

recursive_preds = scaler.inverse_transform(
    np.array(recursive_preds_scaled).reshape(-1, 1)
)

# loss plot 
plt.figure(figsize=(10, 5))
plt.plot(best_history.history["loss"], label="Training Loss") #training loss
plt.plot(best_history.history["val_loss"], label="Validation Loss")#validation loss
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("output/lstm_loss.png")
plt.show()

# fit on full data
preds_all = scaler.inverse_transform(
    best_model.predict(best_X, verbose=0).reshape(-1, 1)
)
actual = scaler.inverse_transform(best_Y.reshape(-1, 1))

plt.figure(figsize=(12, 5))
plt.plot(actual, label="Actual", alpha=0.7)
plt.plot(preds_all, label="Predictions", alpha=0.7)
plt.title("LSTM Fit on Training Data")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.savefig("output/lstm_pred_vs_actual.png")
plt.show()

# forecast plot 
plt.figure(figsize=(12, 6))
plt.plot(range(len(X_raw)), X_raw, label="History")
plt.plot(
    range(len(X_raw), len(X_raw) + PREDICTIONS),
    recursive_preds,
    label="Forecast"
)
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.savefig("output/lstm_forecast_results.png")
plt.show()

print("Best Validation Loss:", min(best_history.history["val_loss"]))
print("Final Validation Loss:", best_history.history["val_loss"][-1])