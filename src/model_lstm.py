import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# Environment setup
# gets rid of annoying warnings and forces TF to use CPU (since GPU is not available in Colab)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# exist=ing_ok=True prevents error if folder already exists
os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# constants
WINDOW_SIZE = 20 #TODO try multiple using hyperparameter grid search 
EPOCHS=100
BATCH_SIZE=32
VAL_SPLIT=0.2
VERBOSE=1 #print epoch loading, otherwise change to 0

# Load and preprocess data
data = scipy.io.loadmat("data/Xtrain.mat")
X_raw = data["Xtrain"].flatten().reshape(-1, 1)
# Scale data 
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_raw).flatten()

# Create sequences for RNN input 
def create_sequences(data, window):
    """Create sequences of length window from data"""
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        Y.append(data[i + window])
    return np.array(X), np.array(Y)

X, Y = create_sequences(X_scaled, WINDOW_SIZE)
X = X.reshape(X.shape[0], X.shape[1], 1)

print("Shape:", X.shape)

# Model building and training
# Try LSTM because it  handles the vanishing gradient problem in time-series...
#  ... and capturing long-term dependencies better than SimpleRNN.
def build_lstm():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        LSTM(50, activation="tanh", return_sequences=False),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

print("Training model...")
model = build_lstm()
history = model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE, validation_split=VAL_SPLIT)
model.save("models/best_lstm_model.keras")

# 4. RECURSIVE PREDICTION (Part c)
# Use the last window of training data to start predicting the next 200 points
print("Generating 200-step recursive forecast...")
last_window = X_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
recursive_preds_scaled = []

current_batch = last_window
for _ in range(200):
    # Predict 1 step ahead
    pred = model.predict(current_batch, verbose=VERBOSE)
    recursive_preds_scaled.append(pred[0, 0])
    
    # Update the window: slide it forward by 1 and insert the prediction
    new_entry = pred.reshape(1, 1, 1)
    current_batch = np.append(current_batch[:, 1:, :], new_entry, axis=1)

# Part (a): Scale back to original units
recursive_preds = scaler.inverse_transform(np.array(recursive_preds_scaled).reshape(-1, 1))

# visualise loss curve 
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Convergence: Training vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("output/loss_lstm.png")
plt.show()

# PREDICTIONS VS ACTUAL (train + validation)
split = int(len(X) * (1 - VAL_SPLIT))
train_preds = scaler.inverse_transform(model.predict(X[:split], verbose=VERBOSE))
val_preds = scaler.inverse_transform(model.predict(X[split:], verbose=VERBOSE))
actual = scaler.inverse_transform(Y.reshape(-1, 1))

plt.figure(figsize=(12, 5))
plt.plot(actual, label="Actual", alpha=0.7)
plt.plot(range(split), train_preds, label="Train pred", alpha=0.7)
plt.plot(range(split, split + len(val_preds)), val_preds, label="Validation pred", alpha=0.8)
plt.axvline(split, color="grey", linestyle="--", linewidth=1)
plt.title("LSTM Predictions vs Actual")
plt.xlabel("Index")
plt.ylabel("Measurement Value")
plt.legend()
plt.savefig("output/pred_vs_actual_lstm.png")
plt.show()

# FORECAST
plt.figure(figsize=(12, 6))
plt.plot(range(len(X_raw)), X_raw, label="Historical Training Data", alpha=0.7)
# 200 predictions start right after the last training point
plt.plot(range(len(X_raw), len(X_raw) + 200), recursive_preds, 
         label="Recursive Forecast (Next 200 steps)", color='red', linewidth=2)
plt.title("Laser Measurement: Full Sequence + Recursive Forecast")
plt.xlabel("Time Steps")
plt.ylabel("Measurement Value")
plt.legend()
plt.savefig("output/forecast_results_lstm.png")
plt.show()

