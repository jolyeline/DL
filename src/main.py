# imports
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Input
import os

#to remove warnings and oneDNN optimizations for better reproducibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

EPOCHS = 100
window_size = 3

os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

# load data
data = scipy.io.loadmat("data/Xtrain.mat")

plt.figure()
plt.plot(data["Xtrain"])
plt.savefig("output/Xtrain.png")
plt.show()

X_raw = data["Xtrain"].flatten()

# windowing
X, Y = [], []
for i in range(len(X_raw) - window_size):
    X.append(X_raw[i:i + window_size])
    Y.append(X_raw[i + window_size])

X, Y = np.array(X), np.array(Y)
X = X.reshape(X.shape[0], X.shape[1], 1)

print("Shape:", X.shape)

# helper to train model
def train_model(cell_type="LSTM"):
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))

    if cell_type == "LSTM":
        model.add(LSTM(20, activation="tanh"))
    else:
        model.add(SimpleRNN(20, activation="relu"))

    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(X, Y, epochs=EPOCHS, batch_size=32, verbose=1)
    return model, history

# train both
lstm_model, lstm_hist = train_model("LSTM")
rnn_model, rnn_hist = train_model("RNN")

# save models
lstm_model.save("models/lstm_model.keras")
rnn_model.save("models/rnn_model.keras")

# plot comparison
plt.figure()
plt.plot(lstm_hist.history["loss"], label="LSTM")
plt.plot(rnn_hist.history["loss"], label="SimpleRNN")
plt.title("Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("output/lstm_vs_rnn.png")
plt.show()

print("done!")