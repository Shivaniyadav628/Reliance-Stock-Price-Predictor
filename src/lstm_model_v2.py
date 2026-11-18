
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "RELIANCE.csv"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
df = df.sort_index()

# Create features
df["Close_Return"] = df["Close"].pct_change()
df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)
df["Open_Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
df["Volume_Change"] = np.log1p(df["Volume"]).diff()

features = [
    "Close_Return",
    "Daily_Range",
    "Open_Gap",
    "Volume_Change"
]

df = df.dropna()

X_raw = df[features].values
y_raw = df["Close_Return"].values.reshape(-1, 1)

LOOKBACK = 60
split = int(len(df) * 0.80)

# Fit scalers on training data only
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

feature_scaler.fit(X_raw[:split])
target_scaler.fit(y_raw[LOOKBACK:split])

X_scaled = feature_scaler.transform(X_raw)
y_scaled = target_scaler.transform(y_raw)

# Create sequences
X_train, y_train = [], []
X_test, y_test = [], []

previous_closes = []
test_dates = []

for i in range(LOOKBACK, len(df)):
    sequence = X_scaled[i-LOOKBACK:i]
    target = y_scaled[i]

    if i < split:
        X_train.append(sequence)
        y_train.append(target)
    else:
        X_test.append(sequence)
        y_test.append(target)

        previous_closes.append(df["Close"].iloc[i-1])
        test_dates.append(df.index[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Build improved LSTM
model = Sequential([
    Input(shape=(LOOKBACK, len(features))),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Stop training when validation performance stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

print("\nTraining LSTM Version 2...")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    shuffle=False,
    verbose=1
)

# Predict percentage returns
predicted_returns_scaled = model.predict(X_test)

predicted_returns = target_scaler.inverse_transform(
    predicted_returns_scaled
).flatten()

actual_returns = target_scaler.inverse_transform(
    y_test
).flatten()

# Convert predicted returns into closing prices
previous_closes = np.array(previous_closes)

predicted_prices = previous_closes * (1 + predicted_returns)
actual_prices = previous_closes * (1 + actual_returns)

# Baseline: predict no price change
baseline_prices = previous_closes.copy()

# Evaluation function
def evaluate(actual, predicted, name):
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE (%)": np.mean(
            np.abs((actual - predicted) / actual)
        ) * 100,
        "R2 Score": r2_score(actual, predicted)
    }

comparison = pd.DataFrame([
    evaluate(actual_prices, predicted_prices, "LSTM V2"),
    evaluate(actual_prices, baseline_prices, "Previous Close Baseline")
])

# Directional accuracy
actual_direction = np.sign(actual_prices - previous_closes)
predicted_direction = np.sign(predicted_prices - previous_closes)

direction_accuracy = np.mean(
    actual_direction == predicted_direction
) * 100

print("\n========== LSTM VERSION 2 RESULTS ==========")
print(comparison.to_string(index=False))
print(f"\nDirectional Accuracy: {direction_accuracy:.2f}%")

# Save model and preprocessing tools
model.save(MODEL_DIR / "reliance_lstm_v2.keras")

import joblib

joblib.dump(
    feature_scaler,
    MODEL_DIR / "reliance_feature_scaler_v2.pkl"
)

joblib.dump(
    target_scaler,
    MODEL_DIR / "reliance_target_scaler_v2.pkl"
)

# Save predictions
results = pd.DataFrame({
    "Actual": actual_prices,
    "LSTM_V2_Predicted": predicted_prices,
    "Previous_Close_Baseline": baseline_prices
}, index=test_dates)

results.to_csv(MODEL_DIR / "predictions_v2.csv")

comparison.to_csv(
    MODEL_DIR / "evaluation_v2.csv",
    index=False
)

# Plot results
plt.figure(figsize=(14, 7))

plt.plot(test_dates, actual_prices, label="Actual Price")
plt.plot(test_dates, predicted_prices, label="LSTM V2")
plt.plot(test_dates, baseline_prices, label="Previous Close Baseline")

plt.title("RELIANCE - LSTM Version 2")
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_DIR / "lstm_prediction_v2.png")
plt.show()

print("\nLSTM VERSION 2 COMPLETED!")
print("Model and evaluation files saved successfully.")
