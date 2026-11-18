
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "RELIANCE.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Load stock data
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
df = df[["Close"]].dropna()

prices = df["Close"].values.reshape(-1, 1)

print("Total stock records:", len(prices))

# Split data chronologically
split = int(len(prices) * 0.80)

# Scale using training data only
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices[:split])

scaled_data = scaler.transform(prices)

# Prepare sequences
LOOKBACK = 60

X_train, y_train = [], []
X_test, y_test = [], []

test_dates = []

for i in range(LOOKBACK, len(scaled_data)):
    X = scaled_data[i-LOOKBACK:i]
    y = scaled_data[i]

    if i < split:
        X_train.append(X)
        y_train.append(y)
    else:
        X_test.append(X)
        y_test.append(y)
        test_dates.append(df.index[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True,
         input_shape=(LOOKBACK, 1)),
    Dropout(0.2),

    LSTM(50),
    Dropout(0.2),

    Dense(25),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()

# Train model
print("\nStarting LSTM training...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    verbose=1
)

# Predict test data
predictions = model.predict(X_test)

# Convert prices back to original scale
predictions = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test)

# Calculate evaluation metrics
mae = mean_absolute_error(actual_prices, predictions)
rmse = np.sqrt(mean_squared_error(actual_prices, predictions))

print("\n========== MODEL RESULTS ==========")
print(f"MAE:  Rs. {mae:.2f}")
print(f"RMSE: Rs. {rmse:.2f}")

# Save trained model
model.save(MODEL_DIR / "reliance_lstm.keras")

# Save scaler
import joblib
joblib.dump(scaler, MODEL_DIR / "reliance_scaler.pkl")

# Save predictions
results = pd.DataFrame({
    "Actual": actual_prices.flatten(),
    "Predicted": predictions.flatten()
}, index=test_dates)

results.to_csv(MODEL_DIR / "predictions.csv")

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))

plt.plot(
    test_dates,
    actual_prices,
    label="Actual Price"
)

plt.plot(
    test_dates,
    predictions,
    label="Predicted Price"
)

plt.title("RELIANCE - LSTM Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_DIR / "lstm_prediction.png")
plt.show()

# Predict next trading day's closing price
last_60_days = scaled_data[-LOOKBACK:]
next_input = last_60_days.reshape(1, LOOKBACK, 1)

next_prediction = model.predict(next_input)
next_price = scaler.inverse_transform(next_prediction)

print("\n===================================")
print("NEXT TRADING DAY PRICE PREDICTION")
print(f"Predicted closing price: Rs. {next_price[0][0]:.2f}")
print("===================================")

print("\nLSTM MODEL TRAINING COMPLETED!")
