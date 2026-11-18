
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# ---------------------------------------
# 1. Project paths and reproducibility
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "training_data.npz"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("Loading prepared training data...")

# ---------------------------------------
# 2. Load training data
# ---------------------------------------

data = np.load(DATA_FILE)

X_train = data["X_market_train"]
X_val = data["X_market_val"]
X_test = data["X_market_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

dates_test = data["dates_test"]

# Load fitted scalers
market_scaler = joblib.load(
    MODEL_DIR / "market_scaler_fusion.pkl"
)

target_scaler = joblib.load(
    MODEL_DIR / "target_scaler_fusion.pkl"
)

print("\nDataset loaded successfully!")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ---------------------------------------
# 3. Build LSTM model
# ---------------------------------------

print("\nBuilding market-only LSTM model...")

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(64, return_sequences=True),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()

# ---------------------------------------
# 4. Train model
# ---------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("\nStarting LSTM training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

# ---------------------------------------
# 5. Predict test data
# ---------------------------------------

print("\nGenerating test predictions...")

predicted_scaled = model.predict(
    X_test,
    verbose=0
)

# Convert predictions back to actual stock prices
predicted_prices = target_scaler.inverse_transform(
    predicted_scaled
).flatten()

actual_prices = target_scaler.inverse_transform(
    y_test
).flatten()

# ---------------------------------------
# 6. Previous-close baseline
# ---------------------------------------

# Inverse-transform the test market sequences
# and take the Close value from the last input day.

test_market_original = market_scaler.inverse_transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)

previous_close_predictions = test_market_original[
    :, -1, 3
]

# ---------------------------------------
# 7. Evaluate both models
# ---------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = mean_absolute_percentage_error(
        actual, predicted
    ) * 100

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


lstm_metrics = calculate_metrics(
    actual_prices,
    predicted_prices
)

baseline_metrics = calculate_metrics(
    actual_prices,
    previous_close_predictions
)

results = pd.DataFrame(
    [lstm_metrics, baseline_metrics],
    index=["Market LSTM", "Previous Close Baseline"]
)

print("\nMODEL EVALUATION RESULTS")
print(results.round(4).to_string())

# Directional accuracy
actual_direction = np.sign(
    actual_prices - previous_close_predictions
)

predicted_direction = np.sign(
    predicted_prices - previous_close_predictions
)

directional_accuracy = np.mean(
    actual_direction == predicted_direction
) * 100

print(
    f"\nLSTM Directional Accuracy: "
    f"{directional_accuracy:.2f}%"
)

# ---------------------------------------
# 8. Save predictions
# ---------------------------------------

predictions_df = pd.DataFrame({
    "Date": dates_test,
    "Actual_Close": actual_prices,
    "LSTM_Predicted_Close": predicted_prices,
    "Previous_Close_Prediction": previous_close_predictions
})

predictions_df.to_csv(
    MODEL_DIR / "market_baseline_predictions.csv",
    index=False
)

results.to_csv(
    MODEL_DIR / "market_baseline_evaluation.csv"
)

# ---------------------------------------
# 9. Save trained model
# ---------------------------------------

model.save(
    MODEL_DIR / "market_baseline.keras"
)

# ---------------------------------------
# 10. Plot actual vs predicted prices
# ---------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    actual_prices,
    label="Actual Close",
    linewidth=2
)

plt.plot(
    predicted_prices,
    label="LSTM Prediction",
    linestyle="--"
)

plt.plot(
    previous_close_predictions,
    label="Previous Close Baseline",
    alpha=0.7
)

plt.title("RELIANCE Market-Only LSTM Prediction")
plt.xlabel("Test Trading Sessions")
plt.ylabel("Stock Price (INR)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "market_baseline_prediction.png",
    dpi=150
)

plt.close()

# ---------------------------------------
# 11. Plot training history
# ---------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Market LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "market_baseline_training.png",
    dpi=150
)

plt.close()

# ---------------------------------------
# 12. Completion message
# ---------------------------------------

print("\n" + "=" * 45)
print("MARKET-ONLY LSTM TRAINING COMPLETED!")
print("=" * 45)

print("\nSaved files:")

print("Model: models/market_baseline.keras")
print("Predictions: models/market_baseline_predictions.csv")
print("Evaluation: models/market_baseline_evaluation.csv")

print("Prediction chart: plots/market_baseline_prediction.png")
print("Training chart: plots/market_baseline_training.png")
