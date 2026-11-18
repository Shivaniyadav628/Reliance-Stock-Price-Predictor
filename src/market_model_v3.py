
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# ---------------------------------------
# 1. Paths and reproducibility
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "market_v3_data.npz"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("Loading Market V3 training data...")

# ---------------------------------------
# 2. Load prepared data
# ---------------------------------------

# allow_pickle=True is needed because dates_test
# was saved as a NumPy object array.
data = np.load(DATA_FILE, allow_pickle=True)

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test_raw = data["y_test_raw"]

dates_test = data["dates_test"]

previous_close = data["previous_close_test"]
actual_close = data["actual_close_test"]

target_scaler = joblib.load(
    MODEL_DIR / "market_v3_target_scaler.pkl"
)

print("\nData loaded successfully!")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ---------------------------------------
# 3. Build LSTM model
# ---------------------------------------

print("\nBuilding Market V3 LSTM...")

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(64, return_sequences=True),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(16, activation="relu"),

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
# 4. Callbacks
# ---------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# ---------------------------------------
# 5. Train model
# ---------------------------------------

print("\nStarting Market V3 training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    shuffle=False,
    verbose=1
)

# ---------------------------------------
# 6. Generate return predictions
# ---------------------------------------

print("\nGenerating test predictions...")

predicted_scaled = model.predict(
    X_test,
    verbose=0
)

predicted_returns = target_scaler.inverse_transform(
    predicted_scaled
).flatten()

actual_returns = y_test_raw.flatten()

# ---------------------------------------
# 7. Convert returns to closing prices
# ---------------------------------------

predicted_prices = (
    previous_close * (1 + predicted_returns)
)

actual_prices = actual_close
baseline_prices = previous_close

# ---------------------------------------
# 8. Evaluation metrics
# ---------------------------------------

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),

        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),

        "MAPE": mean_absolute_percentage_error(
            actual, predicted
        ) * 100,

        "R2": r2_score(actual, predicted)
    }


v3_metrics = calculate_metrics(
    actual_prices,
    predicted_prices
)

baseline_metrics = calculate_metrics(
    actual_prices,
    baseline_prices
)

results = pd.DataFrame(
    [v3_metrics, baseline_metrics],
    index=[
        "Market V3 LSTM",
        "Previous Close Baseline"
    ]
)

print("\n" + "=" * 55)
print("MARKET V3 EVALUATION RESULTS")
print("=" * 55)

print(results.round(4).to_string())

# ---------------------------------------
# 9. Directional accuracy
# ---------------------------------------

actual_direction = np.sign(actual_returns)
predicted_direction = np.sign(predicted_returns)

directional_accuracy = np.mean(
    actual_direction == predicted_direction
) * 100

baseline_direction = np.mean(
    actual_direction == 0
) * 100

print(
    f"\nV3 Directional Accuracy: "
    f"{directional_accuracy:.2f}%"
)

print(
    f"Previous-close direction accuracy "
    f"(zero-return prediction): {baseline_direction:.2f}%"
)

# ---------------------------------------
# 10. Save predictions and evaluation
# ---------------------------------------

predictions_df = pd.DataFrame({
    "Date": dates_test,
    "Previous_Close": previous_close,
    "Actual_Return": actual_returns,
    "Predicted_Return": predicted_returns,
    "Actual_Close": actual_prices,
    "V3_Predicted_Close": predicted_prices,
    "Previous_Close_Prediction": baseline_prices
})

predictions_df.to_csv(
    MODEL_DIR / "market_v3_predictions.csv",
    index=False
)

results.to_csv(
    MODEL_DIR / "market_v3_evaluation.csv"
)

# ---------------------------------------
# 11. Save trained model
# ---------------------------------------

model.save(
    MODEL_DIR / "market_v3.keras"
)

# ---------------------------------------
# 12. Plot actual vs predicted prices
# ---------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    actual_prices,
    label="Actual Close",
    linewidth=2
)

plt.plot(
    predicted_prices,
    label="Market V3 Prediction",
    linestyle="--"
)

plt.plot(
    baseline_prices,
    label="Previous Close Baseline",
    alpha=0.7
)

plt.title("RELIANCE Market V3 Prediction")
plt.xlabel("Test Trading Sessions")
plt.ylabel("Stock Price (INR)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "market_v3_prediction.png",
    dpi=150
)

plt.close()

# ---------------------------------------
# 13. Plot training history
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

plt.title("Market V3 Training History")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "market_v3_training.png",
    dpi=150
)

plt.close()

# ---------------------------------------
# 14. Completion message
# ---------------------------------------

print("\n" + "=" * 55)
print("MARKET V3 TRAINING COMPLETED!")
print("=" * 55)

print("\nSaved files:")
print("Model: models/market_v3.keras")
print("Predictions: models/market_v3_predictions.csv")
print("Evaluation: models/market_v3_evaluation.csv")
print("Prediction chart: plots/market_v3_prediction.png")
print("Training chart: plots/market_v3_training.png")
