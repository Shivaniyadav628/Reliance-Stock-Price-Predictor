
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================
# 1. PATHS AND SETTINGS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "training" / "training_data.npz"

FUSION_FILE = (
    BASE_DIR / "data"
    / "fusion_dataset_with_historical_news.csv"
)

MODEL_FILE = BASE_DIR / "models" / "best_lstm_model.pth"

SCALER_FILE = BASE_DIR / "models" / "target_scaler.pkl"

OUTPUT_FILE = (
    BASE_DIR / "data" / "training"
    / "lstm_test_predictions.csv"
)

DEVICE = torch.device("cpu")

# ============================================
# 2. DEFINE MODEL
# ============================================

class StockLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=12,
            hidden_size=64,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        return self.fc(x)

# ============================================
# 3. LOAD TEST DATA
# ============================================

print("Loading test dataset...")

data = np.load(DATA_FILE)

X_test = torch.tensor(
    data["X_test"],
    dtype=torch.float32
)

y_test_scaled = data["y_test"]

test_dates = data["test_dates"].astype(str)

print("Test input shape:", X_test.shape)
print("Test target shape:", y_test_scaled.shape)

# ============================================
# 4. LOAD MODEL AND SCALER
# ============================================

print("\nLoading trained LSTM model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=True
)

model = StockLSTM().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

target_scaler = joblib.load(SCALER_FILE)

print("Model loaded successfully.")

# ============================================
# 5. MAKE PREDICTIONS
# ============================================

print("\nGenerating predictions...")

with torch.no_grad():
    predictions_scaled = model(
        X_test.to(DEVICE)
    ).cpu().numpy()

predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()

actual_prices = target_scaler.inverse_transform(
    y_test_scaled
).flatten()

# ============================================
# 6. LOAD ACTUAL PREVIOUS CLOSE PRICES
# ============================================

fusion = pd.read_csv(FUSION_FILE)

fusion["trading_date"] = (
    fusion["trading_date"].astype(str)
)

fusion = fusion.sort_values(
    "trading_date"
).reset_index(drop=True)

date_to_index = {
    date: index
    for index, date in enumerate(fusion["trading_date"])
}

previous_closes = []

for date in test_dates:

    index = date_to_index.get(date)

    if index is None or index == 0:
        raise ValueError(
            f"Could not find previous close for {date}"
        )

    previous_closes.append(
        fusion.loc[index - 1, "Close"]
    )

previous_closes = np.array(previous_closes)

# ============================================
# 7. CALCULATE MODEL METRICS
# ============================================

mae = mean_absolute_error(
    actual_prices,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predictions
    )
)

mape = np.mean(
    np.abs(
        (actual_prices - predictions)
        / actual_prices
    )
) * 100

actual_direction = actual_prices > previous_closes

predicted_direction = predictions > previous_closes

direction_accuracy = np.mean(
    actual_direction == predicted_direction
) * 100

# ============================================
# 8. NAIVE BASELINE
# ============================================

baseline_predictions = previous_closes

baseline_mae = mean_absolute_error(
    actual_prices,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        baseline_predictions
    )
)

baseline_mape = np.mean(
    np.abs(
        (actual_prices - baseline_predictions)
        / actual_prices
    )
) * 100

# ============================================
# 9. SAVE PREDICTIONS
# ============================================

results = pd.DataFrame({
    "Date": test_dates,
    "Previous_Close": previous_closes,
    "Actual_Close": actual_prices,
    "Predicted_Close": predictions,
    "Baseline_Prediction": baseline_predictions,
    "Absolute_Error": np.abs(
        actual_prices - predictions
    )
})

results.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================
# 10. FINAL REPORT
# ============================================

print("\n" + "=" * 55)
print("LSTM MODEL EVALUATION RESULTS")
print("=" * 55)

print("\nLSTM MODEL:")
print(f"MAE:  ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(
    f"Directional Accuracy: {direction_accuracy:.2f}%"
)

print("\nNAIVE BASELINE (Previous Close):")
print(f"MAE:  ₹{baseline_mae:.2f}")
print(f"RMSE: ₹{baseline_rmse:.2f}")
print(f"MAPE: {baseline_mape:.2f}%")

print("\nSample predictions:")
print(results.head(10).to_string(index=False))

print("\nSaved predictions:")
print(OUTPUT_FILE)

print("\nEvaluation completed!")