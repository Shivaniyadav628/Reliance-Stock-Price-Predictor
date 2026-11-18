
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTION_PATH = BASE_DIR / "models" / "predictions.csv"
STOCK_PATH = BASE_DIR / "data" / "RELIANCE.csv"

# Load predictions
results = pd.read_csv(
    PREDICTION_PATH,
    index_col=0,
    parse_dates=True
)

# Load historical closing prices
stock = pd.read_csv(
    STOCK_PATH,
    index_col="Date",
    parse_dates=True
)

# Previous trading day's actual closing price
previous_close = stock["Close"].shift(1)

# Align previous prices with test predictions
results["Previous_Close"] = previous_close.reindex(results.index)

results = results.dropna()

actual = results["Actual"].values
predicted = results["Predicted"].values
baseline = results["Previous_Close"].values

# Evaluation function
def evaluate(actual, predicted, model_name):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    r2 = r2_score(actual, predicted)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2 Score": r2
    }

# Evaluate both models
lstm_results = evaluate(
    actual, predicted, "LSTM"
)

baseline_results = evaluate(
    actual, baseline, "Previous Close Baseline"
)

# Directional accuracy
actual_direction = np.sign(actual - baseline)
predicted_direction = np.sign(predicted - baseline)

direction_accuracy = np.mean(
    actual_direction == predicted_direction
) * 100

# Display results
comparison = pd.DataFrame([
    lstm_results,
    baseline_results
])

print("\n========== MODEL EVALUATION ==========")
print(comparison.to_string(index=False))

print(
    f"\nLSTM Directional Accuracy: "
    f"{direction_accuracy:.2f}%"
)

# Compare MAE
if lstm_results["MAE"] < baseline_results["MAE"]:
    print("\nLSTM has lower MAE than the baseline.")
else:
    print("\nBaseline has lower or equal MAE than LSTM.")

# Save evaluation report
output_path = BASE_DIR / "models" / "evaluation_results.csv"

comparison.to_csv(output_path, index=False)

print("\nEvaluation report saved successfully!")
print("Location:", output_path)

print("\nMODEL EVALUATION COMPLETED!")
