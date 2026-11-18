
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load predictions
v3 = pd.read_csv("models/market_v3_predictions.csv")
v4 = pd.read_csv("models/market_v4_predictions.csv")

# Verify matching dates and actual prices
assert v3["Date"].equals(v4["Date"]), "Dates do not match!"
assert np.allclose(v3["Actual_Close"], v4["Actual_Close"]), "Actual prices do not match!"

actual = v3["Actual_Close"]

models = {
    "V3 LSTM": v3["V3_Predicted_Close"],
    "V4 XGBoost": v4["XGBoost_Predicted_Close"],
    "Previous Close Baseline": v3["Previous_Close_Prediction"]
}

results = []

for name, predictions in models.items():
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    r2 = r2_score(actual, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2
    })

comparison = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========\n")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Directional accuracy using the same definition
actual_direction = np.sign(v3["Actual_Return"])

v3_direction = np.sign(v3["Predicted_Return"])
v4_direction = np.sign(v4["XGBoost_Predicted_Return"])

print("\n========== DIRECTIONAL ACCURACY ==========")
print(f"V3 LSTM: {np.mean(actual_direction == v3_direction) * 100:.2f}%")
print(f"V4 XGBoost: {np.mean(actual_direction == v4_direction) * 100:.2f}%")

# Save report
comparison.to_csv("models/final_model_comparison.csv", index=False)

print("\nReport saved: models/final_model_comparison.csv")
