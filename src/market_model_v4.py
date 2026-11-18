
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from xgboost import XGBRegressor

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

print("Loading Market V4 training data...")

# ---------------------------------------
# 2. Load prepared data
# ---------------------------------------

data = np.load(DATA_FILE, allow_pickle=True)

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train_scaled = data["y_train"]
y_val_scaled = data["y_val"]
y_test_raw = data["y_test_raw"].flatten()

dates_test = pd.to_datetime(data["dates_test"])

previous_close = data["previous_close_test"].flatten()
actual_close = data["actual_close_test"].flatten()

target_scaler = joblib.load(
    MODEL_DIR / "market_v3_target_scaler.pkl"
)

# Convert scaled training and validation targets
# back into actual percentage returns.

y_train = target_scaler.inverse_transform(
    y_train_scaled.reshape(-1, 1)
).flatten()

y_val = target_scaler.inverse_transform(
    y_val_scaled.reshape(-1, 1)
).flatten()

print("\nData loaded successfully!")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ---------------------------------------
# 3. Flatten LSTM sequences for XGBoost
# ---------------------------------------

# Each sample contains 60 sessions x 14 features.
# XGBoost receives one row of 840 features per sample.

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("\nFlattened feature shapes:")
print("Training:", X_train_flat.shape)
print("Validation:", X_val_flat.shape)
print("Testing:", X_test_flat.shape)

# ---------------------------------------
# 4. Train and compare XGBoost candidates
# ---------------------------------------

candidates = [
    {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03
    },
    {
        "n_estimators": 500,
        "max_depth": 2,
        "learning_rate": 0.03
    },
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03
    }
]

best_model = None
best_validation_mae = float("inf")
best_parameters = None

validation_results = []

print("\nTraining XGBoost candidates...")

for index, parameters in enumerate(candidates, start=1):

    print(f"\nTraining candidate {index}/{len(candidates)}")
    print("Parameters:", parameters)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        learning_rate=parameters["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(
        X_train_flat,
        y_train,
        eval_set=[(X_val_flat, y_val)],
        verbose=False
    )

    validation_predictions = model.predict(X_val_flat)

    validation_mae = mean_absolute_error(
        y_val,
        validation_predictions
    )

    validation_results.append({
        "Candidate": index,
        "Validation_MAE_Return": validation_mae,
        "Parameters": str(parameters)
    })

    print(
        f"Validation return MAE: {validation_mae:.6f}"
    )

    if validation_mae < best_validation_mae:

        best_validation_mae = validation_mae
        best_model = model
        best_parameters = parameters

print("\n" + "=" * 55)
print("BEST XGBOOST CONFIGURATION")
print("=" * 55)

print("Parameters:", best_parameters)
print(
    f"Validation return MAE: {best_validation_mae:.6f}"
)

validation_df = pd.DataFrame(validation_results)

validation_df.to_csv(
    MODEL_DIR / "market_v4_validation.csv",
    index=False
)

# ---------------------------------------
# 5. Generate final test predictions
# ---------------------------------------

print("\nGenerating XGBoost test predictions...")

predicted_returns = best_model.predict(
    X_test_flat
)

actual_returns = y_test_raw

predicted_prices = (
    previous_close * (1 + predicted_returns)
)

actual_prices = actual_close
baseline_prices = previous_close

# ---------------------------------------
# 6. Calculate evaluation metrics
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


xgb_metrics = calculate_metrics(
    actual_prices,
    predicted_prices
)

baseline_metrics = calculate_metrics(
    actual_prices,
    baseline_prices
)

# ---------------------------------------
# 7. Compare against V3
# ---------------------------------------

v3_file = MODEL_DIR / "market_v3_predictions.csv"

v3_df = pd.read_csv(v3_file)

v3_df["Date"] = pd.to_datetime(
    v3_df["Date"]
).dt.normalize()

test_dates = pd.Series(dates_test).dt.normalize()

v3_df = v3_df.set_index("Date").reindex(
    test_dates
)

v3_prices = v3_df[
    "V3_Predicted_Close"
].to_numpy()

if np.isnan(v3_prices).any():
    raise ValueError(
        "Some V3 predictions could not be matched by date. "
        "Check market_v3_predictions.csv."
    )

v3_metrics = calculate_metrics(
    actual_prices,
    v3_prices
)

results = pd.DataFrame(
    [
        xgb_metrics,
        v3_metrics,
        baseline_metrics
    ],
    index=[
        "Market V4 XGBoost",
        "Market V3 LSTM",
        "Previous Close Baseline"
    ]
)

print("\n" + "=" * 65)
print("MARKET V4 FINAL TEST RESULTS")
print("=" * 65)

print(results.round(4).to_string())

# ---------------------------------------
# 8. Directional accuracy
# ---------------------------------------

actual_direction = np.sign(actual_returns)

xgb_direction = np.sign(predicted_returns)

v3_returns = (
    v3_prices / previous_close
) - 1

v3_direction = np.sign(v3_returns)

xgb_directional_accuracy = np.mean(
    actual_direction == xgb_direction
) * 100

v3_directional_accuracy = np.mean(
    actual_direction == v3_direction
) * 100

print(
    f"\nXGBoost Directional Accuracy: "
    f"{xgb_directional_accuracy:.2f}%"
)

print(
    f"V3 LSTM Directional Accuracy: "
    f"{v3_directional_accuracy:.2f}%"
)

# ---------------------------------------
# 9. Save predictions
# ---------------------------------------

predictions_df = pd.DataFrame({
    "Date": dates_test,
    "Previous_Close": previous_close,
    "Actual_Return": actual_returns,
    "XGBoost_Predicted_Return": predicted_returns,
    "Actual_Close": actual_prices,
    "XGBoost_Predicted_Close": predicted_prices,
    "V3_Predicted_Close": v3_prices,
    "Previous_Close_Prediction": baseline_prices
})

predictions_df.to_csv(
    MODEL_DIR / "market_v4_predictions.csv",
    index=False
)

results.to_csv(
    MODEL_DIR / "market_v4_evaluation.csv"
)

# ---------------------------------------
# 10. Save trained XGBoost model
# ---------------------------------------

joblib.dump(
    best_model,
    MODEL_DIR / "market_v4_xgboost.pkl"
)

# ---------------------------------------
# 11. Plot prediction comparison
# ---------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    actual_prices,
    label="Actual Close",
    linewidth=2
)

plt.plot(
    predicted_prices,
    label="Market V4 XGBoost",
    linestyle="--"
)

plt.plot(
    v3_prices,
    label="Market V3 LSTM",
    linestyle=":"
)

plt.plot(
    baseline_prices,
    label="Previous Close Baseline",
    alpha=0.7
)

plt.title("RELIANCE Market V4 Model Comparison")
plt.xlabel("Test Trading Sessions")
plt.ylabel("Stock Price (INR)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "market_v4_comparison.png",
    dpi=150
)

plt.close()

# ---------------------------------------
# 12. Completion
# ---------------------------------------

print("\n" + "=" * 55)
print("MARKET V4 TRAINING COMPLETED!")
print("=" * 55)

print("\nSaved files:")
print("Model: models/market_v4_xgboost.pkl")
print("Predictions: models/market_v4_predictions.csv")
print("Evaluation: models/market_v4_evaluation.csv")
print("Validation: models/market_v4_validation.csv")
print("Comparison chart: plots/market_v4_comparison.png")
