
import copy
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------
# 1. Paths and settings
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "training" / "training_data.npz"
FUSION_FILE = BASE_DIR / "data" / "fusion_dataset_with_historical_news.csv"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "training"

TARGET_SCALER_FILE = MODEL_DIR / "target_scaler.pkl"

RESULTS_FILE = OUTPUT_DIR / "model_comparison_results.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "model_comparison_predictions.csv"

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 15

MARKET_FEATURE_COUNT = 7

# ----------------------------------------
# 2. Reproducibility
# ----------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

print("=" * 60)
print("RELIANCE STOCK PREDICTION - MODEL COMPARISON")
print("=" * 60)

# ----------------------------------------
# 3. Load saved training data
# ----------------------------------------

data = np.load(DATA_FILE)

X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"].astype(np.float32)

X_val = data["X_val"].astype(np.float32)
y_val = data["y_val"].astype(np.float32)

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.float32)

train_dates = data["train_dates"].astype(str)
val_dates = data["val_dates"].astype(str)
test_dates = data["test_dates"].astype(str)

features = data["features"].astype(str)

print("\nTraining data loaded.")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Features:", list(features))

# ----------------------------------------
# 4. Define LSTM model
# ----------------------------------------

class StockLSTM(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
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

# ----------------------------------------
# 5. Train model with early stopping
# ----------------------------------------

def train_model(name, Xtr, ytr, Xv, yv):

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model = StockLSTM(
        input_size=Xtr.shape[2]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(Xtr),
        torch.tensor(ytr)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    Xv_tensor = torch.tensor(Xv).to(DEVICE)
    yv_tensor = torch.tensor(yv).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_losses = []

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())

        model.eval()

        with torch.no_grad():

            val_predictions = model(Xv_tensor)

            val_loss = criterion(
                val_predictions,
                yv_tensor
            ).item()

        train_loss = np.mean(train_losses)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            patience_counter = 0

        else:

            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}."
            )

            break

    model.load_state_dict(best_state)

    print("\nBest epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)

    return model

# ----------------------------------------
# 6. Prepare both feature sets
# ----------------------------------------

# First 7 features are market data.
# All 12 features include market data and news.

X_train_market = X_train[:, :, :MARKET_FEATURE_COUNT]
X_val_market = X_val[:, :, :MARKET_FEATURE_COUNT]
X_test_market = X_test[:, :, :MARKET_FEATURE_COUNT]

X_train_all = X_train
X_val_all = X_val
X_test_all = X_test

# ----------------------------------------
# 7. Train market-only LSTM
# ----------------------------------------

market_model = train_model(
    "MARKET-ONLY LSTM",
    X_train_market,
    y_train,
    X_val_market,
    y_val
)

# ----------------------------------------
# 8. Train market + news LSTM
# ----------------------------------------

combined_model = train_model(
    "MARKET + NEWS LSTM",
    X_train_all,
    y_train,
    X_val_all,
    y_val
)

# ----------------------------------------
# 9. Generate predictions
# ----------------------------------------

market_model.eval()
combined_model.eval()

with torch.no_grad():

    market_predictions_scaled = market_model(
        torch.tensor(X_test_market)
    ).numpy()

    combined_predictions_scaled = combined_model(
        torch.tensor(X_test_all)
    ).numpy()

# ----------------------------------------
# 10. Convert predictions to rupees
# ----------------------------------------

target_scaler = joblib.load(
    TARGET_SCALER_FILE
)

market_predictions = target_scaler.inverse_transform(
    market_predictions_scaled
).flatten()

combined_predictions = target_scaler.inverse_transform(
    combined_predictions_scaled
).flatten()

actual_prices = target_scaler.inverse_transform(
    y_test
).flatten()

# ----------------------------------------
# 11. Previous-close baseline
# ----------------------------------------

fusion_df = pd.read_csv(FUSION_FILE)

fusion_df["trading_date"] = pd.to_datetime(
    fusion_df["trading_date"]
).dt.strftime("%Y-%m-%d")

fusion_df = fusion_df.sort_values(
    "trading_date"
).reset_index(drop=True)

close_by_date = fusion_df.set_index(
    "trading_date"
)["Close"]

previous_close = []

for date in test_dates:

    position = fusion_df.index[
        fusion_df["trading_date"] == date
    ]

    if len(position) != 1:

        raise ValueError(
            f"Could not uniquely find test date: {date}"
        )

    idx = position[0]

    if idx == 0:

        raise ValueError(
            f"No previous trading session for {date}"
        )

    previous_close.append(
        float(fusion_df.loc[idx - 1, "Close"])
    )

previous_close = np.array(previous_close)

# ----------------------------------------
# 12. Evaluation metrics
# ----------------------------------------

def calculate_metrics(actual, predicted):

    errors = actual - predicted

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(errors ** 2))

    mape = np.mean(
        np.abs(errors / actual)
    ) * 100

    actual_direction = np.sign(
        actual - previous_close
    )

    predicted_direction = np.sign(
        predicted - previous_close
    )

    directional_accuracy = np.mean(
        actual_direction == predicted_direction
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Directional Accuracy": directional_accuracy
    }

results = {}

results["Market-only LSTM"] = calculate_metrics(
    actual_prices,
    market_predictions
)

results["Market + News LSTM"] = calculate_metrics(
    actual_prices,
    combined_predictions
)

results["Previous-close baseline"] = calculate_metrics(
    actual_prices,
    previous_close
)

# ----------------------------------------
# 13. Display comparison
# ----------------------------------------

results_df = pd.DataFrame(results).T

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.round(4).to_string()
)

results_df.to_csv(
    RESULTS_FILE,
    index_label="Model"
)

# ----------------------------------------
# 14. Save individual predictions
# ----------------------------------------

predictions_df = pd.DataFrame({

    "Date": test_dates,

    "Actual_Close": actual_prices,

    "Market_Only_Prediction": market_predictions,

    "Market_News_Prediction": combined_predictions,

    "Previous_Close_Prediction": previous_close

})

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

# ----------------------------------------
# 15. Final summary
# ----------------------------------------

print("\nFiles saved successfully:")

print(RESULTS_FILE)

print(PREDICTIONS_FILE)

print("\nModel comparison completed.")