
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path


# --------------------------------------------------
# 1. Model configuration
# --------------------------------------------------

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "article_count",
    "avg_sentiment",
    "avg_positive_probability",
    "avg_negative_probability",
    "avg_neutral_probability",
]

SEQUENCE_LENGTH = 60


# --------------------------------------------------
# 2. Exact architecture used during training
# --------------------------------------------------

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
        x = self.fc(x)

        return x


# --------------------------------------------------
# 3. Load model, scalers and dataset
# --------------------------------------------------

def predict_next_close(base_dir):

    base_dir = Path(base_dir)

    dataset_path = (
        base_dir / "data" /
        "fusion_dataset_with_historical_news.csv"
    )

    model_path = (
        base_dir / "models" / "best_lstm_model.pth"
    )

    feature_scaler_path = (
        base_dir / "models" / "feature_scaler.pkl"
    )

    target_scaler_path = (
        base_dir / "models" / "target_scaler.pkl"
    )

    required_files = [
        dataset_path,
        model_path,
        feature_scaler_path,
        target_scaler_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------
    # 4. Read and validate dataset
    # --------------------------------------------------

    df = pd.read_csv(dataset_path)

    if "Date" not in df.columns:
        raise ValueError(
            "Dataset must contain a Date column."
        )

    if "Close" not in df.columns:
        raise ValueError(
            "Dataset must contain a Close column."
        )

    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # IMPORTANT:
    # Do not use utc=True here.
    # It shifts midnight IST dates to the previous UTC date.

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(
            "At least 60 valid records are required."
        )

    # --------------------------------------------------
    # 5. Prepare latest 60 model features
    # --------------------------------------------------

    feature_data = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    feature_data = feature_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    latest_feature_data = feature_data.tail(
        SEQUENCE_LENGTH
    )

    if latest_feature_data.isna().any().any():
        raise ValueError(
            "The latest 60 records contain missing or "
            "invalid model features. Prediction cannot continue."
        )

    latest_features = latest_feature_data.to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------
    # 6. Load saved scalers
    # --------------------------------------------------

    feature_scaler = joblib.load(
        feature_scaler_path
    )

    target_scaler = joblib.load(
        target_scaler_path
    )

    if feature_scaler.n_features_in_ != len(FEATURES):
        raise ValueError(
            "Feature scaler has an unexpected input size."
        )

    if target_scaler.n_features_in_ != 1:
        raise ValueError(
            "Target scaler has an unexpected input size."
        )

    # --------------------------------------------------
    # 7. Scale input features
    # --------------------------------------------------

    scaled_features = feature_scaler.transform(
        latest_features
    )

    input_tensor = torch.tensor(
        scaled_features,
        dtype=torch.float32
    ).unsqueeze(0)

    # Expected shape: (1, 60, 12)

    # --------------------------------------------------
    # 8. Load trained LSTM model
    # --------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model = StockLSTM()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------
    # 9. Generate next closing price prediction
    # --------------------------------------------------

    with torch.no_grad():

        scaled_prediction = model(
            input_tensor
        ).numpy()

    predicted_close = target_scaler.inverse_transform(
        scaled_prediction
    )[0, 0]

    if not np.isfinite(predicted_close):
        raise ValueError(
            "The model generated an invalid prediction."
        )

    # --------------------------------------------------
    # 10. Prepare prediction result
    # --------------------------------------------------

    latest_date = df["Date"].iloc[-1]

    last_actual_close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    ).iloc[-1]

    if not np.isfinite(last_actual_close):
        raise ValueError(
            "Latest actual closing price is invalid."
        )

    return {
        "predicted_close": float(predicted_close),

        "latest_data_date": latest_date.strftime(
            "%d %b %Y"
        ),

        "last_actual_close": float(
            last_actual_close
        ),

        "sequence_length": SEQUENCE_LENGTH,

        "model_name": "Market + News LSTM",
    }
