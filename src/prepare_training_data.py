
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ============================================
# 1. PATHS AND SETTINGS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data"
    / "fusion_dataset_with_historical_news.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "training"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 60

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# ============================================
# 2. LOAD DATASET
# ============================================

print("Loading fusion dataset...")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    utc=True
)

df = df.sort_values("Date").reset_index(drop=True)

print("Dataset shape:", df.shape)

# ============================================
# 3. SELECT FEATURES
# ============================================

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
    "avg_neutral_probability"
]

TARGET = "Close"

X_raw = df[FEATURES].astype(float).values
y_raw = df[[TARGET]].astype(float).values

dates = df["trading_date"].astype(str).values

if not np.isfinite(X_raw).all():
    raise ValueError("Input features contain invalid values.")

if not np.isfinite(y_raw).all():
    raise ValueError("Target contains invalid values.")

# ============================================
# 4. CHRONOLOGICAL SPLIT BOUNDARIES
# ============================================

total_rows = len(df)

train_end = int(total_rows * TRAIN_RATIO)

val_end = int(
    total_rows * (TRAIN_RATIO + VAL_RATIO)
)

print("\nSplit boundaries:")
print("Training rows:", train_end)
print("Validation boundary:", val_end)
print("Testing rows:", total_rows - val_end)

# ============================================
# 5. SCALE FEATURES
# FIT ONLY ON TRAINING DATA
# ============================================

print("\nScaling features...")

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(X_raw[:train_end])

target_scaler.fit(y_raw[1:train_end])

X_scaled = feature_scaler.transform(X_raw)

y_scaled = target_scaler.transform(y_raw)

# Save scalers for future predictions
joblib.dump(
    feature_scaler,
    MODEL_DIR / "feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    MODEL_DIR / "target_scaler.pkl"
)

# ============================================
# 6. CREATE SEQUENCES
# ============================================

print("\nCreating LSTM sequences...")

X_sequences = []
y_sequences = []
target_dates = []
target_indices = []

for end_idx in range(
    SEQUENCE_LENGTH - 1,
    total_rows - 1
):
    start_idx = end_idx - SEQUENCE_LENGTH + 1

    X_sequences.append(
        X_scaled[start_idx:end_idx + 1]
    )

    next_idx = end_idx + 1

    y_sequences.append(
        y_scaled[next_idx]
    )

    target_dates.append(dates[next_idx])
    target_indices.append(next_idx)

X_sequences = np.array(X_sequences, dtype=np.float32)

y_sequences = np.array(y_sequences, dtype=np.float32)

target_indices = np.array(target_indices)

target_dates = np.array(target_dates)

print("All input sequences:", X_sequences.shape)
print("All target values:", y_sequences.shape)

# ============================================
# 7. CHRONOLOGICAL TRAIN / VAL / TEST SPLIT
# ============================================

train_mask = target_indices < train_end

val_mask = (
    (target_indices >= train_end)
    & (target_indices < val_end)
)

test_mask = target_indices >= val_end

X_train = X_sequences[train_mask]
y_train = y_sequences[train_mask]

X_val = X_sequences[val_mask]
y_val = y_sequences[val_mask]

X_test = X_sequences[test_mask]
y_test = y_sequences[test_mask]

train_dates = target_dates[train_mask]
val_dates = target_dates[val_mask]
test_dates = target_dates[test_mask]

# ============================================
# 8. SAVE TRAINING DATA
# ============================================

np.savez_compressed(
    OUTPUT_DIR / "training_data.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    train_dates=train_dates,
    val_dates=val_dates,
    test_dates=test_dates,
    features=np.array(FEATURES),
    sequence_length=SEQUENCE_LENGTH
)

# ============================================
# 9. FINAL REPORT
# ============================================

print("\n" + "=" * 50)
print("TRAINING DATA PREPARATION COMPLETED")
print("=" * 50)

print("\nTraining data:", X_train.shape, y_train.shape)
print("Validation data:", X_val.shape, y_val.shape)
print("Testing data:", X_test.shape, y_test.shape)

print("\nTraining target dates:")
print(train_dates[0], "to", train_dates[-1])

print("\nValidation target dates:")
print(val_dates[0], "to", val_dates[-1])

print("\nTesting target dates:")
print(test_dates[0], "to", test_dates[-1])

print("\nNumber of input features:", len(FEATURES))
print("Sequence length:", SEQUENCE_LENGTH)

print("\nSaved files:")
print(OUTPUT_DIR / "training_data.npz")
print(MODEL_DIR / "feature_scaler.pkl")
print(MODEL_DIR / "target_scaler.pkl")

print("\nReady for LSTM model development!")