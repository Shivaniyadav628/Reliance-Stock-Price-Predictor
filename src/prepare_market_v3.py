
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ---------------------------------------
# 1. Paths and settings
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "RELIANCE.csv"
OUTPUT_FILE = BASE_DIR / "data" / "market_v3_data.npz"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 60

FEATURES = [
    "Open_Return",
    "High_Return",
    "Low_Return",
    "Close_Return",
    "Log_Volume",
    "Daily_Return",
    "SMA20_Gap",
    "SMA50_Gap",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volatility_20",
    "Volume_Change"
]

np.random.seed(42)

print("Loading RELIANCE stock data...")

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------------
# 2. Calculate technical indicators
# ---------------------------------------

print("Calculating technical indicators...")

previous_close = df["Close"].shift(1)

# Price movements relative to previous close
df["Open_Return"] = df["Open"] / previous_close - 1
df["High_Return"] = df["High"] / previous_close - 1
df["Low_Return"] = df["Low"] / previous_close - 1
df["Close_Return"] = df["Close"] / previous_close - 1

# Daily return
df["Daily_Return"] = df["Close"].pct_change()

# Moving averages
sma20 = df["Close"].rolling(20).mean()
sma50 = df["Close"].rolling(50).mean()

df["SMA20_Gap"] = df["Close"] / sma20 - 1
df["SMA50_Gap"] = df["Close"] / sma50 - 1

# RSI 14
delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

rs = average_gain / average_loss.replace(0, np.nan)

df["RSI_14"] = 100 - (100 / (1 + rs))

# Handle zero-loss periods
df.loc[
    (average_loss == 0) & (average_gain > 0),
    "RSI_14"
] = 100

df.loc[
    (average_loss == 0) & (average_gain == 0),
    "RSI_14"
] = 50

# MACD
ema12 = df["Close"].ewm(
    span=12, adjust=False
).mean()

ema26 = df["Close"].ewm(
    span=26, adjust=False
).mean()

macd_line = ema12 - ema26

signal_line = macd_line.ewm(
    span=9, adjust=False
).mean()

df["MACD"] = macd_line / df["Close"]
df["MACD_Signal"] = signal_line / df["Close"]
df["MACD_Histogram"] = (
    (macd_line - signal_line) / df["Close"]
)

# Volatility
df["Volatility_20"] = (
    df["Daily_Return"].rolling(20).std()
)

# Volume
df["Log_Volume"] = np.log1p(df["Volume"])

df["Volume_Change"] = (
    df["Volume"].pct_change()
)

df["Volume_Change"] = (
    df["Volume_Change"].replace(
        [np.inf, -np.inf], np.nan
    )
)

# ---------------------------------------
# 3. Create next-session return target
# ---------------------------------------

# For each row i, predict the return from
# previous session close to session i close.

df["Previous_Close"] = df["Close"].shift(1)

df["Target_Return"] = (
    df["Close"] / df["Previous_Close"] - 1
)

# ---------------------------------------
# 4. Remove incomplete rows
# ---------------------------------------

df = df.replace(
    [np.inf, -np.inf], np.nan
)

df = df.dropna(
    subset=FEATURES + [
        "Target_Return",
        "Previous_Close"
    ]
).reset_index(drop=True)

print("Usable rows:", len(df))

# ---------------------------------------
# 5. Create 60-session sequences
# ---------------------------------------

X = []
y = []
dates = []
previous_closes = []
actual_closes = []

for i in range(LOOKBACK, len(df)):

    # All input sessions finish BEFORE target session.
    sequence = df.loc[
        i - LOOKBACK:i - 1,
        FEATURES
    ].values

    target_return = df.loc[
        i, "Target_Return"
    ]

    X.append(sequence)
    y.append(target_return)

    dates.append(df.loc[i, "Date"])

    previous_closes.append(
        df.loc[i, "Previous_Close"]
    )

    actual_closes.append(
        df.loc[i, "Close"]
    )

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

dates = np.array(dates)
previous_closes = np.array(previous_closes)
actual_closes = np.array(actual_closes)

print("\nSequences created:", len(X))
print("Input shape:", X.shape)

# ---------------------------------------
# 6. Chronological split
# ---------------------------------------

total = len(X)

train_end = int(total * 0.80)
val_end = int(total * 0.90)

X_train = X[:train_end]
X_val = X[train_end:val_end]
X_test = X[val_end:]

y_train = y[:train_end]
y_val = y[train_end:val_end]
y_test = y[val_end:]

dates_train = dates[:train_end]
dates_val = dates[train_end:val_end]
dates_test = dates[val_end:]

previous_close_test = previous_closes[val_end:]
actual_close_test = actual_closes[val_end:]

# ---------------------------------------
# 7. Scale input features
# Fit only on training data
# ---------------------------------------

feature_scaler = StandardScaler()

feature_scaler.fit(
    X_train.reshape(-1, len(FEATURES))
)

def scale_features(data):

    original_shape = data.shape

    scaled = feature_scaler.transform(
        data.reshape(-1, len(FEATURES))
    )

    return scaled.reshape(original_shape)

X_train = scale_features(X_train)
X_val = scale_features(X_val)
X_test = scale_features(X_test)

# ---------------------------------------
# 8. Scale target returns
# ---------------------------------------

target_scaler = StandardScaler()

y_train = target_scaler.fit_transform(y_train)
y_val = target_scaler.transform(y_val)
y_test_scaled = target_scaler.transform(y_test)

# ---------------------------------------
# 9. Save scalers
# ---------------------------------------

joblib.dump(
    feature_scaler,
    MODEL_DIR / "market_v3_feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    MODEL_DIR / "market_v3_target_scaler.pkl"
)

# ---------------------------------------
# 10. Save prepared data
# ---------------------------------------

np.savez_compressed(
    OUTPUT_FILE,

    X_train=X_train,
    X_val=X_val,
    X_test=X_test,

    y_train=y_train,
    y_val=y_val,
    y_test=y_test_scaled,

    y_test_raw=y_test,

    dates_train=dates_train,
    dates_val=dates_val,
    dates_test=dates_test,

    previous_close_test=previous_close_test,
    actual_close_test=actual_close_test
)

# ---------------------------------------
# 11. Summary
# ---------------------------------------

print("\n" + "=" * 45)
print("MARKET V3 DATA PREPARATION COMPLETED!")
print("=" * 45)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nFeatures:", len(FEATURES))
print("Lookback:", LOOKBACK)

print("\nSaved dataset:", OUTPUT_FILE)
print("Saved feature scaler.")
print("Saved target scaler.")
