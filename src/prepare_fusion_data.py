
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

STOCK_FILE = BASE_DIR / "data" / "RELIANCE.csv"
NEWS_FILE = BASE_DIR / "data" / "reliance_sentiment_filtered.csv"
OUTPUT_FILE = BASE_DIR / "data" / "fusion_dataset.csv"

print("Loading stock and news datasets...")

# --------------------------------
# 1. Load stock data
# --------------------------------

stock = pd.read_csv(STOCK_FILE)

stock["Date"] = pd.to_datetime(
    stock["Date"], utc=True
).dt.tz_convert("Asia/Kolkata")

stock["trading_date"] = stock["Date"].dt.strftime("%Y-%m-%d")

# Market opening time: 9:15 AM IST
stock["market_open"] = (
    stock["Date"].dt.normalize()
    + pd.Timedelta(hours=9, minutes=15)
)

stock = stock.sort_values("market_open").reset_index(drop=True)

# --------------------------------
# 2. Load news data
# --------------------------------

news = pd.read_csv(NEWS_FILE)

news["publishedAt"] = pd.to_datetime(
    news["publishedAt"],
    utc=True,
    errors="coerce"
).dt.tz_convert("Asia/Kolkata")

news = news.dropna(subset=["publishedAt"])

score_columns = [
    "positive_score",
    "negative_score",
    "neutral_score",
    "sentiment_score"
]

for column in score_columns:
    news[column] = pd.to_numeric(
        news[column], errors="coerce"
    )

news = news.dropna(subset=score_columns)

# --------------------------------
# 3. Assign news to next market open
# --------------------------------

news = news.sort_values("publishedAt")

assigned_news = pd.merge_asof(
    news,
    stock[["trading_date", "market_open"]],
    left_on="publishedAt",
    right_on="market_open",
    direction="forward",
    allow_exact_matches=False
)

# Remove articles with no future trading session
assigned_news = assigned_news.dropna(
    subset=["trading_date"]
)

# --------------------------------
# 4. Aggregate news by trading date
# --------------------------------

daily_news = (
    assigned_news.groupby("trading_date")
    .agg(
        article_count=("title", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        avg_positive_probability=("positive_score", "mean"),
        avg_negative_probability=("negative_score", "mean"),
        avg_neutral_probability=("neutral_score", "mean")
    )
    .reset_index()
)

# --------------------------------
# 5. Merge news with stock data
# --------------------------------

fusion = stock.merge(
    daily_news,
    on="trading_date",
    how="left"
)

news_features = [
    "article_count",
    "avg_sentiment",
    "avg_positive_probability",
    "avg_negative_probability",
    "avg_neutral_probability"
]

fusion[news_features] = fusion[news_features].fillna(0)

fusion["article_count"] = fusion["article_count"].astype(int)

# --------------------------------
# 6. Create previous-session features
# --------------------------------

market_columns = [
    "Open", "High", "Low", "Close", "Volume"
]

for column in market_columns:
    fusion[f"previous_{column.lower()}"] = (
        fusion[column].shift(1)
    )

# Previous session return only
fusion["previous_return"] = (
    fusion["Close"].pct_change().shift(1)
)

# Remove first two rows where previous features
# are unavailable
fusion = fusion.dropna(
    subset=["previous_close", "previous_return"]
).reset_index(drop=True)

# --------------------------------
# 7. Save final dataset
# --------------------------------

fusion.to_csv(OUTPUT_FILE, index=False)

print("\nFUSION DATASET CREATED SUCCESSFULLY!")

print("Total stock trading sessions:", len(fusion))

print(
    "Trading sessions with news:",
    int((fusion["article_count"] > 0).sum())
)

print(
    "News articles assigned:",
    int(fusion["article_count"].sum())
)

print("\nNews and stock data preview:")

print(
    fusion[
        [
            "trading_date",
            "Close",
            "previous_close",
            "previous_return",
            "article_count",
            "avg_sentiment"
        ]
    ].tail(15).to_string(index=False)
)

print("\nSaved to:", OUTPUT_FILE)
