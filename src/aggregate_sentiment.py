
import pandas as pd

from pathlib import Path

# --------------------------------
# 1. Paths and settings
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "sentiment_news"
    / "reliance_news_2024_11_sentiment.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "daily_sentiment"

OUTPUT_FILE = (
    OUTPUT_DIR / "reliance_daily_sentiment_2024_11.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 2. Load sentiment dataset
# --------------------------------

print("Loading FinBERT sentiment dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total articles:", len(df))

# --------------------------------
# 3. Convert timestamps to IST
# --------------------------------

df["published_at"] = pd.to_datetime(
    df["published_at"],
    utc=True,
    errors="coerce"
)

df = df.dropna(subset=["published_at"]).copy()

df["published_date_ist"] = (
    df["published_at"]
    .dt.tz_convert("Asia/Kolkata")
    .dt.strftime("%Y-%m-%d")
)

# --------------------------------
# 4. Aggregate daily sentiment
# --------------------------------

print("\nAggregating daily sentiment...")

daily = (
    df.groupby("published_date_ist")
    .agg(
        article_count=("url", "count"),

        positive_count=(
            "sentiment",
            lambda x: (x == "positive").sum()
        ),

        negative_count=(
            "sentiment",
            lambda x: (x == "negative").sum()
        ),

        neutral_count=(
            "sentiment",
            lambda x: (x == "neutral").sum()
        ),

        avg_positive_score=("positive_score", "mean"),
        avg_negative_score=("negative_score", "mean"),
        avg_neutral_score=("neutral_score", "mean")
    )
    .reset_index()
)

# --------------------------------
# 5. Calculate net sentiment
# --------------------------------

daily["net_sentiment"] = (
    daily["avg_positive_score"]
    - daily["avg_negative_score"]
)

# --------------------------------
# 6. Save daily dataset
# --------------------------------

daily = daily.sort_values("published_date_ist")

daily.to_csv(OUTPUT_FILE, index=False)

print("\nDaily sentiment aggregation completed!")

print("Total articles:", len(df))
print("Total calendar days with news:", len(daily))

print("\nDaily sentiment preview:")
print(daily.head(10).to_string(index=False))

print("\nSaved file:")
print(OUTPUT_FILE)