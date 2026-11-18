
import pandas as pd
from pathlib import Path

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

STOCK_FILE = BASE_DIR / "data" / "RELIANCE.csv"

OLD_NEWS_FILE = (
    BASE_DIR / "data" / "reliance_sentiment_filtered.csv"
)

FINBERT_FILE = (
    BASE_DIR / "data" / "sentiment_news"
    / "reliance_news_2024_11_sentiment.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "aligned_news"

OUTPUT_FILE = (
    BASE_DIR / "data"
    / "fusion_dataset_with_historical_news.csv"
)

ALIGNED_FILE = (
    OUTPUT_DIR / "reliance_all_news_aligned.csv"
)

DAILY_FILE = (
    OUTPUT_DIR / "reliance_combined_daily_sentiment.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 2. LOAD STOCK DATA
# ============================================

print("Loading Reliance stock data...")

stock = pd.read_csv(STOCK_FILE)

stock["Date"] = pd.to_datetime(
    stock["Date"],
    utc=True,
    errors="coerce"
).dt.tz_convert("Asia/Kolkata")

stock = stock.dropna(subset=["Date"]).copy()

stock["trading_date"] = (
    stock["Date"].dt.strftime("%Y-%m-%d")
)

# Market opening time: 9:15 AM IST
stock["market_open"] = (
    stock["Date"].dt.normalize()
    + pd.Timedelta(hours=9, minutes=15)
)

stock = stock.sort_values(
    "market_open"
).reset_index(drop=True)

if stock["trading_date"].duplicated().any():
    raise ValueError("Duplicate stock trading dates detected.")

print("Stock records:", len(stock))

# ============================================
# 3. LOAD AND STANDARDIZE NEWS
# ============================================

def load_news(file_path, timestamp_column):
    news = pd.read_csv(file_path)

    required_columns = [
        timestamp_column,
        "title",
        "url",
        "positive_score",
        "negative_score",
        "neutral_score"
    ]

    missing = [
        col for col in required_columns
        if col not in news.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {file_path.name}: {missing}"
        )

    news = news.rename(
        columns={timestamp_column: "published_at"}
    )

    news["published_at"] = pd.to_datetime(
        news["published_at"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("Asia/Kolkata")

    score_columns = [
        "positive_score",
        "negative_score",
        "neutral_score"
    ]

    for column in score_columns:
        news[column] = pd.to_numeric(
            news[column],
            errors="coerce"
        )

    # Use existing sentiment score when available.
    # Otherwise calculate the FinBERT score.
    if "sentiment_score" in news.columns:
        news["sentiment_score"] = pd.to_numeric(
            news["sentiment_score"],
            errors="coerce"
        )
    else:
        news["sentiment_score"] = pd.NA

    calculated_score = (
        news["positive_score"]
        - news["negative_score"]
    )

    news["sentiment_score"] = (
        news["sentiment_score"]
        .fillna(calculated_score)
    )

    news = news.dropna(
        subset=[
            "published_at",
            "positive_score",
            "negative_score",
            "neutral_score",
            "sentiment_score",
            "url"
        ]
    ).copy()

    news["url"] = news["url"].astype(str).str.strip()

    news = news[news["url"] != ""]

    return news[
        [
            "published_at",
            "title",
            "url",
            "positive_score",
            "negative_score",
            "neutral_score",
            "sentiment_score"
        ]
    ]

print("\nLoading existing historical news...")

old_news = load_news(
    OLD_NEWS_FILE,
    "publishedAt"
)

print("Existing news records:", len(old_news))

print("\nLoading new FinBERT news...")

finbert_news = load_news(
    FINBERT_FILE,
    "published_at"
)

print("FinBERT news records:", len(finbert_news))

# ============================================
# 4. COMBINE AND REMOVE DUPLICATES
# ============================================

print("\nCombining news datasets...")

news = pd.concat(
    [old_news, finbert_news],
    ignore_index=True
)

original_news_count = len(news)

news["url_key"] = (
    news["url"]
    .str.lower()
    .str.strip()
    .str.rstrip("/")
)

news = news.drop_duplicates(
    subset=["url_key"],
    keep="first"
).copy()

news = news.drop(columns=["url_key"])

print("Combined records before deduplication:",
      original_news_count)

print("Duplicate URLs removed:",
      original_news_count - len(news))

print("Unique news records:", len(news))

# ============================================
# 5. ASSIGN NEWS TO NEXT MARKET OPEN
# ============================================

print("\nAligning news with trading sessions...")

news = news.sort_values(
    "published_at"
).reset_index(drop=True)

assigned_news = pd.merge_asof(
    news,
    stock[["trading_date", "market_open"]],
    left_on="published_at",
    right_on="market_open",
    direction="forward",
    allow_exact_matches=False
)

unassigned_count = int(
    assigned_news["trading_date"].isna().sum()
)

assigned_news = assigned_news.dropna(
    subset=["trading_date"]
).copy()

assigned_news["trading_date"] = (
    assigned_news["trading_date"].astype(str)
)

print("News assigned to trading sessions:",
      len(assigned_news))

print("News without a future stock session:",
      unassigned_count)

# ============================================
# 6. SAVE ARTICLE-LEVEL ALIGNED NEWS
# ============================================

assigned_news.to_csv(
    ALIGNED_FILE,
    index=False
)

# ============================================
# 7. AGGREGATE NEWS BY TRADING SESSION
# ============================================

print("\nAggregating daily news features...")

daily_news = (
    assigned_news.groupby("trading_date")
    .agg(
        article_count=("url", "count"),

        avg_sentiment=("sentiment_score", "mean"),

        avg_positive_probability=(
            "positive_score", "mean"
        ),

        avg_negative_probability=(
            "negative_score", "mean"
        ),

        avg_neutral_probability=(
            "neutral_score", "mean"
        )
    )
    .reset_index()
)

daily_news.to_csv(
    DAILY_FILE,
    index=False
)

# ============================================
# 8. MERGE NEWS WITH STOCK DATA
# ============================================

print("\nCreating new fusion dataset...")

fusion = stock.merge(
    daily_news,
    on="trading_date",
    how="left",
    validate="one_to_one"
)

news_features = [
    "article_count",
    "avg_sentiment",
    "avg_positive_probability",
    "avg_negative_probability",
    "avg_neutral_probability"
]

fusion[news_features] = fusion[
    news_features
].fillna(0)

fusion["article_count"] = (
    fusion["article_count"].astype(int)
)

# ============================================
# 9. CREATE PREVIOUS-SESSION FEATURES
# ============================================

market_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for column in market_columns:
    fusion[f"previous_{column.lower()}"] = (
        fusion[column].shift(1)
    )

fusion["previous_return"] = (
    fusion["Close"].pct_change().shift(1)
)

fusion = fusion.dropna(
    subset=[
        "previous_close",
        "previous_return"
    ]
).reset_index(drop=True)

# ============================================
# 10. FINAL VALIDATION
# ============================================

if fusion["trading_date"].duplicated().any():
    raise ValueError(
        "Duplicate dates found in final fusion dataset."
    )

if fusion[news_features].isnull().any().any():
    raise ValueError(
        "Missing values found in news features."
    )

# ============================================
# 11. SAVE NEW FUSION DATASET
# ============================================

fusion.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 50)
print("HISTORICAL NEWS INTEGRATION COMPLETED!")
print("=" * 50)

print("Stock records:", len(stock))
print("Original combined news:", original_news_count)
print("Unique news records:", len(news))
print("Assigned news articles:", len(assigned_news))
print("Unassigned news articles:", unassigned_count)

print(
    "Trading sessions with news:",
    int((fusion["article_count"] > 0).sum())
)

print(
    "Total articles in fusion dataset:",
    int(fusion["article_count"].sum())
)

print("\nFinal dataset shape:", fusion.shape)

print("\nNews feature preview:")

print(
    fusion[
        [
            "trading_date",
            "article_count",
            "avg_sentiment",
            "avg_positive_probability",
            "avg_negative_probability",
            "avg_neutral_probability"
        ]
    ].tail(15).to_string(index=False)
)

print("\nSaved files:")
print(OUTPUT_FILE)
print(ALIGNED_FILE)
print(DAILY_FILE)