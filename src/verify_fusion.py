
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = BASE_DIR / "data" / "fusion_dataset_with_historical_news.csv"

df = pd.read_csv(FILE)

print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\n========== DATE RANGE ==========")

print("First date:", df["Date"].iloc[0])
print("Last date:", df["Date"].iloc[-1])

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

print("\n========== NEWS COVERAGE BY YEAR ==========")

df["year"] = pd.to_datetime(df["Date"], utc=True).dt.year

print(
    df.groupby("year").agg(
        trading_sessions=("Date", "count"),
        total_articles=("article_count", "sum"),
        sessions_with_news=("article_count", lambda x: (x > 0).sum())
    )
)

print("\n========== NEWS FEATURE CHECK ==========")

news_columns = [
    "article_count",
    "avg_sentiment",
    "avg_positive_probability",
    "avg_negative_probability",
    "avg_neutral_probability"
]

print(df[news_columns].describe())

print("\n========== DUPLICATE DATE CHECK ==========")

print("Duplicate dates:", df["Date"].duplicated().sum())

print("\n========== VERIFICATION COMPLETED ==========")