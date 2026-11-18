
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "reliance_sentiment.csv"
OUTPUT_FILE = BASE_DIR / "data" / "reliance_sentiment_filtered.csv"

df = pd.read_csv(INPUT_FILE)

# Explicitly reviewed article indexes from the current 100-article dataset.
# These are indexes from the previous filtered dataset.
keep_indexes = [
    0, 3, 4, 7, 8,
    10, 11, 12, 13, 14,
    15, 16, 17,
    21, 24, 25, 27, 28, 29, 30, 31, 32,
    35, 37, 38, 39, 40, 41
]

# Recreate the original keyword filter to preserve the same index ordering
import re

keywords = [
    r"\breliance industries\b",
    r"\breliance industries limited\b",
    r"\bril\b",
    r"\breliance jio\b",
    r"\bjio platforms\b",
    r"\bjio\b",
    r"\breliance retail\b",
    r"\breliance consumer\b",
    r"\breliance brands\b",
    r"\bmukesh ambani\b",
    r"\breliance new energy\b",
]

pattern = re.compile("|".join(keywords), re.IGNORECASE)

df["search_text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("")
)

original_filtered = df[
    df["search_text"].str.contains(pattern, na=False)
].copy()

original_filtered = original_filtered.drop_duplicates(
    subset=["url"],
    keep="first"
)

original_filtered["publishedAt"] = pd.to_datetime(
    original_filtered["publishedAt"],
    utc=True,
    errors="coerce"
)

original_filtered = original_filtered.dropna(
    subset=["publishedAt"]
)

original_filtered = original_filtered.sort_values("publishedAt")

# Select reviewed articles
result = original_filtered.iloc[keep_indexes].copy()

# Remove temporary column
result = result.drop(columns=["search_text"])

# Save
result.to_csv(OUTPUT_FILE, index=False)

print("STRICT NEWS FILTER COMPLETED!")
print("Original news articles:", len(df))
print("Keyword-filtered articles:", len(original_filtered))
print("Strictly selected articles:", len(result))

print("\nSentiment distribution:")
print(result["sentiment"].value_counts())

print("\nSelected headlines:")
print(result[["title", "sentiment_score"]].to_string(index=False))

print("\nSaved to:", OUTPUT_FILE)

