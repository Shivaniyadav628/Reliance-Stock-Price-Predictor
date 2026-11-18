
import pandas as pd
import re
from pathlib import Path

# --------------------------------
# 1. Project paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "historical_news" /
    "2024" / "reliance_news_2024_11.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed_news"

OUTPUT_FILE = OUTPUT_DIR / "reliance_news_2024_11_cleaned.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 2. Load original news data
# --------------------------------

df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))

# --------------------------------
# 3. Convert publication dates
# --------------------------------

df["published_at"] = pd.to_datetime(
    df["published_at"],
    utc=True,
    errors="coerce"
)

df = df.dropna(subset=["published_at"])

# --------------------------------
# 4. Handle missing values
# --------------------------------

df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")

# --------------------------------
# 5. Clean text
# --------------------------------

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["title"] = df["title"].apply(clean_text)
df["description"] = df["description"].apply(clean_text)

# --------------------------------
# 6. Remove repeated URLs
# --------------------------------

df["url"] = df["url"].fillna("").astype(str).str.strip()

df = df.sort_values("published_at")

df = df.drop_duplicates(
    subset=["url"],
    keep="first"
)

# --------------------------------
# 7. Create publication date
# --------------------------------

df["published_date"] = (
    df["published_at"].dt.strftime("%Y-%m-%d")
)

# --------------------------------
# 8. Combine title and description
# --------------------------------

df["article_text"] = (
    df["title"] + ". " + df["description"]
).str.strip()

# --------------------------------
# 9. Final sorting
# --------------------------------

df = df.sort_values("published_at")

# --------------------------------
# 10. Save cleaned dataset
# --------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nNews preprocessing completed!")

print("Original records: 1,086")
print("Cleaned records:", len(df))
print("Removed repeated URLs:", 1086 - len(df))
print("Missing descriptions:", df["description"].eq("").sum())

print("\nSaved file:")
print(OUTPUT_FILE)