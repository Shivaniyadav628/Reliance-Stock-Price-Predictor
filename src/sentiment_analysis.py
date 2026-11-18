
import pandas as pd
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# --------------------------------
# 1. Paths and settings
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed_news"
    / "reliance_news_2024_11_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "sentiment_news"
OUTPUT_FILE = OUTPUT_DIR / "reliance_news_2024_11_sentiment.csv"

MODEL_NAME = "ProsusAI/finbert"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 2. Load cleaned news
# --------------------------------

print("Loading cleaned news dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total articles:", len(df))

# --------------------------------
# 3. Load FinBERT
# --------------------------------

print("\nLoading FinBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

device = 0 if torch.cuda.is_available() else -1

sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    top_k=None,
    truncation=True,
    max_length=512
)

print("FinBERT loaded successfully!")

# --------------------------------
# 4. Prepare article text
# --------------------------------

df["title"] = df["title"].fillna("").astype(str)

df["description"] = df["description"].fillna("").astype(str)

df["news_text"] = (
    df["title"] + ". " + df["description"]
).str.strip()

# --------------------------------
# 5. Perform sentiment analysis
# --------------------------------

print("\nAnalyzing news sentiment...")

results = sentiment_pipeline(
    df["news_text"].tolist(),
    batch_size=8
)

# --------------------------------
# 6. Extract sentiment scores
# --------------------------------

for label in ["positive", "negative", "neutral"]:
    df[f"{label}_score"] = [
        next(
            item["score"]
            for item in result
            if item["label"].lower() == label
        )
        for result in results
    ]

df["sentiment"] = df[
    ["positive_score", "negative_score", "neutral_score"]
].idxmax(axis=1).str.replace("_score", "", regex=False)

# --------------------------------
# 7. Save results
# --------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nFinBERT sentiment analysis completed!")

print("Total articles analyzed:", len(df))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSaved file:")
print(OUTPUT_FILE)