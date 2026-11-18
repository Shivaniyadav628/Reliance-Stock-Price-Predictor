
import pandas as pd
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
NEWS_PATH = BASE_DIR / "data" / "reliance_news.csv"
OUTPUT_PATH = BASE_DIR / "data" / "reliance_sentiment.csv"

# Check input file
if not NEWS_PATH.exists():
    raise FileNotFoundError("reliance_news.csv not found!")

# Load news
df = pd.read_csv(NEWS_PATH)
df = df.dropna(subset=["title"]).copy()

# Load FinBERT
MODEL_NAME = "ProsusAI/finbert"

print("Loading FinBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("FinBERT loaded successfully!")
print("Running on:", device)

# Store predictions
sentiments = []
positive_scores = []
negative_scores = []
neutral_scores = []
sentiment_scores = []

print("\nAnalysing financial news...")

for index, row in df.iterrows():

    headline = str(row["title"])

    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_class = torch.argmax(probabilities).item()
    label = model.config.id2label[predicted_class]

    positive = probabilities[
        model.config.label2id["positive"]
    ].item()

    negative = probabilities[
        model.config.label2id["negative"]
    ].item()

    neutral = probabilities[
        model.config.label2id["neutral"]
    ].item()

    score = positive - negative

    sentiments.append(label)
    positive_scores.append(positive)
    negative_scores.append(negative)
    neutral_scores.append(neutral)
    sentiment_scores.append(score)

    print(
        f"{index + 1}/{len(df)}: {label} | "
        f"Score: {score:.3f} | {headline[:65]}"
    )

# Add sentiment results
df["sentiment"] = sentiments
df["positive_score"] = positive_scores
df["negative_score"] = negative_scores
df["neutral_score"] = neutral_scores
df["sentiment_score"] = sentiment_scores

# Save results
df.to_csv(OUTPUT_PATH, index=False)

print("\n========== FINBERT RESULTS ==========")
print("Total articles analysed:", len(df))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nAverage sentiment score:")
print(round(df["sentiment_score"].mean(), 4))

print("\nSENTIMENT ANALYSIS COMPLETED!")
print("Saved to:", OUTPUT_PATH)

