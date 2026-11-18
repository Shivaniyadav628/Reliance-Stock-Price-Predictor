import pandas as pd

files = [
    "data/reliance_sentiment_filtered.csv",
    "data/daily_sentiment.csv",
    "data/fusion_dataset.csv"
]

for file in files:
    print("\n" + "=" * 70)
    print("FILE:", file)
    print("=" * 70)

    df = pd.read_csv(file)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    date_columns = [
        col for col in df.columns
        if "date" in col.lower() or "time" in col.lower()
    ]

    for col in date_columns:
        dates = pd.to_datetime(df[col], errors="coerce")
        print(f"\nDate column: {col}")
        print("Start:", dates.min())
        print("End:", dates.max())

    if "article_count" in df.columns:
        print("\nTotal articles:", df["article_count"].sum())
        print("Dates with news:", (df["article_count"] > 0).sum())

        print("\nNews coverage:")
        print(
            df.loc[df["article_count"] > 0]
            .to_string(index=False)
        )

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))