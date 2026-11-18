
import requests
import pandas as pd

url = "https://api.gdeltproject.org/api/v2/doc/doc"

params = {
    "query": '"Reliance Industries"',
    "mode": "ArtList",
    "format": "json",
    "startdatetime": "20241101000000",
    "enddatetime": "20241107235959",
    "maxrecords": 10,
    "sort": "DateDesc"
}

print("Testing GDELT historical news collection...")

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    articles = data.get("articles", [])

    print("\nTotal articles returned:", len(articles))

    if articles:
        df = pd.DataFrame(articles)

        columns = [
            col for col in
            ["seendate", "title", "domain", "url"]
            if col in df.columns
        ]

        print("\nHistorical articles:")
        print(df[columns].head(10).to_string(index=False))
    else:
        print("No articles returned. We will adjust the query.")

except Exception as e:
    print("\nGDELT request failed:")
    print(e)
