
import os
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "reliance_news.csv"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    raise ValueError(
        "NEWS_API_KEY not found. Check your .env file."
    )

URL = "https://newsapi.org/v2/everything"

QUERY = (
    '"Reliance Industries" OR '
    '"Reliance Industries Limited" OR '
    '"Reliance Jio" OR '
    '"Jio Platforms"'
)

# ---------------------------------------
# 2. DATE CONFIGURATION
# ---------------------------------------

START_DATE = "2026-11-01"
END_DATE = "2026-11-13"

PAGE_SIZE = 100
MAX_PAGES = 100

# ---------------------------------------
# 3. SAVE NEWS SAFELY
# ---------------------------------------

def save_news(new_articles):

    if not new_articles:
        return

    new_df = pd.DataFrame(new_articles)

    new_df["publishedAt"] = pd.to_datetime(
        new_df["publishedAt"],
        utc=True,
        errors="coerce"
    )

    new_df = new_df.dropna(
        subset=["publishedAt", "title"]
    )

    if OUTPUT_FILE.exists():

        old_df = pd.read_csv(OUTPUT_FILE)

        old_df["publishedAt"] = pd.to_datetime(
            old_df["publishedAt"],
            utc=True,
            errors="coerce"
        )

        news_df = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )

    else:
        news_df = new_df

    news_df = news_df.drop_duplicates(
        subset=["url"],
        keep="last"
    )

    news_df = news_df.dropna(
        subset=["title"]
    )

    news_df = news_df.sort_values(
        "publishedAt"
    )

    news_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved successfully. "
        f"Total unique articles: {len(news_df)}"
    )


# ---------------------------------------
# 4. FETCH NEWS
# ---------------------------------------

def fetch_news(start_date, end_date):

    headers = {
        "X-Api-Key": API_KEY
    }

    page = 1
    total_collected = 0

    print(
        f"\nFetching Reliance news: "
        f"{start_date} to {end_date}"
    )

    while page <= MAX_PAGES:

        params = {
            "q": QUERY,
            "from": start_date,
            "to": end_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": PAGE_SIZE,
            "page": page
        }

        try:

            response = requests.get(
                URL,
                params=params,
                headers=headers,
                timeout=30
            )

            # Handle API errors without losing saved data
            if response.status_code != 200:

                print("\nNEWS API ERROR!")
                print("Status code:", response.status_code)
                print("Response:", response.text)

                print(
                    "\nStopping collection. "
                    "Previously saved articles are preserved."
                )

                break

            data = response.json()

            if data.get("status") != "ok":

                print("API returned an error:")
                print(data)

                break

            batch = data.get("articles", [])

            if page == 1:

                total_results = data.get(
                    "totalResults", 0
                )

                print(
                    "Total matching articles:",
                    total_results
                )

                if total_results == 0:
                    print("No articles found.")
                    break

            if not batch:
                print("No more articles available.")
                break

            page_articles = []

            for article in batch:

                page_articles.append({

                    "publishedAt": article.get(
                        "publishedAt"
                    ),

                    "source": (
                        article.get("source") or {}
                    ).get("name"),

                    "title": article.get("title"),

                    "description": article.get(
                        "description"
                    ),

                    "url": article.get("url")

                })

            # Save immediately after every successful page
            save_news(page_articles)

            total_collected += len(page_articles)

            print(
                f"Page {page}: "
                f"{len(page_articles)} articles collected"
            )

            if len(batch) < PAGE_SIZE:
                break

            if page * PAGE_SIZE >= total_results:
                break

            page += 1

        except requests.RequestException as error:

            print("\nRequest failed:", error)

            print(
                "Previously saved articles "
                "will remain available."
            )

            break

        except ValueError as error:

            print("Invalid API response:", error)
            break

    print("\n--------------------------------")
    print("NEWS COLLECTION SUMMARY")
    print("--------------------------------")

    print("Articles fetched this run:", total_collected)

    if OUTPUT_FILE.exists():

        final_df = pd.read_csv(OUTPUT_FILE)

        print(
            "Total unique articles in file:",
            len(final_df)
        )

    print("Output file:", OUTPUT_FILE)


# ---------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------

if __name__ == "__main__":

    print("Starting Reliance news collection...")

    fetch_news(
        START_DATE,
        END_DATE
    )

    print("\nNews collection process finished.")

