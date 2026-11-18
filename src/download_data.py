
import yfinance as yf
from pathlib import Path

# Stock symbol for Reliance Industries on NSE
stock_symbol = "RELIANCE.NS"

print("Downloading stock market data...")

try:
    # Download 5 years of daily stock data
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="5y", interval="1d")

    if data.empty:
        print("No data received. Please check your internet connection.")
    else:
        # Find the main project folder
        project_folder = Path(__file__).resolve().parent.parent

        # Create the data folder if it doesn't exist
        data_folder = project_folder / "data"
        data_folder.mkdir(exist_ok=True)

        # Save stock data as a CSV file
        file_path = data_folder / "RELIANCE.csv"
        data.to_csv(file_path)

        print("\nData downloaded successfully!")
        print("Total records:", len(data))
        print("Saved file:", file_path)

        print("\nFirst 5 rows of stock data:")
        print(data.head())

except Exception as error:
    print("Something went wrong:", error)
