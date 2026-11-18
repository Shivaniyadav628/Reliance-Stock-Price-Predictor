
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load stock data
file_path = BASE_DIR / "data" / "RELIANCE.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

print("\n========== STOCK DATA ANALYSIS ==========")

# Display dataset information
print("\nTotal records:", len(df))

print("\nDataset information:")
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nStock Price Statistics:")
print(df[["Open", "High", "Low", "Close", "Volume"]].describe())

# Calculate moving averages
df["MA_50"] = df["Close"].rolling(window=50).mean()
df["MA_200"] = df["Close"].rolling(window=200).mean()

# Create plots folder
plot_dir = BASE_DIR / "plots"
plot_dir.mkdir(exist_ok=True)

# Plot closing price and moving averages
plt.figure(figsize=(14, 7))

plt.plot(df.index, df["Close"], label="Closing Price")
plt.plot(df.index, df["MA_50"], label="50-Day Moving Average")
plt.plot(df.index, df["MA_200"], label="200-Day Moving Average")

plt.title("RELIANCE Stock Price Analysis")
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save graph
graph_path = plot_dir / "reliance_analysis.png"
plt.savefig(graph_path)

print("\nGraph saved successfully!")
print("Graph location:", graph_path)

plt.show()

print("\n========== ANALYSIS COMPLETED ==========")
