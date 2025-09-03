import pandas as pd
import numpy as np

def generate_data(bars=1000):
    np.random.seed(42)
    time = pd.date_range(end=pd.Timestamp.now(), periods=bars, freq="min")
    price = np.cumsum(np.random.randn(bars)) + 100
    df = pd.DataFrame({
        "time": time,
        "open": price + np.random.randn(bars),
        "high": price + np.random.rand(bars),
        "low": price - np.random.rand(bars),
        "close": price,
        "volume": np.random.randint(100, 1000, size=bars)
    })
    df.to_csv("data/historical_data.csv", index=False)
    print("✅ historical_data.csv generated in data/")

if __name__ == "__main__":
    generate_data()
