# feature_engineering.py
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators, single-candle, and multi-candle pattern features.
    Returns ML-ready dataframe with numeric values.
    """
    df = df.copy()

    # --- Simple Moving Averages ---
    df["SMA_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["SMA_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    # --- RSI (14) ---
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD (12,26,9) ---
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["Signal"]

    # --- Bollinger Bands (20,2) ---
    ma20 = df["close"].rolling(window=20, min_periods=1).mean()
    std20 = df["close"].rolling(window=20, min_periods=1).std()
    df["BBM"] = ma20
    df["BBU"] = ma20 + (2 * std20)
    df["BBL"] = ma20 - (2 * std20)

    # --- ATR (14) ---
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=1).mean()

    # --- Candlestick Body/Shadow Ratios ---
    df["body"] = abs(df["close"] - df["open"])
    df["upper_shadow"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_shadow"] = df[["open","close"]].min(axis=1) - df["low"]
    df["body_ratio"] = df["body"] / (df["high"] - df["low"] + 1e-9)
    df["upper_shadow_ratio"] = df["upper_shadow"] / (df["high"] - df["low"] + 1e-9)
    df["lower_shadow_ratio"] = df["lower_shadow"] / (df["high"] - df["low"] + 1e-9)

    # --- Support & Resistance Features ---
    df["resistance"] = df["high"].rolling(20, center=True).max()
    df["support"] = df["low"].rolling(20, center=True).min()
    df["distance_to_resistance"] = df["resistance"] - df["close"]
    df["distance_to_support"] = df["close"] - df["support"]

    # --- Single-Candle Pattern Flags ---
    df["pattern_hammer"] = ((df["lower_shadow"] > 2*df["body"]) & (df["upper_shadow"] < df["body"])).astype(int)
    df["pattern_shooting_star"] = ((df["upper_shadow"] > 2*df["body"]) & (df["lower_shadow"] < df["body"])).astype(int)
    df["pattern_doji"] = (df["body_ratio"] < 0.1).astype(int)

    # --- Multi-Candle Pattern Flags ---
    df["pattern_bullish_engulfing"] = 0
    df["pattern_bearish_engulfing"] = 0
    df["pattern_W"] = 0
    df["pattern_M"] = 0
    df["pattern_three_white_soldiers"] = 0
    df["pattern_three_black_crows"] = 0

    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        # Engulfing
        if prev["close"] < prev["open"] and curr["close"] > curr["open"] and curr["close"] > prev["open"] and curr["open"] < prev["close"]:
            df.at[i,"pattern_bullish_engulfing"] = 1
        elif prev["close"] > prev["open"] and curr["close"] < curr["open"] and curr["open"] > prev["close"] and curr["close"] < prev["open"]:
            df.at[i,"pattern_bearish_engulfing"] = 1

    # W/M pattern (5-candle heuristic)
    for i in range(2, len(df)-2):
        slice_df = df.iloc[i-2:i+3]
        highs = slice_df['high']
        lows = slice_df['low']
        # W pattern: low-high-low higher than previous low
        if lows.iloc[0] > lows.iloc[2] < lows.iloc[4] and highs.iloc[2] < highs.iloc[0]:
            df.at[i,"pattern_W"] = 1
        # M pattern: high-low-high lower than previous high
        elif highs.iloc[0] < highs.iloc[2] > highs.iloc[4] and lows.iloc[2] > lows.iloc[0]:
            df.at[i,"pattern_M"] = 1

    # Three White Soldiers / Three Black Crows (trend continuation)
    for i in range(2, len(df)):
        last3 = df.iloc[i-2:i+1]
        # Three White Soldiers
        if all(last3["close"] > last3["open"]) and \
           last3["close"].iloc[0] < last3["close"].iloc[1] < last3["close"].iloc[2]:
            df.at[i,"pattern_three_white_soldiers"] = 1
        # Three Black Crows
        if all(last3["close"] < last3["open"]) and \
           last3["close"].iloc[0] > last3["close"].iloc[1] > last3["close"].iloc[2]:
            df.at[i,"pattern_three_black_crows"] = 1

    # --- Keep only model’s expected features ---
    keep = [
        "open","high","low","close","volume",
        "SMA_10","SMA_20","RSI","MACD","Signal","MACD_HIST",
        "BBL","BBM","BBU","ATR",
        "body","upper_shadow","lower_shadow","body_ratio","upper_shadow_ratio","lower_shadow_ratio",
        "resistance","support","distance_to_resistance","distance_to_support",
        "pattern_hammer","pattern_shooting_star","pattern_doji",
        "pattern_bullish_engulfing","pattern_bearish_engulfing",
        "pattern_W","pattern_M","pattern_three_white_soldiers","pattern_three_black_crows"
    ]

    return df[keep].bfill().fillna(0)
