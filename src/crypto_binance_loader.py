"""
crypto_binance_loader.py — Download and cache historical 1H and 1D klines from Binance.

Uses python-binance to bypass Yahoo Finance limitations and fetch 
deep history (4+ years) of 1-hour and 1-day candlestick data.
"""

import os
import sys
import time
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure stdout can print emojis on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize public client (no API keys needed for historical klines)
client = Client()

def get_binance_symbol(ticker):
    """
    Convert universe format (BTC-USD or BTCUSDT) to Binance format (BTCUSDT).
    """
    if "-" in ticker:
        coin = ticker.split("-")[0]
        return f"{coin}USDT"
    if not ticker.endswith("USDT") and not ticker.endswith("BTC"):
        return f"{ticker}USDT"
    return ticker

def get_klines_to_df(symbol, interval, start_str, end_str):
    """
    Fetch historical klines from Binance and return a formatted Pandas DataFrame.
    """
    try:
        # get_historical_klines handles pagination automatically
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str
        )
        
        if not klines:
            return None
            
        # Define Binance kline columns
        cols = [
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines, columns=cols)
        
        # Format types
        numeric_cols = cols[1:6] + cols[7:-1]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # Convert timestamp to datetime index
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Keep only essential columns (plus some extras useful for order flow)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 
                 'quote_asset_volume', 'number_of_trades', 
                 'taker_buy_base_asset_volume']]
                 
        # Additional crypto-native feature building relies on taker buy volume
        # (Taker buy base asset volume indicates aggressive buying market orders)
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error fetching {symbol} ({interval}): {e}")
        return None

def download_and_cache_binance(ticker, interval, start_date, end_date, data_dir):
    """
    Download or load cached Binance kline data.
    If interval is 1H, we load the pre-downloaded 2018+ data from tradeBot.
    Else if 1D, we download from Binance.
    """
    os.makedirs(data_dir, exist_ok=True)
    binance_symbol = get_binance_symbol(ticker)
    interval_str = "1h" if "1h" in interval.lower() else "1d"
    
    # --- IF 1H, LOAD DIRECTLY FROM tradeBot ---
    if interval_str == "1h":
        tradebot_file = f"c:/Users/KIIT/Desktop/tradeBot/data/candles/{binance_symbol}_1h.json"
        if os.path.exists(tradebot_file):
            try:
                df_json = pd.read_json(tradebot_file)
                df_json['datetime'] = pd.to_datetime(df_json['time'], unit='ms')
                df_json.set_index('datetime', inplace=True)
                
                # Format to match expected column names
                rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                df_json.rename(columns=rename_map, inplace=True)
                
                # Filter by exact start/end
                if start_date:
                    df_json = df_json[df_json.index >= pd.to_datetime(start_date)]
                if end_date:
                    df_json = df_json[df_json.index <= pd.to_datetime(end_date)]
                    
                print(f"  ✓ {binance_symbol} 1h (loaded from tradeBot, {len(df_json)} candles)")
                return df_json[['Open', 'High', 'Low', 'Close', 'Volume']]
            except Exception as e:
                print(f"  ✗ Error loading tradeBot {binance_symbol} 1h: {e}")
                return None
        else:
            print(f"  ✗ {binance_symbol} 1h missing in tradeBot folder.")
            return None

    # --- IF 1D, DOWNLOAD OR USE OUR OWN CACHE ---
    safe_name = f"{binance_symbol}_{interval_str}"
    cache_file = os.path.join(data_dir, f"{safe_name}.csv")
    
    # Load from cache if it exists
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(df) > 0:
            print(f"  ✓ {binance_symbol} {interval_str} (cached, {len(df)} candles)")
            return df
            
    # Download fresh
    df = get_klines_to_df(binance_symbol, interval, start_date, end_date)
    
    if df is not None and len(df) > 0:
        df.to_csv(cache_file)
        print(f"  ✓ {binance_symbol} {interval_str} (downloaded, {len(df)} candles)")
        # Sleep slightly to respect Binance API limits
        time.sleep(0.1)
        return df
        
    print(f"  ✗ {binance_symbol}: no data found")
    return None

def fetch_binance_universe_data(universe_tickers, start_date, end_date, data_dir, interval="1h"):
    """
    Fetch data for all coins in the universe.
    
    Returns:
        panels: Dict of aligned DataFrames {field: df(dates × tickers)}
        valid_tickers: List of symbols successfully processed
    """
    if interval == "1h":
        binance_interval = Client.KLINE_INTERVAL_1HOUR
        min_history = 24 * 365  # Want at least 1 year of hourly data roughly
    else:
        binance_interval = Client.KLINE_INTERVAL_1DAY
        min_history = 365
        
    print(f"📦 Loading {len(universe_tickers)} assets from Binance ({interval})...")
    
    all_data = {}
    valid_tickers = []
    skipped = 0
    
    for ticker in universe_tickers:
        binance_sym = get_binance_symbol(ticker)
        df = download_and_cache_binance(ticker, binance_interval, start_date, end_date, data_dir)
        
        if df is None or len(df) < min_history:
            if df is not None:
                print(f"  ⚠️ {binance_sym}: only {len(df)} candles (need {min_history}), skipping")
            skipped += 1
            continue
            
        all_data[binance_sym] = df
        valid_tickers.append(binance_sym)
        
    print(f"✅ Loaded {len(valid_tickers)}/{len(universe_tickers)} assets "
          f"({skipped} skipped for insufficient history)")
          
    if len(valid_tickers) == 0:
        return {}, []

    # Build aligned panel DataFrames
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    
    fields = [
        "Open", "High", "Low", "Close", "Volume",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume"
    ]
    panels = {}
    
    for field in fields:
        panel = pd.DataFrame(index=all_dates, columns=valid_tickers)
        for ticker in valid_tickers:
            df = all_data[ticker]
            if field in df.columns:
                panel[ticker] = df[field].reindex(all_dates)
        panels[field] = panel.astype(float)
        
    # Drop dates where all assets are NaN
    valid_mask = panels["Close"].notna().any(axis=1)
    for field in fields:
        panels[field] = panels[field].loc[valid_mask]
        
    return panels, valid_tickers

if __name__ == "__main__":
    # Quick test
    from crypto_config import UNIVERSE_TICKERS, DATA_DIR
    
    # Binance limits might be hit if we do all 65, test with top 3
    test_tickers = UNIVERSE_TICKERS[:3]
    start = "2022-01-01"
    end = "2025-01-01"
    
    print("\n--- Testing 1H Data Fetch ---")
    h1_panels, coins = fetch_binance_universe_data(test_tickers, start, end, DATA_DIR, "1h")
    
    print("\n--- Testing 1D Data Fetch ---")
    d1_panels, coins = fetch_binance_universe_data(test_tickers, start, end, DATA_DIR, "1d")
