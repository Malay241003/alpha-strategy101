"""
plot_fx_equity.py — Plots the equity curve from the FX macro trades.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

TRADE_LOG_PATH = "C:/Users/KIIT/Desktop/Tradebot1/alpha_strategy/macro/results/fx_macro_trades.csv"
OUTPUT_PATH = "C:/Users/KIIT/.gemini/antigravity/brain/a32b73e9-3e9b-4bba-82ad-a2161cfa5f17/fx_equity_curve.png"

def plot_equity():
    if not os.path.exists(TRADE_LOG_PATH):
        print(f"File not found: {TRADE_LOG_PATH}")
        return
        
    df = pd.read_csv(TRADE_LOG_PATH)
    if df.empty:
        print("Empty trade log.")
        return
        
    df['exitDate'] = pd.to_datetime(df['exitDate'])
    df = df.sort_values('exitDate')
    
    # Calculate cumulative R
    df['cumulative_R'] = df['R'].cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['exitDate'].values, df['cumulative_R'].values, color='green', linewidth=2)
    plt.title('FX Systematic Macro: Purged Walk-Forward Equity Curve (Net of Spread)', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Cumulative Profit (R)')
    plt.grid(True, alpha=0.3)
    
    # Save directly to artifact folder so the user can see it
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_equity()
