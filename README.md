# Alpha Combination Strategy

An institutional-grade systematic US Equity trading engine. The bot utilizes a machine learning (LightGBM) model to dynamically weight structural and technical alpha signals across a defined market universe (e.g. S&P 100). The portfolio is risk-managed via a tiered VIX-based Regime Defense system that actively liquidates positions during extended bearish markets or rapid crisis events.

## Key Features

* **Machine Learning Pipeline:** Implements a purged walk-forward cross-validation approach to train LightGBM decision trees on multi-factor alphas and technical indicators without lookahead bias.
* **Alpha Selection:** Integrates multiple normalized technical signals (Alpha-101 formulas, volume/price reversions, structural momentum) evaluated on daily close.
* **Regime Defense System:** Tiered classification (`BULL`, `NEUTRAL`, `BEAR`, `CRISIS`) using VIX thresholds and 200/50-day moving averages to force liquidation or dial down strategy exposure.
* **Per-Stock Execution Engine:** Employs stop-loss and take-profit dynamic holding logic on individual tickers.
* **Visualized Analysis & Stress Testing:** Automated generation of R-multiple trade distributions, rolling Sharpe charts, drawdown logs, and Monte Carlo execution-shock stress tests.

## Architecture

- `config.py` - Core parameter declarations, constants, portfolio size limits, and security universe.
- `run_backtest.py` - Primary master execution script.
- `src/alphas.py` - Library of 10 standalone systematic alpha formulas.
- `src/data_loader.py` - End-of-day data caching pipeline using `yfinance`.
- `src/engine.py` - The order evaluation unit managing positions, TP/SL, and forced liquidation triggers.
- `src/ml_scorer.py` - Model architecture for the LightGBM probability scorer.
- `src/regime_filter.py` - Calculates structural market conditions based on SPY and VIX.
- `src/analysis.py` & `src/revisualize.py` - Analytical reporting suite and manual visual regeneration without simulation overhead.

## Usage

```bash
# 1. Install standard dependencies (pandas, numpy, lightgbm, yfinance, matplotlib, scipy)
pip install -r requirements.txt

# 2. Run the main backtest and simulation pipeline
python run_backtest.py

# 3. Manually refresh output visualizations (bypass full ML retraining)
python src/revisualize.py
```

## Results & Output

Logs, statistical screenings, detailed full-trade records (CSV), and evaluation visualizations (PNG) are saved into the `results/` folder upon completion. Market data caching is stored globally in `data/`.
