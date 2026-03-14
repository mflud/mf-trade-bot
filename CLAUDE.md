# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Copy `.env.example` to `.env` and fill in your Topstep credentials:
```
TOPSTEP_USERNAME=your_email@example.com
TOPSTEP_API_KEY=your_api_key_here
```

Install dependencies:
```bash
pip install -r requirements.txt
pip install numpy pandas openpyxl  # required for backtests (not yet in requirements.txt)
```

## Running Scripts

All scripts are run from the **repo root**:

```bash
# Test API connectivity and fetch recent MES bars
python src/topstep_client.py

# Real-time analytics (range, vol, regime_pl) for current MES contract
python src/analytics.py

# Equity backtest on Polygon data (SPY or QQQ Excel files)
python src/backtest_regime_vol.py polygon_data_SPY.xlsx
python src/backtest_regime_vol.py polygon_data_QQQ.xlsx

# MES 1-min futures backtest (fetches live data from TopstepX)
python src/backtest_mes.py

# Continuation probability analysis (uses cached CSV or fetches from API)
python src/continuation_prob.py
```

## Architecture

**`src/topstep_client.py`** — Core API client for the Topstep (ProjectX) REST API (`https://api.topstepx.com`). Handles auth (API key → Bearer token), contract search, and bar history retrieval. Supports a context manager (`with TopstepClient() as client`). Includes `get_continuous_mes_bars()` which stitches quarterly MES contracts into a back-adjusted continuous series using the Panama/backward method. MES contracts are hardcoded in `MES_CONTRACTS`; extend this list when new quarterly contracts become available.

**`src/analytics.py`** — Real-time analytics using live bars from the client. Computes trailing high/low range, annualised volatility, and `regime_pl` for a given contract and lookback window.

**`src/backtest_regime_vol.py`** — Offline backtest using Polygon 5-min equity bar data (SPY/QQQ Excel files in repo root). Computes `regime_pl` vs. forward volatility across windows of 30min/1h/2h/3h. Filters to regular trading hours (9:30–16:00 ET) only.

**`src/backtest_mes.py`** — Same backtest methodology as above but for MES 1-min bars fetched live from TopstepX, with the CME settlement gap (21:00–22:00 UTC) filtered out.

**`src/continuation_prob.py`** — Given a net price move ≥ threshold over a lookback window, computes the probability price continues vs. reverts by `TARGET_PTS` points. Sweeps vol and `regime_pl` percentile thresholds to find conditioning values that shift continuation probability. Caches raw bars to `mes_1min_bars.csv` to avoid re-fetching.

## Key Concepts

**`regime_pl`** (Price Linearity): `|sum(log_returns)| / sum(|log_returns|)`. Range [0, 1]. Values near 1 indicate strongly trending; near 0 indicate choppy/mean-reverting.

**Bar unit constants** (on `TopstepClient`): `SECOND=1, MINUTE=2, HOUR=3, DAY=4, WEEK=5, MONTH=6`.

**CME settlement gap**: 16:00–17:00 CT = 21:00–22:00 UTC — always filtered out in MES analysis.

**Back-adjustment**: `get_continuous_mes_bars()` uses the Panama backward method — at each contract roll, a cumulative price offset is subtracted from all earlier bars so the series is gap-free. Prices reflect returns accurately but are not actual traded prices.
