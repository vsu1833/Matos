"""
config.py
Central configuration for MATOS.
All tickers, date ranges, thresholds, and model settings live here.
"""

# ── Tickers ──────────────────────────────────────────────────────────────────
TICKERS = [
    "INFY.NS",        # Infosys
    "TCS.NS",         # Tata Consultancy Services
    "WIPRO.NS",       # Wipro
    "HCLTECH.NS",     # HCL Technologies
    "TECHM.NS",       # Tech Mahindra
    "MPHASIS.NS",     # Mphasis
    "PERSISTENT.NS",  # Persistent Systems
]

# LTIM.NS (LTIMindtree) excluded from backtest due to merger in Nov 2022.
# Use only for live signal experiments from Jan 2023 onward.
LTIM_TICKER = "LTIM.NS"
LTIM_START  = "2023-01-01"

# ── Date ranges ──────────────────────────────────────────────────────────────
# Full fetch window: extra 250 trading days before backtest start for warmup
FETCH_START      = "2018-01-01"   # ~250 trading day warmup before backtest
BACKTEST_START   = "2019-01-01"   # first signal date (after warmup)
BACKTEST_END     = "2024-12-31"

# For live signal: fetch 365 calendar days ending today
LIVE_LOOKBACK_DAYS = 365

# ── Indicator parameters ─────────────────────────────────────────────────────
SMA_PERIODS       = [20, 50, 200]
EMA_PERIODS       = [12, 26]
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
ADX_PERIOD        = 14
RSI_PERIOD        = 14
STOCH_K           = 14
STOCH_D           = 3
ROC_PERIOD        = 10
WILLR_PERIOD      = 14
BB_PERIOD         = 20
BB_STD            = 2
ATR_PERIOD        = 14
CMF_PERIOD        = 20
ATR_RATIO_WINDOW  = 20    # rolling window for ATR ratio denominator
PATTERN_WINDOW    = 20    # candles sent to Pattern agent

# ── Regime classifier thresholds ─────────────────────────────────────────────
ADX_TREND_THRESHOLD    = 25     # ADX > 25 → TRENDING
ADX_WEAK_THRESHOLD     = 20     # ADX < 20 → lean toward MEAN_REVERTING
ATR_SPIKE_RATIO        = 1.5    # ATR / 20d ATR_MA > 1.5 → HIGH_VOLATILITY

# ── Routing table ─────────────────────────────────────────────────────────────
# Values: "primary", "secondary", None (inactive)
ROUTING_TABLE = {
    "TRENDING": {
        "trend":      "primary",
        "momentum":   "secondary",
        "volatility": None,
        "pattern":    "secondary",
    },
    "MEAN_REVERTING": {
        "trend":      None,
        "momentum":   "primary",
        "volatility": "secondary",
        "pattern":    "primary",
    },
    "HIGH_VOLATILITY": {
        "trend":      None,
        "momentum":   None,
        "volatility": "primary",
        "pattern":    "primary",
    },
}

# ── Ground truth labelling ────────────────────────────────────────────────────
FORWARD_RETURN_DAYS = 5       # 5-day forward return window
BUY_THRESHOLD       = 0.015   # +1.5% → Buy
SELL_THRESHOLD      = -0.015  # -1.5% → Sell

# ── LLM settings ─────────────────────────────────────────────────────────────
CLAUDE_MODEL        = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS   = 1024
CLAUDE_TEMPERATURE  = 0       # deterministic for reproducibility

# ── Data quality ─────────────────────────────────────────────────────────────
MIN_VOLUME          = 1       # drop rows with volume <= this (non-trading days)
MAX_FFILL_DAYS      = 3       # max consecutive days to forward-fill price gaps
