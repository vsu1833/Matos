"""
data/indicators.py
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import sys
from pathlib import Path

# ✅ IMPORTANT FIX FOR pandas-ta-openbb
# pd.DataFrame.ta = ta.AnalysisIndicators

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    SMA_PERIODS, EMA_PERIODS,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ADX_PERIOD, RSI_PERIOD,
    STOCH_K, STOCH_D,
    ROC_PERIOD, WILLR_PERIOD,
    BB_PERIOD, BB_STD, ATR_PERIOD, CMF_PERIOD,
    ATR_RATIO_WINDOW, PATTERN_WINDOW,
)

# ── Helpers (UNCHANGED) ──────────────────────────────────────────────────────

def _safe(val, decimals: int = 2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)

def _trend(series: pd.Series, window: int = 3) -> str:
    if len(series.dropna()) < window:
        return "insufficient_data"
    recent = series.dropna().iloc[-window:]
    slope = recent.iloc[-1] - recent.iloc[0]
    if slope > 0.01 * abs(recent.mean()):
        return "rising"
    if slope < -0.01 * abs(recent.mean()):
        return "falling"
    return "flat"

def _above_below(value, reference) -> str:
    if value is None or reference is None:
        return "unknown"
    return "above" if value > reference else "below"

def _rsi_zone(rsi_val) -> str:
    if rsi_val is None:
        return "unknown"
    if rsi_val >= 70:
        return "overbought"
    if rsi_val >= 60:
        return "approaching_overbought"
    if rsi_val <= 30:
        return "oversold"
    if rsi_val <= 40:
        return "approaching_oversold"
    return "neutral"

def _stoch_zone(k_val) -> str:
    if k_val is None:
        return "unknown"
    if k_val >= 80:
        return "overbought"
    if k_val <= 20:
        return "oversold"
    return "neutral"

def _willr_zone(wr_val) -> str:
    if wr_val is None:
        return "unknown"
    if wr_val >= -20:
        return "overbought"
    if wr_val <= -80:
        return "oversold"
    return "neutral"

def _vol_zone(atr_ratio) -> str:
    if atr_ratio is None:
        return "unknown"
    if atr_ratio > 1.5:
        return "spike"
    if atr_ratio > 1.2:
        return "elevated"
    if atr_ratio < 0.8:
        return "low"
    return "normal"

def _stoch_crossover(k_series: pd.Series, d_series: pd.Series) -> str:
    try:
        k = k_series.dropna().iloc[-3:]
        d = d_series.dropna().iloc[-3:]
        if len(k) < 2 or len(d) < 2:
            return "none"
        if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1]:
            return "bullish"
        if k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1]:
            return "bearish"
        return "none"
    except Exception:
        return "none"

# ── Trend Indicators ─────────────────────────────────────────────────────────

def compute_trend_indicators(df: pd.DataFrame) -> dict:
    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]

    # ✅ FIXED
    df.ta.sma(close=close, length=20, append=True)
    df.ta.sma(close=close, length=50, append=True)
    df.ta.sma(close=close, length=200, append=True)

    df.ta.ema(close=close, length=12, append=True)
    df.ta.ema(close=close, length=26, append=True)

    df.ta.macd(close=close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True)
    df.ta.adx(high=high, low=low, close=close, length=ADX_PERIOD, append=True)

    sma20, sma50, sma200 = df["SMA_20"], df["SMA_50"], df["SMA_200"]
    ema12, ema26 = df["EMA_12"], df["EMA_26"]

    macd_line = df[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    macd_sig  = df[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    macd_hist = df[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]

    adx_col = f"ADX_{ADX_PERIOD}"
    dmp_col = f"DMP_{ADX_PERIOD}"
    dmn_col = f"DMN_{ADX_PERIOD}"

    return {
        "current_price": _safe(close.iloc[-1]),
        "sma_20": _safe(sma20.iloc[-1]),
        "sma_50": _safe(sma50.iloc[-1]),
        "sma_200": _safe(sma200.iloc[-1]),
        "ema_12": _safe(ema12.iloc[-1]),
        "ema_26": _safe(ema26.iloc[-1]),
        "macd_line": _safe(macd_line.iloc[-1], 3),
        "macd_signal_line": _safe(macd_sig.iloc[-1], 3),
        "macd_histogram": _safe(macd_hist.iloc[-1], 3),
        "adx": _safe(df[adx_col].iloc[-1], 1),
        "plus_di": _safe(df[dmp_col].iloc[-1], 1),
        "minus_di": _safe(df[dmn_col].iloc[-1], 1),
    }

# ── Momentum Indicators ──────────────────────────────────────────────────────

def compute_momentum_indicators(df: pd.DataFrame) -> dict:
    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]

    # ✅ FIXED
    df.ta.rsi(close=close, length=RSI_PERIOD, append=True)
    df.ta.stoch(high=high, low=low, close=close, k=STOCH_K, d=STOCH_D, append=True)
    df.ta.roc(close=close, length=ROC_PERIOD, append=True)
    df.ta.willr(high=high, low=low, close=close, length=WILLR_PERIOD, append=True)

    return {
        "rsi": _safe(df[f"RSI_{RSI_PERIOD}"].iloc[-1], 1),
        "stoch_k": _safe(df.filter(like="STOCHk").iloc[:, -1].iloc[-1], 1),
        "stoch_d": _safe(df.filter(like="STOCHd").iloc[:, -1].iloc[-1], 1),
        "roc": _safe(df[f"ROC_{ROC_PERIOD}"].iloc[-1], 2),
        "williams_r": _safe(df[f"WILLR_{WILLR_PERIOD}"].iloc[-1], 1),
    }

# ── Volatility Indicators ────────────────────────────────────────────────────

def compute_volatility_indicators(df: pd.DataFrame) -> dict:
    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # ✅ FIXED
    df.ta.bbands(close=close, length=BB_PERIOD, std=BB_STD, append=True)
    df.ta.atr(high=high, low=low, close=close, length=ATR_PERIOD, append=True)
    df.ta.obv(close=close, volume=volume, append=True)
    df.ta.cmf(high=high, low=low, close=close, volume=volume, length=CMF_PERIOD, append=True)

    return {
        "bb_upper": _safe(df.filter(like="BBU").iloc[:, -1].iloc[-1]),
        "bb_middle": _safe(df.filter(like="BBM").iloc[:, -1].iloc[-1]),
        "bb_lower": _safe(df.filter(like="BBL").iloc[:, -1].iloc[-1]),
        "atr": _safe(df[f"ATR_{ATR_PERIOD}"].iloc[-1]),
        "obv": _safe(df["OBV"].iloc[-1]),
        "cmf": _safe(df[f"CMF_{CMF_PERIOD}"].iloc[-1]),
    }

# ── Pattern (UNCHANGED) ──────────────────────────────────────────────────────

def compute_pattern_data(df: pd.DataFrame) -> dict:
    window = df.iloc[-PATTERN_WINDOW:].copy()
    return {"candles": len(window)}

# ── Master ───────────────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> dict:
    return {
        "trend": compute_trend_indicators(df),
        "momentum": compute_momentum_indicators(df),
        "volatility": compute_volatility_indicators(df),
        "pattern": compute_pattern_data(df),
    }