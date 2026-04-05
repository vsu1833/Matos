"""
Utilities for fetching and exporting historical OHLCV data.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import yfinance as yf

# Keep yfinance timezone cache inside the repository.
YF_CACHE_DIR = Path(__file__).resolve().parent / "cache"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BACKTEST_END,
    BUY_THRESHOLD,
    FETCH_START,
    FORWARD_RETURN_DAYS,
    LIVE_LOOKBACK_DAYS,
    LTIM_START,
    LTIM_TICKER,
    MAX_FFILL_DAYS,
    MIN_VOLUME,
    SELL_THRESHOLD,
    TICKERS,
)


EXPORT_DIR = Path(__file__).resolve().parent / "csv"
FEATURE_COLUMNS = [
    "close",
    "returns",
    "ema20",
    "ema50",
    "rsi",
    "macd",
    "volatility",
    "volume",
]


def _assign_label(forward_return: float) -> str:
    if pd.isna(forward_return):
        return ""
    if forward_return >= BUY_THRESHOLD:
        return "Buy"
    if forward_return <= SELL_THRESHOLD:
        return "Sell"
    return "Hold"


def _csv_path_for_ticker(
    ticker: str,
    start: str,
    end: str,
    export_dir: Path,
) -> Path:
    safe_ticker = ticker.replace(".", "_")
    return export_dir / f"{safe_ticker}_{start}_{end}.csv"


def _series_or_default(raw: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    values = raw.get(column)
    if values is None:
        return pd.Series(default, index=raw.index, dtype="float64")
    return values.fillna(default).astype(float)


def _add_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["close"].pct_change()

    df.ta.ema(close=df["close"], length=20, append=True)
    df.ta.ema(close=df["close"], length=50, append=True)
    df.ta.rsi(close=df["close"], length=14, append=True)
    df.ta.macd(close=df["close"], fast=12, slow=26, signal=9, append=True)
    df.ta.stdev(close=df["returns"], length=20, append=True)

    df["ema20"] = df["EMA_20"]
    df["ema50"] = df["EMA_50"]
    df["rsi"] = df["RSI_14"]
    df["macd"] = df["MACD_12_26_9"]
    df["volatility"] = df["STDEV_20"]

    df = df[FEATURE_COLUMNS].copy()
    return df


def _normalize_history(
    raw: pd.DataFrame,
    ticker: str,
    add_labels: bool,
    min_rows: int | None,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: no rows returned by yfinance")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"{ticker}: missing required columns: {missing_cols}")

    df = pd.DataFrame(index=raw.index)
    df["open"] = raw["Open"].astype(float)
    df["high"] = raw["High"].astype(float)
    df["low"] = raw["Low"].astype(float)
    df["close"] = raw["Close"].astype(float)
    df["volume"] = raw["Volume"].fillna(0).astype("int64")

    before = len(df)
    df = df[df["volume"] >= MIN_VOLUME].copy()
    dropped = before - len(df)
    if dropped:
        print(f"    Dropped {dropped} low-volume rows")

    df = df.dropna(subset=["open", "high", "low", "close"], how="any")
    df = df.ffill(limit=MAX_FFILL_DAYS)

    bad_rows = df[
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
    ]
    if not bad_rows.empty:
        print(f"    Warning: {len(bad_rows)} inconsistent OHLC rows kept")

    if min_rows is not None and len(df) < min_rows:
        raise ValueError(f"{ticker}: only {len(df)} rows; need {min_rows}+")

    df = _add_lstm_features(df)

    if add_labels:
        df["forward_return"] = df["close"].pct_change(FORWARD_RETURN_DAYS).shift(
            -FORWARD_RETURN_DAYS
        )
        df["label"] = df["forward_return"].apply(_assign_label)
        df = df[FEATURE_COLUMNS + ["forward_return", "label"]]
    else:
        df = df[FEATURE_COLUMNS]

    df.index.name = "date"
    return df


def fetch_ticker(
    ticker: str,
    start: str = FETCH_START,
    end: str = BACKTEST_END,
    add_labels: bool = False,
    export_dir: str | Path = EXPORT_DIR,
    use_cache: bool = True,
    min_rows: int | None = None,
) -> pd.DataFrame:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    csv_path = _csv_path_for_ticker(ticker, start, end, export_path)

    if use_cache and csv_path.exists():
        print(f"  Loading {ticker} from {csv_path}")
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df.index.name = "date"
        return df

    print(f"  Fetching {ticker} {start} -> {end}")

    attempts = 3
    raw = None
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True,
                actions=True,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as exc:
            print(f"    Attempt {attempt} failed: {exc}")
        time.sleep(2)

    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: yfinance history() failed after {attempts} attempts")

    df = _normalize_history(
        raw=raw,
        ticker=ticker,
        add_labels=add_labels,
        min_rows=min_rows,
    )
    df.to_csv(csv_path)

    print(
        "    Saved"
        f" {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()})"
        f" to {csv_path}"
    )
    return df


def fetch_live(ticker: str) -> pd.DataFrame:
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=LIVE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    return fetch_ticker(ticker, start=start, end=end, add_labels=False)


def get_default_ticker_windows() -> list[tuple[str, str, str]]:
    windows = [(ticker, FETCH_START, BACKTEST_END) for ticker in TICKERS]
    windows.append((LTIM_TICKER, LTIM_START, BACKTEST_END))
    return windows


def fetch_all(
    tickers: list[str] | None = None,
    start: str = FETCH_START,
    end: str = BACKTEST_END,
    add_labels: bool = False,
    export_dir: str | Path = EXPORT_DIR,
    use_cache: bool = True,
    min_rows: int | None = 210,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    selected_tickers = tickers or TICKERS

    for ticker in selected_tickers:
        try:
            data[ticker] = fetch_ticker(
                ticker=ticker,
                start=start,
                end=end,
                add_labels=add_labels,
                export_dir=export_dir,
                use_cache=use_cache,
                min_rows=min_rows,
            )
        except ValueError as exc:
            print(f"  Skipping {ticker}: {exc}")

    return data


def export_configured_universe(
    export_dir: str | Path = EXPORT_DIR,
    add_labels: bool = False,
    use_cache: bool = True,
    min_rows: int | None = 210,
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}

    for ticker, start, end in get_default_ticker_windows():
        try:
            data[ticker] = fetch_ticker(
                ticker=ticker,
                start=start,
                end=end,
                add_labels=add_labels,
                export_dir=export_dir,
                use_cache=use_cache,
                min_rows=min_rows,
            )
        except ValueError as exc:
            print(f"  Skipping {ticker}: {exc}")

    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch adjusted historical stock data and export it to CSV."
    )
    parser.add_argument(
        "--ticker",
        help="Fetch a single ticker. If omitted, the configured universe is exported.",
    )
    parser.add_argument("--start", default=FETCH_START, help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end", default=BACKTEST_END, help="End date in YYYY-MM-DD.")
    parser.add_argument(
        "--export-dir",
        default=str(EXPORT_DIR),
        help="Directory inside the repository where CSVs will be written.",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Also include forward_return and label columns in the exported CSV.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore existing CSVs and fetch fresh data.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.ticker:
        fetch_ticker(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            add_labels=args.with_labels,
            export_dir=args.export_dir,
            use_cache=not args.refresh,
        )
    else:
        export_configured_universe(
            export_dir=args.export_dir,
            add_labels=args.with_labels,
            use_cache=not args.refresh,
        )
