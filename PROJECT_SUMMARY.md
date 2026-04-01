# MATOS — Multi-Agent Technical Orchestration System

**Research Project | Department of Computer Engineering, Goa College of Engineering**
**Authors:** Nitesh Naik, Adarsh Naik, Tanish Hede, Roshan Yadav, Viraj Upadhyay

---

## Overview

MATOS is a regime-aware multi-agent LLM orchestration system for generating Buy/Sell/Hold signals on NSE-listed equities using technical analysis. It is the implementation artefact for a research paper targeting a Springer-associated journal.

The core research contribution is **selective agent routing based on detected market regime**. Unlike prior multi-agent trading systems (TradingAgents, FINCON) which activate all agents on every query, MATOS first detects the current market regime (Trending, Mean-Reverting, or High-Volatility) using a deterministic classifier, then activates only the specialist agents that are theoretically appropriate for that regime. An LLM orchestrator performs weighted, conflict-aware fusion of the active agents' outputs to produce a final signal with a confidence score and a written reasoning trace.

---

## Research Context

### Problem Statement

Existing multi-agent LLM trading frameworks use **static agent architectures** — every agent runs on every query regardless of market conditions. A trend-following agent activated during a sideways market adds noise, not signal. No prior work has studied whether regime-aware dynamic routing of agent activation improves signal quality and reduces inference cost.

### What MATOS Does Differently

1. A **non-LLM regime classifier** (ADX + ATR rules) runs first and labels the market state
2. A **routing table** maps regimes to agent activation states (Primary / Secondary / Inactive)
3. Only active agents are called — inactive agents cost zero tokens
4. The **LLM orchestrator** receives only active agent outputs and performs chain-of-thought conflict resolution
5. Every signal includes a **reasoning trace** for interpretability analysis

### Baselines (implemented in Phase 6)

| Baseline | Type |
|---|---|
| MA Crossover | Traditional TA rule |
| RSI + MACD Composite | Traditional TA rule |
| Multi-Indicator Majority Vote | Traditional TA ensemble |
| Zero-Shot LLM | Vanilla LLM |
| Few-Shot LLM | Vanilla LLM |
| Chain-of-Thought LLM | Vanilla LLM |

---

## Target Dataset

**Market:** NSE (National Stock Exchange of India)
**Sector:** Nifty 50 IT sector
**Tickers:**

| Ticker | Company | Notes |
|---|---|---|
| `INFY.NS` | Infosys | Primary benchmark stock |
| `TCS.NS` | Tata Consultancy Services | High stability, good trending regime examples |
| `WIPRO.NS` | Wipro | Mid-volatility |
| `HCLTECH.NS` | HCL Technologies | Strong trend behaviour |
| `TECHM.NS` | Tech Mahindra | Higher volatility |
| `MPHASIS.NS` | Mphasis | Mean-reverting tendencies |
| `PERSISTENT.NS` | Persistent Systems | Strong post-2021 trend |

**Note on LTIM.NS:** LTIMindtree (merged Nov 2022). Excluded from backtesting. Available for live signal experiments from Jan 2023 only.

**Date ranges:**

| Purpose | Start | End |
|---|---|---|
| Data fetch (includes warmup) | 2018-01-01 | 2024-12-31 |
| Backtest warmup period | 2018-01-01 | 2018-12-31 |
| Backtest evaluation period | 2019-01-01 | 2024-12-31 |
| Live signal fetch window | today − 365 days | today |

**Why 2018 as fetch start:** The SMA-200 requires 200 trading days of data to produce its first valid reading. An additional ~50-day warmup buffer is needed for Wilder-smoothed indicators (RSI, ADX) to stabilise. Fetching from 2018 gives a ~250 trading day warmup before the first backtest signal in January 2019. **No signals are generated during the warmup period.**

---

## Architecture

```
matos/
├── config.py                  # Single source of truth for all parameters
├── main.py                    # Entry point: one ticker + date → signal
├── requirements.txt
├── .env                       # ANTHROPIC_API_KEY (never commit this)
│
├── data/
│   ├── fetcher.py             # yfinance download, cleaning, label generation
│   └── indicators.py         # pandas-ta computations → agent-ready dicts
│
├── regime/
│   └── classifier.py          # ADX/ATR rules → TRENDING / MEAN_REVERTING / HIGH_VOLATILITY
│
├── agents/
│   ├── base_agent.py          # Shared Claude API call + JSON parsing
│   ├── trend.py               # Trend Agent
│   ├── momentum.py            # Momentum Agent
│   ├── volatility.py          # Volatility & Volume Agent
│   └── pattern.py             # Pattern Agent
│
├── orchestrator/
│   ├── router.py              # Routing table lookup: regime → {agent: role}
│   └── orchestrator.py        # Calls active agents, fuses via LLM, returns signal
│
├── baselines/
│   ├── ta_rules.py            # Rule-based TA baselines
│   └── vanilla_llm.py         # Zero-shot, few-shot, CoT LLM baselines
│
└── evaluation/
    ├── backtest.py            # Historical signal loop over date range
    └── metrics.py             # F1, accuracy, Sharpe, drawdown, flip rate
```

---

## System Flow (per signal)

```
1. Input: ticker (e.g. "INFY.NS") + analysis date

2. data/fetcher.py
   └── Fetch 365 calendar days of OHLCV via yfinance
   └── close = Adj Close (adjusted for splits/dividends)
   └── open, high, low = raw unadjusted (preserves true intraday ranges)
   └── Drop zero-volume rows (NSE holidays)

3. data/indicators.py
   └── compute_all_indicators(df)
   └── Returns 4 dicts: trend_data, momentum_data, volatility_data, pattern_data

4. regime/classifier.py
   └── classify_regime(df)
   └── Returns: "TRENDING" | "MEAN_REVERTING" | "HIGH_VOLATILITY"
   └── Uses: ATR ratio (spike detection) → checked FIRST
             ADX > 25 → TRENDING
             ADX < 20 + inside Bollinger Bands → MEAN_REVERTING

5. orchestrator/router.py
   └── get_active_agents(regime)
   └── Returns dict: {agent_name: "primary" | "secondary" | None}

6. agents/*.py  (only active agents are called)
   └── Each agent receives its pre-computed indicator dict as JSON
   └── Each agent calls Claude API with a specialist system prompt
   └── Each agent returns: {signal, confidence, reasoning}

7. orchestrator/orchestrator.py
   └── Receives active agent outputs + regime + agent roles
   └── Calls Claude API with orchestrator system prompt
   └── Returns final output JSON (see Output Schema below)
```

---

## Routing Table

| Regime | Trend Agent | Momentum Agent | Volatility Agent | Pattern Agent |
|---|---|---|---|---|
| `TRENDING` | **PRIMARY** | secondary | — inactive — | secondary |
| `MEAN_REVERTING` | — inactive — | **PRIMARY** | secondary | **PRIMARY** |
| `HIGH_VOLATILITY` | — inactive — | — inactive — | **PRIMARY** | **PRIMARY** |

**Regime detection priority order:**
1. `HIGH_VOLATILITY` is checked first — ATR ratio > 1.5 overrides all other signals
2. `TRENDING` — ADX > 25
3. `MEAN_VOLATILITY` — default when neither of the above applies

**Agent role definitions:**
- `primary` — orchestrator treats this agent's signal as the dominant input
- `secondary` — orchestrator uses as supporting context, lower weight
- `None` — agent is not called at all (zero cost)

---

## Indicator Reference

### Trend Agent
| Indicator | pandas-ta call | Purpose |
|---|---|---|
| SMA-20, SMA-50, SMA-200 | `ta.sma(close, length=N)` | Trend direction, golden/death cross |
| EMA-12, EMA-26 | `ta.ema(close, length=N)` | Faster trend tracking |
| MACD (12,26,9) | `ta.macd(close, fast=12, slow=26, signal=9)` | Momentum within trend |
| ADX-14 | `ta.adx(high, low, close, length=14)` | Trend strength (also used in regime classifier) |

### Momentum Agent
| Indicator | pandas-ta call | Purpose |
|---|---|---|
| RSI-14 | `ta.rsi(close, length=14)` | Overbought/oversold |
| Stochastic (14,3) | `ta.stoch(high, low, close, k=14, d=3)` | Short-term momentum shifts |
| ROC-10 | `ta.roc(close, length=10)` | Rate of change, acceleration |
| Williams %R-14 | `ta.willr(high, low, close, length=14)` | Overbought/oversold confirmation |

### Volatility & Volume Agent
| Indicator | pandas-ta call | Purpose |
|---|---|---|
| Bollinger Bands (20,2) | `ta.bbands(close, length=20, std=2)` | Volatility range, squeeze detection |
| ATR-14 | `ta.atr(high, low, close, length=14)` | Absolute volatility |
| OBV | `ta.obv(close, volume)` | Volume trend confirmation |
| CMF-20 | `ta.cmf(high, low, close, volume, length=20)` | Money flow direction |
| BB Bandwidth | derived from bbands | Squeeze detection (BW near 40-day low) |
| ATR Ratio | ATR / 20d ATR MA | Volatility spike detection (also used in regime classifier) |

### Pattern Agent
Receives last 20 OHLCV candles with derived anatomy:
- `body_size` = abs(close − open)
- `upper_wick` = high − max(open, close)
- `lower_wick` = min(open, close) − low
- `doji` = bool (body/range < 0.1)
- `support_level` = lowest low of window
- `resistance_level` = highest high of window
- `price_structure` = "higher_highs_higher_lows" | "lower_highs_lower_lows" | "mixed"

---

## Data Pipeline Details

### Critical: Adjusted vs Raw Prices

```python
df["close"]  = raw["Adj Close"]   # ADJUSTED — used for ALL indicator computation
df["open"]   = raw["Open"]        # RAW — preserves intraday range
df["high"]   = raw["High"]        # RAW — preserves intraday range
df["low"]    = raw["Low"]         # RAW — preserves intraday range
df["volume"] = raw["Volume"]      # RAW
```

**Why this matters:** Over a 6-year backtest, NSE stocks undergo stock splits, bonus issues, and dividend adjustments. Without `Adj Close`, the SMA-200 computed in 2019 uses pre-split prices that are incompatible with post-split prices in 2024, producing invalid crossover signals. The high/low/open are kept raw because ATR, Bollinger Bands, and candlestick body/wick calculations require the true intraday price range. Adjusting those would artificially compress or expand ranges.

### Data Cleaning Steps (in order)
1. Flatten MultiIndex columns from yfinance
2. Map `Adj Close` → `close`
3. Drop rows where `volume <= 0` (NSE holidays that slip through as trading days)
4. Drop rows with NaN in open, high, low, close
5. Forward-fill gaps up to 3 consecutive days (handles rare corporate action anomalies)
6. Drop rows where `high < close` or `low > close` (OHLC consistency check)
7. Raise `ValueError` if fewer than 210 clean rows remain

### Ground Truth Labels
```python
forward_return = close.pct_change(5).shift(-5)   # 5-day lookahead
label = "Buy"  if forward_return >= +0.015
label = "Sell" if forward_return <= -0.015
label = "Hold" otherwise
```

**Important:** Labels are computed using `shift(-5)`, which means the last 5 rows of any DataFrame have `label = None`. These rows must be excluded from backtest evaluation. This is handled automatically in `evaluation/backtest.py`.

### Lookahead Bias Prevention
The regime classifier is computed on a rolling basis using only past data. On any given backtest date `t`, the classifier sees only rows up to and including `t`. The forward return label uses rows `t+1` through `t+5`, which are never visible to any agent or the orchestrator during signal generation.

---

## Output Schema

Every signal produced by the orchestrator conforms to this JSON schema:

```json
{
  "ticker": "INFY.NS",
  "date": "2024-03-15",
  "regime": "TRENDING",
  "active_agents": {
    "trend": "primary",
    "momentum": "secondary",
    "volatility": null,
    "pattern": "secondary"
  },
  "agent_signals": {
    "trend":     {"signal": "Buy",  "confidence": 0.82, "reasoning": "..."},
    "momentum":  {"signal": "Hold", "confidence": 0.55, "reasoning": "..."},
    "pattern":   {"signal": "Buy",  "confidence": 0.71, "reasoning": "..."}
  },
  "orchestrator": {
    "agreements": "trend and pattern both signal Buy",
    "conflicts": "momentum signals Hold citing RSI at 68",
    "conflict_resolution": "In a TRENDING regime, RSI approaching overbought is expected behaviour and does not override a confirmed trend. Momentum signal downweighted.",
    "final_signal": "Buy",
    "confidence": 0.76,
    "justification": "Full natural language justification paragraph..."
  }
}
```

---

## LLM Configuration



**Temperature = 0 is non-negotiable for the paper.** All results must be reproducible. Any reviewer can re-run the backtest and get identical signals.

### Agent Prompt Design Pattern

All agent prompts follow this structure:
1. **Role definition** — "You are a specialist Trend Analysis Agent..."
2. **Regime context** — "The current market regime is: TRENDING. Your role is: PRIMARY."
3. **Indicator data** — JSON block of pre-computed values and interpretations
4. **Output schema** — Exact JSON format required, no other text
5. **Chain-of-thought instruction** — "Reason through each indicator before committing to a signal"

The indicator dicts passed to agents contain both raw numbers and human-readable interpretations (e.g. `"rsi_zone": "approaching_overbought"` alongside `"rsi": 68.4`). This prevents the LLM from needing to perform numerical comparisons, which reduces errors.

---

## Evaluation Metrics

Implemented in `evaluation/metrics.py`:

| Metric | Category | Definition |
|---|---|---|
| Accuracy | Signal quality | Fraction of correct signal labels |
| Macro F1 | Signal quality | F1 averaged across Buy/Sell/Hold classes |
| Per-class F1 | Signal quality | Buy F1, Sell F1, Hold F1 separately |
| Annualised return | Portfolio | Return of portfolio following all signals |
| Sharpe ratio | Portfolio | Annualised excess return / std dev |
| Maximum drawdown | Portfolio | Largest peak-to-trough portfolio loss |
| Signal flip rate | Stability | Frequency of signal changes day-to-day |
| Avg latency (ms) | System | Mean wall-clock time per signal |
| Cost per signal (USD) | System | Estimated Claude API cost per signal |

### Ablation Experiments
The evaluation harness supports these ablation configurations (toggle via config flags):
- MATOS without Trend Agent
- MATOS without Momentum Agent
- MATOS without Volatility Agent
- MATOS without Pattern Agent
- MATOS with majority-vote fusion (replaces LLM orchestrator)
- MATOS with static routing (all agents always active, no regime detection)

---

## Setup Instructions

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd matos
pip install -r requirements.txt
```

### 2. Create .env file

```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

Never commit `.env` to version control. Add it to `.gitignore`.

### 3. Test Phase 1 (data pipeline)

```bash
# Test fetcher
python data/fetcher.py
# Expected output: clean row counts, label distribution for INFY.NS 2022-2024

# Test indicators
python data/indicators.py
# Expected output: all 4 agent indicator dicts for INFY.NS live data
# Cross-check RSI, MACD, Bollinger values against TradingView for the same date
```

### 4. Run a live signal (once all phases are implemented)

```bash
python main.py --ticker INFY.NS
```

### 5. Run backtest

```bash
python evaluation/backtest.py --tickers all --start 2019-01-01 --end 2024-12-31
```

---

## Implementation Status

| Phase | Module | Status |
|---|---|---|
| Phase 1 | `data/fetcher.py`, `data/indicators.py` | Complete |
| Phase 2 | `regime/classifier.py` | Not started |
| Phase 3 | `agents/base_agent.py`, `agents/trend.py`, `agents/momentum.py`, `agents/volatility.py`, `agents/pattern.py` | Not started |
| Phase 4 | `orchestrator/router.py`, `orchestrator/orchestrator.py` | Not started |
| Phase 5 | `evaluation/backtest.py`, `evaluation/metrics.py` | Not started |
| Phase 6 | `baselines/ta_rules.py`, `baselines/vanilla_llm.py` | Not started |

---

## Implementation Notes for the Coding Agent

### Coding conventions
- All modules import from `config.py` for every threshold and parameter. Never hardcode a number inside a module.
- The `data/` layer is entirely LLM-free. No Claude API calls in `fetcher.py` or `indicators.py`.
- Agent functions return Python dicts, not strings. JSON serialisation happens at the API boundary.
- All functions that call the Claude API live inside `agents/` or `orchestrator/`. Nowhere else.
- Use `CLAUDE_TEMPERATURE = 0` everywhere. This is required for reproducibility.

### Phase 2 — Regime Classifier (`regime/classifier.py`)

The function signature must be:
```python
def classify_regime(df: pd.DataFrame) -> str:
    # Returns: "TRENDING" | "MEAN_REVERTING" | "HIGH_VOLATILITY"
```

Logic (check in this exact order):
1. Compute `atr_ratio = ATR_14 / ATR_14.rolling(20).mean()` for the last row
2. If `atr_ratio > ATR_SPIKE_RATIO (1.5)` → return `"HIGH_VOLATILITY"`
3. If current close is outside Bollinger Bands → return `"HIGH_VOLATILITY"`
4. If `ADX_14 > ADX_TREND_THRESHOLD (25)` → return `"TRENDING"`
5. Else → return `"MEAN_REVERTING"`

All thresholds are imported from `config.py`. The function receives the full cleaned DataFrame (not just the last row) because rolling calculations need history.

### Phase 3 — Agents (`agents/`)

`base_agent.py` must implement:
```python
def call_claude(system_prompt: str, user_message: str) -> dict:
    # Calls Claude API, parses JSON response, returns dict
    # Raises ValueError if response is not valid JSON
```

Each specialist agent (e.g. `trend.py`) must implement:
```python
def run(indicator_data: dict, regime: str, role: str) -> dict:
    # Returns: {"signal": "Buy"|"Sell"|"Hold", "confidence": float, "reasoning": str}
```

The `role` parameter ("primary" or "secondary") is injected into the system prompt so the agent knows how its output will be weighted.

### Phase 4 — Orchestrator (`orchestrator/`)

`router.py` must implement:
```python
def get_active_agents(regime: str) -> dict:
    # Returns ROUTING_TABLE[regime] from config.py
    # e.g. {"trend": "primary", "momentum": "secondary", "volatility": None, "pattern": "secondary"}
```

`orchestrator.py` must implement:
```python
def generate_signal(ticker: str, df: pd.DataFrame) -> dict:
    # Runs the full pipeline: indicators → regime → routing → agents → orchestrator
    # Returns the full output JSON conforming to the Output Schema above
```

### Phase 5 — Backtest (`evaluation/backtest.py`)

The backtest loop must:
1. Iterate over trading dates from `BACKTEST_START` to `BACKTEST_END`
2. For each date, slice `df.loc[:date]` to prevent lookahead bias — the agent sees only data up to and including the current date
3. Skip the first 250 rows (warmup period where SMA-200 is not yet valid)
4. Skip rows where `label` is `None` (last 5 rows have no forward return)
5. Log every signal to a results CSV: `date, ticker, regime, final_signal, confidence, true_label`

### Phase 6 — Baselines (`baselines/ta_rules.py`)

Implement these three rule-based systems:

**MA Crossover:**
```python
# Buy: SMA20 crosses above SMA50 (was below yesterday, above today)
# Sell: SMA20 crosses below SMA50
# Hold: otherwise
```

**RSI + MACD Composite:**
```python
# Buy:  RSI < 30 AND MACD histogram > 0
# Sell: RSI > 70 AND MACD histogram < 0
# Hold: otherwise
```

**Multi-Indicator Majority Vote:**
```python
# Run 8 individual indicator rules, each returns Buy/Sell/Hold
# Final signal = majority of the 8
# Tie → Hold
```

---

## Key Design Decisions (Justify in Paper)

### Why ADX-based regime detection rather than learned (HMM)?
ADX thresholds (>25 trending, <20 weak) are established in TA literature (Wilder 1978) and independently citable. Thresholds are not tuned on our data, which avoids a methodological criticism from reviewers. HMM is implemented as a secondary ablation experiment only.

### Why is HIGH_VOLATILITY checked before TRENDING?
A spiking ATR indicates that even a high-ADX trend signal is unreliable because extreme intraday swings corrupt all indicator calculations. Checking ATR ratio first prevents falsely labelling a crash day as "TRENDING."

### Why are open/high/low kept unadjusted?
ATR, Bollinger Bands, and candlestick anatomy all measure intraday price range. Dividend adjustments multiply close prices by a ratio less than 1.0, which would artificially compress historical ranges and produce incorrect ATR and Bollinger values in multi-year backtests.

### Why temperature=0?
Academic reproducibility. Any reviewer can re-run the code and get the same signals on the same dates.

### Why pass both raw values and interpretation strings to agents?
LLMs are unreliable at precise numerical comparisons (is 68.4 above 70?). Pre-computing interpretations (`rsi_zone: "approaching_overbought"`) in Python makes agent outputs deterministic and reduces prompt token count.

---

## Dependencies

```
yfinance==0.2.40          # NSE data via Yahoo Finance
pandas-ta==0.3.14b0       # Technical indicator computations
pandas==2.2.0             # DataFrame operations
numpy==1.26.4             # Numerical operations
anthropic==0.25.0         # Claude API client
hmmlearn==0.3.2           # HMM regime classifier (ablation)
scikit-learn==1.4.2       # F1 score, metrics computation
python-dotenv==1.0.1      # Load ANTHROPIC_API_KEY from .env
```

---

## Paper Citation

When this work is submitted, cite as:

> Naik, N., Naik, A., Hede, T., Yadav, R., & Upadhyay, V. (2025). Regime-Aware Agent Routing in Multi-Agent LLM Systems for Equity Technical Analysis Signal Generation. *[Journal Name]*.