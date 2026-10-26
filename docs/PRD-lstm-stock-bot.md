# PRD: LSTM Stock Predictor Bot (Rebuild)

## 1. Metadata

| Field | Value |
|---|---|
| Title | LSTM Stock Predictor Bot (working name: `stockbot`) |
| Version | 0.1 |
| Date | 2026-10-26 |
| Status | Draft, ready for agent implementation after Open Questions OQ-1 to OQ-5 are reviewed |
| Author | Devon Lopez |
| Intended reader | Claude Code (AI coding agent) |
| Replaces | Prior "Long Short Term Model Stock Predictor Bot" (Nov 2025 to Jun 2026). Nothing from the old codebase is reused. |

---

## 2. Overview

**Problem.** The first version of this bot (LSTM + VADER sentiment on Finnhub news, about 4 trades a day) was built in a single pass, and its reported results (60% prediction accuracy, 2.8% average return per trade) were not produced by a validation setup that rules out look-ahead leakage, survivorship effects, or transaction costs. There is no reproducible pipeline that proves or disproves those numbers.

**Goal.** Rebuild the system from scratch as a reproducible, testable Python pipeline that: ingests S&P 500 daily prices and company news, scores headline sentiment, trains an LSTM that predicts next-session direction, validates it with walk-forward backtesting against baselines, and paper trades the signals through the Alpaca paper API on a schedule.

**Vision (one sentence).** A command-line bot that, every trading day, turns yesterday's prices and headlines into at most 4 paper trades, and can show with honest out-of-sample numbers whether the model has any edge.

---

## 3. Goals and Non-Goals

### Goals

- G-1: Produce daily signals for every active S&P 500 ticker from an LSTM using price features and headline sentiment.
- G-2: Validate the model with walk-forward backtests that include transaction costs and baselines, with confidence intervals on every headline metric.
- G-3: Place paper trades on Alpaca on a fixed daily schedule with idempotent, auditable order handling.
- G-4: Keep the whole system runnable from a CLI, with every stage independently runnable and testable.
- G-5: Make results reportable as-is, whatever they show.

### Non-Goals (hard fence, do not build)

- NG-1: No live trading. The code must not be able to place orders against the live Alpaca endpoint (see FR-2).
- NG-2: No short selling, margin, options, futures, or crypto.
- NG-3: No intraday strategies. Entries are at the next session's open, exits at that session's close.
- NG-4: No web UI, dashboard, or REST API. Interface is CLI plus scheduled runs only.
- NG-5: No hyperparameter search framework. Hyperparameters are fixed in config (see Section 8.4).
- NG-6: No multi-model ensembles, no per-ticker models. One global LSTM.
- NG-7: No use of the old bot's 60% accuracy or 2.8% per-trade return as targets or tuning goals.
- NG-8: No Kalshi, Polymarket, or non-equity data sources.
- NG-9: No portfolio optimization. Sizing is fixed percent of equity.

---

## 4. Target Users and Use Cases

**Primary user: Devon Lopez** (CS student, targeting quant developer and SWE roles). Runs the bot on a MacBook (Apple Silicon) or an always-on home machine.

| Use case | Description |
|---|---|
| UC-1 | Run the nightly pipeline and read tomorrow's picks. |
| UC-2 | Let scheduled jobs place and close paper trades without intervention. |
| UC-3 | Run a backtest after any model or feature change and compare against baselines. |
| UC-4 | Pull honest metrics and a reproducible report for a resume bullet or interview discussion. |

---

## 5. User Stories

| ID | Story | Linked FRs |
|---|---|---|
| US-1 | As Devon, I want one command to load prices and news so the database is current. | FR-7 to FR-15 |
| US-2 | As Devon, I want features and labels built without future data so backtests are trustworthy. | FR-16 to FR-19, FR-31 |
| US-3 | As Devon, I want a walk-forward backtest with baselines so I can see if the LSTM beats simple models. | FR-20 to FR-23, FR-27 to FR-30 |
| US-4 | As Devon, I want nightly signals so I know which 4 stocks the bot wants for tomorrow. | FR-32 to FR-34 |
| US-5 | As Devon, I want paper orders placed and closed automatically so I do not run commands by hand. | FR-35 to FR-40, FR-42 |
| US-6 | As Devon, I want the bot to refuse to touch a real-money account so a config mistake cannot cost money. | FR-2 |
| US-7 | As Devon, I want daily reports of orders and P&L so I can audit what happened. | FR-39, FR-43 |

---

## 6. Functional Requirements

Priority key: **M** Must, **S** Should, **C** Could. Won't items are in Section 3 (Non-Goals).

### 6.1 Config, safety, operations

| ID | Pri | Requirement |
|---|---|---|
| FR-1 | M | Load non-secret settings from `config/default.toml` and secrets from environment variables `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FINNHUB_API_KEY` (read from `.env` via `python-dotenv`). If a required secret is missing, exit with code 2 and print the variable name only. |
| FR-2 | M | Any command that calls Alpaca must verify before the first network call that `ALPACA_PAPER` equals `true` and the base URL equals `https://paper-api.alpaca.markets`. Otherwise exit with code 3. |
| FR-3 | M | `stockbot init-db` creates all tables in Section 9 in `data/stockbot.db`. Running it again must change nothing and exit 0. |
| FR-4 | M | Commands `signal`, `trade`, `exit`, `reconcile`, and `nightly` must check the Alpaca market calendar. If today is not a trading session (for `signal` and `nightly`, if the most recent completed session is already processed), log `market closed` or `already processed`, make no order calls, and exit 0. |
| FR-5 | M | Every command inserts a row into `runs` on start and updates it on finish with `status` of `ok` or `error`. Exit codes: 0 success, 1 unexpected error, 2 bad config or arguments, 3 safety refusal, 4 stale or missing data. |
| FR-6 | M | API keys and secrets must never appear in logs, stdout, stderr, the `runs` table, or reports. |

### 6.2 Universe and prices

| ID | Pri | Requirement |
|---|---|---|
| FR-7 | M | `stockbot load-universe` reads `data/sp500.csv` (columns `symbol,name,sector`) and upserts into `tickers`. It also upserts `SPY` with `is_benchmark=1`. Benchmark symbols are never selected for trading. |
| FR-8 | M | `stockbot ingest-prices` fetches daily OHLCV bars from the Alpaca market data API with `adjustment=all` and `feed=iex` for every active ticker, from `start_date` (default: today minus 5 years) through the latest completed session, and upserts into `bars_daily`. |
| FR-9 | M | Ingestion is incremental: for each symbol, request only sessions after `MAX(session_date)`. If the overlap bar's adjusted close differs from the stored value by more than 0.5%, delete that symbol's bars and refetch its full history (handles splits and dividends). |
| FR-10 | S | If a symbol returns zero bars for the full range, set `active=0` and log a warning containing the symbol. |

### 6.3 News and sentiment

| ID | Pri | Requirement |
|---|---|---|
| FR-11 | M | `stockbot ingest-news` fetches Finnhub company news per active non-benchmark ticker for the date range since the last stored news date (default start: today minus 365 days), throttled to at most 50 requests per minute. |
| FR-12 | M | On HTTP 429 or 5xx responses from Finnhub or Alpaca, retry with delays of 1, 2, 4, 8, and 16 seconds, then fail that symbol, log it, and continue with the next symbol. |
| FR-13 | M | Score each headline with `vaderSentiment` `SentimentIntensityAnalyzer().polarity_scores(headline)["compound"]` and store it in `news.vader_compound`. |
| FR-14 | M | Assign each headline a `session_date`: headlines published at or before 16:00 America/New_York on a trading day belong to that day's session. Later headlines, and headlines on non-trading days, belong to the next trading session. Store `published_at_utc` in ISO 8601 UTC. |
| FR-15 | S | Never delete rows from `news` (Finnhub history is limited, so the local archive is the long-term record). Deduplicate on `(symbol, published_at_utc, headline)`. |

### 6.4 Features and labels

| ID | Pri | Requirement |
|---|---|---|
| FR-16 | M | `stockbot features` computes the 13 features in Section 8.2 for every `(symbol, session_date)` using only data with timestamp at or before the close of `session_date`, and upserts into `features`. |
| FR-17 | M | Sentiment features aggregate headlines whose `session_date` equals the row's `session_date`. For rows with zero headlines: `sent_mean=0`, `sent_max=0`, `sent_min=0`, `news_count=0`, `news_available=0`. |
| FR-18 | M | Label: `label = 1` if the next session's `close/open - 1 > 0`, else `0`. Store the underlying value in `next_oc_ret`. Both are `NULL` for the most recent session. |
| FR-19 | M | Drop rows from `features` that have a `NULL` in any feature column; the first 20 sessions per symbol are expected to drop. Log the dropped count per run. |

### 6.5 Modeling

| ID | Pri | Requirement |
|---|---|---|
| FR-20 | M | Implement a baseline logistic regression (`scikit-learn`, `LogisticRegression(max_iter=1000)`) trained on the last day's 13 features only. It uses the same folds, normalization, and selection rules as the LSTM. |
| FR-21 | M | Implement the LSTM in PyTorch with the architecture and training settings in Section 8.3. Input: tensor of shape `(batch, 20, 13)`. Output: one logit; `p_up = sigmoid(logit)`. |
| FR-22 | M | Compute per-feature mean and standard deviation on the training portion of each fold only, apply them to validation and test data, and save them with the model artifact in `feature_stats.json`. |
| FR-23 | M | Walk-forward folds: the out-of-sample period is the most recent 24 months of available data, split into 8 consecutive 3-month test blocks. For each block, train on all data before the block start, minus a 5-session embargo. Hold out the last 10% of the training dates (by time) for early stopping. |
| FR-24 | M | `stockbot train` trains one final model on all labeled data up to the latest labeled session (same early-stopping split as FR-23) and writes `artifacts/models/<model_id>/model.pt`, `feature_stats.json`, `config_snapshot.toml`, and a `models` table row including the git commit SHA. `model_id` format: `lstm-YYYYMMDD-HHMMSS`. |
| FR-25 | S | Set seeds for `random`, `numpy`, and `torch` from `config.seed` (default 42) and log the seed. |
| FR-26 | S | Select device in this order: `mps` if available, else `cuda`, else `cpu`. Allow override with `config.device`. |

### 6.6 Backtest and reporting

| ID | Pri | Requirement |
|---|---|---|
| FR-27 | M | `stockbot backtest` generates out-of-sample predictions with the FR-23 folds and simulates the live rules: for each session `t`, rank tickers by `p_up`, select up to `N=4` with `p_up >= 0.55` (ties broken by symbol ascending), enter at the open of `t+1`, exit at the close of `t+1`, size each position at 5% of start-of-day equity, and charge 5 bps per side (10 bps round trip). |
| FR-28 | M | Report these metrics for the LSTM, the logistic baseline, SPY buy-and-hold, and a random-pick baseline (4 random tickers per session, mean of 100 seeded runs): directional accuracy at threshold 0.5 with 95% Wilson interval; hit rate of selected trades; mean net return per trade with 95% bootstrap interval (10,000 resamples); annualized Sharpe (daily returns, factor sqrt(252)); maximum drawdown; trade count; average trades per day. |
| FR-29 | M | Run a sentiment ablation: the same LSTM and folds with the 5 sentiment features removed. Report the same metrics beside the full model. |
| FR-30 | M | Write `reports/backtest-<model_id>-<YYYYMMDD>.md` and `reports/backtest-<model_id>-<YYYYMMDD>.json` containing all FR-28 and FR-29 metrics, the config snapshot, the seed, and the git SHA. |
| FR-31 | M | Provide look-ahead protection in `src/stockbot/leakcheck.py`: (a) `assert_no_future_dependence(symbol, t)` recomputes the features for `(symbol, t)` after deleting all `bars_daily` and `news` rows dated after `t` and asserts the values equal the originals to 1e-9; (b) `assert_no_target_proxy(features_df)` fails if any feature column has absolute Pearson correlation above 0.5 with `next_oc_ret` or equals `label`. `backtest` and `train` call both on a 20-symbol sample before running and exit 4 on failure. |

### 6.7 Signals

| ID | Pri | Requirement |
|---|---|---|
| FR-32 | M | `stockbot signal` loads the newest model in `models`, builds the 20-session window for every active non-benchmark ticker ending at the most recent completed session, computes `p_up`, ranks tickers, and writes one row per ticker to `signals` with `selected=1` for the top `N` that meet the threshold. |
| FR-33 | M | Before computing signals, count tickers whose latest `bars_daily.session_date` is not the most recent completed session. If more than 5% of active tickers are stale, exit with code 4 and write no signals. |
| FR-34 | S | If fewer than `N` tickers meet the threshold, select only those that do (zero is valid). Log the count. |

### 6.8 Paper execution

| ID | Pri | Requirement |
|---|---|---|
| FR-35 | M | `stockbot trade` reads `signals` rows with `selected=1` from the most recent completed session and submits one market buy per symbol with `time_in_force=opg`. Quantity is `floor(0.05 * account_equity / prior_close)`; skip the symbol and log if quantity is below 1. |
| FR-36 | M | Set `client_order_id = "{session_date}-{symbol}-{side}"` on every order. Before submitting, check `orders` and Alpaca for that id; if it exists, skip submission. Running `trade` or `exit` twice on the same day must not create duplicate orders. |
| FR-37 | M | `stockbot exit` submits a market sell with `time_in_force=cls` for each position the bot opened today (a filled or open entry order in `orders` for today). Quantity equals the held quantity. If the current time is after 15:50 America/New_York, exit with code 4 and log `exit window missed`. |
| FR-38 | M | Risk checks before every submission: at most 4 open entry orders per day; per-position notional at most 5% of equity; skip all new entries if account equity is 3% or more below the previous day's equity snapshot. A failed check logs the rule name and skips the order. |
| FR-39 | M | `stockbot reconcile` fetches order statuses and fills from Alpaca for all `orders` rows not in a terminal state, updates them, computes realized P&L per closed trade, and inserts one row in `equity_snapshots` for today. |
| FR-40 | M | No code path may submit an order that opens a short position. The only sell allowed is `exit` on a quantity less than or equal to the quantity currently held. |
| FR-41 | S | `trade` and `exit` accept `--dry-run`, which prints the orders that would be submitted and makes no order API calls. |

### 6.9 Orchestration and reporting

| ID | Pri | Requirement |
|---|---|---|
| FR-42 | S | `stockbot nightly` runs `ingest-prices`, `ingest-news`, `features`, `signal` in order and stops at the first failing step, returning that step's exit code. |
| FR-43 | S | `stockbot report` writes `reports/daily-<YYYYMMDD>.md` with: signals (symbol, `p_up`, selected), orders (status, fill price), realized P&L per trade, and equity versus the previous snapshot. |
| FR-44 | S | Provide `docs/SCHEDULING.md` with ready-to-paste crontab lines for the schedule in Section 8.6 and instructions for macOS `launchd` as an alternative. |
| FR-45 | C | Send a failure alert to a webhook URL in `STOCKBOT_ALERT_WEBHOOK` when any scheduled command exits non-zero. |

---

## 7. Non-Functional Requirements

| ID | Category | Requirement | Target |
|---|---|---|---|
| NFR-1 | Performance | `nightly` for 503 tickers, incremental data | Completes in 15 minutes or less (dominated by Finnhub throttle of 50 req/min) |
| NFR-2 | Performance | Full backtest (8 folds, LSTM + baseline + ablation) on MacBook M4 | Completes in 90 minutes or less |
| NFR-3 | Performance | `signal` for 503 tickers | Completes in 2 minutes or less |
| NFR-4 | Memory | Peak RAM during training | 8 GB or less |
| NFR-5 | Reproducibility | Two `train` runs with the same seed, data, and device `cpu` | Final validation loss differs by 1e-4 or less |
| NFR-6 | Reliability | Re-running any command on the same day | Produces no duplicate rows in any table and no duplicate orders |
| NFR-7 | Security | Credentials | Stored only in `.env`; `.env` listed in `.gitignore`; repo contains `.env.example` with placeholder values |
| NFR-8 | Observability | Logging | JSON lines, one object per event, to `logs/YYYYMMDD.log`, with fields `ts`, `level`, `command`, `event` |
| NFR-9 | Code quality | Tests | `pytest` coverage of 80% or more on `src/stockbot/` excluding `cli.py` |
| NFR-10 | Code quality | Lint and types | `ruff check` and `ruff format --check` pass with zero findings |
| NFR-11 | Platform | OS | Runs on macOS (Apple Silicon) and Linux with Python 3.12 |

---

## 8. Technical Specification

### 8.1 Stack

| Component | Choice | Version (minimum; pin exact in Phase 0 via `uv lock`) |
|---|---|---|
| Language | Python | 3.12 |
| Package manager | uv | latest stable |
| ML framework | PyTorch | 2.5 |
| Baseline model and metrics | scikit-learn | 1.5 |
| Data frames | pandas, numpy | pandas 2.2, numpy 1.26 |
| Broker and market data | alpaca-py | 0.30 |
| News | finnhub-python | 2.4 |
| Sentiment | vaderSentiment | 3.3.2 |
| CLI | Typer | 0.12 |
| Config | stdlib `tomllib`, python-dotenv | python-dotenv 1.0 |
| Storage | SQLite via stdlib `sqlite3` (no ORM) | stdlib |
| Time zones | stdlib `zoneinfo` | stdlib |
| Tests | pytest, pytest-cov | pytest 8 |
| Lint and format | ruff | 0.6 |

ASSUMPTION: SQLite is sufficient. 503 tickers times 5 years of daily bars is about 630k rows.

### 8.2 Features (per `symbol`, `session_date` `t`)

`C`, `O`, `H`, `L`, `V` are adjusted close, open, high, low, volume.

| # | Name | Definition |
|---|---|---|
| 1 | `ret_1d` | `ln(C_t / C_{t-1})` |
| 2 | `ret_5d` | `ln(C_t / C_{t-5})` |
| 3 | `ret_20d` | `ln(C_t / C_{t-20})` |
| 4 | `vol_20d` | sample standard deviation of `ret_1d` over sessions `t-19..t` |
| 5 | `vol_z_20d` | `(V_t - mean(V_{t-19..t})) / std(V_{t-19..t})`; 0 if std is 0 |
| 6 | `hl_range` | `(H_t - L_t) / C_t` |
| 7 | `gap` | `ln(O_t / C_{t-1})` |
| 8 | `oc_ret` | `ln(C_t / O_t)` |
| 9 | `sent_mean` | mean `vader_compound` of headlines with `session_date = t` |
| 10 | `sent_max` | max of the same set |
| 11 | `sent_min` | min of the same set |
| 12 | `news_count` | `ln(1 + number of headlines)` |
| 13 | `news_available` | 1 if at least one headline, else 0 |

Sentiment features 9 to 13 are the "5 sentiment features" removed in the FR-29 ablation.

### 8.3 Model and training settings (fixed, defined in `config/default.toml`)

| Setting | Value |
|---|---|
| Sequence length | 20 sessions |
| LSTM layers / hidden size | 2 / 64 |
| Dropout | 0.2 (between LSTM layers and before the head) |
| Head | `Linear(64, 1)` |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | `AdamW`, lr `1e-3`, weight decay `1e-4` |
| Batch size | 512 |
| Max epochs / early stopping | 50 / patience 5 on validation loss |
| Selection | top `N=4`, threshold `p_up >= 0.55` |
| Position size | 5% of equity per position |
| Backtest cost | 5 bps per side |
| Seed | 42 |

ASSUMPTION: a single global model shared across all tickers is used so one model trains on about 600k samples instead of 503 models on about 1,250 samples each.

### 8.4 Architecture

```
Finnhub news --> ingest-news --> news (+VADER) ----\
                                                    +--> features --> features table --> train / backtest
Alpaca bars  --> ingest-prices --> bars_daily -----/                                          |
                                                                                        models + artifacts
                                                                                              |
                                              signal (p_up, rank, selected) --> signals table
                                                                                              |
                         trade (opg buys) --> exit (cls sells) --> reconcile --> orders, equity_snapshots
                                                         (Alpaca PAPER API only)
```

Trading day timeline (all times America/New_York):

1. 17:00 `nightly`: ingest, features, signal for session `t`.
2. 08:45 next session `t+1`: `trade` submits `opg` buys.
3. 09:30: opening auction fills.
4. 15:45: `exit` submits `cls` sells.
5. 16:20: `reconcile` updates fills and P&L.

### 8.5 Directory structure

```
stockbot/
  CLAUDE.md
  README.md
  pyproject.toml
  uv.lock
  .env.example
  .gitignore
  config/
    default.toml
  data/
    sp500.csv                  # symbol,name,sector (committed)
    stockbot.db                # gitignored
  artifacts/models/            # gitignored
  logs/                        # gitignored
  reports/                     # gitignored
  docs/
    PRD-lstm-stock-bot.md
    SCHEDULING.md
  src/stockbot/
    __init__.py
    cli.py                     # Typer app, one function per command
    config.py                  # load TOML + env, paper-only guard (FR-1, FR-2)
    db.py                      # connection, schema, upsert helpers
    calendar.py                # Alpaca market calendar wrapper
    universe.py
    prices.py
    news.py
    sentiment.py
    features.py
    dataset.py                 # windows, folds, normalization
    leakcheck.py               # look-ahead guards (FR-31)
    models.py                  # LSTM + logistic baseline
    train.py
    backtest.py
    metrics.py                 # Wilson, bootstrap, Sharpe, drawdown
    signals.py
    broker.py                  # Alpaca trading client wrapper
    execution.py               # trade, exit, reconcile
    risk.py
    reports.py
    logging_setup.py
  tests/
    fixtures/
    test_*.py                  # one file per module
```

### 8.6 Schedule (machine timezone assumed America/Chicago, so subtract 1 hour from ET)

| Job | ET | CT | Days | Command |
|---|---|---|---|---|
| Nightly pipeline | 17:00 | 16:00 | Mon to Fri | `stockbot nightly` |
| Entry | 08:45 | 07:45 | Mon to Fri | `stockbot trade` |
| Exit | 15:45 | 14:45 | Mon to Fri | `stockbot exit` |
| Reconcile + report | 16:20 | 15:20 | Mon to Fri | `stockbot reconcile && stockbot report` |
| Retrain | 12:00 | 11:00 | Sunday | `stockbot train` |

---

## 9. Data Model

SQLite, file `data/stockbot.db`. Dates are `TEXT` in `YYYY-MM-DD` (exchange date). Timestamps are `TEXT` in ISO 8601 UTC.

```sql
CREATE TABLE tickers (
  symbol        TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  sector        TEXT,
  is_benchmark  INTEGER NOT NULL DEFAULT 0,   -- 1 for SPY
  active        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE bars_daily (
  symbol        TEXT NOT NULL REFERENCES tickers(symbol),
  session_date  TEXT NOT NULL,
  open          REAL NOT NULL,
  high          REAL NOT NULL,
  low           REAL NOT NULL,
  close         REAL NOT NULL,                -- adjusted
  volume        INTEGER NOT NULL,             -- IEX-feed volume
  PRIMARY KEY (symbol, session_date)
);

CREATE TABLE news (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol           TEXT NOT NULL REFERENCES tickers(symbol),
  published_at_utc TEXT NOT NULL,
  session_date     TEXT NOT NULL,
  headline         TEXT NOT NULL,
  source           TEXT,
  url              TEXT,
  vader_compound   REAL NOT NULL,
  UNIQUE (symbol, published_at_utc, headline)
);

CREATE TABLE features (
  symbol         TEXT NOT NULL REFERENCES tickers(symbol),
  session_date   TEXT NOT NULL,
  ret_1d REAL NOT NULL, ret_5d REAL NOT NULL, ret_20d REAL NOT NULL,
  vol_20d REAL NOT NULL, vol_z_20d REAL NOT NULL, hl_range REAL NOT NULL,
  gap REAL NOT NULL, oc_ret REAL NOT NULL,
  sent_mean REAL NOT NULL, sent_max REAL NOT NULL, sent_min REAL NOT NULL,
  news_count REAL NOT NULL, news_available INTEGER NOT NULL,
  next_oc_ret    REAL,                        -- NULL for latest session
  label          INTEGER,                     -- NULL for latest session
  PRIMARY KEY (symbol, session_date)
);

CREATE TABLE models (
  model_id       TEXT PRIMARY KEY,
  trained_at_utc TEXT NOT NULL,
  train_end_date TEXT NOT NULL,
  seed           INTEGER NOT NULL,
  git_sha        TEXT NOT NULL,
  val_loss       REAL NOT NULL,
  artifact_dir   TEXT NOT NULL
);

CREATE TABLE signals (
  session_date   TEXT NOT NULL,               -- session the features ended on
  symbol         TEXT NOT NULL REFERENCES tickers(symbol),
  model_id       TEXT NOT NULL REFERENCES models(model_id),
  p_up           REAL NOT NULL,
  rank           INTEGER NOT NULL,            -- 1 = highest p_up
  selected       INTEGER NOT NULL,
  PRIMARY KEY (session_date, symbol, model_id)
);

CREATE TABLE orders (
  client_order_id  TEXT PRIMARY KEY,          -- "{session_date}-{symbol}-{side}"
  alpaca_order_id  TEXT,
  session_date     TEXT NOT NULL,             -- trading day the order is for
  symbol           TEXT NOT NULL,
  side             TEXT NOT NULL CHECK (side IN ('buy','sell')),
  qty              INTEGER NOT NULL,
  time_in_force    TEXT NOT NULL CHECK (time_in_force IN ('opg','cls')),
  status           TEXT NOT NULL,             -- Alpaca status string
  submitted_at_utc TEXT NOT NULL,
  filled_qty       INTEGER,
  filled_avg_price REAL,
  filled_at_utc    TEXT
);

CREATE TABLE equity_snapshots (
  session_date   TEXT PRIMARY KEY,
  equity         REAL NOT NULL,
  cash           REAL NOT NULL,
  realized_pnl   REAL NOT NULL                -- sum of closed trades that day
);

CREATE TABLE runs (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  command        TEXT NOT NULL,
  started_at_utc TEXT NOT NULL,
  finished_at_utc TEXT,
  status         TEXT,                        -- 'ok' | 'error'
  exit_code      INTEGER,
  error_summary  TEXT
);
```

Relationships: `bars_daily`, `news`, `features`, `signals` reference `tickers`; `signals` references `models`. `orders` links to `signals` by `(session_date, symbol)` through application logic, not a foreign key, because `orders.session_date` is the trade day and `signals.session_date` is the prior session.

---

## 10. Interface Contracts

This system has no HTTP API (NG-4). The contract is the CLI. All commands accept `--config PATH` (default `config/default.toml`).

| Command | Arguments | Reads | Writes | Stdout on success |
|---|---|---|---|---|
| `init-db` | none | none | all tables | `db ready` |
| `load-universe` | none | `data/sp500.csv` | `tickers` | `loaded <n> tickers` |
| `ingest-prices` | `--start YYYY-MM-DD` optional | Alpaca, `tickers` | `bars_daily` | `<n> bars upserted, <m> symbols failed` |
| `ingest-news` | `--start YYYY-MM-DD` optional | Finnhub, `tickers` | `news` | `<n> headlines upserted, <m> symbols failed` |
| `features` | none | `bars_daily`, `news` | `features` | `<n> rows, <d> dropped` |
| `train` | none | `features` | `models`, `artifacts/models/` | `model_id=<id> val_loss=<x>` |
| `backtest` | `--no-ablation` optional | `features` | `reports/` | path of report file |
| `signal` | none | `features`, `models` | `signals` | `selected: <sym>,<sym>,...` or `selected: none` |
| `trade` | `--dry-run` | `signals`, Alpaca | `orders` | one line per order |
| `exit` | `--dry-run` | `orders`, Alpaca | `orders` | one line per order |
| `reconcile` | none | Alpaca | `orders`, `equity_snapshots` | `equity=<x> realized_pnl=<y>` |
| `report` | `--date YYYY-MM-DD` optional | all | `reports/` | path of report file |
| `nightly` | none | see FR-42 | see FR-42 | one line per step |

External APIs consumed:

| Service | Used for | Limit to respect |
|---|---|---|
| Alpaca Market Data (IEX feed) | Daily bars | Batch up to 100 symbols per request |
| Alpaca Trading (paper) | Orders, positions, account, calendar | Paper base URL only |
| Finnhub `company-news` | Headlines | 50 requests per minute (internal cap; free tier is 60) |

---

## 11. UI/UX Requirements

Omitted as a GUI section: the interface is CLI only (NG-4). Output rules instead:

- UX-1: Each command prints one summary line per Section 10 and nothing else on success.
- UX-2: On failure, print one line `ERROR <exit_code>: <message>` to stderr and write the full traceback to the log file only.
- UX-3: Progress for loops over tickers uses a single updating line (`typer.progressbar`), disabled when stdout is not a TTY.

---

## 12. Implementation Plan

Each phase ends with something runnable and all listed ACs passing.

| Phase | Deliverable | FRs | ACs |
|---|---|---|---|
| 0 | Scaffold: `pyproject.toml`, `uv lock`, config loader, paper-only guard, DB schema, run logging, CLI skeleton, `ruff` and `pytest` wired | FR-1 to FR-6 | AC-1 to AC-6 |
| 1 | Universe and prices ingestion | FR-7 to FR-10 | AC-7 to AC-9 |
| 2 | News ingestion and sentiment | FR-11 to FR-15 | AC-10 to AC-13 |
| 3 | Features, labels, dataset windows | FR-16 to FR-19, FR-31 | AC-14 to AC-17 |
| 4 | Logistic baseline, LSTM, walk-forward, `train` | FR-20 to FR-26 | AC-18 to AC-22 |
| 5 | Backtest, baselines, ablation, reports | FR-27 to FR-30 | AC-23 to AC-26 |
| 6 | Signals and Alpaca paper execution | FR-32 to FR-41 | AC-27 to AC-35 |
| 7 | Orchestration, daily report, scheduling docs, 30-session paper run | FR-42 to FR-45 | AC-36 to AC-38 |

Do not start Phase N+1 until Phase N's ACs pass and `ruff check` and `pytest` are green.

---

## 13. Acceptance Criteria

Every Must FR maps to at least one check. All checks run in `pytest` with mocked APIs unless marked **manual**.

| AC | Covers | Check |
|---|---|---|
| AC-1 | FR-1 | With `FINNHUB_API_KEY` unset, `ingest-news` exits 2 and stderr contains `FINNHUB_API_KEY` and no key values. |
| AC-2 | FR-2 | With base URL `https://api.alpaca.markets`, `trade`, `exit`, `reconcile`, and `ingest-prices` exit 3 and the mocked Alpaca client records zero calls. |
| AC-3 | FR-3 | `init-db` run twice: exit 0 both times; `sqlite_master` contents identical. |
| AC-4 | FR-4 | With the calendar mocked to a holiday, `trade`, `exit`, and `nightly` exit 0, log `market closed`, and record zero order calls. |
| AC-5 | FR-5 | Parametrized test triggers exit codes 0, 1, 2, 3, 4; each leaves a `runs` row with matching `exit_code` and `status`. |
| AC-6 | FR-6 | Run all commands with sentinel secrets `SENTINEL_ALPACA_KEY` and `SENTINEL_FINNHUB_KEY`; `grep -r` of `logs/`, stdout capture, and the `runs` table finds zero matches. |
| AC-7 | FR-7 | Loading a 5-row fixture CSV plus SPY yields 6 `tickers` rows; SPY has `is_benchmark=1`. Real `data/sp500.csv` has between 500 and 506 rows. |
| AC-8 | FR-8, FR-9 | First ingest on fixture stores expected bars; second ingest requests bars starting after the stored max date and leaves row count unchanged. |
| AC-9 | FR-9 | Fixture where the overlap bar's close differs by 1% triggers deletion and full refetch for that symbol only. |
| AC-10 | FR-11 | Ingesting 120 mocked symbols takes at least 2 minutes of throttle time (use a fake clock) and never exceeds 50 requests in any 60-second window. |
| AC-11 | FR-12 | A mocked 429 on the first 3 calls succeeds on the 4th; sleep calls are 1, 2, 4 seconds. Five consecutive failures mark the symbol failed and processing continues. |
| AC-12 | FR-13 | Fixture headlines score equal to a direct `polarity_scores` call; stored values match to 1e-9. |
| AC-13 | FR-14 | Headline at 15:59 ET Friday maps to Friday; 16:01 ET Friday maps to Monday; Saturday 10:00 ET maps to Monday; a UTC timestamp across a DST change maps correctly. |
| AC-14 | FR-16, FR-17 | Hand-computed values for a 30-session fixture match all 13 features to 1e-9; a day with no headlines has `news_available=0` and zeros for 9, 10, 11, 12. |
| AC-15 | FR-18 | For a fixture where session `t+1` has open 100 and close 101, `label=1` and `next_oc_ret=0.01`; the latest session has `NULL` for both. |
| AC-16 | FR-19 | The first 20 sessions of each symbol are absent from `features`; log reports the dropped count. |
| AC-17 | FR-31 | On the fixture, both checks pass. After injecting a leaked column (`next_oc_ret` copied into a feature, or `ret_1d` computed with `C_{t+1}`), the matching check raises and `train` exits 4. |
| AC-18 | FR-20 | Baseline trains on a fixture and returns `p_up` in [0, 1] for every row. |
| AC-19 | FR-21 | LSTM forward pass on a `(8, 20, 13)` tensor returns shape `(8, 1)`; a 200-step overfit test on 64 samples reaches training accuracy of 95% or more. |
| AC-20 | FR-22 | A test asserts the saved `feature_stats.json` equals the mean and std of the training slice only (not validation or test). |
| AC-21 | FR-23 | Fold generator on a synthetic 5-year calendar returns 8 test blocks of 3 months; for every fold, `max(train_date) + 5 sessions <= min(test_date)`, and the early-stopping slice is the last 10% of training dates. |
| AC-22 | FR-24 | `train` on a fixture creates all 3 artifact files, inserts a `models` row with a non-empty `git_sha`, and `model_id` matches `^lstm-\d{8}-\d{6}$`. |
| AC-23 | FR-27 | On a hand-built 3-day, 5-ticker fixture with known `p_up`, backtest selects the expected tickers, applies 10 bps round-trip cost, and produces the hand-computed daily return. |
| AC-24 | FR-28 | Metrics module returns hand-checked values for Wilson interval, bootstrap (seeded), Sharpe, and max drawdown on fixture series. |
| AC-25 | FR-29 | Ablation run uses 8 input features (features 1 to 8 only) and the model input dimension reflects it. |
| AC-26 | FR-30 | Backtest writes both report files containing the config snapshot, seed, and git SHA, and every metric listed in FR-28. |
| AC-27 | FR-32 | `signal` on fixture writes one row per active non-benchmark ticker; exactly the top rows meeting threshold have `selected=1`; benchmark symbols never appear. |
| AC-28 | FR-33 | With 6% of tickers stale, `signal` exits 4 and writes zero rows. With 4% stale, it succeeds. |
| AC-29 | FR-35 | With equity 100,000 and prior close 250, `trade` submits quantity 20 with `time_in_force=opg`; a symbol with prior close 6,000 (quantity 0) is skipped and logged. |
| AC-30 | FR-36 | Running `trade` twice makes exactly one submit call per symbol; `client_order_id` equals `{session_date}-{symbol}-buy`. |
| AC-31 | FR-37 | `exit` at 14:00 ET submits `cls` sells equal to held quantity; at 15:51 ET it exits 4 and submits nothing. |
| AC-32 | FR-38 | A 5th entry in one day is rejected; notional above 5% of equity is rejected; equity down 3% from the prior snapshot rejects all entries; each logs the rule name. |
| AC-33 | FR-39 | `reconcile` on mocked fills updates `orders` rows and writes one `equity_snapshots` row; running it twice leaves one row. |
| AC-34 | FR-40 | A test attempts to call `exit` for a symbol with no position and for a quantity above held; both submit nothing. A static test asserts the broker wrapper exposes no function that places a sell without a position check. |
| AC-35 | FR-41 | `trade --dry-run` and `exit --dry-run` print orders and the mocked client records zero submissions. |
| AC-36 | FR-42 | With `ingest-news` mocked to fail, `nightly` stops, returns that exit code, and `features` and `signal` do not run. |
| AC-37 | FR-43, FR-44 | Daily report contains all four required sections; `docs/SCHEDULING.md` contains all 5 jobs from Section 8.6. |
| AC-38 | Section 16 | **Manual:** 30 consecutive trading sessions of scheduled paper runs meet the operational metrics in Section 16. |

---

## 14. Edge Cases and Error Handling

| ID | Situation | Required behavior |
|---|---|---|
| EC-1 | Market holiday or weekend | Per FR-4: log, no orders, exit 0. |
| EC-2 | Half-day session (13:00 ET close) | `exit` uses the Alpaca calendar's actual close. If current time is within 10 minutes of that close or later, exit 4. `cls` orders on half-days still use `time_in_force=cls`. |
| EC-3 | Stock split or large dividend | Handled by FR-9 full refetch. |
| EC-4 | Ticker delisted or renamed | FR-10 sets `active=0`. Existing signals and orders keep their rows. |
| EC-5 | Finnhub returns zero headlines for a symbol | Valid. Feature rule FR-17 applies. |
| EC-6 | Finnhub or Alpaca outage past all retries | Symbol marked failed, run continues, summary line counts failures, exit 0 if under 5% of symbols failed, else exit 4. |
| EC-7 | Alpaca rejects an order (insufficient buying power, halted symbol, invalid TIF) | Store the order with the rejected status and reason, continue with the next symbol, never retry the same `client_order_id`. |
| EC-8 | Partial fill | `reconcile` stores `filled_qty`; `exit` sells `filled_qty` held, not the requested quantity. |
| EC-9 | `trade` runs but no signals exist for the prior session | Log `no signals`, exit 4. |
| EC-10 | Latest model artifact files missing from disk | `signal` exits 4 with the missing path. |
| EC-11 | NaN or infinite value in a feature | Row dropped per FR-19; if more than 1% of rows in a run are dropped beyond warm-up, log a warning. |
| EC-12 | Process killed mid-run | Next run is safe because upserts and `client_order_id` checks are idempotent (NFR-6). |
| EC-13 | Machine asleep during a scheduled job | Job does not run; next `nightly` catches up on missing data. Missed `trade` or `exit` is logged by the next `reconcile`. See OQ-1. |
| EC-14 | Duplicate headline text from different sources at the same timestamp | Deduplicated by the UNIQUE constraint; keep the first inserted. |

---

## 15. Dependencies, Assumptions, Constraints

### Dependencies

Alpaca paper account with API keys, Finnhub free API key, `data/sp500.csv` (created by the user or the agent from a current S&P 500 constituent list; see OQ-3).

### Assumptions (flag if wrong)

- ASSUMPTION A-1: Strategy is long-only, one-session hold, entering at the open and exiting at the close.
- ASSUMPTION A-2: Python 3.12, `uv`, PyTorch.
- ASSUMPTION A-3: One global LSTM, not per-ticker models.
- ASSUMPTION A-4: `N=4`, threshold 0.55, position size 5%. Chosen to roughly match the old bot's 4 trades per day, not derived from data.
- ASSUMPTION A-5: Transaction cost 5 bps per side in backtests.
- ASSUMPTION A-6: Machine timezone is America/Chicago.
- ASSUMPTION A-7: Daily bars are adjusted for splits and dividends.
- ASSUMPTION A-8: Price history of 5 years; news history of about 365 days at first run. Before the first stored headline date, sentiment features are 0 and `news_available=0`.
- ASSUMPTION A-9: Alpaca paper accepts `opg` and `cls` time-in-force for equities (verify per OQ-5).

### Constraints

- Finnhub free tier: 60 requests per minute, limited news history.
- Alpaca IEX feed volume reflects IEX trades only, not consolidated volume.
- `opg` orders must be submitted before 09:28 ET; `cls` orders before 15:50 ET.

### Known research limitations (must appear in every backtest report)

- Survivorship bias: a current-constituents list is applied to history.
- Sentiment history is shorter than price history, which limits the power of the ablation.
- Backtest fills assume the official open and close prices with a flat cost; real slippage may differ.

---

## 16. Success Metrics

### Operational (pass or fail, evaluated over 30 consecutive trading sessions of paper trading)

| ID | Metric | Target |
|---|---|---|
| SM-1 | Scheduled commands exiting 0 | 95% or more of runs |
| SM-2 | Duplicate orders | 0 |
| SM-3 | Orders submitted to a non-paper endpoint | 0 |
| SM-4 | `reconcile` fills matching Alpaca's order history | 100% of orders |
| SM-5 | Positions held overnight by the bot | 0 |

### Research (reported, not targets)

| ID | Metric | How it is used |
|---|---|---|
| SM-6 | LSTM OOS directional accuracy with 95% Wilson interval | Reported next to the logistic baseline and the 50% reference |
| SM-7 | Mean net return per selected trade with 95% bootstrap interval | Reported next to SPY and random-pick baselines |
| SM-8 | Sharpe and max drawdown | Reported next to SPY |
| SM-9 | Sentiment ablation difference | Reported; positive, zero, or negative all count as a valid result |

The only claims that may go on a resume are numbers produced by `stockbot backtest` or the paper run, quoted with their intervals.

---

## 17. Open Questions

| ID | Question | Blocks |
|---|---|---|
| OQ-1 | Where do the scheduled jobs run: the MacBook (sleeps, so jobs can be missed) or an always-on machine (for example the Proxmox homelab in a Docker container)? | FR-44 format only; Phases 0 to 6 unaffected |
| OQ-2 | Stay on Alpaca's free IEX feed (IEX-only volume) or add SIP data? | FR-8, `vol_z_20d` meaning |
| OQ-3 | Source for `data/sp500.csv`, and is a point-in-time constituent history available to reduce survivorship bias? Default if unanswered: a current snapshot copied from a public list, with the limitation in Section 15. | Phase 1 |
| OQ-4 | How far back does Finnhub's free `company-news` actually return per symbol? The agent must measure this in Phase 2 and record it in the README. | Phase 2 |
| OQ-5 | Does the Alpaca paper account accept `opg` and `cls` time-in-force for these orders? The agent must verify against current Alpaca docs in Phase 0. If not, fallback is market orders with `time_in_force=day` submitted at 09:30:30 ET (entry) and 15:55 ET (exit), and FR-35, FR-37, and the Section 8.6 times must be updated before Phase 6. | Phase 6 |

**Agent rule:** if a decision is not covered by this document and affects a Must requirement, stop and ask. Do not guess.
