# CLAUDE.md

Standing rules for this repo. Spec lives in `docs/PRD-lstm-stock-bot.md`. The PRD wins over your assumptions.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions. If an FR is ambiguous, ask. Don't pick.
- Anything in PRD Section 17 (Open Questions) is a stop sign. Ask.
- Name both readings when a requirement has two. Let me choose.
- No guessing on trading rules, label definitions, or time zones.

Ask yourself: could this feature or split see data from after the prediction time? If unsure, stop.

## 2. Simplicity First

Minimum code that meets the FR. Nothing speculative.

- No base classes, plugin systems, or config flags the PRD didn't ask for.
- No ORM. Stdlib `sqlite3`, parameterized SQL only.
- No features from PRD Non-Goals: no shorting, no live trading, no intraday, no dashboard.
- Pythonic first: functions and dataclasses before class hierarchies.

The test: would a senior quant dev call this overbuilt? Cut it.

## 3. Surgical Changes

Touch only what the task needs. Match what's there.

- `ruff` config is the style. Don't reformat files you aren't changing.
- No drive-by refactors or renames.
- Remove only the imports and variables your change orphaned.
- Mention unrelated problems. Don't fix them.

The test: every changed line traces to an FR or a request.

## 4. Goal-Driven Execution

Define success. Loop until it's verified.

- "Add validation" → "Write pytest cases for invalid input, then make them pass"
- "Fix the bug" → "Write a pytest case that reproduces it, then make it pass"
- "Add a command" → "Cite the FR-n and AC-n, write the AC test first, then implement"
- "Change a feature or label" → "Re-run `leakcheck` and `backtest`, report the numbers"

For multi-step work, write a numbered plan with a check per step. Don't start phase N+1 until phase N's ACs pass.

## Project

Python 3.12 CLI that trains an LSTM on S&P 500 prices plus Finnhub headline sentiment, backtests it walk-forward, and paper trades the top picks on Alpaca. Stack: PyTorch, pandas, SQLite, Typer, alpaca-py, uv.

### Commands

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run stockbot init-db
uv run stockbot nightly
uv run stockbot backtest
uv run stockbot trade --dry-run
```

### Architecture

`ingest-prices` and `ingest-news` fill `bars_daily` and `news` in SQLite. `features` builds 13 features and the next-session open-to-close label. `train` and `backtest` use walk-forward folds. `signal` writes top-4 picks. `trade` (opg buys), `exit` (cls sells), `reconcile` run against Alpaca paper. Module map is in PRD Section 8.5.

### Conventions

- Market logic in America/New_York. Stored timestamps in UTC. Dates as `YYYY-MM-DD` exchange dates.
- Every write is an upsert. Every command is safe to run twice.
- Normalization stats come from the training slice only.
- Fixed hyperparameters live in `config/default.toml`. Don't tune them unasked.
- Fixtures in `tests/fixtures/`. Tests never hit real APIs.

### Gotchas and Boundaries

- Never weaken the paper-only guard (FR-2). Never add a code path that opens a short (FR-40).
- Never log, print, or commit secrets. `.env` stays gitignored.
- Finnhub cap is 50 requests per minute. Don't raise it.
- IEX volume is not consolidated volume.
- SPY is a benchmark. It is never traded.
- Don't quote the old bot's 60% accuracy or 2.8% return anywhere. Only numbers from `backtest` or the paper run.
- Don't edit `data/sp500.csv` by hand without saying so.

## Tooling

Use what's listed here. Nothing else applies: no Vercel, Stripe, Figma, Canva, browser, or frontend tooling (NG-4).

### Skills, by trigger

| When | Use |
|---|---|
| Starting a phase (PRD Section 12) | `superpowers:writing-plans`. One step per FR, one AC check per step. |
| Implementing an FR | `superpowers:test-driven-development`. Write the AC-n test, watch it fail, then implement. |
| A test fails or a number looks wrong | `superpowers:systematic-debugging` before proposing a fix. |
| About to call a phase done | `superpowers:verification-before-completion`. Show the `pytest` and `ruff` output. |
| Python, pytest, or PyTorch idioms | `ecc:python-patterns`, `ecc:python-testing`, `ecc:pytorch-patterns`. This file wins on conflict. |
| End of phase | `code-review` on the phase diff. |

### Review agents (end of phase, or when asked)

| Diff touches | Agent | Focus |
|---|---|---|
| `features.py`, `dataset.py`, `leakcheck.py`, `train.py`, `backtest.py` | `ecc:mle-reviewer` | Look-ahead, fold and embargo boundaries, train-only normalization, seeds |
| `config.py`, `broker.py`, `execution.py`, `risk.py`, `logging_setup.py` | `ecc:security-reviewer` | FR-2 paper guard, FR-6 secrets, FR-40 no shorts |
| Any `src/stockbot/` module | `ecc:python-reviewer` | Idioms, types, error paths |
| PyTorch shape, device, or MPS errors | `ecc:pytorch-build-resolver` | Minimal fix |

### MCPs

- **context7**: check SDK signatures before you write a call. Match the version in `uv.lock`.
  - alpaca-py `/alpacahq/alpaca-py`
  - Finnhub `/finnhub-stock-api/finnhub-python`, REST reference `/websites/finnhub_io_api`
  - PyTorch `/pytorch/pytorch`
  - Typer `/websites/typer_tiangolo`
  - Resolve pandas, scikit-learn, and vaderSentiment with `resolve-library-id`.
- **exa** (web search): for facts SDK docs don't cover. The PRD assigns OQ-5 (Alpaca `opg`/`cls` on paper, Phase 0) and OQ-4 (Finnhub history depth, Phase 2) to you. Research them, cite the URLs, and report what you found. Ask before changing an FR based on the answer. OQ-1 to OQ-3 stay with me.
- **github**: only after the repo has a remote. Run `git init` in Phase 0, because FR-24 stores a commit SHA.
- **Never install or call the Alpaca MCP server** (`alpacahq/alpaca-mcp-server`). It can trade live, options, and crypto, which bypasses FR-2, NG-1, and NG-2. All Alpaca access goes through `broker.py`.
- Real Alpaca and Finnhub calls happen only through `stockbot` commands that I run, or a one-off OQ-4 measurement script that I approve.
- No SQLite MCP. Inspect data with `sqlite3 -readonly data/stockbot.db`.

## These guidelines are working if:

- Diffs are small and every line maps to an FR.
- Questions come before code on ambiguous requirements.
- Each phase ends with green `pytest` and `ruff`.
- No backtest number moves without a stated reason.
- No order ever reaches a non-paper endpoint.
