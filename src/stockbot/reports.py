"""Backtest report: markdown, JSON, and an equity-curve chart (FR-30)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {
    "lstm": "LSTM (price + sentiment)",
    "lstm_no_sentiment": "LSTM (price only)",
    "logreg": "Logistic regression",
    "random": "Random 4 picks (mean of 100)",
    "spy": "SPY buy and hold",
}

LIMITATIONS = [
    "Survivorship bias: a current-constituents list is applied to history.",
    "Sentiment history is shorter than price history, which limits the power of the ablation.",
    "Fills assume the official open and close with a flat cost; real slippage may differ.",
    "Strategy exposure is at most 4 x 5% = 20% of equity per day; SPY is 100% invested.",
]


def _pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _ci(ci: tuple[float, float] | None, digits: int = 2) -> str:
    return "n/a" if ci is None else f"[{_pct(ci[0], digits)}, {_pct(ci[1], digits)}]"


def metrics_table(metrics: dict[str, dict[str, Any]]) -> str:
    header = (
        "| Model | Accuracy (95% CI) | Hit rate | Mean net return / trade (95% CI) | "
        "Sharpe | Max DD | Total return | Trades | Trades/day |"
    )
    lines = [header, "|" + "---|" * 9]
    for key in LABELS:
        if key not in metrics:
            continue
        m = metrics[key]
        acc = (
            f"{_pct(m['accuracy'], 1)} {_ci(m.get('accuracy_ci'), 1)}" if "accuracy" in m else "n/a"
        )
        ret = (
            f"{_pct(m['mean_trade_return'], 3)} {_ci(m.get('mean_trade_return_ci'), 3)}"
            if "mean_trade_return" in m
            else "n/a"
        )
        sharpe = m.get("sharpe", math.nan)
        lines.append(
            f"| {LABELS[key]} | {acc} | {_pct(m.get('hit_rate'), 1)} | {ret} | "
            f"{sharpe:.2f} | {_pct(m['max_drawdown'])} | {_pct(m['total_return'])} | "
            f"{m['n_trades']:.0f} | {m['trades_per_day']:.2f} |"
            if "n_trades" in m
            else f"| {LABELS[key]} | {acc} | n/a | n/a | {sharpe:.2f} | "
            f"{_pct(m['max_drawdown'])} | {_pct(m['total_return'])} | n/a | n/a |"
        )
    return "\n".join(lines)


def plot_equity(curves: dict[str, pd.Series], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    for key in LABELS:
        if key not in curves:
            continue
        s = curves[key]
        equity = (1 + s).cumprod()
        equity.index = pd.to_datetime(equity.index)
        style = {"linewidth": 2.2} if key == "lstm" else {"linewidth": 1.3, "alpha": 0.85}
        ax.plot(equity.index, equity.to_numpy(), label=LABELS[key], **style)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Walk-forward out-of-sample equity (net of 10 bps round-trip costs)")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_backtest_report(
    metrics: dict[str, dict[str, Any]],
    curves: dict[str, pd.Series],
    *,
    run_id: str,
    cfg: dict[str, Any],
    git_sha: str,
    out_dir: str | Path = "reports",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"backtest-{run_id}-{datetime.now(UTC):%Y%m%d}"
    sessions = next(iter(curves.values())).index
    config_text = Path(cfg["_path"]).read_text()

    plot_equity(curves, out_dir / f"{stem}.png")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": cfg["seed"],
                "git_sha": git_sha,
                "oos_start": sessions[0],
                "oos_end": sessions[-1],
                "metrics": metrics,
                "config": config_text,
            },
            indent=2,
            default=float,
        )
    )

    md = [
        f"# Backtest {run_id}",
        "",
        f"Out-of-sample: {sessions[0]} to {sessions[-1]} ({len(sessions)} sessions, "
        f"{cfg['walkforward']['n_folds']} walk-forward folds). Seed {cfg['seed']}, "
        f"git `{git_sha[:10]}`.",
        "",
        metrics_table(metrics),
        "",
        f"![equity curve]({stem}.png)",
        "",
        "## Known limitations",
        "",
        *[f"- {line}" for line in LIMITATIONS],
        "",
        "## Config snapshot",
        "",
        "```toml",
        config_text.rstrip(),
        "```",
        "",
    ]
    path = out_dir / f"{stem}.md"
    path.write_text("\n".join(md))
    return path
