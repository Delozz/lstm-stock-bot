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
from matplotlib.ticker import PercentFormatter  # noqa: E402

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


SHORT = {
    "lstm": "LSTM",
    "lstm_no_sentiment": "LSTM price-only",
    "logreg": "Logistic",
    "random": "Random",
    "spy": "SPY",
}
STRATEGIES = ("lstm", "lstm_no_sentiment", "logreg", "random")
COLORS = {
    "lstm": "#2a78d6",
    "lstm_no_sentiment": "#eb6834",
    "logreg": "#1baf7a",
    "random": "#8a8984",
    "spy": "#52514e",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _cumulative(daily: pd.Series) -> pd.Series:
    out = (1 + daily).cumprod() - 1
    out.index = pd.to_datetime(out.index)
    return out * 100


def _end_labels(ax: plt.Axes, ends: dict[str, tuple[pd.Timestamp, float]], min_gap: float) -> None:
    """Direct-label each line at its last point, nudging labels apart so they don't collide."""
    placed: list[float] = []
    for key, (x, y) in sorted(ends.items(), key=lambda kv: -kv[1][1]):
        ly = min(y, placed[-1] - min_gap) if placed else y
        placed.append(ly)
        ax.annotate(
            f"{SHORT[key]}  {y:+.1f}%",
            xy=(x, ly),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK,
            annotation_clip=False,
        )


def plot_equity(curves: dict[str, pd.Series], path: Path) -> None:
    """Strategies on their own scale; SPY in a separate panel so it doesn't flatten them."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.5), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2.4, 1]}
    )
    ends = {}
    for key in STRATEGIES:
        if key not in curves:
            continue
        cum = _cumulative(curves[key])
        lw = 2.2 if key == "lstm" else 1.4
        ls = "--" if key == "random" else "-"
        top.plot(cum.index, cum.to_numpy(), color=COLORS[key], lw=lw, ls=ls, label=LABELS[key])
        top.plot(cum.index[-1], cum.iloc[-1], "o", color=COLORS[key], ms=4)
        ends[key] = (cum.index[-1], float(cum.iloc[-1]))
    top.axhline(0, color=MUTED, lw=0.8)
    lo, hi = top.get_ylim()
    _end_labels(top, ends, min_gap=(hi - lo) * 0.07)
    top.set_title(
        "Strategies: daily top-4 picks, at most 20% of equity invested",
        loc="left",
        fontsize=10.5,
        color=INK,
    )
    top.set_ylabel("Cumulative return", color=MUTED, fontsize=9)
    top.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK)

    if "spy" in curves:
        spy = _cumulative(curves["spy"])
        bottom.plot(spy.index, spy.to_numpy(), color=COLORS["spy"], lw=1.4)
        bottom.plot(spy.index[-1], spy.iloc[-1], "o", color=COLORS["spy"], ms=4)
        _end_labels(bottom, {"spy": (spy.index[-1], float(spy.iloc[-1]))}, min_gap=0)
        bottom.axhline(0, color=MUTED, lw=0.8)
    bottom.set_title(
        "For context: SPY buy and hold, 100% invested", loc="left", fontsize=10.5, color=INK
    )
    bottom.set_ylabel("Cumulative return", color=MUTED, fontsize=9)

    for ax in (top, bottom):
        _style(ax)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    fig.suptitle(
        "Walk-forward out-of-sample results, net of 10 bps round-trip costs",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def decile_returns(preds: pd.DataFrame, model: str = "lstm") -> pd.DataFrame:
    """Mean next-session open-to-close return by each day's p_up decile (10 = most confident)."""
    p = preds[preds["model"] == model].copy()
    p["decile"] = (
        p.groupby("session_date")["p_up"].rank(pct=True, method="first").mul(10).apply(math.ceil)
    )
    g = p.groupby("decile")["next_oc_ret"]
    out = pd.DataFrame({"mean": g.mean(), "se": g.std() / g.count().pow(0.5), "n": g.count()})
    return out * [1e4, 1e4, 1]


def plot_deciles(preds: pd.DataFrame, path: Path) -> None:
    d = decile_returns(preds)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ax.bar(d.index, d["mean"], width=0.7, color=COLORS["lstm"], edgecolor="white", linewidth=2)
    ax.errorbar(d.index, d["mean"], yerr=1.96 * d["se"], fmt="none", ecolor=MUTED, lw=1, capsize=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(d.index, [f"D{i}" for i in d.index])
    ax.set_xlabel("LSTM p_up decile within each session (D10 = most confident)", color=MUTED)
    ax.set_ylabel("Mean next open-to-close return (bps)", color=MUTED, fontsize=9)
    ax.set_title(
        "Next-session return by model confidence, out of sample (bars: mean, whiskers: 95% CI)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def write_backtest_report(
    metrics: dict[str, dict[str, Any]],
    curves: dict[str, pd.Series],
    preds: pd.DataFrame,
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
    plot_deciles(preds, out_dir / f"{stem}-deciles.png")
    pd.DataFrame(curves).to_csv(out_dir / f"{stem}-daily.csv")
    preds.to_csv(out_dir / f"{stem}-predictions.csv.gz", index=False)
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
        f"![return by confidence decile]({stem}-deciles.png)",
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
