"""Block-bootstrap the archived ML Supertrend BUY-email replay."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("analysis/results/archived_replay/10bps_daily.csv"))
    ap.add_argument("--summary", type=Path, default=Path("analysis/results/archived_replay/monte_carlo_summary.json"))
    ap.add_argument("--chart", type=Path, default=Path("docs/images/ml_supertrend_monte_carlo.png"))
    ap.add_argument("--paths", type=int, default=50_000)
    ap.add_argument("--horizon", type=int, default=252)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261102)
    a = ap.parse_args()

    df = pd.read_csv(a.source, parse_dates=["date"]).sort_values("date")
    if df.date.duplicated().any() or len(df) < 100:
        raise ValueError("Need at least 100 unique ordered archived replay marks")
    equity = df.total_value.astype(float).to_numpy()
    returns = np.r_[equity[0] / 100_000.0 - 1.0, equity[1:] / equity[:-1] - 1.0]
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Invalid daily replay returns")

    rng = np.random.default_rng(a.seed)
    terminal = np.empty(a.paths)
    max_drawdown = np.empty(a.paths)
    path_sharpe = np.empty(a.paths)
    path_equity = np.empty((a.paths, a.horizon + 1), dtype=np.float32)
    path_equity[:, 0] = 1.0
    blocks_per_path = (a.horizon + a.block - 1) // a.block
    batch_size = 1_000
    for first in range(0, a.paths, batch_size):
        n = min(batch_size, a.paths - first)
        starts = rng.integers(0, len(returns), size=(n, blocks_per_path))
        indices = (starts[:, :, None] + np.arange(a.block)) % len(returns)
        sampled = returns[indices.reshape(n, -1)[:, :a.horizon]]
        path_sharpe[first:first+n] = np.sqrt(252) * sampled.mean(axis=1) / sampled.std(axis=1, ddof=1)
        wealth = np.cumprod(1.0 + sampled, axis=1)
        path_equity[first:first+n, 1:] = wealth
        peak = np.maximum.accumulate(np.column_stack((np.ones(n), wealth)), axis=1)[:, 1:]
        max_drawdown[first:first+n] = (wealth / peak - 1.0).min(axis=1)
        terminal[first:first+n] = wealth[:, -1] - 1.0

    p05, median, p95 = np.quantile(terminal, [.05, .50, .95])
    summary = {
        "input": "archived ML Supertrend daily BUY-email replay",
        "source_sha256": hashlib.sha256(a.source.read_bytes()).hexdigest(),
        "start": str(df.date.min().date()),
        "end": str(df.date.max().date()),
        "daily_observations": int(len(returns)),
        "horizon_sessions": a.horizon,
        "block_sessions": a.block,
        "paths": a.paths,
        "seed": a.seed,
        "median_terminal_return_pct": float(median * 100),
        "p05_terminal_return_pct": float(p05 * 100),
        "p95_terminal_return_pct": float(p95 * 100),
        "negative_terminal_return_probability_pct": float((terminal < 0).mean() * 100),
        "mean_return_bottom_5_pct": float(terminal[terminal <= p05].mean() * 100),
        "p95_maximum_drawdown_loss_pct": float(-np.quantile(max_drawdown, .05) * 100),
        "median_annualized_sharpe_zero_rf": float(np.median(path_sharpe)),
        "p05_annualized_sharpe_zero_rf": float(np.quantile(path_sharpe, .05)),
        "p95_annualized_sharpe_zero_rf": float(np.quantile(path_sharpe, .95)),
        "negative_sharpe_probability_pct": float((path_sharpe < 0).mean() * 100),
    }
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 7.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": ".25", "text.color": ".1",
        "axes.labelcolor": ".15", "xtick.color": ".2", "ytick.color": ".2",
    })
    fig, ax = plt.subplots(figsize=(8.2, 1.75), layout="constrained")
    x = np.arange(a.horizon + 1)
    segments = np.empty((a.paths, a.horizon + 1, 2), dtype=np.float32)
    segments[:, :, 0] = x
    segments[:, :, 1] = (path_equity - 1.0) * 100.0
    colors = np.where(terminal >= 0, "#24824c", "#bd4545")
    collection = LineCollection(segments, colors=colors, linewidths=.22, alpha=.018,
                                rasterized=True)
    ax.add_collection(collection)
    ax.axhline(0, color=".2", lw=.65, ls=":")
    ax.set_xlim(0, a.horizon)
    low = float(np.quantile(segments[:, :, 1], .005))
    high = float(np.quantile(segments[:, :, 1], .995))
    ax.set_ylim(min(-20, low), max(40, high))
    ax.set(title=f"All {a.paths:,} circular-block bootstrap paths",
           xlabel="Simulated trading session", ylabel="Return from $100,000 (%)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color="#24824c",lw=1,label="Ends above start"),
                       Line2D([0],[0],color="#bd4545",lw=1,label="Ends below start")],
              frameon=False, fontsize=6.2, loc="upper left", ncol=2)
    ax.grid(axis="y",color=".9",lw=.45)
    a.chart.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.chart, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
