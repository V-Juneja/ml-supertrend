"""Block-bootstrap the archived daily ML Supertrend BUY-email replay."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    blocks_per_path = (a.horizon + a.block - 1) // a.block
    batch_size = 1_000
    for first in range(0, a.paths, batch_size):
        n = min(batch_size, a.paths - first)
        starts = rng.integers(0, len(returns), size=(n, blocks_per_path))
        indices = (starts[:, :, None] + np.arange(a.block)) % len(returns)
        sampled = returns[indices.reshape(n, -1)[:, :a.horizon]]
        wealth = np.cumprod(1.0 + sampled, axis=1)
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
    }
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 7.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": ".25", "text.color": ".1",
        "axes.labelcolor": ".15", "xtick.color": ".2", "ytick.color": ".2",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 1.75), layout="constrained")
    axes[0].hist(terminal * 100, bins=54, color=".67", edgecolor="white", linewidth=.25)
    axes[0].axvline(p05 * 100, color="#bd4545", lw=1.15, label=f"5th percentile {p05*100:.1f}%")
    axes[0].axvline(median * 100, color="#24824c", lw=1.15, ls="--", label=f"Median {median*100:.1f}%")
    axes[0].set(title="One-year terminal return", xlabel="Return (%)", ylabel="Paths")
    axes[0].legend(frameon=False, fontsize=6.2)
    dd95 = -np.quantile(max_drawdown, .05) * 100
    axes[1].hist(-max_drawdown * 100, bins=45, color=".67", edgecolor="white", linewidth=.25)
    axes[1].axvline(dd95, color="#bd4545", lw=1.15, label=f"95th percentile loss {dd95:.1f}%")
    axes[1].set(title="Maximum drawdown within path", xlabel="Peak-to-trough loss (%)", ylabel="Paths")
    axes[1].legend(frameon=False, fontsize=6.2)
    a.chart.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.chart, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
