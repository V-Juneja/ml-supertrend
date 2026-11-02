"""Plot the archived ML Supertrend replay; --bars enables its covariance chart."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, default=Path("analysis/results/archived_replay"))
ap.add_argument("--output", type=Path, default=Path("docs/images"))
ap.add_argument("--bars", type=Path, help="the exact split-adjusted provider bars used by the archived replay")
a = ap.parse_args()
a.output.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": ".25", "xtick.color": ".2", "ytick.color": ".2",
    "axes.labelcolor": ".15", "text.color": ".1",
})
d = pd.read_csv(a.results / "10bps_daily.csv", parse_dates=["date"])
b = pd.read_csv(a.results / "qqq_normalized_equity.csv", parse_dates=["date"])
fig, (ax, ddax) = plt.subplots(2, 1, figsize=(8.2, 3.35), sharex=True,
                                gridspec_kw={"height_ratios": [2.1, 1]},
                                layout="constrained")
green = "#24824c"
red = "#bd4545"
ax.plot(d.date, d.total_value / 1000, color=green, lw=1.5,
       label="ML Supertrend BUY-email replay")
ax.plot(b.date, b.value / 1000, color=red, lw=1.2, ls="--",
       label="QQQ benchmark")
for frame, col, color, style in [(d, "total_value", green, "-"),
                                 (b, "value", red, "--")]:
    value = frame[col].to_numpy(float)
    drawdown = (value / np.maximum.accumulate(value) - 1) * 100
    dates = frame.date
    ddax.plot(dates, drawdown, color=color, lw=.9, ls=style)
ax.set_title("ML Supertrend archived replay and QQQ", loc="left", fontsize=10)
ax.set_ylabel("Equity ($000s)")
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=7)
ax.grid(axis="y", color=".88", lw=.55)
ddax.set_ylabel("Drawdown (%)")
ddax.set_xlabel("Session date")
ddax.grid(axis="y", color=".88", lw=.55)
fig.savefig(a.output / "archived_replay_equity.png", dpi=300, bbox_inches="tight")
plt.close(fig)

m = pd.read_csv(a.results / "actual_monthly_returns.csv")
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
pivot = m.pivot(index="year", columns="month", values="return").reindex(index=[2025, 2026], columns=range(1, 13)) * 100
fig, ax = plt.subplots(figsize=(8.2, 1.38), layout="constrained")
values = np.ma.masked_invalid(pivot.to_numpy(float))
rg = LinearSegmentedColormap.from_list("red_white_green", [red, "#ffffff", green])
ax.imshow(values, cmap=rg, norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20), aspect="auto")
ax.set_xticks(np.arange(12), months)
ax.set_yticks(np.arange(2), [str(y) for y in pivot.index])
ax.set_title("Monthly net returns (%)", loc="left", fontsize=9, pad=4)
ax.tick_params(length=0, labelsize=7)
ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        if np.ma.is_masked(values[i, j]):
            continue
        v = float(values[i, j])
        label = f"{v:+.1f}"
        ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) > 11 else ".1")
fig.savefig(a.output / "archived_replay_monthly.png", dpi=300, bbox_inches="tight")
plt.close(fig)

if a.bars:
    from sklearn.covariance import LedoitWolf

    provenance = json.loads((a.results / "provenance.json").read_text())
    bars_hash = hashlib.sha256(a.bars.read_bytes()).hexdigest()
    if bars_hash != provenance["bars_sha256"]:
        raise ValueError("Bars do not match the archived replay's recorded input hash")
    bars = pd.read_csv(a.bars, parse_dates=["date"])
    fills = pd.read_csv(a.results / "10bps_fills.csv", parse_dates=["date"])
    buys = fills.loc[fills.action.eq("BUY"), "symbol"].value_counts()
    replay_dates = pd.to_datetime(d.date)
    close = bars.pivot(index="date", columns="symbol", values="close")
    close = close.reindex(replay_dates)
    # Use the eight most frequently entered names with complete prices for the
    # entire archived replay. IDs follow entry-frequency order; no stock names
    # are printed on the chart.
    selected = [s for s in buys.index if s in close and close[s].notna().all()][:8]
    if len(selected) != 8:
        raise ValueError("Could not find eight replay assets with complete aligned prices")
    returns = close[selected].pct_change().iloc[1:]
    estimator = LedoitWolf(store_precision=False).fit(returns.to_numpy(float))
    covariance = estimator.covariance_
    if not np.isfinite(covariance).all():
        raise ValueError("Ledoit-Wolf covariance estimate contains non-finite values")
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)

    daily_returns = d.total_value.pct_change()
    daily_returns.iloc[0] = d.total_value.iloc[0] / 100000 - 1
    daily_returns = daily_returns.to_numpy(float) * 100
    fig, (rax, ax) = plt.subplots(1, 2, figsize=(8.2, 1.68),
                                  gridspec_kw={"width_ratios": [1.1, 1.8]},
                                  layout="constrained")
    bins = np.linspace(np.floor(daily_returns.min()), np.ceil(daily_returns.max()), 15)
    rax.hist(daily_returns[daily_returns < 0], bins=bins, color=red, alpha=.88,
             edgecolor="white", linewidth=.35, label="Loss sessions")
    rax.hist(daily_returns[daily_returns >= 0], bins=bins, color=green, alpha=.88,
             edgecolor="white", linewidth=.35, label="Gain sessions")
    rax.axvline(0, color=".2", lw=.65, ls=":")
    rax.set_title("Replay daily returns", loc="left", fontsize=8)
    rax.set_xlabel("Return (%)", fontsize=6.5)
    rax.set_ylabel("Sessions", fontsize=6.5)
    rax.grid(axis="y", color=".9", lw=.4)
    rax.tick_params(labelsize=5.8, length=0)
    rax.legend(frameon=False, fontsize=5.7, loc="upper right")
    rg = LinearSegmentedColormap.from_list("red_white_green", [red, "#ffffff", green])
    heat = ax.imshow(correlation, cmap=rg, vmin=-1, vmax=1, aspect="auto")
    labels = [f"{i+1}" for i in range(len(selected))]
    ax.set_xticks(np.arange(len(selected)), labels)
    ax.set_yticks(np.arange(len(selected)), labels)
    ax.set_xlabel("Asset ID (ordered by replay entry count)", fontsize=6.8)
    ax.set_ylabel("Asset ID", fontsize=6.8)
    ax.set_title("Ledoit–Wolf shrinkage correlation", loc="left", fontsize=8)
    ax.set_xticks(np.arange(-.5, len(selected), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(selected), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False, length=0)
    ax.tick_params(labelsize=6.3, length=0)
    for i in range(len(selected)):
        for j in range(len(selected)):
            value = float(correlation[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.7,
                    color="white" if abs(value) > .65 else ".12")
    colorbar = fig.colorbar(heat, ax=ax, fraction=.025, pad=.025)
    colorbar.set_label("Correlation", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.8, length=0)
    fig.savefig(a.output / "ledoit_wolf_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
