"""Plot the archived BUY-ranking replay and its monthly returns in grayscale."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, default=Path("analysis/results/archived_replay"))
ap.add_argument("--output", type=Path, default=Path("docs/images"))
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

# Net trade-return distribution from the same archived replay ledger.
trades = pd.read_csv(a.results / "closed_trade_expectancy.csv")
r = np.sort(trades.net_return_pct.to_numpy(float))
share = np.arange(1, len(r) + 1) / len(r) * 100
median = float(np.median(r))
fig, ax = plt.subplots(figsize=(8.2, 1.42), layout="constrained")
negative = r <= 0
positive = r >= 0
ax.step(r[negative], share[negative], where="post", color=red, lw=1.5, label="Net losses")
ax.step(r[positive], share[positive], where="post", color=green, lw=1.5, label="Net gains")
ax.axvline(0, color=".35", lw=.75, ls=":")
ax.axvline(median, color=".35", lw=.8, ls="--", label=f"Median {median:.2f}%")
ax.set_title("Cumulative distribution of completed-trade returns", loc="left", fontsize=9)
ax.set_xlabel("Net return per trade (%)", fontsize=7.2)
ax.set_ylabel("Trades at or below (%)", fontsize=7.2)
ax.set_xlim(r.min(), r.max())
ax.set_ylim(0, 100)
ax.grid(axis="y", color=".88", lw=.5)
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=6.4)
fig.savefig(a.output / "trade_return_cdf.png", dpi=300, bbox_inches="tight")
plt.close(fig)
