"""Plot the adaptive, fixed-setting and cash-matched benchmark replay."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("results_dir", type=Path, nargs="?", default=Path("analysis/results/declared_portfolio"))
args = ap.parse_args()
root = args.results_dir
series = {}
for model, file, label in [
    ("Adaptive", "adaptive_5bps_daily.csv", "Adaptive ML settings"),
    ("Fixed baseline", "fixed_baseline_5bps_daily.csv", "Fixed 24 / 1.4 settings"),
]:
    d = pd.read_csv(root / file)
    series[label] = (pd.to_datetime(d.session), d.equity.to_numpy(float) / 1000)
benchmark_file = root / "spy_5bps_daily.csv"
d = pd.read_csv(benchmark_file)
series["75% SPY / 25% cash"] = (pd.to_datetime(d.session), d.equity.to_numpy(float) / 1000)

colors = {"Adaptive ML settings": "#168a66", "Fixed 24 / 1.4 settings": "#325c9a",
          "75% SPY / 25% cash": "#87929e"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, ddax) = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1]},
                                layout="constrained")
for label, (dates, values) in series.items():
    color = colors[label]
    ax.plot(dates, values, color=color, lw=1.8, label=label)
    peak = pd.Series(values).cummax().to_numpy()
    ddax.plot(dates, (values / peak - 1) * 100, color=color, lw=1.0)
    ax.annotate(f"${values[-1]:.1f}k", (dates.iloc[-1], values[-1]),
                xytext=(5, 0), textcoords="offset points", color=color,
                va="center", fontsize=8)
ax.set_title("30-minute signal portfolio: adaptive vs fixed settings",
             loc="left", fontsize=12, fontweight="bold")
ax.set_ylabel("Account equity ($000s)")
ax.legend(frameon=False, loc="upper left", ncol=3)
ax.grid(axis="y", color="#dce2e8", lw=.7)
ddax.set_ylabel("Drawdown (%)")
ddax.set_xlabel("Session date")
ddax.grid(axis="y", color="#dce2e8", lw=.7)
fig.text(.01, -.015,
         "Alpaca SIP bars; our calculations. 17-name sample, 2 Jul-9 Oct 2026 (70 sessions); 5 bp per-side costs. "
         "The short, selected sample is not a point-in-time universe or a long-term forecast.",
         fontsize=7.5, color="#505c68")
fig.savefig(root / "equity_drawdown.png", dpi=220, bbox_inches="tight")
plt.close(fig)
