"""Plot 15-session Monte Carlo expected shortfall at the 75% exposure cap."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, default=Path("analysis/results/calibrated/calibrated_scenarios.csv"))
ap.add_argument("--output", type=Path, default=Path("docs/images/monte_carlo_es75.png"))
a = ap.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(a.results)
d = d[d.exposure.round(4) == 0.75].copy()
labels = {
    "Calibrated Gaussian": "Gaussian",
    "Student-t(5)": "Student-t (5)",
    "5-session block bootstrap": "5-session block bootstrap",
    "Crisis: vol x1.5, rho .90": "Crisis: vol x1.5, rho 0.90",
    "20% common opening gap": "20% common opening gap",
}
d["label"] = d.model.map(labels)
d = d.iloc[::-1]
colors = ["#24824c" if x == "Calibrated Gaussian" else "#bd4545" if x == "20% common opening gap" else ".62" for x in d.model]

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 7.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": ".25", "xtick.color": ".2", "ytick.color": ".2",
    "axes.labelcolor": ".15", "text.color": ".1",
})
fig, ax = plt.subplots(figsize=(8.2, 1.45), layout="constrained")
bars = ax.barh(d.label, d.es95_pct, color=colors, height=.62)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=7)
ax.set_title("15-session expected shortfall at 75% exposure", loc="left", fontsize=9)
ax.set_xlabel("Average account loss in worst 5% of simulated outcomes (%)", fontsize=7.2)
ax.set_xlim(0, max(d.es95_pct) * 1.18)
ax.grid(axis="x", color=".88", lw=.5)
ax.set_axisbelow(True)
fig.savefig(a.output, dpi=300, bbox_inches="tight")
plt.close(fig)
