"""Plot deterministic account losses at the declared 75% stock exposure cap."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", type=Path, default=Path("analysis/results/declared_risk/deterministic_shocks.csv"))
ap.add_argument("--output", type=Path, default=Path("docs/images/deterministic_75pct_shock.png"))
a = ap.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(a.results)
d = d[d.exposure.round(4) == 0.75].sort_values("stock_loss_pct")
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": ".25", "xtick.color": ".2", "ytick.color": ".2",
    "axes.labelcolor": ".15", "text.color": ".1",
})
fig, ax = plt.subplots(figsize=(8.2, 2.5), layout="constrained")
ax.plot(d.stock_loss_pct, d.account_loss_pct, color="#bd4545", marker="o", lw=1.6,
        label="Account loss at 75% stock exposure")
ax.axhline(6, color="#24824c", ls="--", lw=1.2, label="6% account halt trigger")
ax.set_title("Simultaneous stock-book shocks at the 75% exposure cap", loc="left", fontsize=10)
ax.set_xlabel("Common stock-price decline (%)")
ax.set_ylabel("Account loss (%)")
ax.set_xticks(d.stock_loss_pct)
ax.grid(axis="y", color=".88", lw=.55)
ax.legend(frameon=False, loc="upper left", fontsize=7)
fig.savefig(a.output, dpi=300, bbox_inches="tight")
plt.close(fig)
