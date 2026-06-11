"""Figure: weighted access-redundancy distribution per city (AD), stacked bar."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

df = pd.read_csv("results/reanchor_perdemand.csv")
df = df[(df.orientation == "AD") & (df.k == 3)]   # multi-entry primary
order = ["Amsterdam", "Barcelona", "Paris", "Seattle", "New York City"]
rows = []
for c in order:
    s = df[df.city == c]; w = s.w.to_numpy(float); k = s.kappa.to_numpy()
    tot = w.sum()
    rows.append([w[k == 0].sum()/tot, w[k == 1].sum()/tot, w[k >= 2].sum()/tot])
A = np.array(rows)

fig, ax = plt.subplots(figsize=(7.2, 3.1))
x = np.arange(len(order))
c0, c1, c2 = "#bdbdbd", "#d6604d", "#4393c3"
ax.bar(x, A[:, 0], color=c0, label=r"$\kappa=0$ (unreachable)", edgecolor="white", linewidth=0.5)
ax.bar(x, A[:, 1], bottom=A[:, 0], color=c1, label=r"$\kappa=1$ (single point of failure)", edgecolor="white", linewidth=0.5)
ax.bar(x, A[:, 2], bottom=A[:, 0]+A[:, 1], color=c2, label=r"$\kappa\geq2$ (edge-redundant)", edgecolor="white", linewidth=0.5)
for i in range(len(order)):
    if A[i, 1] > 0.05:
        ax.text(i, A[i, 0]+A[i, 1]/2, f"{A[i,1]*100:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9)
ax.set_ylabel("Weighted demand share", fontsize=9)
ax.set_ylim(0, 1.0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("../paper/fig-kappa-dist.png", dpi=200, bbox_inches="tight")
print("saved ../paper/fig-kappa-dist.png")
