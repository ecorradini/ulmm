"""
Round-2 #1 (map): NYC single-points-of-failure (the kappa=1 cut edges) overlaid on the
top-decile demand-weighted betweenness (DEB_z) hotspots. Shows the cuts are real streets,
dispersed across the city and away from the busy betweenness hotspots (orthogonality, visual).
"""
import numpy as np, pandas as pd, pickle, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ablations as ab

# cut edges (kappa=1) for NYC
cc = pd.read_csv("results/cut_chars.csv")
nyc = cc[(cc.city == "New York City") & (cc.cut_u >= 0) & (cc.cut_v >= 0)]
u = ab.load_ulmm("New York City, New York, USA"); Gm = u["graph"]
def xy(n):
    return (Gm.nodes[int(n)]["x"], Gm.nodes[int(n)]["y"]) if int(n) in Gm.nodes else (None, None)

# betweenness hotspots: top-decile DEB_z segment centroids
seg = pickle.load(open("cache_nyc_ulmm/segments_fe.pkl", "rb"))
thr = seg["DEB_z"].quantile(0.90)
hot = seg[seg["DEB_z"] >= thr].copy()
try:
    import geopandas as gpd  # noqa
    if getattr(hot, "crs", None) is not None and (hot.crs.to_epsg() or 0) != 4326:
        hot = hot.to_crs(4326)
    hx = np.array([g.x for g in hot.geometry if g is not None], float)
    hy = np.array([g.y for g in hot.geometry if g is not None], float)
    # guard: if coords are clearly projected (not lon/lat), drop the layer
    if hx.size and (np.nanmax(np.abs(hx)) > 360 or np.nanmax(np.abs(hy)) > 90):
        hx = hy = np.array([], float)
except Exception:
    hx = hy = np.array([], float)

fig, ax = plt.subplots(figsize=(6.0, 6.4))
ax.scatter(hx, hy, s=0.6, c="#9ecae1", alpha=0.35, linewidths=0, label="Top-decile betweenness (DEB) hotspots")
n_drawn = 0
for r in nyc.itertuples():
    x0, y0 = xy(r.cut_u); x1, y1 = xy(r.cut_v)
    if x0 is None or x1 is None:
        continue
    ax.plot([x0, x1], [y0, y1], "-", color="#d6201f", linewidth=2.4, solid_capstyle="round")
    n_drawn += 1
# proxy legend handle for SPOF
ax.plot([], [], "-", color="#d6201f", linewidth=2.4, label=f"Single-point-of-failure cut edges ($\\kappa{{=}}1$), n={n_drawn}")
ax.set_aspect(1.0 / np.cos(np.radians(40.71)))   # NYC latitude
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=8, frameon=False, markerscale=6)
plt.tight_layout()
plt.savefig("../paper/fig-spof-map.png", dpi=200, bbox_inches="tight")
print(f"drew {n_drawn} SPOF cut edges over {len(hot)} hotspot segments; saved fig-spof-map.png")
