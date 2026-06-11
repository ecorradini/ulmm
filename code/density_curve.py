"""
Round-2 #2 / Round-3 Minor 4: per-city access-density sweep, 0.5x-10x, under the primary
k=3 multi-entry anchoring. For f<=1 we subsample the real access set; for f>1 we add
synthetic access points drawn from commercially plausible street nodes: candidates are
sampled with probability proportional to the POI weight of the nearest demand cell (within
500 m), so densification lands where lockers and pickup points could plausibly be sited,
not on arbitrary street nodes. If the kappa=1 share is flat WITHIN each city while the
cross-city ordering tracks access proximity, the binding constraint is local network
structure, not the access count. DA orientation, van.
"""
import numpy as np, pandas as pd, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree
import ablations as ab

VEH = "van"; BIG = 1 << 24; KCAP = 50; KNN = 3
FRACS = [0.5, 1.0, 2.0, 5.0, 10.0]
CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
os.makedirs("results", exist_ok=True)


def kappa_shares(N, eu, ev, acc_idx, entry, zone_node, zone_w):
    SRC, SINK = N, N + 1; memo = {}
    for n in set(zone_node):
        ent = entry[n]
        sr = list(eu) + [SRC] * len(ent) + acc_idx; co = list(ev) + ent + [SINK] * len(acc_idx)
        vv = [1] * len(eu) + [BIG] * (len(ent) + len(acc_idx))
        M = csr_matrix((np.array(vv, np.int64), (np.array(sr), np.array(co))), shape=(N + 2, N + 2))
        memo[n] = int(min(maximum_flow(M, SRC, SINK).flow_value, KCAP))
    k = np.array([memo[n] for n in zone_node]); w = zone_w; t = w.sum()
    return float(w[k == 1].sum() / t), float(w[k >= 2].sum() / t)


def main():
    rows = []
    for c in CITIES:
        short = c.split(",")[0]; print(f"[{c}]", flush=True)
        u = ab.load_ulmm(c); Gm = u["graph"]; G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
        lat0 = lat.mean(); XY = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
        tree = cKDTree(XY)
        eu = np.array([idx[a] for a, b in G.edges()], np.int64); ev = np.array([idx[b] for a, b in G.edges()], np.int64)
        acc_all = [int(a) for a in u["access"]["i_node"].tolist() if int(a) in idx]
        # synthetic-access candidates: street nodes carrying edges, excluding existing access,
        # sampled proportionally to the POI weight of the nearest demand cell (within 500 m)
        cand = [n for n in nodes if G.out_degree(n) >= 1 and n not in set(acc_all)]
        zn = [int(n) for n in u["demand"]["i_node"].tolist()]; zw = u["demand"]["w"].astype(float).to_numpy()
        keep = np.array([n in idx for n in zn]); zone_node = [n for n, m in zip(zn, keep) if m]; zone_w = zw[keep]
        ztree = cKDTree(np.array([XY[idx[n]] for n in zone_node]))
        cd, cj = ztree.query(np.array([XY[idx[n]] for n in cand]))
        cprob = np.where(cd <= 500.0, np.maximum(zone_w[cj], 1e-9), 1e-9)
        cprob = cprob / cprob.sum()
        entry = {}
        for n in set(zone_node):
            _, nn = tree.query(XY[idx[n]], k=KNN); entry[n] = sorted(set([idx[n]] + [int(x) for x in np.atleast_1d(nn)]))
        rng = np.random.RandomState(11); base = len(acc_all)
        for f in FRACS:
            if f <= 1.0:
                sub = acc_all if f == 1.0 else list(rng.choice(acc_all, max(1, int(round(f * base))), replace=False))
            else:
                extra = min(len(cand), int(round((f - 1.0) * base)))
                pick = rng.choice(len(cand), extra, replace=False, p=cprob)
                sub = acc_all + [cand[int(i)] for i in pick]
            acc_idx = [idx[a] for a in sub]
            w1, w2 = kappa_shares(N, eu, ev, acc_idx, entry, zone_node, zone_w)
            rows.append(dict(city=short, frac=f, n_access=len(sub), w1=round(w1, 3), w2=round(w2, 3)))
            print(f"  {f}x |A|={len(sub)}: kappa1={w1:.3f}", flush=True)
        pd.DataFrame(rows).to_csv("results/density_curve.csv", index=False)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    mk = {"Amsterdam": "o", "Barcelona": "s", "Paris": "^", "Seattle": "D", "New York City": "v"}
    for short in mk:
        d = df[df.city == short]
        ax.plot(d.frac, d.w1, marker=mk[short], label=short, linewidth=1.4, markersize=5)
    ax.set_xscale("log"); ax.set_xticks(FRACS); ax.set_xticklabels([f"{f:g}x" for f in FRACS])
    ax.set_xlabel("Access-set size relative to the modeled inventory", fontsize=9)
    ax.set_ylabel(r"Weighted share with $\kappa=1$", fontsize=9); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, ncol=2); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("../paper/fig-kappa-density.png", dpi=200, bbox_inches="tight")
    print("\n", df.to_string(index=False)); print("saved fig-kappa-density.png\nDONE", flush=True)


if __name__ == "__main__":
    main()
