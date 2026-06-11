"""
Round-3 Major 3: anchoring sensitivity sweep.

The kappa=1 share moves 31% -> 7% from k=1 to k=3 entry nodes; the reviewer asks whether
k=3 is a plateau or an arbitrary knob. We sweep k in {1,2,3,4,5} nearest entry nodes plus
an "all street nodes within 250 m of the snapped centroid" variant (the demand zone is a
500 m cell, so r=250 approximates anchoring to the whole cell). DA orientation, van.

kappa is monotone non-decreasing in the entry set (uncapacitated connectors), so along the
nested k-chain the per-node values must be non-decreasing; asserted as a free correctness
check. Outputs: results/ksweep_perdemand.csv, results/ksweep_summary.csv
"""
import numpy as np, pandas as pd, time, os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree
import ablations as ab

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; BIG = 1 << 24; KCAP = 50; KMAX = 5; R = 250.0
LABELS = ["1", "2", "3", "4", "5", "r250"]
os.makedirs("results", exist_ok=True)


def boot_wshare(w, mask, B=2000, seed=123):
    n = len(w)
    if n == 0 or w.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed); m = mask.astype(float); v = np.empty(B)
    for b in range(B):
        i = rng.randint(0, n, n); s = w[i].sum()
        v[b] = (w[i] * m[i]).sum() / s if s > 0 else np.nan
    return (round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3))


def main():
    perd, summ = [], []
    for c in CITIES:
        short = c.split(",")[0]; print(f"[{c}] ...", flush=True); t0 = time.time()
        u = ab.load_ulmm(c); Gm = u["graph"]; G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
        lat0 = lat.mean(); XY = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
        tree = cKDTree(XY)
        eu = np.array([idx[a] for a, b in G.edges()], np.int64)
        ev = np.array([idx[b] for a, b in G.edges()], np.int64)
        acc = [idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx]
        zn = [int(n) for n in u["demand"]["i_node"].tolist()]
        zw = u["demand"]["w"].astype(float).to_numpy()
        keep = np.array([n in idx for n in zn])
        zone_node = [n for n, m in zip(zn, keep) if m]; zone_w = zw[keep]
        unique = sorted(set(zone_node))

        # one k=5 query per node (prefixes give k=1..5); one ball query for r250
        near, ball = {}, {}
        for n in unique:
            _, nn = tree.query(XY[idx[n]], k=KMAX)
            near[n] = [int(x) for x in np.atleast_1d(nn)]
            ball[n] = [int(x) for x in tree.query_ball_point(XY[idx[n]], r=R)]

        SRC, SINK = N, N + 1

        def kappa_DA(ent):
            ent = sorted(set(ent))
            sr = list(eu) + [SRC] * len(ent) + acc
            co = list(ev) + ent + [SINK] * len(acc)
            vv = [1] * len(eu) + [BIG] * (len(ent) + len(acc))
            M = csr_matrix((np.array(vv, np.int64), (np.array(sr), np.array(co))), shape=(N + 2, N + 2))
            return int(min(maximum_flow(M, SRC, SINK).flow_value, KCAP))

        memo = {lab: {} for lab in LABELS}
        for n in unique:
            base = idx[n]; prev = -1
            for k in range(1, KMAX + 1):
                kap = kappa_DA([base] + near[n][:k])
                assert kap >= prev, f"monotonicity violated at {short} d={n} k={k}: {kap} < {prev}"
                memo[str(k)][n] = kap; prev = kap
            memo["r250"][n] = kappa_DA([base] + ball[n])

        for lab in LABELS:
            kapz = np.array([memo[lab][n] for n in zone_node]); w = zone_w
            ent_sizes = [len(set([idx[n]] + (near[n][:int(lab)] if lab != "r250" else ball[n]))) for n in unique]
            lo, hi = boot_wshare(w, kapz == 1)
            perd.append(pd.DataFrame(dict(city=short, k_label=lab, d=zone_node, w=zone_w, kappa=kapz)))
            res = dict(city=short, k_label=lab, n=len(kapz), n_entry_med=float(np.median(ent_sizes)),
                       w0=round(float(w[kapz == 0].sum() / w.sum()), 3),
                       w1=round(float(w[kapz == 1].sum() / w.sum()), 3),
                       w2=round(float(w[kapz >= 2].sum() / w.sum()), 3), w1_ci=f"[{lo},{hi}]")
            summ.append(res)
            print(f"  k={lab}: w1={res['w1']} {res['w1_ci']} |entry|med={res['n_entry_med']}", flush=True)
        pd.concat(perd).to_csv("results/ksweep_perdemand.csv", index=False)
        pd.DataFrame(summ).to_csv("results/ksweep_summary.csv", index=False)
        print(f"  [{short} done {time.time()-t0:.0f}s]", flush=True)

    allp = pd.concat(perd)
    for lab in LABELS:
        s = allp[allp.k_label == lab]; w = s.w.to_numpy(); kap = s.kappa.to_numpy()
        lo, hi = boot_wshare(w, kap == 1)
        summ.append(dict(city="POOLED", k_label=lab, n=len(s), n_entry_med=float("nan"),
                         w0=round(float(w[kap == 0].sum() / w.sum()), 3),
                         w1=round(float(w[kap == 1].sum() / w.sum()), 3),
                         w2=round(float(w[kap >= 2].sum() / w.sum()), 3), w1_ci=f"[{lo},{hi}]"))
    pd.DataFrame(summ).to_csv("results/ksweep_summary.csv", index=False)
    print("\n=== anchoring k-sweep (kappa=1 weighted share) ===", flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
