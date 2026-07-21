"""
R10 (panel): anchor-densification test. Under plausible ADDITION of unmapped access
points (+10% and +20% synthetic anchors, sited as in density_curve.py: street nodes
sampled proportionally to nearby demand weight), how many currently kappa=1 zones flip
to kappa>=2? Complements the removal-direction sweep with the direction that matters
for OSM-inventory incompleteness.

Output: results/densify_test.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree

import ablations as ab

VEH = "van"; BIG = 1 << 24; KCAP = 50; KNN = 3
FRACS = [1.1, 1.2]
CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
os.makedirs("results", exist_ok=True)


def kappa_per_zone(N, eu, ev, acc_idx, entry, zone_set):
    SRC, SINK = N, N + 1
    memo = {}
    for n in zone_set:
        ent = entry[n]
        sr = list(eu) + [SRC] * len(ent) + acc_idx
        co = list(ev) + ent + [SINK] * len(acc_idx)
        vv = [1] * len(eu) + [BIG] * (len(ent) + len(acc_idx))
        M = csr_matrix((np.array(vv, np.int64), (np.array(sr), np.array(co))),
                       shape=(N + 2, N + 2))
        memo[n] = int(min(maximum_flow(M, SRC, SINK).flow_value, KCAP))
    return memo


def main():
    rows = []
    for c in CITIES:
        short = c.split(",")[0]
        print(f"[{c}]", flush=True)
        u = ab.load_ulmm(c)
        Gm = u["graph"]
        G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
        XY = np.column_stack([lon * np.cos(np.radians(lat.mean())) * 111320.0, lat * 110540.0])
        tree = cKDTree(XY)
        eu = np.array([idx[a] for a, b in G.edges()], np.int64)
        ev = np.array([idx[b] for a, b in G.edges()], np.int64)
        acc_all = [int(a) for a in u["access"]["i_node"].tolist() if int(a) in idx]
        cand = [n for n in nodes if G.out_degree(n) >= 1 and n not in set(acc_all)]
        zn = [int(n) for n in u["demand"]["i_node"].tolist()]
        zw = u["demand"]["w"].astype(float).to_numpy()
        keep = np.array([n in idx for n in zn])
        zone_node = [n for n, m in zip(zn, keep) if m]; zone_w = zw[keep]
        ztree = cKDTree(np.array([XY[idx[n]] for n in zone_node]))
        cd, cj = ztree.query(np.array([XY[idx[n]] for n in cand]))
        cprob = np.where(cd <= 500.0, np.maximum(zone_w[cj], 1e-9), 1e-9)
        cprob = cprob / cprob.sum()
        entry = {}
        for n in set(zone_node):
            _, nn = tree.query(XY[idx[n]], k=KNN)
            entry[n] = sorted(set([idx[n]] + [int(x) for x in np.atleast_1d(nn)]))

        base_memo = kappa_per_zone(N, eu, ev, [idx[a] for a in acc_all], entry, set(zone_node))
        k1_zones = [n for n in set(zone_node) if base_memo[n] == 1]
        kz = np.array([base_memo[n] for n in zone_node])
        base_w1 = float(zone_w[kz == 1].sum() / zone_w.sum())
        rng = np.random.RandomState(11)
        base = len(acc_all)
        for f in FRACS:
            extra = min(len(cand), int(round((f - 1.0) * base)))
            pick = rng.choice(len(cand), extra, replace=False, p=cprob)
            sub = acc_all + [cand[int(i)] for i in pick]
            acc_idx = [idx[a] for a in sub]
            # only kappa=1 zones can flip upward; recompute those
            memo2 = kappa_per_zone(N, eu, ev, acc_idx, entry, set(k1_zones))
            flips = sum(1 for n in k1_zones if memo2[n] >= 2)
            kz2 = np.array([memo2.get(n, base_memo[n]) for n in zone_node])
            new_w1 = float(zone_w[(kz2 == 1)].sum() / zone_w.sum())
            rows.append(dict(city=short, frac=f, n_extra=extra, n_k1_zones=len(k1_zones),
                             flips=flips, flip_share=round(flips / max(len(k1_zones), 1), 3),
                             w1_before=round(base_w1, 3), w1_after=round(new_w1, 3)))
            print(f"  {f}x: +{extra} anchors, {flips}/{len(k1_zones)} kappa=1 zones flip; "
                  f"w1 {base_w1:.3f}->{new_w1:.3f}", flush=True)
        pd.DataFrame(rows).to_csv("results/densify_test.csv", index=False)
    df = pd.DataFrame(rows)
    for f in FRACS:
        s = df[df.frac == f]
        print(f"pooled flip share at {f}x: {s.flips.sum()}/{s.n_k1_zones.sum()} = "
              f"{s.flips.sum()/max(s.n_k1_zones.sum(),1):.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
