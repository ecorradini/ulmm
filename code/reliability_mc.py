"""
E-REL: Monte-Carlo check of the disconnection-exponent proposition on Barcelona.

Proposition (paper): under i.i.d. edge failures with probability p, the probability that
a demand unit is disconnected from the access set satisfies P_p(disc) = Theta(p^kappa) as
p -> 0, i.e. kappa(d) is the leading-order exponent: log P / log p -> kappa(d).

Design: Barcelona substrate (smallest city), primary multi-entry anchoring (k=3, exactly
as reanchor_kappa.py), collection orientation (DA: entry set -> access). For each failure
probability p we draw B random edge-failure replicates; a zone is disconnected when no
entry node can reach any access anchor in the surviving graph (one multi-source reverse
BFS from a super-access node per replicate covers all zones at once). We then fit, per
kappa class and per zone, the slope of log P-hat vs log p.

Co-located zones (entry node coincides with an access anchor; kappa = infinity by the
paper's convention) can never disconnect and are reported as class 'inf'.

Output: results/reliability_mc.csv (long: level/kappa/p/B/events/phat + slope rows)
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
from scipy.spatial import cKDTree

import ablations as ab

CITY = "Barcelona, Spain"
VEH = "van"; KNN = 3; KCAP = 50
PGRID = [(0.10, 30_000), (0.05, 50_000), (0.02, 100_000), (0.01, 100_000), (0.005, 100_000)]
SEED = 20260719
os.makedirs("results", exist_ok=True)


def main():
    t0 = time.time()
    u = ab.load_ulmm(CITY)
    Gm = u["graph"]
    G, _, _ = ab.collapsed_graph(u, VEH)
    nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
    lat0 = lat.mean()
    XY = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
    tree = cKDTree(XY)
    eu = np.array([idx[a] for a, b in G.edges()], np.int64)
    ev = np.array([idx[b] for a, b in G.edges()], np.int64)
    m = len(eu)
    acc = np.array(sorted({idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx}), np.int64)

    zn = [int(n) for n in u["demand"]["i_node"].tolist()]
    keep = [n in idx for n in zn]
    zone_node = [n for n, kkeep in zip(zn, keep) if kkeep]
    unique = sorted(set(zone_node))
    ui = {n: i for i, n in enumerate(unique)}
    nu = len(unique)

    # entry sets exactly as reanchor_kappa.py
    ent_list = []
    for n in unique:
        _, nn = tree.query(XY[idx[n]], k=KNN)
        ent_list.append(sorted(set([idx[n]] + [int(x) for x in np.atleast_1d(nn)])))
    max_ent = max(len(e) for e in ent_list)
    ENT = np.zeros((nu, max_ent), np.int64)
    for i, e in enumerate(ent_list):
        ENT[i, :] = (e + [e[0]] * (max_ent - len(e)))  # pad with a repeat

    # kappa labels from the primary run
    rk = pd.read_csv("results/reanchor_perdemand.csv")
    rk = rk[(rk.city == "Barcelona") & (rk.orientation == "DA") & (rk.k == KNN)]
    rk = rk.drop_duplicates(subset=["d"])
    kap = {int(r.d): int(r.kappa) for r in rk.itertuples()}
    kap_u = np.array([kap.get(n, -1) for n in unique])
    print(f"zones={len(zone_node)} unique={nu} kappa classes: "
          f"{dict(pd.Series(kap_u).value_counts().sort_index())}", flush=True)

    # reversed substrate arcs (reach access <=> reached from access on reversed graph)
    # super-access node A = N with permanent arcs A -> acc
    A = N
    rev_rows_full = ev
    rev_cols_full = eu
    acc_rows = np.full(len(acc), A, np.int64)

    rng = np.random.default_rng(SEED)
    rows_out = []
    for p, B in PGRID:
        t1 = time.time()
        disc = np.zeros(nu, np.int64)
        for b in range(B):
            keep_mask = rng.random(m) >= p
            rr = np.concatenate([rev_rows_full[keep_mask], acc_rows])
            cc = np.concatenate([rev_cols_full[keep_mask], acc])
            M = csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(N + 1, N + 1))
            order = breadth_first_order(M, A, directed=True, return_predecessors=False)
            reached = np.zeros(N + 1, bool)
            reached[order] = True
            disc += ~reached[ENT].any(axis=1)
        for i in range(nu):
            rows_out.append(dict(level="zone", node=unique[i], kappa=int(kap_u[i]),
                                 p=p, B=B, events=int(disc[i]), phat=disc[i] / B))
        # class aggregates (real kappa classes only; KCAP = co-located = 'inf')
        for kv in sorted(set(kap_u)):
            sel = kap_u == kv
            ev_c = int(disc[sel].sum()); n_c = int(sel.sum())
            label = "inf" if kv >= KCAP else str(kv)
            rows_out.append(dict(level="class", node=-1, kappa=label,
                                 p=p, B=B, events=ev_c, phat=ev_c / (B * n_c)))
        cls = {("inf" if kv >= KCAP else str(kv)): disc[kap_u == kv].sum() / (B * (kap_u == kv).sum())
               for kv in sorted(set(kap_u))}
        print(f"p={p} B={B}: class P-hat {cls} ({time.time()-t1:.0f}s)", flush=True)
        pd.DataFrame(rows_out).to_csv("results/reliability_mc.csv", index=False)

    df = pd.DataFrame(rows_out)
    # slope fits: class level, using points with >=10 events
    for kv in sorted(df[df.level == "class"].kappa.unique(), key=str):
        sub = df[(df.level == "class") & (df.kappa == kv) & (df.events >= 10)]
        if len(sub) >= 2 and kv not in ("inf", "0", "-1"):
            x = np.log(sub.p.to_numpy()); y = np.log(sub.phat.to_numpy())
            A_ = np.vstack([x, np.ones_like(x)]).T
            slope, _ = np.linalg.lstsq(A_, y, rcond=None)[0]
            resid = y - A_ @ np.linalg.lstsq(A_, y, rcond=None)[0]
            se = float(np.sqrt(resid.var() / max(len(x) - 2, 1) / x.var())) if len(x) > 2 else np.nan
            rows_out.append(dict(level="slope_class", node=-1, kappa=kv, p=np.nan,
                                 B=len(sub), events=-1, phat=round(float(slope), 3)))
            print(f"class kappa={kv}: fitted exponent {slope:.3f} (se~{se:.3f}, {len(sub)} p-points)", flush=True)
    # zone-level slopes for kappa in {1,2}
    for kv in (1, 2):
        zs = []
        for n in df[(df.level == "zone") & (df.kappa == kv)].node.unique():
            sub = df[(df.level == "zone") & (df.node == n) & (df.events >= 10)]
            if len(sub) >= 3:
                x = np.log(sub.p.to_numpy()); y = np.log((sub.events / sub.B).to_numpy())
                A_ = np.vstack([x, np.ones_like(x)]).T
                zs.append(float(np.linalg.lstsq(A_, y, rcond=None)[0][0]))
        if zs:
            rows_out.append(dict(level="slope_zone_median", node=-1, kappa=str(kv), p=np.nan,
                                 B=len(zs), events=-1, phat=round(float(np.median(zs)), 3)))
            print(f"zone-level kappa={kv}: median fitted exponent {np.median(zs):.3f} "
                  f"(IQR {np.percentile(zs,25):.2f}-{np.percentile(zs,75):.2f}, n={len(zs)} zones)", flush=True)
    pd.DataFrame(rows_out).to_csv("results/reliability_mc.csv", index=False)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
