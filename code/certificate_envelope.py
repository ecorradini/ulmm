"""
R6 (panel): completeness envelope for the certificate D across generator families
and budgets. The manuscript's AUC/AP 1.00 was obtained under one friendly generator
(iterated penalized shortest paths, rho=4, K=15); reviewers require the envelope.

Generator cells (multi-entry configuration, k=3 labels from reanchor_perdemand.csv):
  ipsp_rho4   iterated penalized Dijkstra, rho=4, beta=1 (paper baseline)
  ipsp_rho125 iterated penalized Dijkstra, rho=1.25, beta=1 (weak diversification)
  yen         exact Yen K-shortest loopless paths (networkx shortest_simple_paths)
  eps10/eps25 Yen routes restricted to cost <= (1+eps) * L1 (budget-capped ensembles)
  API-like    = yen at K=3 (few, overlap-heavy alternates)
Each x K in {3, 5, 15}.

Per cell: n, coverage P(D>=2 | kappa>=2), soundness violations (must be 0),
AUC(D -> kappa>=2), AP(-D -> kappa=1), median D by kappa class.
Sample: all demands in the four smaller cities + 250 NYC (seed 7), as in the audit.

Output: results/certificate_envelope.csv
"""
import os
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

import ablations as ab
import misranking2
from misranking_certificate import max_disjoint

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; KNN = 3; EPSV = 1e-6; KCAP = 50
SAMPLE = {"New York City": 250}
KS = [3, 5, 15]
YEN_CAP = 150       # per-city cap for the exact-Yen family (disclosed; Yen is costly)
YEN_SEED = 11
os.makedirs("results", exist_ok=True)


def ipsp_routes(Mv, VS, access_set, K, pen):
    """Iterated penalized Dijkstra with configurable multiplicative penalty."""
    old = misranking2.PEN
    misranking2.PEN = pen
    try:
        routes = misranking2.k_routes(Mv, VS, access_set, K)
    finally:
        misranking2.PEN = old
    return routes


def build_yen_graph(G, access_nodes):
    """One copy per city: substrate + zero-cost sink over the access set."""
    H = G.copy()
    SINK = "__SINK__"
    H.add_node(SINK)
    for a in set(access_nodes):
        if a in H:
            H.add_edge(a, SINK, weight=0.0)
    return H, SINK


def yen_routes(H, SINK, ent_nodes, K):
    """Exact Yen loopless K shortest via a per-demand virtual source (added/removed)."""
    VS = "__VSRC__"
    for n in ent_nodes:
        H.add_edge(VS, n, weight=EPSV)
    try:
        gen = nx.shortest_simple_paths(H, VS, SINK, weight="weight")
        paths = []
        for _, p in zip(range(K), gen):
            if len(p) >= 3 and p[0] == VS and p[-1] == SINK:
                paths.append(p[1:-1])
    except nx.NetworkXNoPath:
        paths = []
    finally:
        H.remove_node(VS)
    return paths


def route_cost(G, path):
    return sum(G[a][b]["weight"] for a, b in zip(path[:-1], path[1:]))


def route_cost_idx(G, nodes, path):
    """Cost of a path given as CSR indices (IPSP output)."""
    return sum(G[nodes[a]][nodes[b]]["weight"]
               for a, b in zip(path[:-1], path[1:]))


def horizon_ipsp(L1, h1, K, rho, beta=1.0):
    """Lemma 1 a priori horizon: rho^(K-1) (L1 + (K-1) beta h1)."""
    if not np.isfinite(L1):
        return np.inf
    return (rho ** (K - 1)) * (L1 + (K - 1) * beta * h1)


def cell_stats(rows):
    df = pd.DataFrame(rows)
    y2 = (df.kappa >= 2).astype(int).to_numpy()
    y1 = (df.kappa == 1).astype(int).to_numpy()
    D = df.D.to_numpy()
    real = df.kappa.to_numpy() < KCAP
    out = dict(n=len(df),
               violations=int(((D > df.kappa.to_numpy()) & real).sum()),
               coverage=round(float((D[y2 == 1] >= 2).mean()), 4) if y2.any() else np.nan,
               auc=round(float(roc_auc_score(y2, D)), 4) if len(np.unique(y2)) > 1 else np.nan,
               ap_spof=round(float(average_precision_score(y1, -D.astype(float))), 4)
               if len(np.unique(y1)) > 1 else np.nan)
    for kv, lbl in ((1, "medD_k1"), (2, "medD_k2")):
        sel = df[df.kappa == kv]
        out[lbl] = float(np.median(sel.D)) if len(sel) else np.nan
    sel = df[(df.kappa >= 3) & (df.kappa < KCAP)]
    out["medD_k3plus"] = float(np.median(sel.D)) if len(sel) else np.nan
    return out


PERDEMAND = []


def main():
    t0 = time.time()
    rk = pd.read_csv("results/reanchor_perdemand.csv")
    rk = rk[(rk.orientation == "DA") & (rk.k == KNN)].drop_duplicates(subset=["city", "d"])
    kap_multi = {(r.city, int(r.d)): int(r.kappa) for r in rk.itertuples()}

    cells = defaultdict(list)   # (generator, K) -> per-demand rows
    for c in CITIES:
        short = c.split(",")[0]
        print(f"[{c}]", flush=True)
        t1 = time.time()
        u = ab.load_ulmm(c)
        Gm = u["graph"]
        G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        mr = np.array([idx[a] for a, b in G.edges()], np.int64)
        mc = np.array([idx[b] for a, b in G.edges()], np.int64)
        md = np.array([float(G[a][b]["weight"]) for a, b in G.edges()], float)
        lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
        XY = np.column_stack([lon * np.cos(np.radians(lat.mean())) * 111320.0, lat * 110540.0])
        tree = cKDTree(XY)
        access_set = {idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx}
        access_nodes = [int(a) for a in u["access"]["i_node"].tolist() if int(a) in idx]
        d_nodes = [int(n) for n in u["demand"]["i_node"].tolist() if int(n) in idx]
        cap = SAMPLE.get(short)
        if cap and len(d_nodes) > cap:
            d_nodes = sorted(np.random.RandomState(7).choice(d_nodes, cap, replace=False).tolist())
        labeled = [d for d in d_nodes if (short, d) in kap_multi]
        yen_set = set(labeled if len(labeled) <= YEN_CAP else
                      sorted(np.random.RandomState(YEN_SEED).choice(labeled, YEN_CAP,
                                                                    replace=False).tolist()))
        Hyen, SINK = build_yen_graph(G, access_nodes)
        VS = N
        done = 0
        for d in labeled:
            kappa = kap_multi[(short, d)]
            _, nn = tree.query(XY[idx[d]], k=KNN)
            ent = sorted(set([idx[d]] + [int(x) for x in np.atleast_1d(nn)]))
            ent_nodes = [nodes[i] for i in ent]
            rows_ = np.concatenate([mr, np.full(len(ent), VS, np.int64)])
            cols_ = np.concatenate([mc, np.array(ent, np.int64)])
            vals_ = np.concatenate([md, np.full(len(ent), EPSV, float)])
            Mv = csr_matrix((vals_, (rows_, cols_)), shape=(N + 1, N + 1))

            # IPSP families (index paths on the CSR; strip virtual source)
            for pen, gname in ((4.0, "ipsp_rho4"), (1.25, "ipsp_rho125")):
                full = [p[1:] for p in ipsp_routes(Mv, VS, access_set, max(KS), pen)
                        if len(p) > 1 and p[0] == VS]
                fcost = [route_cost_idx(G, nodes, p) for p in full]
                iL1 = min(fcost) if fcost else np.inf
                ih1 = (len(full[int(np.argmin(fcost))]) - 1) if fcost else 0
                for K in KS:
                    sub = full[:K]
                    Es = [set(zip(p[:-1], p[1:])) for p in sub]
                    Dv = max_disjoint(Es) if Es else 0
                    cells[(gname, K)].append(dict(city=short, d=d, kappa=kappa, D=Dv))
                    PERDEMAND.append(dict(
                        city=short, d=d, kappa=kappa, generator=gname, K=K, D=Dv,
                        Kprime=len(sub), L1=iL1, h1=ih1,
                        Bmax=max(fcost[:K]) if fcost[:K] else np.nan,
                        Btheory=horizon_ipsp(iL1, ih1, K, pen)))
            # Yen family on the disclosed per-city subsample (node-label paths; eps caps)
            if d in yen_set:
                yfull = yen_routes(Hyen, SINK, ent_nodes, max(KS))
                ycosts = [route_cost(G, p) for p in yfull]
                L1 = min(ycosts) if ycosts else np.inf
                yh1 = (len(yfull[int(np.argmin(ycosts))]) - 1) if ycosts else 0
                for K in KS:
                    sub = yfull[:K]
                    Es = [set(zip(p[:-1], p[1:])) for p in sub]
                    Dv = max_disjoint(Es) if Es else 0
                    cells[("yen", K)].append(dict(city=short, d=d, kappa=kappa, D=Dv))
                    PERDEMAND.append(dict(
                        city=short, d=d, kappa=kappa, generator="yen", K=K, D=Dv,
                        Kprime=len(sub), L1=L1, h1=yh1,
                        Bmax=max(ycosts[:K]) if ycosts[:K] else np.nan,
                        Btheory=max(ycosts[:K]) if ycosts[:K] else np.inf))
                    for eps, gname in ((0.10, "eps10"), (0.25, "eps25")):
                        subc = [p for p, cst in zip(yfull, ycosts) if cst <= (1 + eps) * L1][:K]
                        cc = [c for c in ycosts if c <= (1 + eps) * L1][:K]
                        Es = [set(zip(p[:-1], p[1:])) for p in subc]
                        Dv = max_disjoint(Es) if Es else 0
                        cells[(gname, K)].append(dict(city=short, d=d, kappa=kappa, D=Dv))
                        PERDEMAND.append(dict(
                            city=short, d=d, kappa=kappa, generator=gname, K=K, D=Dv,
                            Kprime=len(subc), L1=L1, h1=yh1,
                            Bmax=max(cc) if cc else np.nan,
                            Btheory=(1 + eps) * L1))
            done += 1
            if done % 100 == 0:
                print(f"  {done} demands ({time.time()-t1:.0f}s)", flush=True)
        print(f"  city done: {done} demands ({time.time()-t1:.0f}s)", flush=True)
        # checkpoint after every city
        summ = [dict(generator=g, K=K, **cell_stats(rows)) for (g, K), rows in sorted(cells.items())]
        pd.DataFrame(summ).to_csv("results/certificate_envelope.csv", index=False)

    summ = [dict(generator=g, K=K, **cell_stats(rows)) for (g, K), rows in sorted(cells.items())]
    out = pd.DataFrame(summ)
    out.to_csv("results/certificate_envelope.csv", index=False)
    pd.DataFrame(PERDEMAND).to_csv("results/certificate_envelope_perdemand.csv",
                                   index=False)
    print(out.to_string(index=False), flush=True)
    print(f"TOTAL VIOLATIONS: {int(out.violations.sum())} (must be 0)", flush=True)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
