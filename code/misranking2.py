"""
E2 (fast, scipy): do route-diversity / redundancy indices recover exact access
redundancy kappa?  For a sample of demands per city we build K near-optimal routes
to the access set by iterated penalized Dijkstra (C-speed; bounded, no Yen blow-ups),
compute the diversity indices on them, and compare to kappa (from E1, DA orientation):

  e^RPE  = exp(Shannon entropy of Gibbs weights over the K routes)   [entropy effective #]
  RPE'   = 1 / sum_ij p_i p_j Jaccard(E_i,E_j)                       [similarity-corrected #]
  Li_div = mean pairwise edge-DISSIMILARITY (1 - mean Jaccard)       [route-diversity index]
  Deng_pm= number of distinct routes found                           [path multiplicity]

Headline: ROC-AUC of each index for predicting genuine redundancy (kappa>=2) and
Spearman(index, kappa).  If the indices are miscalibrated (Prop. 3.x), AUC ~ 0.5.
"""
import numpy as np, pandas as pd, time, os
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import ablations as ab

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; K = 15; ETA = 1.0; PEN = 4.0
SAMPLE = {"New York City": 250}      # per-city demand cap; others use all
os.makedirs("results", exist_ok=True)


def build_csr(G):
    nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    eu = np.array([idx[u] for u, v in G.edges()], np.int64)
    ev = np.array([idx[v] for u, v in G.edges()], np.int64)
    w = np.array([float(G[u][v]["weight"]) for u, v in G.edges()], float)
    M = csr_matrix((w, (eu, ev)), shape=(N, N))
    return M, idx, N


def k_routes(M, src, access_set, K):
    """K near-optimal src->(nearest access) routes by iterated penalized Dijkstra.
    Returns list of routes as node-index lists; lengths use the ORIGINAL weights."""
    data = M.data.copy(); indptr = M.indptr; indices = M.indices
    N = M.shape[0]; acc = np.fromiter(access_set, np.int64)
    routes = []
    for _ in range(K):
        Mw = csr_matrix((data, indices, indptr), shape=(N, N))
        dist, pred = dijkstra(Mw, indices=src, return_predecessors=True)
        da = dist[acc]
        if not np.isfinite(da).any():
            break
        best = int(acc[np.argmin(da)])
        path = []; v = best
        while v != src and v >= 0:
            path.append(v); v = pred[v]
        if v != src:
            break
        path.append(src); path.reverse()
        routes.append(path)
        for a_, b_ in zip(path[:-1], path[1:]):
            for jj in range(indptr[a_], indptr[a_ + 1]):
                if indices[jj] == b_:
                    data[jj] = data[jj] * PEN + 1.0; break
    return routes


def measures(routes, Morig):
    n = len(routes)
    if n < 1:
        return None
    Es = [set(zip(p[:-1], p[1:])) for p in routes]
    # original lengths
    lens = []
    for p in routes:
        L = 0.0
        for a_, b_ in zip(p[:-1], p[1:]):
            L += Morig[a_, b_]
        lens.append(L)
    lens = np.array(lens, float)
    pr = np.exp(-ETA * (lens - lens.min())); pr /= pr.sum()
    H = -float((pr * np.log(np.clip(pr, 1e-12, 1))).sum())
    if n == 1:
        Deff = 1.0; meanJ = 1.0
    else:
        quad = float((pr * pr).sum()); Js = []
        for i, j in combinations(range(n), 2):
            uni = len(Es[i] | Es[j]); jac = (len(Es[i] & Es[j]) / uni) if uni else 0.0
            quad += 2 * pr[i] * pr[j] * jac; Js.append(jac)
        Deff = 1.0 / quad if quad > 0 else 1.0
        meanJ = float(np.mean(Js))
    return dict(expH=float(np.exp(H)), rpe_prime=Deff, Li_div=1.0 - meanJ, Deng_pm=float(n))


def auc_safe(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def main():
    kp = pd.read_csv("results/kappa_perdemand.csv")
    kap = {(r.city, int(r.d)): int(r.kappa) for r in kp[kp.orientation == "DA"].itertuples()}
    rows, perd = [], []
    for c in CITIES:
        short = c.split(",")[0]; print(f"[{c}] ...", flush=True); t0 = time.time()
        u = ab.load_ulmm(c); G, _, _ = ab.collapsed_graph(u, VEH)
        M, idx, N = build_csr(G)
        Mlil = M.tolil()   # fast element access for lengths
        d_nodes = [int(n) for n in u["demand"]["i_node"].tolist() if int(n) in idx]
        cap = SAMPLE.get(short)
        if cap and len(d_nodes) > cap:
            d_nodes = sorted(np.random.RandomState(7).choice(d_nodes, cap, replace=False).tolist())
        access_set = {idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx}
        recs = []
        for d in d_nodes:
            if (short, d) not in kap:
                continue
            routes = k_routes(M, idx[d], access_set, K)
            m = measures(routes, Mlil)
            if m is None:
                continue
            m.update(city=short, d=d, kappa=kap[(short, d)]); recs.append(m)
        df = pd.DataFrame(recs); perd.append(df)
        print(f"  n={len(df)} ({time.time()-t0:.1f}s) frac(kappa>=2)={float((df.kappa>=2).mean()):.3f}", flush=True)
        if len(df) >= 10:
            y = (df.kappa >= 2).astype(int)
            res = dict(city=short, n=len(df), frac_redund=round(float(y.mean()), 3))
            for col in ["expH", "rpe_prime", "Li_div", "Deng_pm"]:
                res[f"auc_{col}"] = round(auc_safe(y, df[col].to_numpy()), 3)
                res[f"sp_{col}"] = round(float(spearmanr(df[col], df.kappa).correlation), 3)
            rows.append(res); print("  ", {k: v for k, v in res.items() if k.startswith("auc")}, flush=True)
        pd.concat(perd).to_csv("results/misranking_perdemand.csv", index=False)
        pd.DataFrame(rows).to_csv("results/misranking_summary.csv", index=False)
    allp = pd.concat(perd); y = (allp.kappa >= 2).astype(int)
    pooled = dict(city="POOLED", n=len(allp), frac_redund=round(float(y.mean()), 3))
    for col in ["expH", "rpe_prime", "Li_div", "Deng_pm"]:
        pooled[f"auc_{col}"] = round(auc_safe(y, allp[col].to_numpy()), 3)
        pooled[f"sp_{col}"] = round(float(spearmanr(allp[col], allp.kappa).correlation), 3)
    rows.append(pooled); pd.DataFrame(rows).to_csv("results/misranking_summary.csv", index=False)
    print("\n=== E2 misranking (AUC for predicting kappa>=2; ~0.5 = miscalibrated) ===", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
