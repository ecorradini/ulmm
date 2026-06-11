"""
Round-2 #3: K-sensitivity of the route-diversity (edge-dissimilarity, Li) index.
The path-multiplicity/entropy indices are K-insensitive by argument; the one baseline with
signal (Li, AUC 0.71 at K=15) is the one where K could matter, so we sweep K in {5,15,30,50}
on a reduced per-city sample and report the Li ROC-AUC vs kappa>=2. kappa labels reused from
results/kappa_perdemand.csv (DA), matching the misranking table.
"""
import numpy as np, pandas as pd, os, time
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import roc_auc_score
import ablations as ab

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; KS = [5, 15, 30, 50]; PEN = 4.0; SAMPLE = 60
os.makedirs("results", exist_ok=True)


def build_csr(G, idx, N):
    eu = np.array([idx[a] for a, b in G.edges()], np.int64); ev = np.array([idx[b] for a, b in G.edges()], np.int64)
    w = np.array([float(G[a][b]["weight"]) for a, b in G.edges()], float)
    return csr_matrix((w, (eu, ev)), shape=(N, N))


def k_routes(M, src, access_set, K):
    data = M.data.copy(); indptr = M.indptr; indices = M.indices; N = M.shape[0]
    acc = np.fromiter(access_set, np.int64); routes = []
    for _ in range(K):
        Mw = csr_matrix((data, indices, indptr), shape=(N, N))
        dist, pred = dijkstra(Mw, indices=src, return_predecessors=True)
        da = dist[acc]
        if not np.isfinite(da).any():
            break
        best = int(acc[np.argmin(da)]); path = []; v = best
        while v != src and v >= 0:
            path.append(v); v = pred[v]
        if v != src:
            break
        path.append(src); path.reverse(); routes.append(path)
        for a_, b_ in zip(path[:-1], path[1:]):
            for jj in range(indptr[a_], indptr[a_ + 1]):
                if indices[jj] == b_:
                    data[jj] = data[jj] * PEN + 1.0; break
    return routes


def li_div(routes):
    n = len(routes)
    if n < 2:
        return 0.0
    Es = [set(zip(p[:-1], p[1:])) for p in routes]; Js = []
    for i, j in combinations(range(n), 2):
        uni = len(Es[i] | Es[j]); Js.append((len(Es[i] & Es[j]) / uni) if uni else 0.0)
    return 1.0 - float(np.mean(Js))


def main():
    kp = pd.read_csv("results/kappa_perdemand.csv")
    kap = {(r.city, int(r.d)): int(r.kappa) for r in kp[kp.orientation == "DA"].itertuples()}
    perd = {K: [] for K in KS}
    for c in CITIES:
        short = c.split(",")[0]; print(f"[{c}]", flush=True); t0 = time.time()
        u = ab.load_ulmm(c); G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        M = build_csr(G, idx, N)
        access_set = {idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx}
        d_nodes = [int(n) for n in u["demand"]["i_node"].tolist() if int(n) in idx and (short, int(n)) in kap]
        d_nodes = sorted(set(d_nodes))
        if len(d_nodes) > SAMPLE:
            d_nodes = sorted(np.random.RandomState(3).choice(d_nodes, SAMPLE, replace=False).tolist())
        for K in KS:
            for d in d_nodes:
                r = k_routes(M, idx[d], access_set, K)
                perd[K].append((li_div(r), kap[(short, d)]))
        print(f"  n={len(d_nodes)} ({time.time()-t0:.0f}s)", flush=True)
    rows = []
    for K in KS:
        a = np.array([x[0] for x in perd[K]]); k = np.array([x[1] for x in perd[K]])
        y = (k >= 2).astype(int)
        auc = float(roc_auc_score(y, a)) if len(np.unique(y)) > 1 else np.nan
        rows.append(dict(K=K, n=len(a), frac_redund=round(float(y.mean()), 3), auc_Li=round(auc, 3)))
        print(f"K={K}: n={len(a)} Li AUC(kappa>=2)={auc:.3f}", flush=True)
    pd.DataFrame(rows).to_csv("results/k_sensitivity.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
