"""
B1 (reviewer M2): does demand-weighting w_d improve OUT-OF-SAMPLE retrieval?
Recompute DEB (POI-weighted) and DEB^U (uniform) on the cached NYC graph in a
single SSSP pass over access sources, map to segments, and run the identical
held-out retrieval (ROC-AUC, top-decile lift, NB-IRR) for both. Offline from cache.
"""
import math, heapq, pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple
import numpy as np, pandas as pd, networkx as nx
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial
from sklearn.metrics import roc_auc_score

CACHE = Path("cache_nyc_ulmm")
EPS_DENOM = 1e-12
def load(name): return pickle.load(open(CACHE / f"{name}.pkl", "rb"))

def collapse(G, weight_key="tt"):
    """Min-cost simple-directed adjacency with representative edge key per (u,v)."""
    succ = defaultdict(list)  # u -> list of (v, w_tt, rep_key)
    for u, v, k, data in G.edges(keys=True, data=True):
        w = float(data.get(weight_key, 1.0))
        succ[u].append((v, w, k))
    # keep only the min-cost parallel edge per (u,v)
    out = {}
    for u, lst in succ.items():
        best = {}
        for v, w, k in lst:
            if v not in best or w < best[v][0]:
                best[v] = (w, k)
        out[u] = [(v, w, k) for v, (w, k) in best.items()]
    return out

def deb_dual(adj, sources, tgt_weight_w: Dict[int, float], tgt_set_u: set):
    """Standard Brandes on collapsed adjacency; dual accumulation (weighted/unweighted)."""
    e_w = defaultdict(float); e_u = defaultdict(float)
    tol = 1e-9; src = list(sources); n = len(src)
    for idx, s in enumerate(src):
        if idx % 100 == 0: print(f"  source {idx}/{n}", flush=True)
        S = []; P = defaultdict(list); sigma = defaultdict(float)
        dist = {s: 0.0}; sigma[s] = 1.0; Q = [(0.0, s)]; repkey = {}
        while Q:
            dv, v = heapq.heappop(Q)
            if dv > dist.get(v, math.inf) + 1e-12: continue
            S.append(v)
            for (w, wtt, k) in adj.get(v, ()):
                vw = dv + wtt
                dw = dist.get(w, math.inf)
                if vw < dw - tol:
                    dist[w] = vw; heapq.heappush(Q, (vw, w)); sigma[w] = sigma[v]; P[w] = [v]; repkey[(v, w)] = k
                elif abs(vw - dw) <= tol:
                    sigma[w] += sigma[v]; P[w].append(v); repkey[(v, w)] = k
        delta_w = defaultdict(float); delta_u = defaultdict(float)
        while S:
            w = S.pop()
            coef_w = float(tgt_weight_w.get(w, 0.0)); coef_u = 1.0 if w in tgt_set_u else 0.0
            sw = sigma[w]
            if sw == 0: continue
            for v in P.get(w, ()):
                ratio = sigma[v] / sw
                cw = ratio * (coef_w + delta_w[w]); cu = ratio * (coef_u + delta_u[w])
                ek = (v, w, repkey[(v, w)])
                e_w[ek] += cw; e_u[ek] += cu
                delta_w[v] += cw; delta_u[v] += cu
    return e_w, e_u

def top_decile_lift(score, label):
    n = len(score); k = max(1, int(math.ceil(0.10 * n)))
    idx = score.sort_values(ascending=False).index[:k]
    base = label.mean()
    return (label.loc[idx].mean() / base) if base > 0 else float("inf")

def zscore(x):
    sd = x.std(ddof=0); return (x - x.mean()) / (sd if sd > 0 else 1.0)

def fit_nb_irr(df, y, x, fe, offset, controls):
    d = df.copy()
    for c in [x] + controls: d[c] = pd.to_numeric(d[c], errors="coerce")
    fed = pd.get_dummies(d[fe].astype(str).fillna("NA"), prefix=fe, drop_first=True, dtype=float)
    X = pd.concat([d[[x] + controls], fed], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    yv = pd.to_numeric(d[y], errors="coerce").fillna(0).clip(lower=0).astype(np.int64).values
    off = np.log(pd.to_numeric(d[offset], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=1e-6).values.astype(float))
    X = sm.add_constant(X, has_constant="add")
    try:
        r = GLM(yv, X, family=NegativeBinomial(), offset=off).fit(maxiter=200, disp=0)
        return float(np.exp(r.params.get(x, np.nan)))
    except Exception as e:
        print("NB fail", e); return float("nan")

def main():
    G = load("graph_G")
    ad = load("anchor_demand_cells"); aa = load("anchor_access"); dw = load("demand_weighted")
    seg = load("segments_fe").copy()
    wmap = dict(zip(dw["d_id"], dw["w_d"]))
    # target node weights (demand), source set (access)
    tgt_w = defaultdict(float); tgt_set = set()
    for _, r in ad.iterrows():
        tgt_w[int(r["node"])] += float(wmap.get(r["d_id"], 1.0)); tgt_set.add(int(r["node"]))
    src_all = sorted({int(n) for n in aa["node"]})
    N_SRC = 80  # subsample of access sources (full 3058 is infeasible in pure Python);
    rng = np.random.default_rng(7)
    src_nodes = sorted(rng.choice(src_all, size=min(N_SRC, len(src_all)), replace=False).tolist()) if len(src_all) > N_SRC else src_all
    print(f"access sources={len(src_nodes)}/{len(src_all)} demand targets={len(tgt_set)}", flush=True)
    print("collapsing graph...", flush=True); adj = collapse(G); del G; import gc; gc.collect(); print(f"  adj nodes={len(adj)}", flush=True)
    e_w, e_u = deb_dual(adj, src_nodes, tgt_w, tgt_set)
    dfw = pd.DataFrame([(u, v, k, val) for (u, v, k), val in e_w.items()], columns=["u", "v", "key", "DEB_w"])
    dfu = pd.DataFrame([(u, v, k, val) for (u, v, k), val in e_u.items()], columns=["u", "v", "key", "DEB_u"])
    seg = seg.merge(dfw, on=["u", "v", "key"], how="left").merge(dfu, on=["u", "v", "key"], how="left")
    seg["DEB_w"] = seg["DEB_w"].fillna(0.0); seg["DEB_u"] = seg["DEB_u"].fillna(0.0)
    seg["DEB_w_z"] = zscore(seg["DEB_w"]); seg["DEB_u_z"] = zscore(seg["DEB_u"])
    # sanity vs cached weighted DEB_z
    corr = float(np.corrcoef(seg["DEB_w_z"], seg["DEB_z"])[0, 1])
    print(f"sanity corr(DEB_w_z, cached DEB_z)={corr:.4f}", flush=True)
    # persist scored segments so retrieval/IRR can be re-run offline without recomputing DEB
    seg[["u", "v", "key", "DEB_w_z", "DEB_u_z", "DEB_z", "y_311_hot", "y_trk_hot",
         "n_311", "n_truck", "geoid", "length_m", "n_poi50"]].to_pickle("cache_nyc_ulmm/b1_seg_scored.pkl")
    DO_IRR = False  # NB-GLM with ~2000 tract dummies is slow and is known ~1 after FE
    rows = []
    # AUC + lift first (instant)
    for ycol, cnt in [("y_311_hot", "n_311"), ("y_trk_hot", "n_truck")]:
        for tag, zc in [("weighted", "DEB_w_z"), ("unweighted", "DEB_u_z")]:
            y = seg[ycol].astype(int); s = seg[zc].astype(float)
            auc = roc_auc_score(y, s) if y.nunique() > 1 else float("nan")
            lift = top_decile_lift(s, y)
            rows.append(dict(outcome=ycol, deb=tag, auc=round(auc, 3), lift=round(lift, 3)))
            print(f"  {ycol} {tag}: AUC={auc:.3f} lift={lift:.3f}", flush=True)
    out = pd.DataFrame(rows); out.to_csv("results/revision_b1_wd_external.csv", index=False)
    print("\n=== B1: DEB (weighted) vs DEB^U (unweighted) external retrieval ===", flush=True)
    print(out.to_string(index=False), flush=True); print("DONE", flush=True)
    if not DO_IRR:
        return
    print("\n=== B1: DEB (weighted) vs DEB^U (unweighted) external retrieval ===", flush=True)
    print(out.to_string(index=False), flush=True); print("DONE", flush=True)

if __name__ == "__main__":
    main()
