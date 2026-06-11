"""
E3 + E5: demand-weighted CUT CRITICALITY and its held-out NYC external validation.

On the NYC street graph (cache_nyc_ulmm/graph_G), for each demand cell d (anchored at
a node, weighted w_d) we compute kappa(d) = max edge-disjoint paths d -> access set
(super-sink over all access anchors). For every SINGLE-POINT-OF-FAILURE demand (kappa=1)
the min cut is a SINGLE substrate edge e*(d): removing it isolates d. We accumulate

    cutcrit(e) = sum_{d: kappa(d)=1, e=e*(d)} w_d      (weighted demand isolated by e)

These are the structurally critical edges. EXTERNAL TEST (non-circular): do high-cutcrit
edges coincide with OBSERVED disruption (2024 NYPD truck-involved crashes, 311 illegal-
parking complaints) that never entered the model? We report ROC-AUC, top-decile lift, and
negative-binomial IRR (census-tract FE + log-length offset + POI control), with the cached
DEB_z as the comparison baseline.

Engine: scipy maximum_flow (C); residual BFS for the kappa=1 min cut. Self-contained on the
external cache so cut edges (u,v) map directly onto segments_fe rows.
"""
import numpy as np, pandas as pd, pickle, time, os
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from sklearn.metrics import roc_auc_score
import b1_wd_external as b1   # reuse top_decile_lift, zscore, fit_nb_irr

CACHE = "cache_nyc_ulmm/"
BIG = 1 << 24
os.makedirs("results", exist_ok=True)


def load(name):
    return pickle.load(open(CACHE + name + ".pkl", "rb"))


def main():
    t0 = time.time()
    G = load("graph_G")                 # MultiDiGraph
    ad = load("anchor_demand_cells"); aa = load("anchor_access"); dw = load("demand_weighted")
    seg = load("segments_fe").copy()
    wmap = dict(zip(dw["d_id"], dw["w_d"]))

    # collapse MultiDiGraph -> simple directed edge set keyed by (u,v)
    edge_uv = {}
    for u, v, d in G.edges(data=True):
        w = float(d.get("travel_time", 1.0))
        if (u, v) not in edge_uv or w < edge_uv[(u, v)]:
            edge_uv[(u, v)] = w
    nodes = sorted(set([x for e in edge_uv for x in e]))
    idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes); TERM = N
    eu = np.array([idx[u] for (u, v) in edge_uv], np.int64)
    ev = np.array([idx[v] for (u, v) in edge_uv], np.int64)
    uv_list = list(edge_uv.keys())
    print(f"nodes={N} edges={len(uv_list)} ({time.time()-t0:.1f}s)", flush=True)

    # demand target nodes + weights, access source nodes
    tgt_w = defaultdict(float)
    for _, r in ad.iterrows():
        tgt_w[int(r["node"])] += float(wmap.get(r["d_id"], 1.0))
    dem_nodes = [n for n in tgt_w if n in idx]
    acc_nodes = sorted({int(n) for n in aa["node"] if int(n) in idx})
    print(f"demand nodes={len(dem_nodes)} access nodes={len(acc_nodes)}", flush=True)

    # capacity matrix: substrate u->v cap1, a->SINK capBIG
    rows = list(eu) + [idx[a] for a in acc_nodes]
    cols = list(ev) + [TERM] * len(acc_nodes)
    vals = [1] * len(eu) + [BIG] * len(acc_nodes)
    M = csr_matrix((np.array(vals, np.int64), (np.array(rows), np.array(cols))), shape=(N + 1, N + 1))

    # base capacity arrays for residual (forward)
    cutcrit = defaultdict(float)
    kappa_w = defaultdict(float)        # weighted kappa histogram
    n1 = 0; t1 = time.time()
    for j, d in enumerate(dem_nodes):
        di = idx[d]; w = tgt_w[d]
        res = maximum_flow(M, di, TERM)
        k = int(res.flow_value)
        kappa_w[min(k, 4)] += w
        if k == 1:
            n1 += 1
            F = res.flow
            # residual adjacency: forward (cap-flow>0) OR backward (flow>0)
            Rf = (M - F)
            resid = ((Rf > 0).astype(np.int8) + (F > 0).astype(np.int8).transpose()).tocsr()
            order = breadth_first_order(resid, di, directed=True, return_predecessors=False)
            Rmask = np.zeros(N + 1, bool); Rmask[order] = True
            cut = np.where(Rmask[eu] & ~Rmask[ev])[0]    # substrate cut edges (u in R, v not in R)
            for ci in cut:
                cutcrit[uv_list[ci]] += w
        if (j + 1) % 1000 == 0:
            print(f"  {j+1}/{len(dem_nodes)} demands  kappa=1 so far={n1}  ({time.time()-t1:.0f}s)", flush=True)

    tot_w = sum(kappa_w.values())
    print("\nweighted kappa distribution (external 3058-access setup):", flush=True)
    for k in sorted(kappa_w):
        lab = f"{k}+" if k == 4 else str(k)
        print(f"  kappa={lab}: {kappa_w[k]/tot_w:.3f}", flush=True)

    # map cut-criticality onto segments and validate
    crit_df = pd.DataFrame([(u, v, c) for (u, v), c in cutcrit.items()], columns=["u", "v", "cutcrit"])
    seg = seg.merge(crit_df, on=["u", "v"], how="left")
    seg["cutcrit"] = seg["cutcrit"].fillna(0.0)
    seg["is_spof"] = (seg["cutcrit"] > 0).astype(int)
    seg["cutcrit_z"] = b1.zscore(seg["cutcrit"])
    print(f"\nSPOF edges (segments with cutcrit>0): {int(seg.is_spof.sum())} / {len(seg)}", flush=True)
    seg[["u", "v", "key", "cutcrit", "cutcrit_z", "is_spof", "DEB_z", "n_truck", "n_311",
         "y_trk_hot", "y_311_hot", "nta2020", "length_m", "n_poi50"]].to_pickle("cache_nyc_ulmm/cutcrit_seg.pkl")

    rows_out = []
    for y_hot, y_cnt, name in [("y_trk_hot", "n_truck", "Truck-crash hotspot"),
                               ("y_311_hot", "n_311", "311 hotspot")]:
        yb = seg[y_hot].astype(int)
        for score, sname in [("cutcrit_z", "cut-criticality"), ("is_spof", "is-SPOF (binary)"),
                             ("DEB_z", "DEB (baseline)")]:
            auc = float(roc_auc_score(yb, seg[score])) if yb.nunique() > 1 else np.nan
            lift = b1.top_decile_lift(seg[score], yb)
            print(f"  {name:22s} {sname:18s} AUC={auc:.3f} lift={lift:.2f}", flush=True)
            irr = np.nan
            if score != "is_spof":
                irr = b1.fit_nb_irr(seg, y=y_cnt, x=score, fe="nta2020", offset="length_m", controls=["n_poi50"])
                print(f"      -> NB-IRR (nta FE) = {irr:.3f}", flush=True)
            rows_out.append(dict(outcome=name, score=sname, roc_auc=round(auc, 3),
                                 lift_top10=round(float(lift), 2),
                                 nb_irr=(round(float(irr), 3) if irr == irr else np.nan)))
        pd.DataFrame(rows_out).to_csv("results/external_spof.csv", index=False)

    pd.DataFrame([(f"kappa={k}", kappa_w[k] / tot_w) for k in sorted(kappa_w)],
                 columns=["bucket", "w_share"]).to_csv("results/external_kappa_dist.csv", index=False)
    print("\n=== external SPOF validation ===", flush=True)
    print(pd.DataFrame(rows_out).to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
