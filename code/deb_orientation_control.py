"""Same-orientation control for the held-out betweenness null (EIC re-review, MAJOR #2).

Sec. V-E.1 explains the below-chance retrieval of kappa=1 cut edges by an
orientation mismatch: the min cuts separate demand->access, while the reported
demand-weighted betweenness (deb_exact.py) accumulates over access->demand
shortest paths, rooted at the 2,657 access-anchor nodes.

That explanation was never controlled. This script computes betweenness in the
MATCHING orientation, rooted at the demand anchors and accumulating toward the
access set, and re-runs the same retrieval statistics against the same kappa=1
cut edges. Two variants:

  DEB_fwd_w  demand-weighted: each demand source contributes w_d
  DEB_fwd_u  unweighted: each demand source contributes 1

If the null survives in the matching orientation, the "structurally invisible to
activity-based exposure" claim is not an artifact of comparing mismatched
directed quantities. If it does not, the mechanism paragraph is wrong.

Reuses the audited Brandes accumulation of b1_wd_external.deb_dual, modified only
to scale each source's contribution by a per-source weight (the accumulation is
linear in the source, so this is the same estimator with source weights).

Outputs:
  cache_nyc_ulmm/deb_fwd_seg.pkl
  results/deb_orientation_control.csv
"""
import heapq
import math
import os
import pickle
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import b1_wd_external as b1

CACHE = "cache_nyc_ulmm/"
N_WORKERS = 8
os.makedirs("results", exist_ok=True)

_ADJ = None
_TGT_SET = None
_SRC_W = None


def deb_sourceweighted(adj, sources, tgt_set_u, src_w):
    """Brandes with per-source weights; targets counted uniformly.

    Identical to b1_wd_external.deb_dual except that the dependency accumulated
    from source s is scaled by src_w[s] (weighted variant) and by 1 (unweighted
    variant). Accumulation is linear in the source, so this equals the
    demand-weighted betweenness of the demand->access orientation.
    """
    e_w = defaultdict(float)
    e_u = defaultdict(float)
    tol = 1e-9
    src = list(sources)
    n = len(src)
    for idx, s in enumerate(src):
        if idx % 200 == 0:
            print(f"  source {idx}/{n}", flush=True)
        ws = float(src_w.get(s, 0.0))
        S = []
        P = defaultdict(list)
        sigma = defaultdict(float)
        dist = {s: 0.0}
        sigma[s] = 1.0
        Q = [(0.0, s)]
        repkey = {}
        while Q:
            dv, v = heapq.heappop(Q)
            if dv > dist.get(v, math.inf) + 1e-12:
                continue
            S.append(v)
            for (w, wtt, k) in adj.get(v, ()):
                vw = dv + wtt
                dw = dist.get(w, math.inf)
                if vw < dw - tol:
                    dist[w] = vw
                    heapq.heappush(Q, (vw, w))
                    sigma[w] = sigma[v]
                    P[w] = [v]
                    repkey[(v, w)] = k
                elif abs(vw - dw) <= tol:
                    sigma[w] += sigma[v]
                    P[w].append(v)
                    repkey[(v, w)] = k
        delta = defaultdict(float)
        while S:
            w = S.pop()
            coef = 1.0 if w in tgt_set_u else 0.0
            sw = sigma[w]
            if sw == 0:
                continue
            for v in P.get(w, ()):
                ratio = sigma[v] / sw
                c = ratio * (coef + delta[w])
                ek = (v, w, repkey[(v, w)])
                e_w[ek] += ws * c
                e_u[ek] += c
                delta[v] += c
    return e_w, e_u


def _worker(chunk):
    ew, eu = deb_sourceweighted(_ADJ, chunk, _TGT_SET, _SRC_W)
    return dict(ew), dict(eu)


def main():
    global _ADJ, _TGT_SET, _SRC_W
    t0 = time.time()
    G = b1.load("graph_G")
    ad = b1.load("anchor_demand_cells")
    aa = b1.load("anchor_access")
    dw = b1.load("demand_weighted")
    seg = b1.load("segments_fe").copy()

    wmap = dict(zip(dw["d_id"], dw["w_d"]))
    # sources = demand anchors, weighted by demand mass (matching orientation)
    src_w = defaultdict(float)
    for _, r in ad.iterrows():
        src_w[int(r["node"])] += float(wmap.get(r["d_id"], 1.0))
    src_all = sorted(src_w)
    # targets = access anchors
    tgt_set = {int(n) for n in aa["node"]}
    print(f"demand sources={len(src_all)} access targets={len(tgt_set)}", flush=True)

    print("collapsing graph...", flush=True)
    _ADJ = b1.collapse(G)
    _TGT_SET = tgt_set
    _SRC_W = dict(src_w)

    chunks = [src_all[i::N_WORKERS] for i in range(N_WORKERS)]
    print(f"running {len(src_all)} sources on {N_WORKERS} fork workers", flush=True)
    ctx = get_context("fork")
    e_w = defaultdict(float)
    e_u = defaultdict(float)
    with ctx.Pool(N_WORKERS) as pool:
        for i, (ew, eu) in enumerate(pool.imap_unordered(_worker, chunks)):
            for k, v in ew.items():
                e_w[k] += v
            for k, v in eu.items():
                e_u[k] += v
            print(f"[merge] worker {i+1}/{N_WORKERS} ({time.time()-t0:.0f}s)", flush=True)

    dfw = pd.DataFrame([(u, v, k, val) for (u, v, k), val in e_w.items()],
                       columns=["u", "v", "key", "DEB_fwd_w"])
    dfu = pd.DataFrame([(u, v, k, val) for (u, v, k), val in e_u.items()],
                       columns=["u", "v", "key", "DEB_fwd_u"])
    seg = (seg.merge(dfw, on=["u", "v", "key"], how="left")
              .merge(dfu, on=["u", "v", "key"], how="left"))
    seg["DEB_fwd_w"] = seg["DEB_fwd_w"].fillna(0.0)
    seg["DEB_fwd_u"] = seg["DEB_fwd_u"].fillna(0.0)

    # labels: the same kappa=1 cut edges used by external_spof_stats --exact
    cut = pd.read_pickle(CACHE + "cutcrit_seg_exact.pkl")
    keycols = ["u", "v", "key"]
    ycol = [c for c in cut.columns if c.lower() in ("y_spof", "is_spof", "spof")]
    if not ycol:
        cand = [c for c in cut.columns if "spof" in c.lower() or "cut" in c.lower()]
        raise SystemExit(f"cannot find SPOF label column; candidates={cand}")
    y = cut[keycols + [ycol[0]]].rename(columns={ycol[0]: "y_spof"})
    seg = seg.merge(y, on=keycols, how="left")
    seg["y_spof"] = seg["y_spof"].fillna(0).astype(int)

    yv = seg["y_spof"].to_numpy()
    rows = [dict(metric="n_segments", value=len(seg)),
            dict(metric="n_spof", value=int(yv.sum())),
            dict(metric="n_demand_sources", value=len(src_all)),
            dict(metric="base_rate", value=round(float(yv.mean()), 4))]
    for label, col in (("fwd_weighted", "DEB_fwd_w"), ("fwd_unweighted", "DEB_fwd_u"),
                       ("reverse_weighted_exact", "DEB_w_exact")):
        if col not in seg.columns:
            if col == "DEB_w_exact":
                ex = pd.read_pickle(CACHE + "deb_exact_seg.pkl")
                seg = seg.merge(ex[keycols + ["DEB_w_exact"]], on=keycols, how="left")
                seg["DEB_w_exact"] = seg["DEB_w_exact"].fillna(0.0)
            else:
                continue
        s = seg[col].to_numpy()
        rows += [dict(metric=f"auc_{label}", value=round(float(roc_auc_score(yv, s)), 4)),
                 dict(metric=f"ap_{label}", value=round(float(average_precision_score(yv, s)), 4)),
                 dict(metric=f"ap_neg_{label}", value=round(float(average_precision_score(yv, -s)), 4)),
                 dict(metric=f"zero_share_{label}", value=round(float((s == 0).mean()), 4)),
                 dict(metric=f"spof_share_in_zero_{label}",
                      value=round(float((s[yv == 1] == 0).mean()), 4))]
    out = pd.DataFrame(rows)
    out.to_csv("results/deb_orientation_control.csv", index=False)
    seg[keycols + ["DEB_fwd_w", "DEB_fwd_u", "y_spof"]].to_pickle(CACHE + "deb_fwd_seg.pkl")
    print(out.to_string(index=False))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
