"""
R5 (panel): EXACT demand-weighted betweenness on the held-out NYC extract.

The manuscript's held-out DEB was computed with an undisclosed 80-of-3,058
access-source subsample (b1_wd_external.py, N_SRC=80). Per the revision decision,
this script computes DEB exactly over ALL access sources, reusing the audited
Brandes implementation (b1_wd_external.deb_dual) unchanged and parallelizing over
sources with fork-based multiprocessing.

Outputs:
  cache_nyc_ulmm/deb_exact_seg.pkl  (u,v,key, DEB_w_exact, DEB_u_exact, z-scores,
                                     plus the cached subsampled DEB_z for comparison)
  results/deb_exact_summary.csv     (sanity + retrieval AUC/lift, exact vs subsampled)

Modes:
  python3 deb_exact.py --probe N     time N sources single-process and exit
  python3 deb_exact.py --sanity      run the SAME 80-source subsample (seed 7) and
                                     report correlation with cached DEB_z, then exit
  python3 deb_exact.py               full exact run (all sources, multiprocessing)
"""
import gc
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import b1_wd_external as b1

CACHE = "cache_nyc_ulmm/"
N_WORKERS = 8
os.makedirs("results", exist_ok=True)

# module-level state for fork workers (copy-on-write)
_ADJ = None
_TGT_W = None
_TGT_SET = None


def _worker(chunk):
    ew, eu = b1.deb_dual(_ADJ, chunk, _TGT_W, _TGT_SET)
    return dict(ew), dict(eu)


def load_inputs():
    G = b1.load("graph_G")
    ad = b1.load("anchor_demand_cells")
    aa = b1.load("anchor_access")
    dw = b1.load("demand_weighted")
    seg = b1.load("segments_fe").copy()
    wmap = dict(zip(dw["d_id"], dw["w_d"]))
    tgt_w = defaultdict(float)
    tgt_set = set()
    for _, r in ad.iterrows():
        tgt_w[int(r["node"])] += float(wmap.get(r["d_id"], 1.0))
        tgt_set.add(int(r["node"]))
    src_all = sorted({int(n) for n in aa["node"]})
    print(f"sources(all)={len(src_all)} demand targets={len(tgt_set)}", flush=True)
    print("collapsing graph...", flush=True)
    adj = b1.collapse(G)
    del G
    gc.collect()
    print(f"  adj nodes={len(adj)}", flush=True)
    return adj, dict(tgt_w), tgt_set, src_all, seg


def main():
    global _ADJ, _TGT_W, _TGT_SET
    t0 = time.time()
    adj, tgt_w, tgt_set, src_all, seg = load_inputs()
    _ADJ, _TGT_W, _TGT_SET = adj, tgt_w, tgt_set

    if "--probe" in sys.argv:
        n = int(sys.argv[sys.argv.index("--probe") + 1])
        t1 = time.time()
        b1.deb_dual(adj, src_all[:n], tgt_w, tgt_set)
        dt = time.time() - t1
        est = dt / n * len(src_all)
        print(f"PROBE: {n} sources in {dt:.1f}s -> {dt/n:.2f}s/source; "
              f"full single-process ~{est/60:.0f} min; /{N_WORKERS} workers ~{est/60/N_WORKERS:.0f} min",
              flush=True)
        return

    if "--sanity" in sys.argv:
        rng = np.random.default_rng(7)
        src80 = sorted(rng.choice(src_all, size=80, replace=False).tolist())
        ew, eu = b1.deb_dual(adj, src80, tgt_w, tgt_set)
        dfw = pd.DataFrame([(u, v, k, val) for (u, v, k), val in ew.items()],
                           columns=["u", "v", "key", "DEB_w"])
        s = seg.merge(dfw, on=["u", "v", "key"], how="left")
        s["DEB_w"] = s["DEB_w"].fillna(0.0)
        corr = float(np.corrcoef(b1.zscore(s["DEB_w"]), s["DEB_z"])[0, 1])
        print(f"SANITY: corr(80-source recompute, cached DEB_z) = {corr:.6f}", flush=True)
        return

    # ---------------- full exact run ----------------
    chunks = [src_all[i::N_WORKERS] for i in range(N_WORKERS)]
    print(f"running {len(src_all)} sources on {N_WORKERS} fork workers "
          f"(chunks of ~{len(chunks[0])})", flush=True)
    ctx = get_context("fork")
    e_w = defaultdict(float)
    e_u = defaultdict(float)
    with ctx.Pool(N_WORKERS) as pool:
        for i, (ew, eu) in enumerate(pool.imap_unordered(_worker, chunks)):
            for k, v in ew.items():
                e_w[k] += v
            for k, v in eu.items():
                e_u[k] += v
            print(f"[merge] worker {i+1}/{N_WORKERS} done ({time.time()-t0:.0f}s)", flush=True)

    dfw = pd.DataFrame([(u, v, k, val) for (u, v, k), val in e_w.items()],
                       columns=["u", "v", "key", "DEB_w_exact"])
    dfu = pd.DataFrame([(u, v, k, val) for (u, v, k), val in e_u.items()],
                       columns=["u", "v", "key", "DEB_u_exact"])
    seg = seg.merge(dfw, on=["u", "v", "key"], how="left").merge(dfu, on=["u", "v", "key"], how="left")
    seg["DEB_w_exact"] = seg["DEB_w_exact"].fillna(0.0)
    seg["DEB_u_exact"] = seg["DEB_u_exact"].fillna(0.0)
    seg["DEB_exact_z"] = b1.zscore(seg["DEB_w_exact"])
    seg["DEB_u_exact_z"] = b1.zscore(seg["DEB_u_exact"])

    corr = float(np.corrcoef(seg["DEB_exact_z"], seg["DEB_z"])[0, 1])
    zero_share = float((seg["DEB_w_exact"] == 0.0).mean())
    print(f"corr(exact, subsampled cached) = {corr:.4f}; exact zero-DEB share = {zero_share:.4f}",
          flush=True)

    keep = ["u", "v", "key", "DEB_w_exact", "DEB_u_exact", "DEB_exact_z", "DEB_u_exact_z",
            "DEB_z", "y_311_hot", "y_trk_hot", "n_311", "n_truck", "geoid", "nta2020",
            "length_m", "n_poi50"]
    keep = [c for c in keep if c in seg.columns]
    seg[keep].to_pickle(CACHE + "deb_exact_seg.pkl")

    rows = [dict(metric="corr_exact_vs_subsampled", value=round(corr, 4)),
            dict(metric="zero_deb_share_exact", value=round(zero_share, 4)),
            dict(metric="n_sources", value=len(src_all))]
    for ycol in ("y_trk_hot", "y_311_hot"):
        y = seg[ycol].astype(int)
        for tag, col in (("exact", "DEB_exact_z"), ("subsampled", "DEB_z")):
            auc = roc_auc_score(y, seg[col]) if y.nunique() > 1 else float("nan")
            lift = b1.top_decile_lift(seg[col], y)
            rows.append(dict(metric=f"auc_{ycol}_{tag}", value=round(float(auc), 3)))
            rows.append(dict(metric=f"lift_{ycol}_{tag}", value=round(float(lift), 3)))
            print(f"  {ycol} {tag}: AUC={auc:.3f} lift={lift:.3f}", flush=True)
    pd.DataFrame(rows).to_csv("results/deb_exact_summary.csv", index=False)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
