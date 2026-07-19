"""
E-EXT: corrected statistics for the NYC betweenness -> single-point-of-failure block
(the reverse direction of Table "external"; the raw ROC-AUC 0.34 previously reported).

Reviewer 1 argued the 0.34 is an artifact of the 0.96% label imbalance and of spatial
distribution rather than a meaningful negative association. This script quantifies that
in full from cache_nyc_ulmm/cutcrit_seg.pkl (209,152 segments; is_spof = positive
cut-criticality; DEB_z = demand-weighted betweenness z-score):

  1. raw ROC-AUC + segment-bootstrap CI (replicates the manuscript number)
  2. average precision (PR) in BOTH directions vs the base rate  (retrieval usefulness)
  3. top-decile lift in both directions
  4. spatial decomposition: between-NTA AUC (NTA-mean betweenness as score) vs
     within-NTA AUC (NTA-demeaned betweenness), + per-NTA stratified AUC (median/IQR)
  5. within-NTA permutation test (labels permuted within NTA) for the within component
  6. matched-negative AUC: 5 same-NTA controls per SPOF, nearest in (length, POI density)
  7. distributional decomposition: SPOF betweenness percentile quartiles, share in the
     bottom quartile / top decile of DEB_z, class means

Persists every number as a tidy row so the manuscript block is fully traceable.
Output: results/external_spof_stats.csv
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CACHE = "cache_nyc_ulmm/cutcrit_seg.pkl"
OUT = "results/external_spof_stats.csv"
B_BOOT = 2000
B_PERM = 1000
SEED = 4242
os.makedirs("results", exist_ok=True)

rows = []


def put(metric, value, lo=np.nan, hi=np.nan, note=""):
    rows.append(dict(metric=metric, value=round(float(value), 4),
                     lo=(round(float(lo), 4) if lo == lo else np.nan),
                     hi=(round(float(hi), 4) if hi == hi else np.nan), note=note))
    print(f"  {metric:42s} {value:.4f}" + (f"  [{lo:.4f},{hi:.4f}]" if lo == lo else "") +
          (f"  ({note})" if note else ""), flush=True)


def main():
    t0 = time.time()
    with open(CACHE, "rb") as f:
        seg = pickle.load(f)
    seg = seg.reset_index(drop=True)
    y = seg.is_spof.astype(int).to_numpy()
    s = seg.DEB_z.astype(float).to_numpy()
    nta = seg.nta2020.astype(str).to_numpy()
    n = len(seg)
    base = y.mean()
    print(f"segments={n}  SPOF={y.sum()} (base rate {base:.4f})", flush=True)

    put("n_segments", n)
    put("n_spof", y.sum())
    put("base_rate", base)

    # 1. raw AUC + segment bootstrap
    auc = roc_auc_score(y, s)
    rng = np.random.RandomState(SEED)
    bs = []
    for _ in range(B_BOOT):
        i = rng.randint(0, n, n)
        if y[i].min() != y[i].max():
            bs.append(roc_auc_score(y[i], s[i]))
    put("auc_raw", auc, np.percentile(bs, 2.5), np.percentile(bs, 97.5),
        "manuscript 0.34 [0.33,0.35]")

    # 2. average precision, both directions
    put("ap_deb", average_precision_score(y, s), note="retrieval with high betweenness")
    put("ap_neg_deb", average_precision_score(y, -s), note="retrieval with LOW betweenness")

    # 3. top-decile lift, both directions
    for name, score in (("lift_top_decile_deb", s), ("lift_top_decile_neg_deb", -s)):
        thr = np.quantile(score, 0.9)
        top = score >= thr
        put(name, y[top].mean() / base)

    # 4. spatial decomposition
    nta_mean = pd.Series(s).groupby(nta).transform("mean").to_numpy()
    put("auc_between_nta", roc_auc_score(y, nta_mean), note="NTA-mean betweenness as score")
    within = s - nta_mean
    auc_within = roc_auc_score(y, within)
    put("auc_within_nta", auc_within, note="NTA-demeaned betweenness")
    per = []
    for g in np.unique(nta):
        m = nta == g
        if y[m].min() != y[m].max():
            per.append(roc_auc_score(y[m], s[m]))
    put("auc_stratified_median", np.median(per), np.percentile(per, 25), np.percentile(per, 75),
        f"per-NTA AUC over {len(per)} NTAs with both classes; lo/hi = IQR")

    # 5. within-NTA permutation test for the demeaned AUC
    perm = np.empty(B_PERM)
    order = np.argsort(nta, kind="stable")
    inv = np.empty_like(order); inv[order] = np.arange(n)
    y_sorted = y[order]
    bounds = np.searchsorted(nta[order], np.unique(nta))
    bounds = np.append(bounds, n)
    for b in range(B_PERM):
        yp = y_sorted.copy()
        for gi in range(len(bounds) - 1):
            lo_, hi_ = bounds[gi], bounds[gi + 1]
            rng.shuffle(yp[lo_:hi_])
        perm[b] = roc_auc_score(yp[inv], within)
    p_two = (np.sum(np.abs(perm - 0.5) >= abs(auc_within - 0.5)) + 1) / (B_PERM + 1)
    put("perm_within_null_lo", np.percentile(perm, 2.5))
    put("perm_within_null_hi", np.percentile(perm, 97.5))
    put("perm_within_pvalue", p_two, note="two-sided, within-NTA label permutation")

    # 6. matched-negative AUC (5 same-NTA controls per SPOF nearest in length & POI density)
    L = seg.length_m.astype(float).to_numpy()
    P = seg.n_poi50.astype(float).to_numpy()
    zL = (L - L.mean()) / L.std()
    zP = (P - P.mean()) / P.std()
    mi, ms, my = [], [], []
    for g in np.unique(nta[y == 1]):
        gm = nta == g
        pos = np.where(gm & (y == 1))[0]
        neg = np.where(gm & (y == 0))[0]
        if len(neg) == 0:
            continue
        for i in pos:
            dist = (zL[neg] - zL[i]) ** 2 + (zP[neg] - zP[i]) ** 2
            take = neg[np.argsort(dist)[:5]]
            mi.append(i); ms.extend([s[i]] + list(s[take])); my.extend([1] + [0] * len(take))
    my = np.array(my); ms = np.array(ms)
    auc_m = roc_auc_score(my, ms)
    bs = []
    for _ in range(500):
        i = rng.randint(0, len(my), len(my))
        if my[i].min() != my[i].max():
            bs.append(roc_auc_score(my[i], ms[i]))
    put("auc_matched", auc_m, np.percentile(bs, 2.5), np.percentile(bs, 97.5),
        f"{len(mi)} SPOFs, 5 same-NTA controls each, matched on length+POI")

    # 7. distributional decomposition
    pct = pd.Series(s).rank(pct=True).to_numpy()
    sp = pct[y == 1]
    put("spof_deb_pctile_q25", np.percentile(sp, 25))
    put("spof_deb_pctile_median", np.median(sp), note="manuscript diagnosis: ~24th pctile")
    put("spof_deb_pctile_q75", np.percentile(sp, 75))
    put("spof_share_bottom_quartile_deb", (sp <= 0.25).mean())
    put("spof_share_top_decile_deb", (sp >= 0.9).mean())
    put("mean_debz_spof", s[y == 1].mean(), note="manuscript diagnosis: +0.10")
    put("mean_debz_nonspof", s[y == 0].mean())
    put("n_ntas_with_spof", len(np.unique(nta[y == 1])))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
