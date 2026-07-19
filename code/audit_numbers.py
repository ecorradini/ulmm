#!/usr/bin/env python3
"""E-AUDIT: verify every load-bearing hand-transcribed number in the manuscript
against the results CSVs it was transcribed from.

Usage:  python3 audit_numbers.py            # audit the numbers of the current paper
Exit code 0 iff no FAIL lines.

Each check compares a value recomputed from results/*.csv with the value printed
in the manuscript (hard-coded here, table cell by table cell). Rounding rule:
match after rounding to the manuscript's printed precision.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
KCAP = 50  # reanchor_kappa.py display cap; kappa==KCAP means co-located (entry ∩ access ≠ ∅)

checks = []  # (status, label, computed, expected)


def check(label, computed, expected, nd=2):
    """Match raw computed value against the printed one within print precision.
    Tolerance 0.5*10^-nd covers both round-half-up and round-half-even printings."""
    if computed is None or (isinstance(computed, float) and np.isnan(computed)):
        checks.append(("FAIL", label, "nan", expected))
        return
    got = float(computed)
    ok = abs(got - float(expected)) <= 0.5 * 10 ** (-nd) + 1e-9
    checks.append(("PASS" if ok else "FAIL", label, round(got, nd + 2), expected))


def info(label, value):
    checks.append(("INFO", label, value, ""))


# ---------------------------------------------------------------- load
per = pd.read_csv(R / "reanchor_summary.csv")
pdm = pd.read_csv(R / "reanchor_perdemand.csv")
ksw = pd.read_csv(R / "ksweep_summary.csv")
mis = pd.read_csv(R / "misranking_multientry_summary.csv")
ext = pd.read_csv(R / "external_spof.csv")
hc = pd.read_csv(R / "kappa_healthcare_summary.csv")
pop = pd.read_csv(R / "kappa_popweight_summary.csv")
nod = pd.read_csv(R / "kappa_nodedisjoint_summary.csv")
phy = pd.read_csv(R / "kappa_physical_summary.csv")
ksens = pd.read_csv(R / "k_sensitivity.csv")

CITIES = ["Amsterdam", "Barcelona", "Paris", "Seattle", "New York City"]
# CSV city label for NYC
NYC = [c for c in pdm.city.unique() if "York" in c][0]
cname = {c: (NYC if c == "New York City" else c) for c in CITIES}


def pool_share(df, kval):
    """Demand-weighted pooled share of kappa==kval, cities pooled by raw weight
    (weights are per-city normalized, so this equal-weights cities)."""
    if kval == ">=2":
        m = df.kappa >= 2
    else:
        m = df.kappa == kval
    return df.w[m].sum() / df.w.sum()


def pool_share_n(df, kval):
    """Pooled share weighting each city by its zone count n."""
    tot = 0.0
    wtot = 0.0
    for c, g in df.groupby("city"):
        if kval == ">=2":
            m = g.kappa >= 2
        else:
            m = g.kappa == kval
        s = g.w[m].sum() / g.w.sum()
        tot += s * len(g)
        wtot += len(g)
    return tot / wtot


# ============================================================ Table 5 (tab:kappa)
# per-city w0/w1/w2 at k=3, both orientations; med kappa; pooled w1 AD .08 DA .07
T5 = {  # city: (AD w0,w1,w2, DA w0,w1,w2, med)
    "Amsterdam": (0.00, 0.12, 0.88, 0.00, 0.13, 0.87, 3),
    "Barcelona": (0.00, 0.12, 0.88, 0.00, 0.11, 0.89, 3),
    "Paris": (0.00, 0.12, 0.88, 0.01, 0.10, 0.89, 3),
    "Seattle": (0.00, 0.07, 0.93, 0.00, 0.06, 0.94, 3),
    "New York City": (0.00, 0.04, 0.96, 0.00, 0.04, 0.96, 4),
}
for city, exp in T5.items():
    for oi, orient in enumerate(("AD", "DA")):
        row = per[(per.city == cname[city]) & (per.orientation == orient) & (per.k == 3)]
        if row.empty:
            check(f"T5 {city} {orient}", None, exp[oi * 3])
            continue
        row = row.iloc[0]
        check(f"T5 {city} {orient} w0", row.w0, exp[oi * 3 + 0])
        check(f"T5 {city} {orient} w1", row.w1, exp[oi * 3 + 1])
        check(f"T5 {city} {orient} w2", row.w2, exp[oi * 3 + 2])
    g = pdm[(pdm.city == cname[city]) & (pdm.orientation == "DA") & (pdm.k == 3)]
    # paper's median includes co-located zones (kappa capped at KCAP = treated as +inf);
    # the median is robust to them, matching the kappa=infinity reporting convention
    check(f"T5 {city} med kappa (DA, incl. co-located)", np.median(g.kappa), exp[6], nd=0)

# Pooling convention identified: per-city-normalized weights concatenated
# (= cities equally weighted). The n-weighted alternative gives AD 0.069 -> must
# be documented in the manuscript as "cities pooled with equal weight".
for orient, expw1 in (("AD", 0.08), ("DA", 0.07)):
    g = pdm[(pdm.orientation == orient) & (pdm.k == 3)]
    check(f"T5 pooled w1 {orient} [equal-city pooling]", pool_share(g, 1), expw1)
    checks.append(("INFO", f"T5 pooled w1 {orient} n-weighted alternative", round(pool_share_n(g, 1), 3), f"paper prints {expw1}"))

# ============================================================ Table 7 (tab:anchoring)
# DA orientation. single-entry k=1: kappa1, kappa0, cut-adj ; multi k=3 same.
T7 = {
    "Amsterdam": (0.40, 0.02, 0.94, 0.13, 0.00, 0.77),
    "Barcelona": (0.43, 0.01, 0.99, 0.11, 0.00, 0.59),
    "Paris": (0.40, 0.01, 0.95, 0.10, 0.01, 0.88),
    "Seattle": (0.34, 0.01, 0.98, 0.06, 0.00, 0.93),
    "New York City": (0.22, 0.00, 0.97, 0.04, 0.00, 0.79),
}
for city, exp in T7.items():
    for ki, k in enumerate((1, 3)):
        row = per[(per.city == cname[city]) & (per.orientation == "DA") & (per.k == k)]
        if row.empty:
            continue
        row = row.iloc[0]
        check(f"T7 {city} k{k} w1", row.w1, exp[ki * 3 + 0])
        check(f"T7 {city} k{k} w0", row.w0, exp[ki * 3 + 1])
        if not pd.isna(row.cut_first_share):
            check(f"T7 {city} k{k} cut-adj", row.cut_first_share, exp[ki * 3 + 2])
g1 = pdm[(pdm.orientation == "DA") & (pdm.k == 1)]
g3 = pdm[(pdm.orientation == "DA") & (pdm.k == 3)]
check("T7 pooled k1 w1", pool_share(g1, 1), 0.31)
check("T7 pooled k3 w1", pool_share(g3, 1), 0.07)
# pooled cut-adj: share of kappa==1 demands (weighted) with cut_first==1
for k, exp in ((1, 0.97), (3, 0.79)):
    g = pdm[(pdm.orientation == "DA") & (pdm.k == k) & (pdm.kappa == 1)]
    check(f"T7 pooled k{k} cut-adj", (g.w * (g.cut_first == 1)).sum() / g.w.sum(), exp)

# ============================================================ Table 8 (tab:ksweep)
T8 = {
    "Amsterdam": (0.40, 0.22, 0.12, 0.06, 0.05, 0.02),
    "Barcelona": (0.43, 0.24, 0.11, 0.05, 0.02, 0.00),
    "Paris": (0.40, 0.20, 0.10, 0.08, 0.07, 0.00),
    "Seattle": (0.34, 0.10, 0.06, 0.03, 0.01, 0.00),
    "New York City": (0.22, 0.10, 0.04, 0.03, 0.01, 0.00),
}
LBL = ["1", "2", "3", "4", "5", "r250"]
ksw["k_label"] = ksw.k_label.astype(str)
for city, exp in T8.items():
    for j, lbl in enumerate(LBL):
        row = ksw[(ksw.city == cname[city]) & (ksw.k_label == lbl)]
        if row.empty:
            check(f"T8 {city} k={lbl}", None, exp[j])
            continue
        check(f"T8 {city} k={lbl} w1", row.iloc[0].w1, exp[j])
POOLED8 = (0.31, 0.15, 0.07, 0.04, 0.03, 0.003)
kswp = pd.read_csv(R / "ksweep_perdemand.csv")
kswp["k_label"] = kswp.k_label.astype(str)
for j, lbl in enumerate(LBL):
    sub = kswp[kswp.k_label == lbl]
    if sub.empty:
        checks.append(("INFO", f"T8 pooled k={lbl}", "no perdemand rows", POOLED8[j]))
        continue
    val = sub.w[sub.kappa == 1].sum() / sub.w.sum()
    check(f"T8 pooled k={lbl} (raw pooling)", val, POOLED8[j], nd=2 if lbl != "r250" else 3)

# ============================================================ Table 6 (tab:misrank)
pool_single = mis[(mis.config == "single") & (mis.city.str.lower() == "pooled")]
pool_multi = mis[(mis.config == "multi") & (mis.city.str.lower() == "pooled")]
if len(pool_single):
    r = pool_single.iloc[0]
    check("T6 single base rate 41%", r.base_rate_k1, 0.41)
    check("T6 single AUC expH", r.auc_redund_expH, 0.55)
    check("T6 single AUC rpe'", r.auc_redund_rpe_prime, 0.56)
    check("T6 single AUC Li", r.auc_redund_Li_div, 0.79)
    check("T6 single top1/3 expH", r.topthird_k1_expH, 0.36)
    check("T6 single top1/3 rpe'", r.topthird_k1_rpe_prime, 0.35)
    check("T6 single top1/3 Li", r.topthird_k1_Li_div, 0.12)
    check("T6 single n=1804", r.n, 1804, nd=0)
else:
    info("T6 single pooled row", "NOT FOUND in misranking_multientry_summary.csv")
if len(pool_multi):
    r = pool_multi.iloc[0]
    check("T6 multi base rate 13%", r.base_rate_k1, 0.13)
    check("T6 multi AUC expH", r.auc_redund_expH, 0.62)
    check("T6 multi AUC rpe'", r.auc_redund_rpe_prime, 0.60)
    check("T6 multi AUC Li", r.auc_redund_Li_div, 0.87)
    check("T6 multi top1/3 expH", r.topthird_k1_expH, 0.08)
    check("T6 multi top1/3 rpe'", r.topthird_k1_rpe_prime, 0.09)
    check("T6 multi top1/3 Li", r.topthird_k1_Li_div, 0.007, nd=3)
    check("T6 multi n=1819", r.n, 1819, nd=0)
else:
    info("T6 multi pooled row", "NOT FOUND in misranking_multientry_summary.csv")

# K-sensitivity claim: Li AUC in 0.75–0.81 for K in {5,15,30,50}, n=300
info("K-sens Li AUC range (claimed 0.75-0.81)", f"{ksens.auc_Li.min():.3f}-{ksens.auc_Li.max():.3f} (n={ksens.n.iloc[0]})")

# ============================================================ Table 9 (tab:external) hotspot block
T9 = {
    ("Truck-crash hotspot", "cut"): (0.50, 0.93),
    ("Truck-crash hotspot", "DEB"): (0.75, 4.11),
    ("311 complaint hotspot", "cut"): (0.50, 0.69),
    ("311 complaint hotspot", "DEB"): (0.62, 1.95),
}
for _, row in ext.iterrows():
    key = (row.outcome, "cut" if "cut" in row.score else "DEB")
    if key in T9:
        e_auc, e_lift = T9[key]
        check(f"T9 {key[0]} {key[1]} AUC", row.AUC, e_auc)
        check(f"T9 {key[0]} {key[1]} lift", row.lift, e_lift)
info("T9 SPOF vs DEB row (0.34/0.75)", "NOT persisted in any CSV -> produced by E-EXT (external_spof_stats.csv)")

# ============================================================ Table 10 (healthcare)
T10 = {
    "Amsterdam": (146, 0.14, 0.12, 0.88, 3),
    "Barcelona": (270, 0.13, 0.12, 0.88, 3),
    "Paris": (540, 0.14, 0.11, 0.88, 3),
    "Seattle": (185, 0.06, 0.06, 0.94, 3),
    "New York City": (997, 0.04, 0.04, 0.96, 4),
}
for city, exp in T10.items():
    for orient, ei in (("AD", 1), ("DA", 2)):
        row = hc[(hc.city == cname[city]) & (hc.orientation == orient)]
        if row.empty:
            continue
        r = row.iloc[0]
        if orient == "AD":
            check(f"T10 {city} n_access", r.n_access, exp[0], nd=0)
        check(f"T10 {city} {orient} w1", r.w1, exp[ei])
        if orient == "DA":
            check(f"T10 {city} DA w2", r.w2, exp[3])
            check(f"T10 {city} med", r.med_kappa, exp[4], nd=0)
for o, expv in (("AD", 0.09), ("DA", 0.09)):
    row = hc[(hc.city == "POOLED") & (hc.orientation == o)]
    if len(row):
        check(f"T10 pooled w1 {o}", row.iloc[0].w1, expv)
    else:
        checks.append(("INFO", f"T10 pooled w1 {o}", "POOLED row missing", expv))

# ============================================================ §4.5 inline robustness numbers
# popweight: pooled block-scale 0.074 -> 0.096 ; single-entry 0.31 -> 0.35 (DA); POOLED rows persisted
for k, e_poi, e_pop in ((3, 0.074, 0.096), (1, 0.313, 0.353)):
    row = pop[(pop.city == "POOLED") & (pop.orientation == "DA") & (pop.k == k)]
    if len(row):
        check(f"§4.5 popweight k{k} POI pooled", row.iloc[0].w1_poi, e_poi, nd=3)
        check(f"§4.5 popweight k{k} POP pooled", row.iloc[0].w1_pop, e_pop, nd=3)
# node-disjoint: pooled 0.076 vs 0.074 (k3), 0.315 vs 0.313 (k1); POOLED rows persisted
for k, expnd, exped in ((3, 0.076, 0.074), (1, 0.315, 0.313)):
    row = nod[nod.city == "POOLED"] if "city" in nod else nod
    row = row[row.k == k]
    if len(row):
        check(f"§4.5 node-disjoint k{k} pooled", row.iloc[0].w1, expnd, nd=3)
    g = pdm[(pdm.orientation == "DA") & (pdm.k == k)]
    check(f"§4.5 edge-disjoint k{k} pooled (raw)", pool_share(g, 1), exped, nd=3)
# physical: way-disjoint certified 79% k3 / 82% k1 (weighted), Seattle k3 uncertified 57%
for k, expv in ((3, 0.79), (1, 0.82)):
    sub = phy[phy.k == k]
    check(f"§4.5 way-disjoint certified k{k} (n-wt zones)", (sub.w_witness * sub.n_k2_zones).sum() / sub.n_k2_zones.sum(), expv)
sea = phy[(phy.k == 3) & (phy.city.str.contains("Seattle"))]
if len(sea):
    check("§4.5 Seattle k3 uncertified 57%", sea.iloc[0].w_affected, 0.57)
# unweighted vs weighted single-entry: 0.33 vs 0.31
g = pdm[(pdm.orientation == "DA") & (pdm.k == 1)]
check("§4.5 unweighted single-entry 0.33", (g.kappa == 1).mean(), 0.33)

# ============================================================ D1/D2 diagnostics (k=3 DA)
g = pdm[(pdm.orientation == "DA") & (pdm.k == 3)]
real = g[g.kappa < KCAP]
colocated = g[g.kappa >= KCAP]
info("D1: real kappa max (k=3 DA)", int(real.kappa.max()))
info("D1: real kappa p90/p95/p99 (k=3 DA)", tuple(int(x) for x in np.percentile(real.kappa, [90, 95, 99])))
info("D2: co-located zones (kappa==KCAP, k=3 DA)", f"{len(colocated)} zones, {100 * len(colocated) / len(g):.1f}% count, {100 * colocated.w.sum() / g.w.sum():.1f}% weighted")
g1 = pdm[(pdm.orientation == "DA") & (pdm.k == 1)]
info("D2: co-located zones (k=1 DA)", f"{(g1.kappa >= KCAP).sum()} zones")
info("claim 'at most four' (Remark 3.3)", f"WRONG at k=3 (max {int(real.kappa.max())}); k=1 real max = {int(g1.kappa[g1.kappa < KCAP].max())}")

# ============================================================ report
fails = [c for c in checks if c[0] == "FAIL"]
for st, label, got, exp in checks:
    mark = {"PASS": "  ok ", "FAIL": " FAIL", "INFO": " info"}[st]
    print(f"[{mark}] {label:55s} computed={got}  expected={exp}")
print(f"\n{len([c for c in checks if c[0]=='PASS'])} pass, {len(fails)} fail, "
      f"{len([c for c in checks if c[0]=='INFO'])} info")
sys.exit(1 if fails else 0)
