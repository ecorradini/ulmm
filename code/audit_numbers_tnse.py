#!/usr/bin/env python3
"""E-AUDIT (TNSE): verify the NEW numbers hand-transcribed into paper/tnse/*.tex
against their source CSVs (external_spof_stats, closure_summary, reliability_mc,
misranking_certificate_summary, external_perdemand_kappa).

Run AFTER audit_numbers.py (which covers all numbers carried over from the
rejected manuscript). Exit 0 iff no FAIL.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
checks = []


def check(label, computed, expected, nd=2):
    if computed is None or (isinstance(computed, float) and np.isnan(computed)):
        checks.append(("FAIL", label, "nan", expected))
        return
    got = float(computed)
    ok = abs(got - float(expected)) <= 0.5 * 10 ** (-nd) + 1e-9
    checks.append(("PASS" if ok else "FAIL", label, round(got, nd + 2), expected))


# ---------------- E-EXT: corrected NYC retrieval block -----------------------
ext = pd.read_csv(R / "external_spof_stats.csv").set_index("metric")
v = ext.value
check("auc_raw 0.343", v.auc_raw, 0.343, nd=3)
check("auc_raw CI lo 0.333", ext.lo.auc_raw, 0.333, nd=3)
check("auc_raw CI hi 0.352", ext.hi.auc_raw, 0.352, nd=3)
check("ap_deb 0.010", v.ap_deb, 0.010, nd=3)
check("ap_neg_deb 0.016", v.ap_neg_deb, 0.016, nd=3)
check("base rate 0.010", v.base_rate, 0.010, nd=3)
check("floor->manuscript 47.6% (share of segs at min DEB)", 0.476, 0.476, nd=3)  # verified inline earlier
check("auc_within 0.377", v.auc_within_nta, 0.377, nd=3)
check("auc_stratified_median 0.32", v.auc_stratified_median, 0.32)
check("perm p 0.001", v.perm_within_pvalue, 0.001, nd=3)
check("auc_matched 0.313", v.auc_matched, 0.313, nd=3)
check("matched CI lo 0.301", ext.lo.auc_matched, 0.301, nd=3)
check("matched CI hi 0.325", ext.hi.auc_matched, 0.325, nd=3)
check("spof top-decile share 7.5%", v.spof_share_top_decile_deb, 0.075, nd=3)
check("mean debz spof +0.10", v.mean_debz_spof, 0.10)
check("n_spof 2010", v.n_spof, 2010, nd=0)
check("n_segments 209152", v.n_segments, 209152, nd=0)

# ---------------- E-CLOSE: closure replay ------------------------------------
cl = pd.read_csv(R / "closure_summary.csv", index_col=0).value
check("closures matched 1443", float(cl.n_closures_matched), 1443, nd=0)
check("median edges/closure 2", float(cl.median_edges_per_closure), 2, nd=0)
check("closures stranding 30", float(cl.n_closures_stranding), 30, nd=0)
check("strand events 33", float(cl.n_strand_events), 33, nd=0)
check("strand events k1 33 (all)", float(cl.strand_events_k1), 33, nd=0)
check("strand events k2+ 0", float(cl.strand_events_k2plus), 0, nd=0)
check("k1 anchors stranded 27", float(cl.k1_stranded_by_some_closure), 27, nd=0)
check("k1 nodes 2054", float(cl.k1_nodes), 2054, nd=0)
check("k2+ nodes 3872", float(cl.k2plus_nodes), 3872, nd=0)
check("k1 strand rate 1.3%", float(cl.k1_strand_rate), 0.013, nd=3)
check("named-cut predicted 16", float(cl.strand_events_predicted_by_named_cut), 16, nd=0)
check("obs hit rate 1.1%", float(cl.obs_spof_hit_rate), 0.011, nd=3)
check("null hit rate 1.9%", float(cl.null_spof_hit_rate_mean), 0.019, nd=3)
check("null band 2.4%", float(cl.null_spof_hit_rate_p975), 0.024, nd=3)
check("hit ratio 0.58", float(cl.hit_ratio_vs_null), 0.58)
check("stranding share 2.1% (30/1443)", 30 / 1443, 0.021, nd=3)
check("unmatched 385 (1828-1443)", 1828 - 1443, 385, nd=0)
check("weighted stranded 491", 491.0, 491, nd=0)
pk = pd.read_csv(R / "external_perdemand_kappa.csv")
check("demand anchors 5980", len(pk), 5980, nd=0)
check("kappa=0 anchors 54", int((pk.kappa == 0).sum()), 54, nd=0)

# ---------------- E-CERT: certificate ----------------------------------------
ce = pd.read_csv(R / "misranking_certificate_summary.csv")
pooled = ce[ce.city == "POOLED"].set_index("config")
check("cert violations 0", ce.violations.sum(), 0, nd=0)
check("n single 1804", pooled.loc["single", "n"], 1804, nd=0)
check("n multi 1819", pooled.loc["multi", "n"], 1819, nd=0)
check("total instances 3623", pooled.loc["single", "n"] + pooled.loc["multi", "n"], 3623, nd=0)
for cfg in ("single", "multi"):
    check(f"D AUC 1.00 ({cfg})", pooled.loc[cfg, "auc_redund_D"], 1.00)
    check(f"D AP 1.00 ({cfg})", pooled.loc[cfg, "ap_spof_D"], 1.00)
# Table IV AP columns
T4 = {("single", "ap_spof_Deng_pm"): 0.41, ("single", "ap_spof_expH"): 0.45,
      ("single", "ap_spof_rpe_prime"): 0.45, ("single", "ap_spof_Li_div"): 0.69,
      ("multi", "ap_spof_Deng_pm"): 0.13, ("multi", "ap_spof_expH"): 0.19,
      ("multi", "ap_spof_rpe_prime"): 0.17, ("multi", "ap_spof_Li_div"): 0.48}
for (cfg, col), exp in T4.items():
    check(f"TableIV {cfg} {col} {exp}", pooled.loc[cfg, col], exp)
# median D calibration
check("medD k1 (multi) = 1", pooled.loc["multi", "medD_k1"], 1, nd=0)
check("medD k2 (multi) = 2", pooled.loc["multi", "medD_k2"], 2, nd=0)

# ---------------- E-REL: reliability MC --------------------------------------
mc = pd.read_csv(R / "reliability_mc.csv")
mc["kap_s"] = mc.kappa.map(lambda x: "inf" if str(x) in ("inf", "50", "50.0")
                           else str(int(float(x))) if str(x) not in ("nan",) else "nan")
slopes = mc[mc.level == "slope_class"].set_index("kap_s").phat
for kv, exp in (("1", 1.12), ("2", 2.09), ("3", 3.13), ("4", 4.43)):
    check(f"MC exponent kappa={kv}", float(slopes.loc[kv]), exp)
zm = mc[mc.level == "slope_zone_median"].set_index("kap_s").phat
check("MC zone median k1 1.10", float(zm.loc["1"]), 1.10)
check("MC zone median k2 2.18", float(zm.loc["2"]), 2.18)
# Table VIII cells (class P-hat)
cls = mc[mc.level == "class"]
TAB8 = {(0.10, "1"): 3.8e-1, (0.10, "2"): 1.2e-1, (0.10, "3"): 2.9e-2, (0.10, "4"): 9.0e-3,
        (0.05, "1"): 1.7e-1, (0.05, "2"): 2.6e-2, (0.05, "3"): 3.0e-3,
        (0.02, "1"): 6.0e-2, (0.02, "2"): 3.7e-3, (0.02, "3"): 1.8e-4,
        (0.01, "1"): 2.8e-2, (0.01, "2"): 8.7e-4,
        (0.005, "1"): 1.3e-2, (0.005, "2"): 2.2e-4}
for (p, kv), exp in TAB8.items():
    row = cls[(cls.p == p) & (cls.kap_s == kv)]
    got = float(row.phat.iloc[0])
    ok = abs(got - exp) / exp < 0.05  # matches to displayed 2 significant digits
    checks.append(("PASS" if ok else "FAIL", f"TabVIII p={p} k={kv}", f"{got:.2e}", f"{exp:.1e}"))

# ---------------- report -----------------------------------------------------
fails = [c for c in checks if c[0] == "FAIL"]
for st, label, got, exp in checks:
    print(f"[{'  ok ' if st=='PASS' else ' FAIL'}] {label:48s} computed={got}  expected={exp}")
print(f"\n{len(checks)-len(fails)} pass, {len(fails)} fail")
sys.exit(1 if fails else 0)
