"""Fase 0 -- the gate itself: does the certification-frontier claim survive?

Two tests, deliberately distinct (a failure of T1 kills the theorem; a failure of
T2 only makes the a priori corollary uninformative).

T1 -- POINTWISE NECESSITY (falsification test).
    For every demand with D >= r under some generator cell, the realized horizon
    Bmax(P) = max route cost must satisfy Bmax >= B*_r. This needs no hypothesis
    beyond non-co-location and edge-simple routes: each route p in P is itself an
    S->A route of cost <= Bmax, so every edge of p lies in G^(Bmax), hence r
    disjoint routes inside G^(Bmax) force kappa(G^(Bmax)) >= r.
    A single violation, after the guards pass, falsifies the package.

T2 -- A PRIORI CEILING (informativeness test).
    coverage = P(D>=2 | kappa>=2) must be <= P(Btheory >= B*_2 | kappa>=2),
    computed cell by cell ON THE SAME DEMAND SET as the cell (the eps/yen cells
    use the disclosed 150/city subsample, so pooling against a different set is
    invalid -- that mistake is what this script exists to avoid).
    Reported alongside the realized-horizon ceiling using Bmax.

Outputs results/frontier_gate.csv and prints the verdict.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Layout-agnostic root: in the working tree the scripts sit beside results/;
# in the released repository they live in code/ with results/ one level up.
if not (HERE / "results").exists() and (HERE.parent / "results").exists():
    HERE = HERE.parent
R = HERE / "results"
KCAP = 50
TOL = 1e-9

hor = pd.read_csv(R / "certification_horizon.csv")
pdm = pd.read_csv(R / "certificate_envelope_perdemand.csv")
env = pd.read_csv(R / "certificate_envelope.csv")

# G9: several demand cells can snap to the SAME street node, so both frames carry
# duplicate (city,d) keys. B*_r and D depend only on (city,d[,generator,K]), so the
# duplicates are identical rows -- but merging without dedup multiplies them and
# silently reweights every rate. Deduplicate before joining.
dup_h = int(hor.duplicated(["city", "d"]).sum())
dup_p = int(pdm.duplicated(["city", "d", "generator", "K"]).sum())
hor = hor.drop_duplicates(["city", "d"])
pdm = pdm.drop_duplicates(["city", "d", "generator", "K"])
print(f"G9 dedup: dropped {dup_h} horizon and {dup_p} envelope duplicate rows")

# guards G4/G5: B*_r is undefined for co-located demands, meaningless if unreachable
hor = hor[(~hor.colocated) & hor.reachable]
m = pdm.merge(hor[["city", "d", "L1", "Bstar2", "Bstar3"]],
              on=["city", "d"], how="inner", suffixes=("", "_hor"))
assert not m.duplicated(["city", "d", "generator", "K"]).any(), "merge inflated rows"
print(f"merged rows: {len(m)}  (envelope {len(pdm)}, horizon {len(hor)})")

# real (non-co-located, non-sentinel) kappa only
m = m[m.kappa < KCAP]

# ---------------- T1: pointwise necessity ------------------------------------
viol = []
for r, bcol in ((2, "Bstar2"), (3, "Bstar3")):
    sel = m[(m.D >= r) & m[bcol].notna() & np.isfinite(m[bcol])]
    bad = sel[sel.Bmax < sel[bcol] - TOL]
    viol.append(dict(r=r, tested=len(sel), violations=len(bad)))
    if len(bad):
        print(f"\nT1 VIOLATION r={r}: {len(bad)} of {len(sel)}")
        print(bad[["city", "d", "generator", "K", "kappa", "D", "Bmax", bcol]]
              .head(10).to_string(index=False))
t1 = pd.DataFrame(viol)
print("\nT1 pointwise necessity (Bmax >= B*_r whenever D >= r):")
print(t1.to_string(index=False))

# ---------------- T2: per-cell ceilings --------------------------------------
rows = []
for (g, K), sub in m.groupby(["generator", "K"]):
    red = sub[sub.kappa >= 2]
    if not len(red):
        continue
    cov = float((red.D >= 2).mean())
    # Table XII's own coverage is computed on a DIFFERENT population: it keeps
    # co-located demands (kappa sentinel >= 2, which trivially certify) and one row per
    # demand cell rather than per snapped node. B*_r is undefined there, so the frontier
    # comparison cannot use it. Carry the envelope's number alongside so the two are
    # never conflated.
    er = env[(env.generator == g) & (env.K == K)]
    cov_tabXII = float(er.coverage.iloc[0]) if len(er) else np.nan
    n_tabXII = int(er.n.iloc[0]) if len(er) else -1
    ceil_th = float((red.Btheory >= red.Bstar2 - TOL).mean())
    ceil_re = float((red.Bmax >= red.Bstar2 - TOL).mean())
    rows.append(dict(generator=g, K=K, n=len(red), n_tabXII=n_tabXII,
                     coverage_tabXII=round(cov_tabXII, 4),
                     coverage=round(cov, 4),
                     ceiling_theory=round(ceil_th, 4),
                     ceiling_realized=round(ceil_re, 4),
                     slack_theory=round(ceil_th - cov, 4),
                     holds=bool(cov <= ceil_th + 1e-6),
                     vacuous=bool(ceil_th >= 0.95)))
t2 = pd.DataFrame(rows).sort_values(["generator", "K"])
print("\nT2 per-cell ceiling (same demand set per cell):")
print(t2.to_string(index=False))
t2.to_csv(R / "frontier_gate.csv", index=False)

# ---------------- verdict ----------------------------------------------------
n_viol = int(t1.violations.sum())
broken = t2[~t2.holds]
informative = t2[(~t2.vacuous) & t2.holds]
print("\n" + "=" * 68)
if n_viol:
    print(f"VERDICT: FALSIFIED -- {n_viol} pointwise violations. Theorem does not enter.")
elif len(broken):
    print(f"VERDICT: T2 BREACH in {len(broken)} cells (necessity holds pointwise, so "
          f"this indicates a horizon-formula or demand-set mismatch, not a false theorem).")
    print(broken.to_string(index=False))
else:
    print(f"VERDICT: HOLDS. 0 pointwise violations over {int(t1.tested.sum())} tests.")
    print(f"  informative cells (ceiling < 0.95): {len(informative)}/{len(t2)}")
    if len(informative):
        print(informative[["generator", "K", "coverage", "ceiling_theory",
                           "slack_theory"]].to_string(index=False))
print("=" * 68)
