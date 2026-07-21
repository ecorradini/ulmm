"""Fase 0b -- direct empirical test of the achievability half (Prop. 8) + guard G8.

Prop. 8 claims: at B = B*_r the r-round augmenting-path generator returns r
pairwise edge-disjoint routes whenever kappa >= r, hence D = min(kappa, r). That
is the only half of the package the T1/T2 gate does NOT test, because T1 tests
necessity and T2 tests the ceiling.

Test: for a sample of non-co-located demands with kappa >= 2, build G^(B*_2),
run the max-flow capped at 2, decompose the integral flow into routes, and assert
(a) exactly 2 routes come out, (b) they share no substrate arc, i.e. D = 2.

G8 positive control: recompute B*_2 by exhaustive ascending sweep over the
candidate thresholds (no binary search) and assert it matches, plus assert
kappa(G^(B)) is non-decreasing along the sorted thresholds.

Reuses the flow decomposition of kappa_physical.py rather than reimplementing it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

import certification_frontier as cf

HERE = Path(__file__).resolve().parent
# Layout-agnostic root: in the working tree the scripts sit beside results/;
# in the released repository they live in code/ with results/ one level up.
if not (HERE / "results").exists() and (HERE.parent / "results").exists():
    HERE = HERE.parent
R = HERE / "results"
SAMPLE_PER_CITY = 40
SEED = 11


def flow_routes(C, ent, mask, r):
    """Capped max-flow on G^(B); decompose into r arc-disjoint routes."""
    N, eu, ev, acc = C["N"], C["eu"], C["ev"], C["acc"]
    VS, CAP, VT = N, N + 1, N + 2
    su, sv = eu[mask], ev[mask]
    rows = np.concatenate([su, np.full(len(ent), VS, np.int64), acc,
                           np.array([CAP], np.int64)])
    cols = np.concatenate([sv, np.asarray(ent, np.int64),
                           np.full(len(acc), CAP, np.int64),
                           np.array([VT], np.int64)])
    vals = np.concatenate([np.ones(len(su), np.int64),
                           np.full(len(ent), cf.BIG, np.int64),
                           np.full(len(acc), cf.BIG, np.int64),
                           np.array([r], np.int64)])
    M = csr_matrix((vals, (rows, cols)), shape=(N + 3, N + 3), dtype=np.int64)
    res = maximum_flow(M.tocsr(), VS, VT)
    if int(res.flow_value) < r:
        return None
    F = res.flow.tocoo()
    pos = {(int(i), int(j)): int(v) for i, j, v in zip(F.row, F.col, F.data) if v > 0}
    # cancel antiparallel substrate flow (kappa_physical.cancel_antiparallel)
    for (i, j) in list(pos):
        if (j, i) in pos and i < N and j < N:
            c = min(pos[(i, j)], pos[(j, i)])
            for key in ((i, j), (j, i)):
                pos[key] -= c
                if pos[key] <= 0:
                    pos.pop(key, None)
    # decompose VS->VT paths (kappa_physical.decompose)
    routes, rem = [], dict(pos)
    for _ in range(r):
        path, seen, steps = [VS], {VS: 0}, 0
        while path[-1] != VT and steps < 10 * N:
            i = path[-1]
            nxt = next((j for (a, j) in rem if a == i and rem[(a, j)] > 0), None)
            if nxt is None:
                return None
            rem[(i, nxt)] -= 1
            if rem[(i, nxt)] == 0:
                del rem[(i, nxt)]
            if nxt in seen:
                path = path[:seen[nxt] + 1]
                seen = {v: k for k, v in enumerate(path)}
            else:
                path.append(nxt)
                seen[nxt] = len(path) - 1
            steps += 1
        if path[-1] != VT:
            return None
        routes.append([n for n in path if n < N])
    return routes


def main():
    hor = pd.read_csv(R / "certification_horizon.csv")
    hor = hor[(~hor.colocated) & hor.reachable & (hor.kappa >= 2)
              & (hor.kappa < cf.KCAP) & hor.Bstar2.notna()]
    rng = np.random.RandomState(SEED)
    out = []
    for city in cf.CITIES:
        short = cf.city_short(city)
        sub = hor[hor.city == short]
        if not len(sub):
            continue
        take = sub.iloc[rng.choice(len(sub), min(SAMPLE_PER_CITY, len(sub)),
                                   replace=False)]
        C = cf.build_city(city)
        dA = cf.dist_to_access(C)
        ok8 = bad8 = ok_prop8 = bad_prop8 = 0
        for _, rec in take.iterrows():
            d = int(rec.d)
            if d not in C["idx"]:
                continue
            ent = cf.entry_set(C, d)
            dS = cf.dist_from_entry(C, ent)
            mu = dS[C["eu"]] + C["ew"] + dA[C["ev"]]
            finite = np.isfinite(mu)
            vals = np.unique(mu[finite])
            vals = vals[vals >= rec.L1 - 1e-9]

            # G8: exhaustive ascending sweep + monotonicity
            prev, sweep = -1, None
            for b in vals:
                kv = cf.kappa_capped(C, ent, mu <= b, 2)
                if kv < prev:
                    bad8 += 1
                    break
                prev = kv
                if kv >= 2:
                    sweep = float(b)
                    break
            if sweep is not None and abs(sweep - float(rec.Bstar2)) < 1e-6:
                ok8 += 1
            elif sweep is not None:
                bad8 += 1
                print(f"  G8 FAIL {short} d={d}: sweep={sweep} binary={rec.Bstar2}")

            # Prop. 8: augmenting generator at B*_2 must yield D = 2
            routes = flow_routes(C, ent, mu <= rec.Bstar2 + 1e-9, 2)
            if routes is None or len(routes) != 2:
                bad_prop8 += 1
                print(f"  PROP8 FAIL {short} d={d}: routes={routes if routes is None else len(routes)}")
                continue
            e0 = set(zip(routes[0][:-1], routes[0][1:]))
            e1 = set(zip(routes[1][:-1], routes[1][1:]))
            if e0 & e1:
                bad_prop8 += 1
                print(f"  PROP8 FAIL {short} d={d}: routes share {len(e0 & e1)} arcs")
            else:
                ok_prop8 += 1
        print(f"[{short}] G8 ok={ok8} bad={bad8} | Prop8 D=2 ok={ok_prop8} bad={bad_prop8}",
              flush=True)
        out.append(dict(city=short, g8_ok=ok8, g8_bad=bad8,
                        prop8_ok=ok_prop8, prop8_bad=bad_prop8))
    df = pd.DataFrame(out)
    df.to_csv(R / "frontier_achievability.csv", index=False)
    print("\n" + df.to_string(index=False))
    tot_bad = int(df.g8_bad.sum() + df.prop8_bad.sum())
    print(f"\nTOTAL FAILURES: {tot_bad} (must be 0)")
    sys.exit(1 if tot_bad else 0)


if __name__ == "__main__":
    main()
