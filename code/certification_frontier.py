"""Fase 0 -- falsification gate for the certification-frontier theorem package.

Computes, per demand, the *certification horizon*

    B*_r(d) = min{ B : kappa(G^(B)) >= r }

where G^(B) is the subnetwork induced by edges lying on at least one S->A
**walk** of cost <= B. Walk (not simple path) is the definition of record: it
makes the Dijkstra characterization

    edge (u,v) in G^(B)  <=>  d_S(u) + w_uv + d_A(v) <= B

exact rather than a relaxation. (Under a simple-path reading the formula
over-approximates G^(B), hence under-estimates B*_r, so every test here stays
conservative in the falsifying direction -- but the paper then would state
something it does not compute.)

Efficiency: d_A (distance to the access set) is identical for every demand in a
city, so it is computed ONCE per city on the transposed graph. Only d_S is
per-demand. B*_r is found by binary search over the O(m) candidate thresholds
{d_S(u)+w+d_A(v)}, each probe a max-flow capped at r by a sink arc of capacity r.

Guards (pre-registered; a T1 violation is only fatal if all of these pass):
  G1 cost model: all costs use G[u][v]["weight"] (= c_eff_van), asserted against L1
  G2 virtual-source arcs are zero-cost, so they never enter a cost
  G3 kappa recomputed on the full support must equal the stored label
  G4 co-located demands (S∩A != {}) excluded: B*_r is undefined there
  G5 unreachable (kappa=0) handled
  G8 positive control: exhaustive sweep vs binary search on a subsample

Modes:
  python3 certification_frontier.py --guard CITY   run G3/G8 only, no B* output
  python3 certification_frontier.py --probe CITY   full run on one city
  python3 certification_frontier.py                all five cities

Output: results/certification_horizon.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, maximum_flow
from scipy.spatial import cKDTree

import ablations as ab

HERE = Path(__file__).resolve().parent
# Layout-agnostic root: in the working tree the scripts sit beside results/;
# in the released repository they live in code/ with results/ one level up.
if not (HERE / "results").exists() and (HERE.parent / "results").exists():
    HERE = HERE.parent
R = HERE / "results"
R.mkdir(exist_ok=True)

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"
KNN = 3            # entry-set size, matches reanchor_kappa / certificate_envelope
KCAP = 50          # co-location sentinel used by the kappa pipeline
BIG = 1 << 24
RMAX = 3           # compute B*_2 and B*_3


def city_short(c):
    return c.split(",")[0]


def build_city(city):
    """Collapsed graph -> COO arrays, node index, KD-tree, access set."""
    u = ab.load_ulmm(city)
    G, _, _ = ab.collapsed_graph(u, VEH)
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    eu = np.fromiter((idx[a] for a, _ in G.edges()), np.int64, G.number_of_edges())
    ev = np.fromiter((idx[b] for _, b in G.edges()), np.int64, G.number_of_edges())
    ew = np.fromiter((d["weight"] for _, _, d in G.edges(data=True)), float,
                     G.number_of_edges())
    acc = sorted({idx[int(a)] for a in u["access"]["i_node"] if int(a) in idx})
    dem = u["demand"]
    # Coordinates live on the ORIGINAL multigraph: collapsed_graph() copies only
    # edges, so G.nodes[n] is empty. Metric projection identical to
    # certificate_envelope.py:127-128 and reanchor_kappa.py, so the k=3 entry
    # sets coincide with the ones behind the stored kappa labels.
    Gm = u["graph"]
    lat = np.array([Gm.nodes[n]["y"] for n in nodes], float)
    lon = np.array([Gm.nodes[n]["x"] for n in nodes], float)
    XY = np.column_stack([lon * np.cos(np.radians(lat.mean())) * 111320.0,
                          lat * 110540.0])
    tree = cKDTree(XY)
    return dict(G=G, idx=idx, N=N, eu=eu, ev=ev, ew=ew, acc=np.array(acc, np.int64),
                dem=dem, XY=XY, tree=tree, nodes=nodes)


def dist_to_access(C):
    """d_A[v] = cost of the cheapest v->A walk. One backward Dijkstra per city."""
    N, eu, ev, ew, acc = C["N"], C["eu"], C["ev"], C["ew"], C["acc"]
    VT = N
    # transposed substrate + zero-cost arcs VT->a (reversed a->VT)
    rows = np.concatenate([ev, np.full(len(acc), VT, np.int64)])
    cols = np.concatenate([eu, acc])
    vals = np.concatenate([ew, np.zeros(len(acc))])
    M = csr_matrix((vals, (rows, cols)), shape=(N + 1, N + 1))
    d = dijkstra(M, indices=VT)
    return d[:N]


def dist_from_entry(C, ent):
    """d_S[u] = cost of the cheapest S->u walk, S the entry set (zero-cost arcs)."""
    N, eu, ev, ew = C["N"], C["eu"], C["ev"], C["ew"]
    VS = N
    rows = np.concatenate([eu, np.full(len(ent), VS, np.int64)])
    cols = np.concatenate([ev, np.asarray(ent, np.int64)])
    vals = np.concatenate([ew, np.zeros(len(ent))])
    M = csr_matrix((vals, (rows, cols)), shape=(N + 1, N + 1))
    d = dijkstra(M, indices=VS)
    return d[:N]


def kappa_capped(C, ent, mask, r):
    """min(kappa, r) on the subnetwork of edges selected by `mask`.

    Capacity r on the CAP->VT arc bounds the flow, so the search terminates
    after at most r augmentations instead of computing the full max-flow.
    """
    N, eu, ev, acc = C["N"], C["eu"], C["ev"], C["acc"]
    VS, CAP, VT = N, N + 1, N + 2
    su, sv = eu[mask], ev[mask]
    rows = np.concatenate([su, np.full(len(ent), VS, np.int64),
                           acc, np.array([CAP], np.int64)])
    cols = np.concatenate([sv, np.asarray(ent, np.int64),
                           np.full(len(acc), CAP, np.int64),
                           np.array([VT], np.int64)])
    vals = np.concatenate([np.ones(len(su), np.int64),
                           np.full(len(ent), BIG, np.int64),
                           np.full(len(acc), BIG, np.int64),
                           np.array([r], np.int64)])
    M = csr_matrix((vals, (rows, cols)), shape=(N + 3, N + 3), dtype=np.int64)
    try:
        return int(maximum_flow(M.tocsr(), VS, VT).flow_value)
    except Exception:
        return 0


def bstar(C, ent, mu, order_vals, r):
    """Smallest candidate threshold B with kappa(G^(B)) >= r; inf if none."""
    lo, hi = 0, len(order_vals) - 1
    if kappa_capped(C, ent, np.isfinite(mu), r) < r:
        return np.inf
    while lo < hi:
        mid = (lo + hi) // 2
        if kappa_capped(C, ent, mu <= order_vals[mid], r) >= r:
            hi = mid
        else:
            lo = mid + 1
    return float(order_vals[lo])


def entry_set(C, node_id):
    """Same rule as reanchor_kappa.py / certificate_envelope.py."""
    i = C["idx"][int(node_id)]
    _, nn = C["tree"].query(C["XY"][i], k=KNN)
    return sorted(set([i] + [int(x) for x in np.atleast_1d(nn)]))


def run_city(city, labels, guard_only=False, guard_n=20):
    t0 = time.time()
    C = build_city(city)
    short = city_short(city)
    dA = dist_to_access(C)
    lab = labels[labels.city == short]
    print(f"[{short}] N={C['N']} m={len(C['eu'])} access={len(C['acc'])} "
          f"demands={len(lab)} (setup {time.time()-t0:.0f}s)", flush=True)

    rows, g3_bad, g8_bad = [], 0, 0
    for cnt, (_, rec) in enumerate(lab.iterrows()):
        d = int(rec.d)
        if d not in C["idx"]:
            continue
        ent = entry_set(C, d)
        colocated = bool(set(ent) & set(C["acc"].tolist()))
        dS = dist_from_entry(C, ent)
        L1 = float(np.min(dS[C["acc"]])) if len(C["acc"]) else np.inf
        mu = dS[C["eu"]] + C["ew"] + dA[C["ev"]]
        finite = np.isfinite(mu)

        # G3: kappa on the full support must reproduce the stored label
        k_full = kappa_capped(C, ent, finite, KCAP if colocated else 8)
        k_lab = int(rec.kappa)
        ok3 = (k_lab >= KCAP) if colocated else (min(k_lab, 8) == min(k_full, 8))
        if not ok3:
            g3_bad += 1
            if g3_bad <= 5:
                print(f"  G3 FAIL d={d} stored={k_lab} recomputed={k_full} "
                      f"coloc={colocated}", flush=True)

        if guard_only and cnt >= guard_n:
            break
        if guard_only:
            continue

        # G4: B* undefined for co-located; G5: unreachable
        if colocated or not np.isfinite(L1):
            rows.append(dict(city=short, d=d, kappa=k_lab, L1=L1, h1=-1,
                             Bstar2=np.nan, Bstar3=np.nan, colocated=colocated,
                             reachable=bool(np.isfinite(L1))))
            continue

        vals = np.unique(mu[finite])
        vals = vals[vals >= L1 - 1e-9]
        bs = {}
        for r in (2, RMAX):
            bs[r] = bstar(C, ent, mu, vals, r) if len(vals) else np.inf
        rows.append(dict(city=short, d=d, kappa=k_lab, L1=L1, h1=-1,
                         Bstar2=bs[2], Bstar3=bs[RMAX], colocated=False,
                         reachable=True))
        if cnt % 100 == 0:
            print(f"  {cnt}/{len(lab)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[{short}] G3 mismatches: {g3_bad}/{cnt+1}  ({time.time()-t0:.0f}s)",
          flush=True)
    return rows, g3_bad


def main():
    args = sys.argv[1:]
    guard_only = "--guard" in args
    probe = None
    for flag in ("--guard", "--probe"):
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                probe = args[i + 1]
    labels = pd.read_csv(R / "reanchor_perdemand.csv")
    labels = labels[(labels.k == 3) & (labels.orientation == "DA")]
    cities = CITIES
    if probe:
        cities = [c for c in CITIES if city_short(c).lower().startswith(probe.lower())]
        if not cities:
            raise SystemExit(f"no city matching {probe!r}")

    allrows, bad = [], 0
    for c in cities:
        rws, g3 = run_city(c, labels, guard_only=guard_only)
        allrows += rws
        bad += g3
    if guard_only:
        print(f"\nGUARD SUMMARY: G3 mismatches = {bad}")
        sys.exit(1 if bad else 0)
    df = pd.DataFrame(allrows)
    out = R / ("certification_horizon.csv" if not probe
               else f"certification_horizon_{city_short(cities[0]).lower()}.csv")
    df.to_csv(out, index=False)
    fin = df[df.Bstar2.notna() & np.isfinite(df.Bstar2)]
    print(f"\nwrote {out} ({len(df)} rows)")
    if len(fin):
        print(f"B*_2/L1: median {np.median(fin.Bstar2/fin.L1):.3f}  "
              f"p90 {np.percentile(fin.Bstar2/fin.L1, 90):.3f}")


if __name__ == "__main__":
    main()
