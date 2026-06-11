"""
Round-3 Major 2: graph-edge disjointness vs physical street independence.

Theory (stated as a Proposition in the paper): on the collapsed digraph, both directions of
a two-way street are antiparallel unit-capacity arcs; by flow cancellation, an integral
max-flow never needs both directions of one segment, so the directed kappa ALREADY equals
the maximum number of routes disjoint on undirected physical segments (same node pair), and
its min cut is a set of segments. Antiparallel twins therefore do not inflate kappa.

The residual case is divided roads / dual carriageways mapped as separate node chains that
share the same OSM way id. For every kappa_dir = 2 demand (DA, k in {1,3}) we:
  1. run max-flow, cancel antiparallel flow (implementing the Proposition), decompose into
     the two routes, and check whether they share any OSM way id;
  2. if they share, remove all arcs of the shared way(s) and retry (<=3 rounds): if two
     routes exist avoiding the shared ways and are way-disjoint -> certified "witness";
     if the flow drops below 2 -> "affected" (conservatively counts long same-way streets
     and genuine dual carriageways alike, an upper bound on the divided-road exposure).
Outputs: results/kappa_physical_perdemand.csv, results/kappa_physical_summary.csv
"""
import numpy as np, pandas as pd, time, os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree
import ablations as ab

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; BIG = 1 << 24; KCAP = 50; KNN = 3; MAXROUNDS = 3
os.makedirs("results", exist_ok=True)


def flow_run(N, eu, ev, ent, acc, keep_mask=None):
    """DA max-flow; returns (value, positive-flow dict)."""
    SRC, SINK = N, N + 1
    if keep_mask is None:
        keep_mask = np.ones(len(eu), bool)
    e1, e2 = eu[keep_mask], ev[keep_mask]
    ent = sorted(set(ent))
    sr = list(e1) + [SRC] * len(ent) + list(acc)
    co = list(e2) + ent + [SINK] * len(acc)
    vv = [1] * len(e1) + [BIG] * (len(ent) + len(acc))
    M = csr_matrix((np.array(vv, np.int64), (np.array(sr), np.array(co))), shape=(N + 2, N + 2))
    res = maximum_flow(M, SRC, SINK)
    F = res.flow.tocoo(); pos = {}
    for i, j, v in zip(F.row, F.col, F.data):
        if v > 0:
            pos[(int(i), int(j))] = int(v)
    return int(min(res.flow_value, KCAP)), pos


def cancel_antiparallel(pos):
    """Subtract min flow on jointly-used antiparallel pairs (the Proposition, computationally).
    Returns the cancelled dict and the number of pairs that were jointly used."""
    twins = 0
    for (i, j) in [p for p in list(pos) if (p[1], p[0]) in pos and p[0] < p[1]]:
        a, b = pos.get((i, j), 0), pos.get((j, i), 0)
        m = min(a, b)
        if m > 0:
            twins += 1
            for p, v in (((i, j), a - m), ((j, i), b - m)):
                if v > 0:
                    pos[p] = v
                else:
                    pos.pop(p, None)
    return pos, twins


def decompose(pos, SRC, SINK, value, nmax):
    """Decompose a value-`value` integral flow into SRC->SINK paths (loops spliced out)."""
    rem = dict(pos); paths = []
    for _ in range(value):
        path = [SRC]; seen = {SRC: 0}; steps = 0
        while path[-1] != SINK and steps < nmax:
            i = path[-1]
            nxt = next((j for (a, j) in rem if a == i and rem[(a, j)] > 0), None)
            if nxt is None:
                return None
            rem[(i, nxt)] -= 1
            if rem[(i, nxt)] == 0:
                del rem[(i, nxt)]
            if nxt in seen:                      # splice the loop out, keep arcs consumed
                path = path[:seen[nxt] + 1]
                seen = {v: k for k, v in enumerate(path)}
            else:
                path.append(nxt); seen[nxt] = len(path) - 1
            steps += 1
        if path[-1] != SINK:
            return None
        paths.append(path[1:-1])                  # strip SRC, SINK -> substrate nodes only
    return paths


def way_keys(path, arckey):
    return {arckey[(a, b)] for a, b in zip(path[:-1], path[1:]) if (a, b) in arckey}


def selftest():
    # Cancellation toy: d->u, d->v, u->v, v->u, u->a, v->a. kappa=2, and after cancellation
    # the two decomposed routes must not share an undirected node-pair segment.
    names = ["d", "u", "v", "a"]; ix = {s: i for i, s in enumerate(names)}; N = len(names)
    E = [("d", "u"), ("d", "v"), ("u", "v"), ("v", "u"), ("u", "a"), ("v", "a")]
    eu = np.array([ix[a] for a, b in E], np.int64); ev = np.array([ix[b] for a, b in E], np.int64)
    val, pos = flow_run(N, eu, ev, [ix["d"]], [ix["a"]])
    assert val == 2, f"selftest: cancellation toy kappa expected 2, got {val}"
    pos, _ = cancel_antiparallel(pos)
    paths = decompose(pos, N, N + 1, 2, 50)
    assert paths is not None and len(paths) == 2, "selftest: decomposition failed"
    segs = [ {frozenset(p_) for p_ in zip(p[:-1], p[1:])} for p in paths ]
    assert not (segs[0] & segs[1]), f"selftest: routes share a segment after cancellation: {segs}"
    # Divided-road toy: two chains d->p1->p2->a / d->q1->q2->a with (p1,p2),(q1,q2) on way W.
    names2 = ["d", "p1", "p2", "q1", "q2", "a"]; ix2 = {s: i for i, s in enumerate(names2)}
    E2 = [("d", "p1"), ("p1", "p2"), ("p2", "a"), ("d", "q1"), ("q1", "q2"), ("q2", "a")]
    eu2 = np.array([ix2[a] for a, b in E2], np.int64); ev2 = np.array([ix2[b] for a, b in E2], np.int64)
    key2 = {(ix2[a], ix2[b]): ("W" if (a, b) in [("p1", "p2"), ("q1", "q2")] else f"u{a}{b}") for a, b in E2}
    val2, pos2 = flow_run(len(names2), eu2, ev2, [ix2["d"]], [ix2["a"]])
    pos2, _ = cancel_antiparallel(pos2)
    paths2 = decompose(pos2, len(names2), len(names2) + 1, 2, 50)
    shared = way_keys(paths2[0], key2) & way_keys(paths2[1], key2)
    assert shared == {"W"}, f"selftest: divided-road toy expected shared way W, got {shared}"
    print("selftest OK (cancellation toy segment-disjoint; divided-road toy flagged)", flush=True)


def main():
    selftest()
    rk = pd.read_csv("results/reanchor_perdemand.csv")
    rk = rk[rk.orientation == "DA"].drop_duplicates(subset=["city", "k", "d"])
    kdir = {(r.city, int(r.k), int(r.d)): int(r.kappa) for r in rk.itertuples()}
    perd, summ = [], []
    for c in CITIES:
        short = c.split(",")[0]; print(f"[{c}] ...", flush=True); t0 = time.time()
        u = ab.load_ulmm(c); Gm = u["graph"]; G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
        lat0 = lat.mean(); XY = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
        tree = cKDTree(XY)
        edges = list(G.edges())
        eu = np.array([idx[a] for a, b in edges], np.int64)
        ev = np.array([idx[b] for a, b in edges], np.int64)
        # OSM way key per collapsed arc (the min-effective-cost parallel arc, as in the collapse)
        wattr = f"c_eff_{VEH}"; arckey = {}; ekey = np.empty(len(edges), object)
        for t, (a, b) in enumerate(edges):
            dd = Gm.get_edge_data(a, b)
            best = min(dd.values(), key=lambda d: d.get(wattr, float("inf")))
            o = best.get("osmid")
            kk = (min(o) if isinstance(o, (list, tuple, set)) else o)
            if kk is None:
                kk = ("noway", int(idx[a]), int(idx[b]))
            arckey[(int(idx[a]), int(idx[b]))] = kk; ekey[t] = kk
        acc = [idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx]
        zn = [int(n) for n in u["demand"]["i_node"].tolist()]
        zw = u["demand"]["w"].astype(float).to_numpy()
        keep = np.array([n in idx for n in zn])
        zone_node = [n for n, m in zip(zn, keep) if m]; zone_w = zw[keep]
        unique = sorted(set(zone_node))
        entry = {}
        for n in unique:
            _, nn = tree.query(XY[idx[n]], k=KNN)
            entry[n] = sorted(set([idx[n]] + [int(x) for x in np.atleast_1d(nn)]))

        for k in (1, KNN):
            t1 = time.time(); memo = {}; twins_used = 0; n_k2 = 0
            for n in unique:
                if kdir.get((short, k, n)) != 2:
                    continue
                n_k2 += 1
                ent = [idx[n]] if k == 1 else entry[n]
                val, pos = flow_run(N, eu, ev, ent, acc)
                if val < 2:
                    memo[n] = "affected"; continue
                pos, tw = cancel_antiparallel(pos); twins_used += int(tw > 0)
                status = None; removed = set()
                for _ in range(MAXROUNDS):
                    paths = decompose(pos, N, N + 1, 2, N + 4)
                    if paths is None or len(paths) < 2:
                        status = "affected"; break
                    shared = way_keys(paths[0], arckey) & way_keys(paths[1], arckey)
                    if not shared:
                        status = "witness"; break
                    removed |= shared
                    keepm = ~np.isin(ekey, list(removed))
                    val2, pos = flow_run(N, eu, ev, ent, acc, keep_mask=keepm)
                    if val2 < 2:
                        status = "affected"; break
                    pos, _ = cancel_antiparallel(pos)
                memo[n] = status or "affected"
            zmask = np.array([kdir.get((short, k, n)) == 2 for n in zone_node])
            stz = np.array([memo.get(n, "") for n in zone_node], object)
            w = zone_w
            wk2 = w[zmask].sum()
            res = dict(city=short, k=k, n_k2_zones=int(zmask.sum()),
                       w_witness=round(float(w[zmask & (stz == "witness")].sum() / wk2), 3) if wk2 > 0 else float("nan"),
                       w_affected=round(float(w[zmask & (stz == "affected")].sum() / wk2), 3) if wk2 > 0 else float("nan"),
                       frac_twin_flow=round(twins_used / n_k2, 3) if n_k2 else float("nan"))
            summ.append(res)
            for n in unique:
                if n in memo:
                    perd.append(dict(city=short, k=k, d=n, status=memo[n]))
            print(f"  k={k}: kappa_dir=2 zones={res['n_k2_zones']} witness={res['w_witness']} "
                  f"affected={res['w_affected']} twin_flow_frac={res['frac_twin_flow']} ({time.time()-t1:.0f}s)", flush=True)
        pd.DataFrame(perd).to_csv("results/kappa_physical_perdemand.csv", index=False)
        pd.DataFrame(summ).to_csv("results/kappa_physical_summary.csv", index=False)
        print(f"  [{short} done {time.time()-t0:.0f}s]", flush=True)

    # pooled: zone-count-weighted average of the per-city shares
    for k in (1, KNN):
        s = [r for r in summ if r["k"] == k and r["city"] != "POOLED"]
        wts = [r["n_k2_zones"] for r in s]
        if sum(wts):
            summ.append(dict(city="POOLED", k=k, n_k2_zones=int(sum(wts)),
                             w_witness=round(float(np.average([r["w_witness"] for r in s], weights=wts)), 3),
                             w_affected=round(float(np.average([r["w_affected"] for r in s], weights=wts)), 3),
                             frac_twin_flow=round(float(np.average([r["frac_twin_flow"] for r in s], weights=wts)), 3)))
    pd.DataFrame(summ).to_csv("results/kappa_physical_summary.csv", index=False)
    print("\n=== physical-segment diagnostics (kappa_dir=2 demands) ===", flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
