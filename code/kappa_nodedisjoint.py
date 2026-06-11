"""
Round-3 Minor 8: node-disjoint access redundancy.

Node-splitting gadget: each substrate node v becomes (v_in -> v_out); interior nodes get
split capacity 1 (the node is the failure unit), entry/access endpoints get BIG (terminals
are not interior failure units). Substrate arcs u->w become u_out -> w_in with capacity 1
(node-disjoint paths are automatically edge-disjoint, and the unit arc cap only forbids the
degenerate reuse of one physical street between two unsplit terminals). Max-flow then counts
internally node-disjoint entry-to-access paths (vertex Menger).

kappa_node <= kappa_edge per demand by construction (same arcs + extra node caps); asserted
against results/reanchor_perdemand.csv. DA orientation, van, k in {1,3}.
Outputs: results/kappa_nodedisjoint_perdemand.csv, results/kappa_nodedisjoint_summary.csv
"""
import numpy as np, pandas as pd, time, os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree
import ablations as ab

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; BIG = 1 << 24; KCAP = 50; KNN = 3
os.makedirs("results", exist_ok=True)


def boot_wshare(w, mask, B=2000, seed=123):
    n = len(w)
    if n == 0 or w.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed); m = mask.astype(float); v = np.empty(B)
    for b in range(B):
        i = rng.randint(0, n, n); s = w[i].sum()
        v[b] = (w[i] * m[i]).sum() / s if s > 0 else np.nan
    return (round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3))


def kappa_node(N, eu, ev, ent, acc):
    """Internally node-disjoint max-flow from entry set to access set on the split graph."""
    ent = sorted(set(ent)); SRC, SINK = 2 * N, 2 * N + 1
    in_ = lambda v: 2 * v
    out_ = lambda v: 2 * v + 1
    split_cap = np.ones(N, np.int64)
    for a in acc:
        split_cap[a] = BIG
    for e in ent:
        split_cap[e] = BIG
    rows = list(out_(eu)) + [in_(v) for v in range(N)] + [SRC] * len(ent) + [out_(a) for a in acc]
    cols = list(in_(ev)) + [out_(v) for v in range(N)] + [in_(e) for e in ent] + [SINK] * len(acc)
    vals = [1] * len(eu) + list(split_cap) + [BIG] * len(ent) + [BIG] * len(acc)
    M = csr_matrix((np.array(vals, np.int64), (np.array(rows), np.array(cols))),
                   shape=(2 * N + 2, 2 * N + 2))
    return int(min(maximum_flow(M, SRC, SINK).flow_value, KCAP))


def selftest():
    # Toy: d->u->m->v->a and d->x->m->y->a share interior node m:
    # edge-disjoint kappa = 2, node-disjoint kappa = 1.
    names = ["d", "u", "m", "v", "a", "x", "y"]; ix = {s: i for i, s in enumerate(names)}
    E = [("d", "u"), ("u", "m"), ("m", "v"), ("v", "a"), ("d", "x"), ("x", "m"), ("m", "y"), ("y", "a")]
    eu = np.array([ix[a] for a, b in E], np.int64); ev = np.array([ix[b] for a, b in E], np.int64)
    kn = kappa_node(len(names), eu, ev, [ix["d"]], [ix["a"]])
    assert kn == 1, f"selftest: node-disjoint toy expected 1, got {kn}"
    # Two genuinely node-disjoint routes: d->u->a, d->v->a -> kappa_node = 2.
    names2 = ["d", "u", "v", "a"]; ix2 = {s: i for i, s in enumerate(names2)}
    E2 = [("d", "u"), ("u", "a"), ("d", "v"), ("v", "a")]
    eu2 = np.array([ix2[a] for a, b in E2], np.int64); ev2 = np.array([ix2[b] for a, b in E2], np.int64)
    kn2 = kappa_node(len(names2), eu2, ev2, [ix2["d"]], [ix2["a"]])
    assert kn2 == 2, f"selftest: disjoint toy expected 2, got {kn2}"
    print("selftest OK (shared-node toy: kappa_node=1; disjoint toy: kappa_node=2)", flush=True)


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
        eu = np.array([idx[a] for a, b in G.edges()], np.int64)
        ev = np.array([idx[b] for a, b in G.edges()], np.int64)
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
            t1 = time.time(); memo = {}
            for n in unique:
                ent = [idx[n]] if k == 1 else entry[n]
                kn = kappa_node(N, eu, ev, ent, acc)
                ke = kdir.get((short, k, n))
                if ke is not None:
                    assert kn <= ke, f"kappa_node>kappa_edge at {short} d={n} k={k}: {kn}>{ke}"
                memo[n] = kn
            kapz = np.array([memo[n] for n in zone_node]); w = zone_w
            lo, hi = boot_wshare(w, kapz == 1)
            perd.append(pd.DataFrame(dict(city=short, k=k, d=zone_node, w=zone_w, kappa_node=kapz)))
            res = dict(city=short, k=k, n=len(kapz),
                       w0=round(float(w[kapz == 0].sum() / w.sum()), 3),
                       w1=round(float(w[kapz == 1].sum() / w.sum()), 3),
                       w2=round(float(w[kapz >= 2].sum() / w.sum()), 3), w1_ci=f"[{lo},{hi}]")
            summ.append(res)
            print(f"  k={k}: node-disjoint w1={res['w1']} {res['w1_ci']} ({time.time()-t1:.0f}s)", flush=True)
        pd.concat(perd).to_csv("results/kappa_nodedisjoint_perdemand.csv", index=False)
        pd.DataFrame(summ).to_csv("results/kappa_nodedisjoint_summary.csv", index=False)
        print(f"  [{short} done {time.time()-t0:.0f}s]", flush=True)

    allp = pd.concat(perd)
    for k in (1, KNN):
        s = allp[allp.k == k]; w = s.w.to_numpy(); kap = s.kappa_node.to_numpy()
        lo, hi = boot_wshare(w, kap == 1)
        summ.append(dict(city="POOLED", k=k, n=len(s),
                         w0=round(float(w[kap == 0].sum() / w.sum()), 3),
                         w1=round(float(w[kap == 1].sum() / w.sum()), 3),
                         w2=round(float(w[kap >= 2].sum() / w.sum()), 3), w1_ci=f"[{lo},{hi}]"))
    pd.DataFrame(summ).to_csv("results/kappa_nodedisjoint_summary.csv", index=False)
    print("\n=== node-disjoint kappa ===", flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
