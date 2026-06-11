"""
E6: robustness of the kappa law.
 (a) Topological invariance (T2): kappa is IDENTICAL across friction salience lambda
     in {0.5,1,2} -- only edge costs change, never edge existence. Decisive empirical T2.
 (b) Anchor-artifact control: connect each demand to a MULTI-ENTRY set
     {anchor} u N_in(anchor) u N_out(anchor) (1-hop), removing the single-anchor-edge
     bottleneck. If kappa=1 still dominates, the single point of failure is a genuine
     network bottleneck, not an artifact of nearest-node anchoring.
 (c) Node-disjoint variant (intersection failures) via node-splitting, small cities.
DA orientation, van. scipy maximum_flow.
"""
import numpy as np, pandas as pd, networkx as nx, os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import ablations as ab

SMALL = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France", "Seattle, Washington, USA"]
VEH = "van"; BIG = 1 << 24
os.makedirs("results", exist_ok=True)


def kappa_multi(G, idx, N, entry_sets, weights, access_idx):
    """kappa for each demand whose ENTRY set (list of node-idx) feeds a super-source;
    sink = super-sink over access_idx. Returns array of kappa."""
    TERM = N            # sink
    SRC = N + 1         # source
    eu = [idx[u] for u, v in G.edges()]; ev = [idx[v] for u, v in G.edges()]
    base_r = eu + list(access_idx); base_c = ev + [TERM] * len(access_idx)
    base_v = [1] * len(eu) + [BIG] * len(access_idx)
    out = []
    for ent, w in zip(entry_sets, weights):
        rows = base_r + [SRC] * len(ent); cols = base_c + list(ent); vals = base_v + [BIG] * len(ent)
        M = csr_matrix((np.array(vals, np.int64), (np.array(rows), np.array(cols))), shape=(N + 2, N + 2))
        out.append(int(min(maximum_flow(M, SRC, TERM).flow_value, 50)))
    return np.array(out)


def wshare(w, mask):
    return float(w[mask].sum() / w.sum()) if w.sum() > 0 else float("nan")


def main():
    rows = []
    # (a) lambda-invariance on Barcelona (has non-trivial kappa>=2 and kappa=0)
    print("=== (a) lambda-invariance (Barcelona) ===", flush=True)
    u = ab.load_ulmm("Barcelona, Spain")
    prev = None
    for lam in [0.5, 1.0, 2.0]:
        G, _, _ = ab.collapsed_graph(u, VEH, lambda_override=lam)
        idx = {n: i for i, n in enumerate(G.nodes())}; N = len(idx); TERM = N
        eu = [idx[a] for a, b in G.edges()]; ev = [idx[b] for a, b in G.edges()]
        acc = [idx[int(n)] for n in u["access"]["i_node"] if int(n) in idx]
        M = csr_matrix((np.array([1] * len(eu) + [BIG] * len(acc), np.int64),
                        (np.array(eu + acc), np.array(ev + [TERM] * len(acc)))), shape=(N + 1, N + 1))
        d_nodes = [int(n) for n in u["demand"]["i_node"] if int(n) in idx]
        kap = np.array([int(min(maximum_flow(M, idx[d], TERM).flow_value, 50)) for d in d_nodes])
        print(f"  lambda={lam}: median={np.median(kap)} mean={kap.mean():.3f} "
              f"share(k=1)={(kap==1).mean():.3f} share(k>=2)={(kap>=2).mean():.3f}", flush=True)
        if prev is not None:
            print(f"    identical to lambda=prev: {np.array_equal(kap, prev)}", flush=True)
        prev = kap

    # (b) anchor-artifact control: single-entry vs multi-entry (1-hop)
    print("\n=== (b) anchor-artifact control: single vs multi-entry kappa=1 share ===", flush=True)
    for c in SMALL:
        short = c.split(",")[0]; u = ab.load_ulmm(c)
        G, _, _ = ab.collapsed_graph(u, VEH)
        idx = {n: i for i, n in enumerate(G.nodes())}; N = len(idx)
        acc = [idx[int(n)] for n in u["access"]["i_node"] if int(n) in idx]
        d_nodes = [int(n) for n in u["demand"]["i_node"] if int(n) in idx]
        w = u["demand"]["w"].astype(float).to_numpy()[[int(n) in idx for n in u["demand"]["i_node"]]]
        single = [[idx[d]] for d in d_nodes]
        multi = []
        for d in d_nodes:
            ent = {idx[d]}
            for nb in list(G.successors(d)) + list(G.predecessors(d)):
                if nb in idx: ent.add(idx[nb])
            multi.append(sorted(ent))
        k_s = kappa_multi(G, idx, N, single, w, acc)
        k_m = kappa_multi(G, idx, N, multi, w, acc)
        print(f"  {short}: single  w(k=1)={wshare(w,k_s==1):.3f} w(k=0)={wshare(w,k_s==0):.3f} | "
              f"multi  w(k=1)={wshare(w,k_m==1):.3f} w(k=0)={wshare(w,k_m==0):.3f}", flush=True)
        rows.append(dict(city=short, single_w_spof=round(wshare(w, k_s == 1), 3),
                         single_w_unreach=round(wshare(w, k_s == 0), 3),
                         multi_w_spof=round(wshare(w, k_m == 1), 3),
                         multi_w_unreach=round(wshare(w, k_m == 0), 3)))
    pd.DataFrame(rows).to_csv("results/e6_anchor.csv", index=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
