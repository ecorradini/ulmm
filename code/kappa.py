"""
E1 (decisive): exact Access Redundancy kappa(d) on all 5 cities incl. NYC.

kappa(d) = max number of EDGE-DISJOINT directed paths between the demand anchor pi(d)
and the access set A (super-terminal over every access anchor rho(a); unit capacity on
substrate edges, big capacity on access<->terminal arcs). By Menger / max-flow-min-cut,
kappa(d) = min number of substrate edges whose removal isolates d from ALL access.

Orientations:
  DA: demand -> access  (collection: demand reaches service)  flow(d  -> SINK),  a->SINK
  AD: access -> demand  (delivery:  service reaches demand)   flow(SRC -> d),    SRC->a
(They differ only through one-way streets; kappa ignores edge costs -> topological invariant.)

Engine: scipy.sparse.csgraph.maximum_flow (C, ~5 ms/call). Build the capacity matrix once
per city/orientation; call per demand. ~minutes for NYC.

Outputs:
  results/kappa_perdemand.csv : city, orientation, d, w, kappa, reachable, in_lscc
  results/kappa_summary.csv   : per-city + pooled weighted shares (kappa=0/1/>=2) + bootstrap CIs
"""
import numpy as np, pandas as pd, networkx as nx, time, os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import ablations as ab

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"
BIG = 1 << 24
KCAP = 50          # display cap (kappa is small; guards d-in-access pathology)
os.makedirs("results", exist_ok=True)


def build_matrices(G, access_nodes):
    """Return (M_DA, M_AD, idx, N): CSR capacity matrices of size (N+1)."""
    nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    TERM = N
    eu = [idx[u] for u, v in G.edges()]
    ev = [idx[v] for u, v in G.edges()]
    aset = [idx[a] for a in set(int(x) for x in access_nodes) if a in idx]
    # DA: substrate u->v (cap1) + a->SINK (BIG); flow(d, SINK)
    rows = eu + aset
    cols = ev + [TERM] * len(aset)
    vals = [1] * len(eu) + [BIG] * len(aset)
    M_DA = csr_matrix((np.array(vals, np.int64), (np.array(rows), np.array(cols))),
                      shape=(N + 1, N + 1))
    # AD: substrate u->v (cap1) + SRC->a (BIG); flow(SRC, d)
    rows = eu + [TERM] * len(aset)
    cols = ev + aset
    vals = [1] * len(eu) + [BIG] * len(aset)
    M_AD = csr_matrix((np.array(vals, np.int64), (np.array(rows), np.array(cols))),
                      shape=(N + 1, N + 1))
    return M_DA, M_AD, idx, N, len(aset)


def kappa_series(M, idx, N, demand_nodes, ori):
    TERM = N
    out = []
    for d in demand_nodes:
        if d not in idx:
            out.append(0); continue
        di = idx[d]
        try:
            if ori == "DA":
                fv = maximum_flow(M, di, TERM).flow_value
            else:  # AD
                fv = maximum_flow(M, TERM, di).flow_value
        except Exception:
            fv = 0
        out.append(int(min(fv, KCAP)))
    return np.array(out, int)


def wshare(w, mask):
    tot = w.sum()
    return float(w[mask].sum() / tot) if tot > 0 else float("nan")


def boot_wshare(w, mask, B=2000, seed=12345):
    n = len(w)
    if n == 0 or w.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    m = mask.astype(float); vals = np.empty(B)
    for b in range(B):
        i = rng.randint(0, n, n)
        s = w[i].sum()
        vals[b] = (w[i] * m[i]).sum() / s if s > 0 else np.nan
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def main():
    perd, summ = [], []
    for c in CITIES:
        short = c.split(",")[0]
        print(f"[{c}] loading...", flush=True)
        u = ab.load_ulmm(c)
        G, _, _ = ab.collapsed_graph(u, VEH)
        demand = u["demand"]
        d_nodes = [int(n) for n in demand["i_node"].tolist()]
        w = demand["w"].astype(float).to_numpy()
        a_nodes = [int(n) for n in u["access"]["i_node"].tolist()]
        lscc = max(nx.strongly_connected_components(G), key=len)
        in_lscc = np.array([1 if d in lscc else 0 for d in d_nodes])
        t0 = time.time()
        M_DA, M_AD, idx, N, apresent = build_matrices(G, a_nodes)
        print(f"  |V|={G.number_of_nodes()} |E|={G.number_of_edges()} |D|={len(d_nodes)} "
              f"|A|={apresent} |LSCC|={len(lscc)} (build {time.time()-t0:.1f}s)", flush=True)

        for ori, M in (("AD", M_AD), ("DA", M_DA)):
            t1 = time.time()
            kap = kappa_series(M, idx, N, d_nodes, ori)
            dt = time.time() - t1
            perd.append(pd.DataFrame(dict(city=short, orientation=ori, d=d_nodes, w=w,
                                          kappa=kap, reachable=(kap > 0).astype(int), in_lscc=in_lscc)))
            lo, hi = boot_wshare(w, kap == 1)
            res = dict(city=short, orientation=ori, n=len(d_nodes), access=apresent, secs=round(dt, 1),
                       w_unreach=round(wshare(w, kap == 0), 4),
                       w_spof=round(wshare(w, kap == 1), 4),
                       w_redund=round(wshare(w, kap >= 2), 4),
                       w_spof_ci=f"[{lo:.3f},{hi:.3f}]",
                       med_kappa=float(np.median(kap)), mean_kappa=round(float(kap.mean()), 3),
                       p90_kappa=float(np.percentile(kap, 90)), max_kappa=int(kap.max()),
                       frac_in_lscc=round(float(in_lscc.mean()), 3),
                       w_spof_lscc=round(wshare(w[in_lscc == 1], (kap[in_lscc == 1] == 1)), 4))
            summ.append(res)
            print(f"  {ori}: w(k=0)={res['w_unreach']}  w(k=1)={res['w_spof']} {res['w_spof_ci']}  "
                  f"w(k>=2)={res['w_redund']}  med={res['med_kappa']} max={res['max_kappa']} ({dt:.1f}s)", flush=True)

        pd.concat(perd).to_csv("results/kappa_perdemand.csv", index=False)
        pd.DataFrame(summ).to_csv("results/kappa_summary.csv", index=False)

    allp = pd.concat(perd)
    for ori in ("AD", "DA"):
        s = allp[allp.orientation == ori]; w = s.w.to_numpy(float); k = s.kappa.to_numpy()
        lo, hi = boot_wshare(w, k == 1)
        summ.append(dict(city="POOLED", orientation=ori, n=len(s),
                         w_unreach=round(wshare(w, k == 0), 4), w_spof=round(wshare(w, k == 1), 4),
                         w_redund=round(wshare(w, k >= 2), 4), w_spof_ci=f"[{lo:.3f},{hi:.3f}]",
                         med_kappa=float(np.median(k)), mean_kappa=round(float(k.mean()), 3),
                         max_kappa=int(k.max())))
    pd.DataFrame(summ).to_csv("results/kappa_summary.csv", index=False)
    print("\n=== kappa summary ===", flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
