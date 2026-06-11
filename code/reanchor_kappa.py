"""
Part A: multi-entry anchoring as the PRIMARY kappa definition + cut-location + degree.

A demand zone (500 m cell) connects to its k nearest street nodes (multi-entry); kappa is
the min number of substrate edges separating that entry set from the access set. Several
zones may share a nearest node, so kappa is memoized per unique node while shares are
weighted per ZONE (fixes a weighting bug vs. a node-keyed dict).

Per city, van, both orientations, for k in {1,3}:
  - kappa(d) via scipy max-flow (super-source over one side, super-sink over the other)
  - demand-weighted shares kappa in {0,1,>=2} with demand-bootstrap CIs (over zones)
  - anchor out-degree; for kappa=1 zones: cut edge FIRST-out-of-entry vs interior + hops
Cut-location is reported in the DA framing (source = demand entry), where 'first edge out
of the anchor' is well defined.

Outputs: results/reanchor_perdemand.csv, results/reanchor_summary.csv
"""
import numpy as np, pandas as pd, time, os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
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


def main():
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
        # zones: (node, weight) preserving duplicates; unique nodes memoized
        zn = [int(n) for n in u["demand"]["i_node"].tolist()]
        zw = u["demand"]["w"].astype(float).to_numpy()
        keep = np.array([n in idx for n in zn])
        zone_node = [n for n, m in zip(zn, keep) if m]; zone_w = zw[keep]
        unique = sorted(set(zone_node))
        outdeg = {n: G.out_degree(n) for n in unique}
        entry = {}
        for n in unique:
            _, nn = tree.query(XY[idx[n]], k=KNN)
            entry[n] = sorted(set([idx[n]] + [int(x) for x in np.atleast_1d(nn)]))

        SRC, SINK = N, N + 1
        for ori in ("AD", "DA"):
            for k in (1, KNN):
                t1 = time.time(); memo = {}
                for n in unique:
                    ent = [idx[n]] if k == 1 else entry[n]
                    if ori == "DA":
                        sr = list(eu) + [SRC] * len(ent) + acc; co = list(ev) + ent + [SINK] * len(acc)
                    else:
                        sr = list(eu) + [SRC] * len(acc) + ent; co = list(ev) + acc + [SINK] * len(ent)
                    vv = [1] * len(eu) + [BIG] * (len(ent) + len(acc))
                    M = csr_matrix((np.array(vv, np.int64), (np.array(sr), np.array(co))), shape=(N + 2, N + 2))
                    res = maximum_flow(M, SRC, SINK); kap = int(min(res.flow_value, KCAP))
                    cf, ch = -1, -1
                    if kap == 1 and ori == "DA":   # cut-location only meaningful from the demand source
                        F = res.flow
                        resid = (((M - F) > 0).astype(np.int8) + (F > 0).astype(np.int8).transpose()).tocsr()
                        order, pred = breadth_first_order(resid, SRC, directed=True, return_predecessors=True)
                        Rm = np.zeros(N + 2, bool); Rm[order] = True
                        cut = np.where(Rm[eu] & ~Rm[ev])[0]
                        if len(cut):
                            tail = int(eu[cut[0]]); cf = int(tail in set(ent))
                            h = 0; v = tail
                            while v != SRC and v >= 0 and h < 10**6:
                                v = pred[v]; h += 1
                            ch = max(h - 1, 0) if v == SRC else -1
                    memo[n] = (kap, cf, ch)
                # expand to zones
                kapz = np.array([memo[n][0] for n in zone_node])
                cfz = np.array([memo[n][1] for n in zone_node]); chz = np.array([memo[n][2] for n in zone_node])
                odz = np.array([outdeg[n] for n in zone_node])
                df = pd.DataFrame(dict(city=short, orientation=ori, k=k, d=zone_node, w=zone_w,
                                       kappa=kapz, outdeg=odz, cut_first=cfz, cut_hops=chz))
                perd.append(df)
                w = zone_w; lo, hi = boot_wshare(w, kapz == 1)
                spofm = (kapz == 1)
                cf_share = (float((w[spofm & (cfz == 1)].sum()) / w[spofm].sum())
                            if (ori == "DA" and spofm.any()) else float("nan"))
                res = dict(city=short, orientation=ori, k=k, n=len(df),
                           w0=round(float(w[kapz == 0].sum() / w.sum()), 3),
                           w1=round(float(w[kapz == 1].sum() / w.sum()), 3),
                           w2=round(float(w[kapz >= 2].sum() / w.sum()), 3),
                           w1_ci=f"[{lo},{hi}]", med_kappa=float(np.median(kapz)), max_kappa=int(kapz.max()),
                           cut_first_share=round(cf_share, 3) if cf_share == cf_share else float("nan"))
                summ.append(res)
                tag = "" if ori == "AD" else f" cut_first={res['cut_first_share']}"
                print(f"  {ori} k={k}: w1={res['w1']} {res['w1_ci']} w0={res['w0']} w2={res['w2']}{tag} ({time.time()-t1:.0f}s)", flush=True)
        pd.concat(perd).to_csv("results/reanchor_perdemand.csv", index=False)
        pd.DataFrame(summ).to_csv("results/reanchor_summary.csv", index=False)
        print(f"  [{short} done {time.time()-t0:.0f}s]", flush=True)
    allp = pd.concat(perd)
    for ori in ("AD", "DA"):
        for k in (1, KNN):
            s = allp[(allp.orientation == ori) & (allp.k == k)]; w = s.w.to_numpy(); kap = s.kappa.to_numpy()
            lo, hi = boot_wshare(w, kap == 1)
            spm = (kap == 1); cfz = s.cut_first.to_numpy()
            cf = (float(w[spm & (cfz == 1)].sum() / w[spm].sum()) if ori == "DA" and spm.any() else float("nan"))
            summ.append(dict(city="POOLED", orientation=ori, k=k, n=len(s),
                             w0=round(float(w[kap == 0].sum() / w.sum()), 3),
                             w1=round(float(w[kap == 1].sum() / w.sum()), 3),
                             w2=round(float(w[kap >= 2].sum() / w.sum()), 3), w1_ci=f"[{lo},{hi}]",
                             cut_first_share=round(cf, 3) if cf == cf else float("nan")))
    pd.DataFrame(summ).to_csv("results/reanchor_summary.csv", index=False)
    print("\n=== multi-entry (k=3) vs single-entry (k=1) kappa ===", flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
