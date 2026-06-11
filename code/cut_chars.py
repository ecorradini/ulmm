"""
Round-2 #1: characterize the kappa=1 min cut under the primary k=3 multi-entry anchoring.

Reviewer's correct observation: with uncapacitated super-source connectors to the 3 entry
nodes, a kappa=1 cut CANNOT be a connector (severing one connector leaves two); it is
necessarily a single SUBSTRATE edge downstream of all three entries -- a genuine shared
bottleneck. So 'cut at the entry set' is a PROXIMITY statement, not an artifact. We measure:
  - cut edge (cut_u, cut_v)  (node ids, for the NYC map)
  - cut_first : is the cut edge incident to (leaving) an entry node? (adjacent vs interior)
  - cut_hops  : hops from the entry set to the cut edge tail
  - entry_diam_m : spatial diameter (max pairwise metres) of the 3-node entry set
An ADJACENT cut on a SPREAD entry set is a real single-exit pocket; on a CLUSTERED entry set
it degrades toward a geometry artifact. We report the diameter distribution by cut location.

DA orientation (demand source), k=3, van. Outputs results/cut_chars.csv + a printed summary.
"""
import numpy as np, pandas as pd, os, time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from scipy.spatial import cKDTree
import ablations as ab

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; BIG = 1 << 24; KCAP = 50; KNN = 3
os.makedirs("results", exist_ok=True)


def main():
    rows = []
    for c in CITIES:
        short = c.split(",")[0]; print(f"[{c}] ...", flush=True); t0 = time.time()
        u = ab.load_ulmm(c); Gm = u["graph"]; G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
        lat0 = lat.mean(); XY = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
        tree = cKDTree(XY)
        eu = np.array([idx[a] for a, b in G.edges()], np.int64); ev = np.array([idx[b] for a, b in G.edges()], np.int64)
        acc = [idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx]
        d_nodes = sorted({int(n) for n in u["demand"]["i_node"].tolist() if int(n) in idx})
        SRC, SINK = N, N + 1
        for n in d_nodes:
            _, nn = tree.query(XY[idx[n]], k=KNN); ent = sorted(set([idx[n]] + [int(x) for x in np.atleast_1d(nn)]))
            sr = list(eu) + [SRC] * len(ent) + acc; co = list(ev) + ent + [SINK] * len(acc)
            vv = [1] * len(eu) + [BIG] * (len(ent) + len(acc))
            M = csr_matrix((np.array(vv, np.int64), (np.array(sr), np.array(co))), shape=(N + 2, N + 2))
            res = maximum_flow(M, SRC, SINK); kap = int(min(res.flow_value, KCAP))
            if kap != 1:
                continue
            F = res.flow; resid = (((M - F) > 0).astype(np.int8) + (F > 0).astype(np.int8).transpose()).tocsr()
            order, pred = breadth_first_order(resid, SRC, directed=True, return_predecessors=True)
            Rm = np.zeros(N + 2, bool); Rm[order] = True
            cut = np.where(Rm[eu] & ~Rm[ev])[0]
            if not len(cut):
                continue
            ci = cut[0]; tu, tv = int(eu[ci]), int(ev[ci])
            cut_first = int(tu in set(ent))
            h = 0; v = tu
            while v != SRC and v >= 0 and h < 10**6:
                v = pred[v]; h += 1
            cut_hops = max(h - 1, 0) if v == SRC else -1
            # entry-set spatial diameter (max pairwise metres)
            P = XY[ent]; diam = 0.0
            for i_ in range(len(P)):
                for j_ in range(i_ + 1, len(P)):
                    diam = max(diam, float(np.hypot(*(P[i_] - P[j_]))))
            # cut-edge endpoints as node ids (for mapping); both must be substrate nodes
            both_sub = int(tu < N and tv < N)
            rows.append(dict(city=short, d=n, cut_u=(nodes[tu] if tu < N else -1),
                             cut_v=(nodes[tv] if tv < N else -1), cut_first=cut_first,
                             cut_hops=cut_hops, entry_diam_m=round(diam, 1), cut_substrate=both_sub))
        df = pd.DataFrame(rows); df.to_csv("results/cut_chars.csv", index=False)
        cur = df[df.city == short]
        adj = cur[cur.cut_first == 1]; inte = cur[cur.cut_first == 0]
        print(f"  kappa=1 nodes={len(cur)}  cut_substrate(all?)={cur.cut_substrate.mean():.3f}  "
              f"adjacent={len(adj)} interior={len(inte)}  ({time.time()-t0:.0f}s)", flush=True)
        if len(adj):
            print(f"    entry-set diameter (m): adjacent median={adj.entry_diam_m.median():.0f} "
                  f"[{adj.entry_diam_m.quantile(.25):.0f},{adj.entry_diam_m.quantile(.75):.0f}]  "
                  f"interior median={inte.entry_diam_m.median():.0f}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== POOLED ===", flush=True)
    print("kappa=1 cuts are single substrate edges (share with both endpoints substrate):",
          round(float(df.cut_substrate.mean()), 4), flush=True)
    adj = df[df.cut_first == 1]; inte = df[df.cut_first == 0]
    print(f"adjacent cuts: {len(adj)} ({len(adj)/len(df):.2f}); entry diameter median={adj.entry_diam_m.median():.0f} m "
          f"(IQR {adj.entry_diam_m.quantile(.25):.0f}-{adj.entry_diam_m.quantile(.75):.0f})", flush=True)
    print(f"interior cuts: {len(inte)} ({len(inte)/len(df):.2f}); median hops={inte.cut_hops.median():.0f}, "
          f"entry diameter median={inte.entry_diam_m.median():.0f} m", flush=True)
    print(f"share of adjacent cuts whose entry set spans >50 m (genuine pocket): {float((adj.entry_diam_m>50).mean()):.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
