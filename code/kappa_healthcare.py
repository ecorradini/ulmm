"""
Round-3 Major 6: second instantiation -- population-weighted healthcare access.

Same model, different coupling: access = hospitals, clinics, doctors; demand weight =
WorldPop population at the cell centroid. Amsterdam/Barcelona/Paris use the cached
*-Healthcare ULMM variants (healthcare access already anchored); Seattle and New York City
reuse the last-mile substrate and snap a fresh OSM healthcare query to it. kappa with the
primary multi-entry anchoring (k=3), both orientations, demand-bootstrap CIs.
Outputs: results/kappa_healthcare_perdemand.csv, results/kappa_healthcare_summary.csv
"""
import numpy as np, pandas as pd, time, os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree
import ablations as ab
from pop_weights import pop_weights

EU = {"Amsterdam": "Amsterdam-Healthcare", "Barcelona": "Barcelona-Healthcare",
      "Paris": "Paris-Healthcare"}
US = {"Seattle": "Seattle, Washington, USA", "New York City": "New York City, New York, USA"}
TAGS = {"amenity": ["hospital", "clinic", "doctors"]}
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


def us_health_access(u, XY, tree, idx_nodes):
    """OSM healthcare features snapped to the existing last-mile substrate."""
    import osmnx as ox
    from shapely.geometry import MultiPoint
    Gm = u["graph"]
    pts = MultiPoint([(Gm.nodes[n]["x"], Gm.nodes[n]["y"]) for n in list(Gm.nodes())[::5]])
    hull = pts.convex_hull
    feats = ox.features_from_polygon(hull, TAGS)
    geo = feats.geometry.representative_point()
    lon = geo.x.to_numpy(); lat = geo.y.to_numpy()
    lat0 = np.mean([Gm.nodes[n]["y"] for n in list(Gm.nodes())[:2000]])
    P = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
    _, nn = tree.query(P)
    return sorted(set(int(x) for x in np.atleast_1d(nn))), len(feats)


def main():
    perd, summ = [], []
    for short in list(EU) + list(US):
        key = EU.get(short, US.get(short))
        print(f"[{short} healthcare] ...", flush=True); t0 = time.time()
        u = ab.load_ulmm(key); Gm = u["graph"]; G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
        lat0 = lat.mean(); XY = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
        tree = cKDTree(XY)
        eu_ = np.array([idx[a] for a, b in G.edges()], np.int64)
        ev_ = np.array([idx[b] for a, b in G.edges()], np.int64)
        if short in EU:
            acc = sorted(set(idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx))
            n_fac = len(u["access"])
        else:
            acc, n_fac = us_health_access(u, XY, tree, idx)
        print(f"  access: {n_fac} facilities -> {len(acc)} anchored nodes", flush=True)
        zn = [int(n) for n in u["demand"]["i_node"].tolist()]
        keep = np.array([n in idx for n in zn])
        zone_node = [n for n, m in zip(zn, keep) if m]
        pw = pop_weights(u, short)[keep]
        unique = sorted(set(zone_node))
        entry = {}
        for n in unique:
            _, nn = tree.query(XY[idx[n]], k=KNN)
            entry[n] = sorted(set([idx[n]] + [int(x) for x in np.atleast_1d(nn)]))
        SRC, SINK = N, N + 1
        for ori in ("AD", "DA"):
            t1 = time.time(); memo = {}
            for n in unique:
                ent = entry[n]
                if ori == "DA":
                    sr = list(eu_) + [SRC] * len(ent) + acc; co = list(ev_) + ent + [SINK] * len(acc)
                else:
                    sr = list(eu_) + [SRC] * len(acc) + ent; co = list(ev_) + acc + [SINK] * len(ent)
                vv = [1] * len(eu_) + [BIG] * (len(ent) + len(acc))
                M = csr_matrix((np.array(vv, np.int64), (np.array(sr), np.array(co))), shape=(N + 2, N + 2))
                memo[n] = int(min(maximum_flow(M, SRC, SINK).flow_value, KCAP))
            kapz = np.array([memo[n] for n in zone_node]); w = pw
            lo, hi = boot_wshare(w, kapz == 1)
            perd.append(pd.DataFrame(dict(city=short, orientation=ori, d=zone_node, w=w, kappa=kapz)))
            res = dict(city=short, orientation=ori, n=len(kapz), n_access=len(acc),
                       w0=round(float(w[kapz == 0].sum() / w.sum()), 3),
                       w1=round(float(w[kapz == 1].sum() / w.sum()), 3),
                       w2=round(float(w[kapz >= 2].sum() / w.sum()), 3),
                       w1_ci=f"[{lo},{hi}]", med_kappa=float(np.median(kapz)))
            summ.append(res)
            print(f"  {ori}: w1={res['w1']} {res['w1_ci']} w2={res['w2']} med={res['med_kappa']} "
                  f"({time.time()-t1:.0f}s)", flush=True)
        pd.concat(perd).to_csv("results/kappa_healthcare_perdemand.csv", index=False)
        pd.DataFrame(summ).to_csv("results/kappa_healthcare_summary.csv", index=False)
        print(f"  [{short} done {time.time()-t0:.0f}s]", flush=True)
    allp = pd.concat(perd)
    for ori in ("AD", "DA"):
        s = allp[allp.orientation == ori]; w = s.w.to_numpy(); kap = s.kappa.to_numpy()
        lo, hi = boot_wshare(w, kap == 1)
        summ.append(dict(city="POOLED", orientation=ori, n=len(s), n_access=float("nan"),
                         w0=round(float(w[kap == 0].sum() / w.sum()), 3),
                         w1=round(float(w[kap == 1].sum() / w.sum()), 3),
                         w2=round(float(w[kap >= 2].sum() / w.sum()), 3), w1_ci=f"[{lo},{hi}]",
                         med_kappa=float(np.median(kap))))
    pd.DataFrame(summ).to_csv("results/kappa_healthcare_summary.csv", index=False)
    print("\n=== healthcare instantiation (population-weighted kappa) ===", flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
