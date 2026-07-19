"""
E-CERT: the sound disjoint-subfamily certificate D(P) on the misranking route ensembles.

For every demand in BOTH misranking configurations (single-entry and multi-entry,
replicating the exact sampling of misranking2.py / misranking_multientry.py: all demands
in the four smaller cities, 250 random NYC demands with seed 7), re-enumerate the K=15
penalized-Dijkstra routes and compute

  D(P) = size of the largest pairwise-edge-disjoint subfamily of the returned routes
         (exact, bitmask DP over <=15 routes; substrate edges only, virtual-source
          arcs stripped in the multi configuration).

Theory (paper Sec. certification): D(P) <= kappa always (a pairwise-disjoint subfamily is
an integral feasible flow; super-source connectors are uncapacitated). D is therefore a
SOUND lower-bound certificate computable from the ensemble alone, and the script's hard
acceptance gate is zero violations of D <= kappa. Completeness fails (D=1 with kappa>=2
whenever every returned pair overlaps), which the coverage numbers quantify.

Also recomputed per demand: the four diversity indices (replication check against the
persisted results/misranking_multientry_perdemand.csv values) and average-precision
metrics for flagging kappa=1 with each negated index vs the certificate.

Outputs:
  results/misranking_certificate_perdemand.csv
  results/misranking_certificate_summary.csv
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

import ablations as ab
from misranking2 import K, auc_safe, k_routes, measures

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; KNN = 3; EPSV = 1e-6; KCAP = 50
SAMPLE = {"New York City": 250}
os.makedirs("results", exist_ok=True)


def max_disjoint(edge_sets):
    """Exact size of the largest pairwise-disjoint subfamily (bitmask DP, n<=15)."""
    n = len(edge_sets)
    if n == 0:
        return 0
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not (edge_sets[i] & edge_sets[j]):
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    best = 1
    feas = bytearray(1 << n)
    feas[0] = 1
    for S in range(1, 1 << n):
        i = (S & -S).bit_length() - 1
        Sp = S ^ (1 << i)
        if feas[Sp] and (Sp & ~adj[i]) == 0:
            feas[S] = 1
            c = bin(S).count("1")
            if c > best:
                best = c
    return best


def ap_safe(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def run_city(c, kap_single, kap_multi):
    short = c.split(",")[0]
    print(f"[{c}] ...", flush=True)
    t0 = time.time()
    u = ab.load_ulmm(c)
    Gm = u["graph"]
    G, _, _ = ab.collapsed_graph(u, VEH)
    nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    mr = np.array([idx[a] for a, b in G.edges()], np.int64)
    mc = np.array([idx[b] for a, b in G.edges()], np.int64)
    md = np.array([float(G[a][b]["weight"]) for a, b in G.edges()], float)
    Msub = csr_matrix((md, (mr, mc)), shape=(N, N))
    Mlil = Msub.tolil()
    lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
    lat0 = lat.mean()
    XY = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
    tree = cKDTree(XY)
    access_set = {idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx}
    d_nodes = [int(n) for n in u["demand"]["i_node"].tolist() if int(n) in idx]
    cap = SAMPLE.get(short)
    if cap and len(d_nodes) > cap:
        d_nodes = sorted(np.random.RandomState(7).choice(d_nodes, cap, replace=False).tolist())

    recs = []
    VS = N
    for d in d_nodes:
        # ---- single-entry configuration (routes from the snapped node)
        if (short, d) in kap_single:
            routes = k_routes(Msub, idx[d], access_set, K)
            m = measures(routes, Mlil)
            if m is not None:
                Es = [set(zip(p[:-1], p[1:])) for p in routes]
                m.update(config="single", city=short, d=d,
                         kappa=kap_single[(short, d)], D=max_disjoint(Es))
                recs.append(m)
        # ---- multi-entry configuration (virtual source over the k=3 entry set)
        if (short, d) in kap_multi:
            _, nn = tree.query(XY[idx[d]], k=KNN)
            ent = sorted(set([idx[d]] + [int(x) for x in np.atleast_1d(nn)]))
            rows = np.concatenate([mr, np.full(len(ent), VS, np.int64)])
            cols = np.concatenate([mc, np.array(ent, np.int64)])
            vals = np.concatenate([md, np.full(len(ent), EPSV, float)])
            Mv = csr_matrix((vals, (rows, cols)), shape=(N + 1, N + 1))
            routes = [p[1:] for p in k_routes(Mv, VS, access_set, K) if len(p) > 1 and p[0] == VS]
            m = measures(routes, Mlil)
            if m is not None:
                Es = [set(zip(p[:-1], p[1:])) for p in routes]
                m.update(config="multi", city=short, d=d,
                         kappa=kap_multi[(short, d)], D=max_disjoint(Es))
                recs.append(m)
    print(f"  n={len(recs)} rows ({time.time()-t0:.1f}s)", flush=True)
    return recs


def summarize(df, config, city):
    y2 = (df.kappa >= 2).astype(int).to_numpy()
    y1 = (df.kappa == 1).astype(int).to_numpy()
    D = df.D.to_numpy()
    kap = df.kappa.to_numpy()
    real = kap < KCAP  # exclude co-located (kappa = infinity by convention) from violation stats
    res = dict(config=config, city=city, n=len(df),
               base_rate_k1=round(float(y1.mean()), 3),
               violations=int(((D > kap) & real).sum()),
               cover_D2_given_redund=round(float((D[y2 == 1] >= 2).mean()), 3) if y2.any() else np.nan,
               certified_share=round(float((D >= 2).mean()), 3),
               spof_share_in_uncertified=round(float(y1[D == 1].mean()), 3) if (D == 1).any() else np.nan,
               auc_redund_D=round(auc_safe(y2, D), 3),
               ap_spof_D=round(ap_safe(y1, -D.astype(float)), 3))
    for m in ["expH", "rpe_prime", "Li_div", "Deng_pm"]:
        x = df[m].to_numpy()
        res[f"auc_redund_{m}"] = round(auc_safe(y2, x), 3)
        res[f"ap_spof_{m}"] = round(ap_safe(y1, -x), 3)
    # Li_div residual signal within the uncertified stratum (D == 1)
    sub = df[df.D == 1]
    if len(sub) >= 10 and len(np.unique((sub.kappa >= 2))) > 1:
        res["auc_redund_Li_within_D1"] = round(auc_safe((sub.kappa >= 2).astype(int), sub.Li_div.to_numpy()), 3)
    # calibration: median D by kappa class
    for kv in (1, 2, 3):
        sel = df[(df.kappa == kv)] if kv < 3 else df[(df.kappa >= 3) & (df.kappa < KCAP)]
        if len(sel):
            res[f"medD_k{'3plus' if kv == 3 else kv}"] = float(np.median(sel.D))
    return res


def main():
    kp = pd.read_csv("results/kappa_perdemand.csv")
    kap_single = {(r.city, int(r.d)): int(r.kappa) for r in kp[kp.orientation == "DA"].itertuples()}
    rk = pd.read_csv("results/reanchor_perdemand.csv")
    rk = rk[(rk.orientation == "DA") & (rk.k == KNN)].drop_duplicates(subset=["city", "d"])
    kap_multi = {(r.city, int(r.d)): int(r.kappa) for r in rk.itertuples()}

    perd = []
    for c in CITIES:
        perd.extend(run_city(c, kap_single, kap_multi))
        pd.DataFrame(perd).to_csv("results/misranking_certificate_perdemand.csv", index=False)

    df = pd.DataFrame(perd)
    df.to_csv("results/misranking_certificate_perdemand.csv", index=False)

    # replication check against the persisted per-demand indices
    old = pd.read_csv("results/misranking_multientry_perdemand.csv")
    j = df.merge(old, on=["config", "city", "d"], suffixes=("", "_old"))
    for m in ["expH", "rpe_prime", "Li_div", "Deng_pm"]:
        dev = (j[m] - j[f"{m}_old"]).abs().max()
        print(f"replication {m}: n={len(j)} max|new-old|={dev:.2e}", flush=True)

    summ = []
    for config in ("single", "multi"):
        sub = df[df.config == config]
        for city in sub.city.unique():
            s = sub[sub.city == city]
            if len(s) >= 10:
                summ.append(summarize(s, config, city))
        summ.append(summarize(sub, config, "POOLED"))
    out = pd.DataFrame(summ)
    out.to_csv("results/misranking_certificate_summary.csv", index=False)
    print(out.to_string(index=False), flush=True)
    viol = int(out.violations.sum())
    print(f"\nTOTAL SOUNDNESS VIOLATIONS (D > kappa, real kappa only): {viol}", flush=True)
    print("ACCEPTANCE: " + ("PASS" if viol == 0 else "FAIL"), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
