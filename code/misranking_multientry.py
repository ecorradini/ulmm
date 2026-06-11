"""
Round-3 Major 1: misranking re-anchored to the paper's own primary labels.

Two configurations, side by side:
  config="single": existing results/misranking_perdemand.csv (single-entry routes,
                   single-entry kappa labels, k=1 DA) -- reused, not recomputed.
  config="multi":  routes generated from a virtual super-source over the SAME k=3 entry set
                   that defines the primary multi-entry kappa (epsilon-cost virtual arcs,
                   leading VS stripped before the measures), labelled by multi-entry kappa
                   (k=3, DA, from results/reanchor_perdemand.csv, deduplicated).

Per index (Deng_pm, expH, rpe_prime, Li_div) and config: ROC-AUC for kappa>=2, ROC-AUC for
kappa=1 on the NEGATED index (the decision-relevant rare class), Spearman, and the kappa=1
share among the top third by the index vs the base rate. Pooled rows carry bootstrap CIs.
The demand sample replicates misranking2.py exactly (all demands in the four smaller
cities, 250 random NYC demands, seed 7), so the two configs share the demand multiset.
Outputs: results/misranking_multientry_perdemand.csv, results/misranking_multientry_summary.csv
"""
import numpy as np, pandas as pd, time, os
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import ablations as ab
from misranking2 import k_routes, measures, auc_safe, K

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"; KNN = 3; EPSV = 1e-6
SAMPLE = {"New York City": 250}
METRICS = ["Deng_pm", "expH", "rpe_prime", "Li_div"]
os.makedirs("results", exist_ok=True)


def summarize(df, config, city, boot=False, B=600, seed=11):
    y2 = (df.kappa >= 2).astype(int).to_numpy()
    y1 = (df.kappa == 1).astype(int).to_numpy()
    res = dict(config=config, city=city, n=len(df),
               frac_redund=round(float(y2.mean()), 3), base_rate_k1=round(float(y1.mean()), 3))
    kap = df.kappa.to_numpy()
    for m in METRICS:
        x = df[m].to_numpy()
        res[f"auc_redund_{m}"] = round(auc_safe(y2, x), 3)
        res[f"auc_spof_{m}"] = round(auc_safe(y1, -x), 3)
        sp = spearmanr(x, kap).correlation if len(np.unique(x)) > 1 else float("nan")
        res[f"sp_{m}"] = round(float(sp), 3) if sp == sp else float("nan")
        thr = np.quantile(x, 2.0 / 3.0)
        top = x >= thr
        res[f"topthird_k1_{m}"] = round(float(y1[top].mean()), 3) if top.any() else float("nan")
        if boot:
            rng = np.random.RandomState(seed); a2, a1 = [], []
            n = len(df)
            for _ in range(B):
                i = rng.randint(0, n, n)
                if y2[i].min() != y2[i].max():
                    a2.append(roc_auc_score(y2[i], x[i]))
                if y1[i].min() != y1[i].max():
                    a1.append(roc_auc_score(y1[i], -x[i]))
            if a2:
                res[f"auc_redund_{m}_ci"] = f"[{np.percentile(a2,2.5):.2f},{np.percentile(a2,97.5):.2f}]"
            if a1:
                res[f"auc_spof_{m}_ci"] = f"[{np.percentile(a1,2.5):.2f},{np.percentile(a1,97.5):.2f}]"
    return res


def main():
    rk = pd.read_csv("results/reanchor_perdemand.csv")
    rk = rk[(rk.orientation == "DA") & (rk.k == KNN)].drop_duplicates(subset=["city", "d"])
    kap_me = {(r.city, int(r.d)): int(r.kappa) for r in rk.itertuples()}

    perd = []
    for c in CITIES:
        short = c.split(",")[0]; print(f"[{c}] ...", flush=True); t0 = time.time()
        u = ab.load_ulmm(c); Gm = u["graph"]; G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
        mr = np.array([idx[a] for a, b in G.edges()], np.int64)
        mc = np.array([idx[b] for a, b in G.edges()], np.int64)
        md = np.array([float(G[a][b]["weight"]) for a, b in G.edges()], float)
        Msub = csr_matrix((md, (mr, mc)), shape=(N, N))
        Mlil = Msub.tolil()
        lat = np.array([Gm.nodes[n]["y"] for n in nodes]); lon = np.array([Gm.nodes[n]["x"] for n in nodes])
        lat0 = lat.mean(); XY = np.column_stack([lon * np.cos(np.radians(lat0)) * 111320.0, lat * 110540.0])
        tree = cKDTree(XY)
        access_set = {idx[int(a)] for a in u["access"]["i_node"].tolist() if int(a) in idx}
        d_nodes = [int(n) for n in u["demand"]["i_node"].tolist() if int(n) in idx]
        cap = SAMPLE.get(short)
        if cap and len(d_nodes) > cap:
            d_nodes = sorted(np.random.RandomState(7).choice(d_nodes, cap, replace=False).tolist())
        VS = N
        for d in d_nodes:
            if (short, d) not in kap_me:
                continue
            _, nn = tree.query(XY[idx[d]], k=KNN)
            ent = sorted(set([idx[d]] + [int(x) for x in np.atleast_1d(nn)]))
            rows = np.concatenate([mr, np.full(len(ent), VS, np.int64)])
            cols = np.concatenate([mc, np.array(ent, np.int64)])
            vals = np.concatenate([md, np.full(len(ent), EPSV, float)])
            Mv = csr_matrix((vals, (rows, cols)), shape=(N + 1, N + 1))
            routes = [p[1:] for p in k_routes(Mv, VS, access_set, K) if len(p) > 1 and p[0] == VS]
            m = measures(routes, Mlil)
            if m is None:
                continue
            m.update(config="multi", city=short, d=d, kappa=kap_me[(short, d)])
            perd.append(m)
        print(f"  n={sum(1 for r in perd if r['city']==short)} ({time.time()-t0:.1f}s)", flush=True)
        pd.DataFrame(perd).to_csv("results/misranking_multientry_perdemand.csv", index=False)

    multi = pd.DataFrame(perd)
    single = pd.read_csv("results/misranking_perdemand.csv").copy()
    single["config"] = "single"
    both = pd.concat([single, multi], ignore_index=True)
    both.to_csv("results/misranking_multientry_perdemand.csv", index=False)

    summ = []
    for config, df in (("single", single), ("multi", multi)):
        for city in [c.split(",")[0] for c in CITIES]:
            s = df[df.city == city]
            if len(s) >= 10:
                summ.append(summarize(s, config, city))
        summ.append(summarize(df, config, "POOLED", boot=True))
        pd.DataFrame(summ).to_csv("results/misranking_multientry_summary.csv", index=False)
    out = pd.DataFrame(summ)
    print("\n=== misranking, single-entry vs multi-entry configuration ===", flush=True)
    cols = ["config", "city", "n", "frac_redund", "base_rate_k1"] + \
           [f"auc_redund_{m}" for m in METRICS] + [f"topthird_k1_{m}" for m in METRICS]
    print(out[ [c for c in cols if c in out.columns] ].to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
