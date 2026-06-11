"""
Round-3 Major 6 (sensitivity): population-weighted (alpha_pop=1) last-mile shares.

kappa(d) does not depend on the demand weight, so the alpha_pop=1 variant only re-aggregates
the existing per-demand kappa (reanchor, k=3 and k=1, both orientations) with WorldPop-based
population weights in place of the POI weights. Alignment is positional: reanchor rows for a
(city, orientation, k) block preserve the original zone order filtered by graph membership,
which we reconstruct and assert.
Output: results/kappa_popweight_summary.csv
"""
import numpy as np, pandas as pd, os
import ablations as ab
from pop_weights import pop_weights

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"
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
    rk = pd.read_csv("results/reanchor_perdemand.csv")
    summ, pooled = [], {}
    for c in CITIES:
        short = c.split(",")[0]; print(f"[{c}] ...", flush=True)
        u = ab.load_ulmm(c); G, _, _ = ab.collapsed_graph(u, VEH)
        idx = set(G.nodes())
        zn = [int(n) for n in u["demand"]["i_node"].tolist()]
        keep = np.array([n in idx for n in zn])
        pw_all = pop_weights(u, short)
        pw = pw_all[keep]
        zone_node = [n for n, m in zip(zn, keep) if m]
        for ori in ("AD", "DA"):
            for k in (1, 3):
                s = rk[(rk.city == short) & (rk.orientation == ori) & (rk.k == k)]
                assert len(s) == len(zone_node) and list(s.d) == zone_node, \
                    f"alignment failed {short} {ori} k={k}"
                kap = s.kappa.to_numpy(); wp = pw
                lo, hi = boot_wshare(wp, kap == 1)
                w1_poi = float(s.w[kap == 1].sum() / s.w.sum())
                w1_pop = float(wp[kap == 1].sum() / wp.sum()) if wp.sum() > 0 else float("nan")
                summ.append(dict(city=short, orientation=ori, k=k, n=len(s),
                                 w1_poi=round(w1_poi, 3), w1_pop=round(w1_pop, 3),
                                 w1_pop_ci=f"[{lo},{hi}]"))
                pooled.setdefault((ori, k), []).append((s.kappa.to_numpy(), s.w.to_numpy(), wp))
                print(f"  {ori} k={k}: w1 POI={w1_poi:.3f} pop={w1_pop:.3f} [{lo},{hi}]", flush=True)
    for (ori, k), blocks in pooled.items():
        kap = np.concatenate([b[0] for b in blocks]); wpoi = np.concatenate([b[1] for b in blocks])
        wpop = np.concatenate([b[2] for b in blocks])
        lo, hi = boot_wshare(wpop, kap == 1)
        summ.append(dict(city="POOLED", orientation=ori, k=k, n=len(kap),
                         w1_poi=round(float(wpoi[kap == 1].sum() / wpoi.sum()), 3),
                         w1_pop=round(float(wpop[kap == 1].sum() / wpop.sum()), 3),
                         w1_pop_ci=f"[{lo},{hi}]"))
    pd.DataFrame(summ).to_csv("results/kappa_popweight_summary.csv", index=False)
    print("\n=== alpha_pop=1 sensitivity (kappa=1 share, POI vs population weights) ===", flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
