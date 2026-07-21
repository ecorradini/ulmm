"""Per-city entry-set spatial diameter by anchoring granularity k.

Regenerates results/entry_diameters.csv, which backs the manuscript's statement
that block-scale entry sets span a median 55-91 m (Section III-C, "the
granularity of door-level service") and is checked by audit_numbers_tnse.py.

The entry set is the same object used everywhere else: the k street nodes
nearest the demand's snapped node, found on the metric projection of lon/lat
(identical rule to reanchor_kappa.py and certificate_envelope.py). The diameter
is the maximum pairwise distance within that set; we report the per-city median
over demands, for k = 2..5.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import ablations as ab

HERE = Path(__file__).resolve().parent
# Layout-agnostic root: in the working tree the scripts sit beside results/;
# in the released repository they live in code/ with results/ one level up.
if not (HERE / "results").exists() and (HERE.parent / "results").exists():
    HERE = HERE.parent
CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
VEH = "van"
KS = (2, 3, 4, 5)


def main():
    rows = []
    for city in CITIES:
        u = ab.load_ulmm(city)
        Gm = u["graph"]
        G, _, _ = ab.collapsed_graph(u, VEH)
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        lat = np.array([Gm.nodes[n]["y"] for n in nodes], float)
        lon = np.array([Gm.nodes[n]["x"] for n in nodes], float)
        XY = np.column_stack([lon * np.cos(np.radians(lat.mean())) * 111320.0,
                              lat * 110540.0])
        tree = cKDTree(XY)
        d_nodes = sorted({int(n) for n in u["demand"]["i_node"] if int(n) in idx})

        rec = {"city": city.split(",")[0]}
        for k in KS:
            diams = []
            for n in d_nodes:
                i = idx[n]
                _, nn = tree.query(XY[i], k=k)
                ent = sorted(set([i] + [int(x) for x in np.atleast_1d(nn)]))
                P = XY[ent]
                d = 0.0
                for a in range(len(P)):
                    for b in range(a + 1, len(P)):
                        d = max(d, float(np.hypot(*(P[a] - P[b]))))
                diams.append(d)
            rec[f"diam_k{k}"] = round(float(np.median(diams)), 1)
        rows.append(rec)
        print(f"[{rec['city']}] " +
              "  ".join(f"k={k}: {rec[f'diam_k{k}']}m" for k in KS), flush=True)

    df = pd.DataFrame(rows)
    out = HERE / "results" / "entry_diameters.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"k=3 span across cities: {df.diam_k3.min():.0f}-{df.diam_k3.max():.0f} m")


if __name__ == "__main__":
    main()
