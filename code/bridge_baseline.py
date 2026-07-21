"""
S1 (panel, DA-M6): the cheap structural baseline. Undirected bridge decomposition
(Tarjan, O(m)) enumerates edges whose undirected removal disconnects the graph. How
much of the per-demand kappa=1 structure does it reproduce, and where does it fail?

Per city: share of kappa=1 named cut edges (cut_chars.csv, block-scale k=3, DA) that
are undirected bridges of the collapsed substrate; total bridge count (specificity:
bridges vastly outnumber demand-relevant cuts, and carry no per-demand kappa value,
no orientation, and no demand attribution).

Output: results/bridge_baseline.csv
"""
import os

import networkx as nx
import numpy as np
import pandas as pd

import ablations as ab

VEH = "van"
CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France",
          "Seattle, Washington, USA", "New York City, New York, USA"]
os.makedirs("results", exist_ok=True)


def main():
    cc = pd.read_csv("results/cut_chars.csv")
    rows = []
    for c in CITIES:
        short = c.split(",")[0]
        u = ab.load_ulmm(c)
        G, _, _ = ab.collapsed_graph(u, VEH)
        UG = nx.Graph()
        UG.add_edges_from((a, b) for a, b in G.edges())
        bridges = set()
        for a, b in nx.bridges(UG):
            bridges.add((a, b)); bridges.add((b, a))
        sub = cc[cc.city == short]
        if len(sub) == 0:
            continue
        is_br = [(int(r.cut_u), int(r.cut_v)) in bridges for r in sub.itertuples()]
        rows.append(dict(city=short, n_cut_edges=len(sub),
                         cuts_that_are_bridges=int(np.sum(is_br)),
                         share=round(float(np.mean(is_br)), 3),
                         total_undirected_bridges=len(bridges) // 2,
                         substrate_edges=G.number_of_edges()))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/bridge_baseline.csv", index=False)
    tot = df.n_cut_edges.sum()
    hit = df.cuts_that_are_bridges.sum()
    print(f"pooled: {hit}/{tot} = {hit/tot:.3f} of kappa=1 cut edges are undirected bridges; "
          f"bridges per city median {int(df.total_undirected_bridges.median())}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
