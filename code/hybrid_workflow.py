"""
S6 (panel, DA-M1): the hybrid workflow comparator. Screen demands by the
mean-edge-dissimilarity index (least diverse first), compute exact kappa only on the
flagged fraction q; report the recall of kappa=1 demands within the flagged tail as a
function of q (= fraction of exact computations spent). From the persisted multi-entry
audit rows; no recomputation.

Also: co-location sensitivity (S7): pooled block-scale kappa=1 share with the 85
co-located (kappa=inf) zones excluded from the pool.

Output: results/hybrid_workflow.csv (+ prints the co-location sensitivity line)
"""
import os

import numpy as np
import pandas as pd

KCAP = 50
os.makedirs("results", exist_ok=True)


def main():
    df = pd.read_csv("results/misranking_certificate_perdemand.csv")
    m = df[df.config == "multi"].copy()
    y1 = (m.kappa == 1).astype(int).to_numpy()
    x = m.Li_div.to_numpy()
    order = np.argsort(x)  # least diverse first
    n = len(m)
    rows = []
    for q in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = max(1, int(round(q * n)))
        flagged = order[:k]
        recall = float(y1[flagged].sum() / max(y1.sum(), 1))
        precision = float(y1[flagged].mean())
        rows.append(dict(q=q, flagged=k, recall_k1=round(recall, 3),
                         precision_k1=round(precision, 3)))
        print(f"screen q={q:.0%}: recall of kappa=1 = {recall:.3f}, precision {precision:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv("results/hybrid_workflow.csv", index=False)

    # ---- S7: co-location sensitivity on the primary headline ----
    pdm = pd.read_csv("results/reanchor_perdemand.csv")
    for ori in ("AD", "DA"):
        g = pdm[(pdm.orientation == ori) & (pdm.k == 3)]
        full = g.w[g.kappa == 1].sum() / g.w.sum()
        ex = g[g.kappa < KCAP]
        excl = ex.w[ex.kappa == 1].sum() / ex.w.sum()
        print(f"co-location sensitivity {ori}: pooled w1 with co-located included "
              f"{full:.4f}, excluded {excl:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
