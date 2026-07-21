"""
R4 (panel): extended permitted-closure replay.

Adds to closure_replay.py, reusing its persisted outputs:
  (i)   multi-street closures: how many matched closures remove segments of >= 2
        distinct named streets (join matched edges to graph_edges_gdf names) — the
        only closures geometrically capable of severing a kappa=2 pair on their own;
  (ii)  DAILY CONCURRENT replay: for every day in the permit range, remove the union
        of all closures active that day and recompute reachability. This gives the
        test power the one-at-a-time replay lacks (dozens of concurrent closures) and
        yields per-anchor severed-day counts (durations) directly;
  (iii) exposure magnitudes: weighted-demand-days of lost modeled drive access;
  (iv)  snapshot split: closures already opened / already finished by the snapshot
        date (2026-07-19) vs. wholly future.

Inputs: results/closure_match.csv, results/external_perdemand_kappa.csv,
        cache_nyc_ulmm/{graph_G,graph_edges_gdf,anchor_access}.pkl
Outputs: results/closure_daily.csv, results/closure_severed_days.csv,
         results/closure_ext_summary.csv
"""
import pickle
import time
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

CACHE = "cache_nyc_ulmm/"
SNAPSHOT = date(2026, 7, 19)


def load(name):
    return pickle.load(open(CACHE + name + ".pkl", "rb"))


def build_substrate():
    G = load("graph_G")
    edge_uv = {}
    for u, v, d in G.edges(data=True):
        w = float(d.get("travel_time", 1.0))
        if (u, v) not in edge_uv or w < edge_uv[(u, v)]:
            edge_uv[(u, v)] = w
    nodes = sorted(set(x for e in edge_uv for x in e))
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    eu = np.array([idx[u] for (u, v) in edge_uv], np.int64)
    ev = np.array([idx[v] for (u, v) in edge_uv], np.int64)
    uv_index = {uv: i for i, uv in enumerate(edge_uv)}
    return idx, N, eu, ev, uv_index


def main():
    t0 = time.time()
    idx, N, eu, ev, uv_index = build_substrate()
    perd = pd.read_csv("results/external_perdemand_kappa.csv")
    mdf = pd.read_csv("results/closure_match.csv")
    matched = mdf[mdf.n_matched_edges > 0].copy()
    print(f"substrate ok; matched closures: {len(matched)}", flush=True)

    # ---------- (i) multi-street closures ----------
    eg = load("graph_edges_gdf")
    name_of = {}
    for _, r in eg.iterrows():
        uv = (r["u"], r["v"])
        if uv not in name_of:
            nm = r.get("name")
            if isinstance(nm, list):
                nm = tuple(sorted(str(x) for x in nm))
            name_of[uv] = str(nm) if nm is not None and str(nm) != "nan" else None

    def parse_edges(s):
        out = []
        for e in s.split(";"):
            a, b = e.split("|")
            a = int(a) if a.lstrip("-").isdigit() else a
            b = int(b) if b.lstrip("-").isdigit() else b
            out.append((a, b))
        return out

    matched["edge_list"] = matched.edges.map(parse_edges)
    def n_streets(edges):
        names = {name_of.get(uv) for uv in edges} - {None}
        return max(1, len(names))
    matched["n_streets"] = matched.edge_list.map(n_streets)
    multi_street = int((matched.n_streets >= 2).sum())
    print(f"multi-street closures (>=2 named streets): {multi_street}/{len(matched)}", flush=True)

    # ---------- (iv) snapshot split ----------
    matched["start_d"] = pd.to_datetime(matched.start).dt.date
    matched["end_d"] = pd.to_datetime(matched.end).dt.date
    opened = int((matched.start_d <= SNAPSHOT).sum())
    finished = int((matched.end_d < SNAPSHOT).sum())
    future = int((matched.start_d > SNAPSHOT).sum())
    print(f"windows: opened-by-snapshot {opened}, finished {finished}, wholly-future {future}", flush=True)

    # ---------- (ii) daily concurrent replay ----------
    aa = load("anchor_access")
    acc = np.array(sorted({idx[int(n)] for n in aa["node"] if int(n) in idx}), np.int64)
    dem_nodes = perd.node.to_numpy()
    dem_idx = np.array([idx[n] for n in dem_nodes], np.int64)
    kap = perd.kappa.to_numpy()
    wts = perd.w.to_numpy()
    A = N
    acc_rows = np.full(len(acc), A, np.int64)

    def reach(closed_eids):
        keep = np.ones(len(eu), bool)
        if len(closed_eids):
            keep[np.fromiter(closed_eids, np.int64)] = False
        rr = np.concatenate([ev[keep], acc_rows])
        cc = np.concatenate([eu[keep], acc])
        M = csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(N + 1, N + 1))
        order = breadth_first_order(M, A, directed=True, return_predecessors=False)
        reached = np.zeros(N + 1, bool)
        reached[order] = True
        return reached[dem_idx]

    base_ok = reach([])
    matched["eids"] = matched.edge_list.map(
        lambda es: [uv_index[uv] for uv in es if uv in uv_index])

    d0 = matched.start_d.min()
    d1 = matched.end_d.max()
    days = (d1 - d0).days + 1
    print(f"daily replay over {days} days ({d0}..{d1})", flush=True)
    daily = []
    severed_days = defaultdict(int)   # anchor row-index -> days severed
    day = d0
    t1 = time.time()
    while day <= d1:
        active = matched[(matched.start_d <= day) & (matched.end_d >= day)]
        eids = set()
        for lst in active.eids:
            eids.update(lst)
        now_ok = reach(eids) if eids else base_ok
        sev = np.where(base_ok & ~now_ok)[0]
        for j in sev:
            severed_days[j] += 1
        daily.append(dict(day=day.isoformat(), n_active=len(active),
                          n_edges_closed=len(eids), n_severed=len(sev),
                          sev_k1=int((kap[sev] == 1).sum()),
                          sev_k2plus=int((kap[sev] >= 2).sum()),
                          w_severed=float(wts[sev].sum()) if len(sev) else 0.0))
        day += timedelta(days=1)
    print(f"daily replay done ({time.time()-t1:.0f}s)", flush=True)

    dd = pd.DataFrame(daily)
    dd.to_csv("results/closure_daily.csv", index=False)
    sd = pd.DataFrame([dict(node=int(dem_nodes[j]), kappa=int(kap[j]), w=float(wts[j]),
                            days_severed=n, w_days=float(wts[j]) * n)
                       for j, n in severed_days.items()])
    sd.to_csv("results/closure_severed_days.csv", index=False)

    k1_sev = int((sd.kappa == 1).sum()) if len(sd) else 0
    k2_sev = int((sd.kappa >= 2).sum()) if len(sd) else 0
    summ = dict(
        matched_closures=len(matched),
        multi_street_closures=multi_street,
        windows_opened_by_snapshot=opened,
        windows_finished_by_snapshot=finished,
        windows_wholly_future=future,
        replay_days=days,
        peak_concurrent_closures=int(dd.n_active.max()),
        peak_edges_closed=int(dd.n_edges_closed.max()),
        anchors_ever_severed=len(sd),
        severed_k1=k1_sev,
        severed_k2plus=k2_sev,
        median_days_severed=float(sd.days_severed.median()) if len(sd) else 0.0,
        max_days_severed=int(sd.days_severed.max()) if len(sd) else 0,
        total_weighted_demand_days=round(float(sd.w_days.sum()), 1) if len(sd) else 0.0,
    )
    pd.DataFrame([summ]).T.rename(columns={0: "value"}).to_csv("results/closure_ext_summary.csv")
    print("\n=== extended replay summary ===", flush=True)
    for k, v in summ.items():
        print(f"  {k:32s} {v}", flush=True)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
