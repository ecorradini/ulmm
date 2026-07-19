"""
E-CLOSE: predictive-validity replay of recorded NYC DOT full street closures.

Data: cache_closures/nyc_full_closures_by_block_2026-07-19.geojson — the DOT "Street
Closures due to Construction Activities by Block" dataset (snapshot 2026-07-19; 3,627
permitted FULL closures restricting through traffic, work windows Apr-Oct 2026), matched
spatially onto the independent October-2025 NYC OpenStreetMap extract used by the
external-validation pipeline (cutcrit_external.py).

Three questions:
  A. Do recorded full closures strike the kappa=1 min-cut edges (cutcrit>0) more often
     than size-matched random contiguous segment groups? (structural hit rate vs null)
  B. Replay: removing each closure's matched directed segments from the graph, which
     demand cells lose ALL access? Rate for kappa=1 vs kappa>=2 demands (the
     predictive-validity headline).
  C. Agreement: are the stranded demands exactly those whose named min-cut edge was hit
     (prediction by cut localization), and what explains the residual (series cuts /
     whole-block severance of kappa=2 pairs)?

Stage 1 recomputes per-demand kappa + named min-cut edge on the external extract
(single-node anchoring, exactly as cutcrit_external.py) and persists it:
  results/external_perdemand_kappa.csv   (node, w, kappa, cut_u, cut_v)
Stage 2 matches closures to directed segments (endpoint-trimmed 12 m buffer, >=60%
length coverage). Stage 3 replays. Outputs:
  results/closure_match.csv     (per closure: matched edges, hit flags)
  results/closure_replay.csv    (per closure x stranded-demand events)
  results/closure_summary.csv   (headline numbers)
"""
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from shapely.geometry import shape
from shapely.strtree import STRtree

CACHE = "cache_nyc_ulmm/"
CLOSURES = "cache_closures/nyc_full_closures_by_block_2026-07-19.geojson"
PERD = "results/external_perdemand_kappa.csv"
BIG = 1 << 24
BUF = 12.0        # matching buffer, metres
TRIM = 10.0       # endpoint trim, metres
COVER = 0.60      # minimum share of a segment's length inside the buffer
NPERM = 100
SEED = 77
os.makedirs("results", exist_ok=True)


def load(name):
    return pickle.load(open(CACHE + name + ".pkl", "rb"))


def build_substrate():
    G = load("graph_G")
    edge_uv = {}
    for u, v, d in G.edges(data=True):
        w = float(d.get("travel_time", 1.0))
        if (u, v) not in edge_uv or w < edge_uv[(u, v)]:
            edge_uv[(u, v)] = w
    nodes = sorted(set([x for e in edge_uv for x in e]))
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    eu = np.array([idx[u] for (u, v) in edge_uv], np.int64)
    ev = np.array([idx[v] for (u, v) in edge_uv], np.int64)
    uv_list = list(edge_uv.keys())
    return nodes, idx, N, eu, ev, uv_list


def stage1_perdemand(idx, N, eu, ev, uv_list):
    """kappa + named min-cut edge per demand node (single-node anchoring), as in
    cutcrit_external.py, persisted per demand node."""
    ad = load("anchor_demand_cells"); aa = load("anchor_access"); dw = load("demand_weighted")
    wmap = dict(zip(dw["d_id"], dw["w_d"]))
    tgt_w = defaultdict(float)
    for _, r in ad.iterrows():
        tgt_w[int(r["node"])] += float(wmap.get(r["d_id"], 1.0))
    dem_nodes = [n for n in tgt_w if n in idx]
    acc_nodes = sorted({int(n) for n in aa["node"] if int(n) in idx})
    TERM = N
    rows = list(eu) + [idx[a] for a in acc_nodes]
    cols = list(ev) + [TERM] * len(acc_nodes)
    vals = [1] * len(eu) + [BIG] * len(acc_nodes)
    M = csr_matrix((np.array(vals, np.int64), (np.array(rows), np.array(cols))), shape=(N + 1, N + 1))
    out = []
    t1 = time.time()
    for j, d in enumerate(dem_nodes):
        di = idx[d]
        res = maximum_flow(M, di, TERM)
        k = int(res.flow_value)
        cu, cv = -1, -1
        if k == 1:
            F = res.flow
            resid = (((M - F) > 0).astype(np.int8) + (F > 0).astype(np.int8).transpose()).tocsr()
            order = breadth_first_order(resid, di, directed=True, return_predecessors=False)
            Rmask = np.zeros(N + 1, bool); Rmask[order] = True
            cut = np.where(Rmask[eu] & ~Rmask[ev])[0]
            if len(cut):
                cu, cv = uv_list[cut[0]]
        out.append(dict(node=d, w=tgt_w[d], kappa=k, cut_u=cu, cut_v=cv))
        if (j + 1) % 1000 == 0:
            print(f"  stage1 {j+1}/{len(dem_nodes)} ({time.time()-t1:.0f}s)", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(PERD, index=False)
    print(f"stage1 done: {len(df)} demand nodes, kappa=1: {(df.kappa==1).sum()} "
          f"({time.time()-t1:.0f}s)", flush=True)
    return df


def stage2_match(uv_list, idx):
    """Match closure geometries to directed substrate edges via the edges GeoDataFrame.
    The cached edges gdf is in Web Mercator (EPSG:3857); closures arrive as WGS84
    GeoJSON and are reprojected to 3857. Buffers/trims are scaled by the Mercator
    factor at NYC's latitude; length RATIOS are scale-invariant."""
    import shapely
    from pyproj import Transformer
    eg = load("graph_edges_gdf").reset_index()
    tr = Transformer.from_crs(4326, 3857, always_xy=True)
    def proj(geom):
        return shapely.transform(geom, lambda a: np.column_stack(tr.transform(a[:, 0], a[:, 1])))
    SCALE = 1.0 / np.cos(np.radians(40.7128))   # ground metres -> Mercator units
    buf_m = BUF * SCALE
    trim_m = TRIM * SCALE
    eg["g"] = list(eg.geometry)
    eg["len_m"] = [g.length for g in eg["g"]]
    tree = STRtree(list(eg["g"]))
    gj = json.load(open(CLOSURES))
    feats = gj["features"]
    # dedupe by segmentid (several permits can cover one block)
    seen = {}
    for f in feats:
        p = f["properties"]
        sid = p.get("segmentid") or p.get("uniqueid")
        if sid not in seen:
            seen[sid] = dict(segmentid=sid, purpose=p.get("purpose", ""),
                             start=p.get("work_start_date", ""), end=p.get("work_end_date", ""),
                             on=p.get("onstreetname", ""), geom=shape(f["geometry"]), n_permits=1)
        else:
            seen[sid]["n_permits"] += 1
    print(f"closures: {len(feats)} permit rows -> {len(seen)} unique blocks", flush=True)
    uv_index = {uv: i for i, uv in enumerate(uv_list)}
    match_rows = []
    from shapely.ops import substring
    for sid, c in seen.items():
        g = proj(c["geom"])
        merged = shapely.line_merge(g)
        if merged.geom_type == "LineString" and merged.length > 2 * trim_m + 5:
            # trim both ends to avoid grabbing cross streets at the endpoints
            core = substring(merged, trim_m, merged.length - trim_m)
        else:
            core = g
        buf = core.buffer(buf_m)
        cand = tree.query(buf)
        hits = []
        for ci in cand:
            seg_g = eg["g"].iloc[ci]
            if eg["len_m"].iloc[ci] < 1:
                continue
            inter = seg_g.intersection(buf).length
            if inter / eg["len_m"].iloc[ci] >= COVER:
                hits.append(ci)
        edges = []
        for ci in hits:
            r = eg.iloc[ci]
            uv = (r["u"], r["v"])
            if uv in uv_index:
                edges.append(uv)
        edges = sorted(set(edges))
        match_rows.append(dict(segmentid=sid, purpose=c["purpose"], start=c["start"][:10],
                               end=c["end"][:10], on=c["on"], n_permits=c["n_permits"],
                               n_matched_edges=len(edges),
                               edges=";".join(f"{u}|{v}" for u, v in edges)))
    mdf = pd.DataFrame(match_rows)
    mdf.to_csv("results/closure_match.csv", index=False)
    print(f"matched: {(mdf.n_matched_edges>0).sum()}/{len(mdf)} closures to >=1 directed edge "
          f"(median edges/closure among matched: {mdf[mdf.n_matched_edges>0].n_matched_edges.median()})", flush=True)
    return mdf


def stage3_replay(perd, mdf, nodes, idx, N, eu, ev, uv_list):
    aa = load("anchor_access")
    acc = np.array(sorted({idx[int(n)] for n in aa["node"] if int(n) in idx}), np.int64)
    uv_index = {uv: i for i, uv in enumerate(uv_list)}
    cutcrit_edges = {(r.cut_u, r.cut_v) for r in perd[perd.kappa == 1].itertuples()}
    dem_nodes = perd.node.to_numpy()
    dem_idx = np.array([idx[n] for n in dem_nodes], np.int64)
    kap = perd.kappa.to_numpy()
    wts = perd.w.to_numpy()
    cut_uv = list(zip(perd.cut_u, perd.cut_v))

    A = N  # super-access
    rev_rows = ev; rev_cols = eu
    acc_rows = np.full(len(acc), A, np.int64)

    # baseline reachability (no closure): kappa==0 demands are unreachable already
    def reach(closed_eids):
        keep = np.ones(len(eu), bool)
        if len(closed_eids):
            keep[np.array(sorted(closed_eids), np.int64)] = False
        rr = np.concatenate([rev_rows[keep], acc_rows])
        cc = np.concatenate([rev_cols[keep], acc])
        M = csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(N + 1, N + 1))
        order = breadth_first_order(M, A, directed=True, return_predecessors=False)
        reached = np.zeros(N + 1, bool)
        reached[order] = True
        return reached

    base_reached = reach([])
    base_ok = base_reached[dem_idx]
    print(f"baseline: {int(base_ok.sum())}/{len(dem_idx)} demand nodes reach access "
          f"(kappa==0 count {int((kap==0).sum())})", flush=True)

    events = []
    per_closure = []
    t1 = time.time()
    matched = mdf[mdf.n_matched_edges > 0].reset_index(drop=True)
    for i, r in matched.iterrows():
        edges = [tuple(e.split("|")) for e in r.edges.split(";")]
        edges = [(int(u) if u.lstrip("-").isdigit() else u, int(v) if v.lstrip("-").isdigit() else v)
                 for u, v in edges]
        eids = [uv_index[uv] for uv in edges if uv in uv_index]
        hit_spof = any(uv in cutcrit_edges for uv in edges)
        reached = reach(eids)
        now_ok = reached[dem_idx]
        stranded = np.where(base_ok & ~now_ok)[0]
        pred = {j for j in range(len(dem_nodes))
                if kap[j] == 1 and cut_uv[j] in set(edges)}
        for j in stranded:
            events.append(dict(segmentid=r.segmentid, node=int(dem_nodes[j]), w=float(wts[j]),
                               kappa=int(kap[j]), predicted=int(j in pred)))
        per_closure.append(dict(segmentid=r.segmentid, n_edges=len(eids), hit_spof=int(hit_spof),
                                n_stranded=len(stranded),
                                w_stranded=float(wts[stranded].sum()) if len(stranded) else 0.0,
                                n_predicted=len(pred)))
        if (i + 1) % 500 == 0:
            print(f"  replay {i+1}/{len(matched)} ({time.time()-t1:.0f}s)", flush=True)
    ev_df = pd.DataFrame(events)
    pc = pd.DataFrame(per_closure)
    ev_df.to_csv("results/closure_replay.csv", index=False)
    if not len(pc):
        print("NO MATCHED CLOSURES - aborting summary", flush=True)
        return

    # ---- null model for the SPOF hit rate: random contiguous edge groups (same sizes)
    rng = np.random.RandomState(SEED)
    succ = defaultdict(list)
    for k_ in range(len(eu)):
        succ[eu[k_]].append(k_)
    eids_all = np.arange(len(eu))
    spof_eids = {uv_index[uv] for uv in cutcrit_edges if uv in uv_index}
    sizes = pc.n_edges.to_numpy()
    obs_rate = pc.hit_spof.mean()
    null_rates = []
    for b in range(NPERM):
        hits = 0
        for s_ in sizes:
            e0 = rng.choice(eids_all)
            grp = {e0}
            frontier = e0
            steps = 0
            while len(grp) < s_ and steps < 10 * s_ + 20:  # step bound: walks can cycle
                steps += 1
                nxt = succ.get(ev[frontier], [])
                if not nxt:
                    break
                frontier = nxt[rng.randint(len(nxt))]
                grp.add(frontier)
            if grp & spof_eids:
                hits += 1
        null_rates.append(hits / len(sizes))
    null_rates = np.array(null_rates)

    # ---- headline summary
    strand_by_class = ev_df.groupby("kappa").agg(n=("node", "nunique"), w=("w", "sum")) \
        if len(ev_df) else pd.DataFrame()
    n_k1 = int((kap == 1).sum()); n_k2 = int((kap >= 2).sum())
    k1_stranded = ev_df[ev_df.kappa == 1].node.nunique() if len(ev_df) else 0
    k2_stranded = ev_df[ev_df.kappa >= 2].node.nunique() if len(ev_df) else 0
    summ = dict(
        n_closures_matched=len(matched),
        median_edges_per_closure=float(pc.n_edges.median()),
        obs_spof_hit_rate=round(float(obs_rate), 4),
        null_spof_hit_rate_mean=round(float(null_rates.mean()), 4),
        null_spof_hit_rate_p975=round(float(np.percentile(null_rates, 97.5)), 4),
        hit_ratio_vs_null=round(float(obs_rate / null_rates.mean()), 2) if null_rates.mean() > 0 else np.nan,
        n_closures_stranding=int((pc.n_stranded > 0).sum()),
        n_strand_events=len(ev_df),
        k1_nodes=n_k1, k1_stranded_by_some_closure=int(k1_stranded),
        k1_strand_rate=round(k1_stranded / n_k1, 4) if n_k1 else np.nan,
        k2plus_nodes=n_k2, k2plus_stranded_by_some_closure=int(k2_stranded),
        k2plus_strand_rate=round(k2_stranded / n_k2, 5) if n_k2 else np.nan,
        strand_events_predicted_by_named_cut=int(ev_df.predicted.sum()) if len(ev_df) else 0,
        strand_events_k1=int((ev_df.kappa == 1).sum()) if len(ev_df) else 0,
        strand_events_k2plus=int((ev_df.kappa >= 2).sum()) if len(ev_df) else 0,
    )
    pd.DataFrame([summ]).T.rename(columns={0: "value"}).to_csv("results/closure_summary.csv")
    print("\n=== closure replay summary ===", flush=True)
    for k_, v_ in summ.items():
        print(f"  {k_:38s} {v_}", flush=True)
    if len(strand_by_class):
        print(strand_by_class.to_string(), flush=True)
    print("DONE", flush=True)


def main():
    t0 = time.time()
    nodes, idx, N, eu, ev, uv_list = build_substrate()
    print(f"substrate: {N} nodes, {len(uv_list)} directed edges ({time.time()-t0:.0f}s)", flush=True)
    if os.path.exists(PERD):
        perd = pd.read_csv(PERD)
        print(f"stage1: reusing {PERD} ({len(perd)} rows)", flush=True)
    else:
        perd = stage1_perdemand(idx, N, eu, ev, uv_list)
    mdf = stage2_match(uv_list, idx)
    stage3_replay(perd, mdf, nodes, idx, N, eu, ev, uv_list)
    print(f"ALL DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
