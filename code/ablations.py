#!/usr/bin/env python
# coding: utf-8

# In[8]:


# ablations_and_sensitivity.py — H1–H4 metrics, parameter sweeps, LaTeX table rows
# Minimal deps: numpy, pandas, networkx, pickle

import math
import heapq
import warnings
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx
import pickle

# -----------------------------
# Config
# -----------------------------
PICKLE_DIR   = Path("ulmm_pickles")
CACHE_DIR    = Path("cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)

CITY         = "Barcelona, Spain"
VEHICLE      = "van"         # "van" | "cargo_bike"
ORIS         = ["AD", "DA"]  # evaluate both directions

# Reference params (paper defaults)
REF = dict(lambda_=1.0, gamma=1.0, kappa=1.0, K=15, eta=1.0)

# Sweeps to probe (match the table rows in the manuscript)
SWEEPS = [
    dict(name=r"$\lambda\uparrow$", lambda_=2.0),
    dict(name=r"$\gamma=0$",        gamma=0.0),
    dict(name=r"$\gamma=2$",        gamma=2.0),
    dict(name=r"$\kappa\uparrow$",  kappa=2.0),
    dict(name=r"$K\uparrow$",       K=25),
    dict(name=r"$\eta\uparrow$",    eta=2.0),
]

# Coverage thresholds (seconds) for H2 summary; short-time regime
TAUS = [300, 600]  # 5 min, 10 min

BUDGET_PCT = 2.0   # % of edges removed for targeted shock in H3
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)

# -----------------------------
# Cache helpers
# -----------------------------
def slugify(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")

def ulmm_path_for_city(city_name: str) -> Path:
    return PICKLE_DIR / f"ulmm_{slugify(city_name)}.pkl"

def ulmm_fingerprint(ulmm_pkl_path: Path) -> str:
    try:
        st = ulmm_pkl_path.stat()
        return f"{st.st_mtime_ns}-{st.st_size}"
    except FileNotFoundError:
        return "missing"

def cache_file(prefix: str, **kwargs) -> Path:
    key = "__".join(f"{k}-{slugify(v)}" for k, v in sorted(kwargs.items()))
    return CACHE_DIR / f"{prefix}__{key}.pkl"

def cache_save(path: Path, data, meta: dict):
    payload = {"__meta__": meta, "__data__": data}
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

def cache_load(path: Path, expected_meta: dict):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        meta = payload.get("__meta__", {})
        if meta == expected_meta:
            return payload.get("__data__")
        return None
    except Exception:
        return None

# -----------------------------
# ULMM I/O
# -----------------------------
_RESNAP_CACHE = {}


def _utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _resnap_demand(ulmm: dict) -> dict:
    """Fix a build bug in the demand anchoring. Demand zone centroids are stored in the
    city's projected UTM CRS (columns lat=northing, lon=easting), but the original build
    snapped them to street nodes with a CRS mismatch, collapsing every zone onto ~20-36
    nodes. We re-snap each zone to its true nearest street node. Access is unaffected
    (its lat/lon are WGS84 and correctly snapped)."""
    dem = ulmm.get("demand")
    if dem is None or "lat" not in dem.columns:
        return ulmm
    if float(np.abs(dem["lat"]).max()) <= 1000.0:
        return ulmm  # already WGS84 degrees -> no fix needed
    import pyproj
    from scipy.spatial import cKDTree
    Gm = ulmm["graph"]; nodes = list(Gm.nodes())
    lon = np.array([Gm.nodes[n]["x"] for n in nodes], float)
    lat = np.array([Gm.nodes[n]["y"] for n in nodes], float)
    epsg = _utm_epsg(float(lon.mean()), float(lat.mean()))
    tf = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    nx_, ny_ = tf.transform(lon, lat)
    tree = cKDTree(np.column_stack([nx_, ny_]))
    _, ii = tree.query(np.column_stack([dem["lon"].to_numpy(float), dem["lat"].to_numpy(float)]), k=1)
    dem = dem.copy(); dem["i_node"] = [nodes[i] for i in ii]
    out = dict(ulmm); out["demand"] = dem
    return out


def load_ulmm(city_name: str) -> dict:
    p = ulmm_path_for_city(city_name)
    key = str(p)
    if key in _RESNAP_CACHE:
        return _RESNAP_CACHE[key]
    with open(p, "rb") as f:
        u = pickle.load(f)
    u = _resnap_demand(u)
    _RESNAP_CACHE[key] = u
    return u

# -----------------------------
# Collapsed graph (MultiDiGraph → DiGraph)
# -----------------------------
def _guess_base_cost_key(Gm: nx.MultiDiGraph, veh: str) -> Tuple[str, str]:
    phi_key = f"phi_{veh}"
    candidates = ["c_base", "c0", "t_free", "time_free", "travel_time_free",
                  "travel_time", "base_time", "base_cost", "c"]
    for _, _, _, d in list(Gm.edges(keys=True, data=True))[:500]:
        for k in candidates:
            v = d.get(k, None)
            if v is not None and np.isfinite(v):
                return k, phi_key
    raise KeyError("No base-cost attribute found on edges (lambda sweeps unavailable).")

def collapsed_graph(ulmm: dict, veh: str, lambda_override: float | None = None, force_recompute: bool = False):
    city = ulmm["city"]
    meta = {
        "fingerprint": ulmm_fingerprint(ulmm_path_for_city(city)),
        "city": city, "veh": veh,
        "lambda": None if lambda_override is None else float(lambda_override),
        "ver": "v2",
    }
    cpath = cache_file("collapsed_graph", city=city, veh=veh,
                       lam=("none" if lambda_override is None else f"{lambda_override:.6g}"),
                       fp=meta["fingerprint"])
    if not force_recompute:
        cached = cache_load(cpath, meta)
        if cached is not None:
            return cached

    Gm = ulmm["graph"]
    G = nx.DiGraph()

    if lambda_override is None:
        weight_attr = f"c_eff_{veh}"
        for u, v, k, d in Gm.edges(keys=True, data=True):
            w = d.get(weight_attr, None)
            if w is None or not np.isfinite(w):
                continue
            w = float(w)
            if (u, v) not in G or w < G[u][v]["weight"]:
                G.add_edge(u, v, weight=w)
    else:
        base_key, phi_key = _guess_base_cost_key(Gm, veh)
        for u, v, k, d in Gm.edges(keys=True, data=True):
            c0  = d.get(base_key, None)
            if c0 is None or not np.isfinite(c0):
                continue
            c0 = float(c0)
            phi = float(d.get(phi_key, 0.0))
            w = c0 * (1.0 + float(lambda_override) * phi)
            if (u, v) not in G or w < G[u][v]["weight"]:
                G.add_edge(u, v, weight=w)

    out = (G, None, None)  # only G is used downstream
    cache_save(cpath, out, meta)
    return out

# -----------------------------
# φ proximity at an access node
# -----------------------------
def mean_phi_prox(Gm: nx.MultiDiGraph, node: int, veh: str, hops: int = 1) -> float:
    phis = []
    if hops <= 1:
        for _, _, _, d in Gm.edges(node, keys=True, data=True):
            v = d.get(f"phi_{veh}")
            if v is not None: phis.append(float(v))
        for _, _, _, d in Gm.in_edges(node, keys=True, data=True):
            v = d.get(f"phi_{veh}")
            if v is not None: phis.append(float(v))
    else:
        q = deque([(node,0)]); seen = {node}; nodes = {node}
        while q:
            u, h = q.popleft()
            if h == hops: continue
            for w in set(list(Gm.successors(u)) + list(Gm.predecessors(u))):
                if w not in seen:
                    seen.add(w); nodes.add(w); q.append((w, h+1))
        for u in nodes:
            for _, _, _, d in Gm.edges(u, keys=True, data=True):
                v = d.get(f"phi_{veh}")
                if v is not None: phis.append(float(v))
    return float(np.mean(phis)) if phis else 0.0

# -----------------------------
# ServC (both orientations) [CACHE]
# -----------------------------
def compute_servc(ulmm: dict, veh: str, kappa: float = 0.0, lambda_override: float | None = None,
                  force_recompute: bool = False) -> pd.DataFrame:
    city = ulmm["city"]
    meta = {
        "fingerprint": ulmm_fingerprint(ulmm_path_for_city(city)),
        "city": city, "veh": veh, "kappa": float(kappa),
        "lambda": None if lambda_override is None else float(lambda_override),
        "ver": "v2",
    }
    cpath = cache_file("servc", city=city, veh=veh, kappa=kappa,
                       lam=("none" if lambda_override is None else f"{lambda_override:.6g}"),
                       fp=meta["fingerprint"])
    if not force_recompute:
        cached = cache_load(cpath, meta)
        if cached is not None:
            return cached

    G, _, _ = collapsed_graph(ulmm, veh, lambda_override=lambda_override)
    Gm = ulmm["graph"]
    demand, access = ulmm["demand"], ulmm["access"]

    d_nodes = demand["i_node"].astype(int).tolist()
    a_nodes = access["i_node"].astype(int).tolist()
    w_d     = demand["w"].astype(float).to_numpy()
    d_w = {int(n): float(w) for n, w in zip(d_nodes, w_d)}

    a_phi = {int(a): mean_phi_prox(Gm, int(a), veh) for a in a_nodes}
    if all(v == 0.0 for v in a_phi.values()):
        # simple fallback to lanes scarcity if φ is missing
        for a in a_nodes:
            vals=[]
            for _,_,_,d in Gm.edges(int(a), keys=True, data=True):
                x=d.get("f_lanes_scarcity")
                if x is not None and np.isfinite(x): vals.append(float(x))
            for _,_,_,d in Gm.in_edges(int(a), keys=True, data=True):
                x=d.get("f_lanes_scarcity")
                if x is not None and np.isfinite(x): vals.append(float(x))
            a_phi[int(a)] = float(np.nanmean(vals)) if vals else 0.0

    # AD: access -> demand
    servc_AD = {}
    for a in a_nodes:
        dist = nx.single_source_dijkstra_path_length(G, int(a), weight="weight")
        val = 0.0
        for dn in d_nodes:
            t = dist.get(int(dn), np.inf)
            if np.isfinite(t):
                val += d_w[int(dn)] / (t + kappa*a_phi[int(a)] + 1e-9)
        servc_AD[int(a)] = val

    # DA: demand -> access (reverse graph)
    G_rev = G.reverse(copy=False)
    servc_DA = {}
    for a in a_nodes:
        dist = nx.single_source_dijkstra_path_length(G_rev, int(a), weight="weight")
        val = 0.0
        for dn in d_nodes:
            t = dist.get(int(dn), np.inf)
            if np.isfinite(t):
                val += d_w[int(dn)] / (t + kappa*a_phi[int(a)] + 1e-9)
        servc_DA[int(a)] = val

    df = pd.DataFrame({
        "a_node": [int(n) for n in a_nodes],
        "a_id": access["a_id"].tolist(),
        "lat":   access["lat"].tolist(),
        "lon":   access["lon"].tolist(),
        "phi_prox": [a_phi[int(n)] for n in a_nodes],
        "servc_AD": [servc_AD[int(n)] for n in a_nodes],
        "servc_DA": [servc_DA[int(n)] for n in a_nodes],
    })
    cache_save(cpath, df, meta)
    return df

# -----------------------------
# DEB (pair-weighted, with optional exponential damping γ) [CACHE]
# -----------------------------
def deb_pairweighted(ulmm: dict, veh: str, orientation: str = "AD",
                     lambda_override: float | None = None, gamma: float = 0.0,
                     force_recompute: bool = False):
    """
    Pair-weighted, orientation-aware DEB on the collapsed graph.
    For AD: sources=Access (1.0), targets=Demand (weighted by w_d).
    For DA: sources=Demand (weighted by w_d), targets=Access (1.0).
    Damping γ inflates weights: w' = w * exp(γ * φ_uv), φ_uv = mean curb friction on (u,v).
    """
    city = ulmm["city"]
    meta = {
        "fingerprint": ulmm_fingerprint(ulmm_path_for_city(city)),
        "city": city, "veh": veh, "ori": orientation,
        "lambda": None if lambda_override is None else float(lambda_override),
        "gamma": float(gamma),
        "ver": "v5",
    }
    cpath = cache_file("deb_pairweighted", city=city, veh=veh, ori=orientation,
                       lam=("none" if lambda_override is None else f"{lambda_override:.6g}"),
                       gamma=f"{gamma:.6g}", fp=meta["fingerprint"])
    if not force_recompute:
        cached = cache_load(cpath, meta)
        if cached is not None:
            return cached

    G, _, _ = collapsed_graph(ulmm, veh, lambda_override=lambda_override)
    Gm = ulmm["graph"]
    demand, access = ulmm["demand"], ulmm["access"]
    d_nodes = demand["i_node"].astype(int).tolist()
    a_nodes = access["i_node"].astype(int).tolist()
    w_d = {int(n): float(w) for n, w in zip(demand["i_node"], demand["w"])}

    # φ per collapsed edge (mean over multiedges)
    phi_key = f"phi_{veh}"
    phi_uv_map: Dict[Tuple[int,int], float] = {}
    for u, v in G.edges():
        phis_uv = []
        try:
            for _, _, _, d in Gm.edges(u, v, keys=True, data=True):
                p = d.get(phi_key)
                if p is not None:
                    phis_uv.append(float(p))
        except Exception:
            pass
        phi_uv_map[(u, v)] = float(np.mean(phis_uv)) if phis_uv else 0.0

    # Optionally apply damping: w' = w * exp(γ * φ)
    if gamma and gamma != 0.0:
        G_use = nx.DiGraph()
        for n in G.nodes(): G_use.add_node(n)
        for u, v in G.edges():
            w = G[u][v]["weight"]
            phi = phi_uv_map.get((u, v), 0.0)
            G_use.add_edge(u, v, weight=float(w) * math.exp(float(gamma) * float(phi)))
    else:
        G_use = G

    if orientation.upper() == "AD":
        sources = a_nodes
        targets_set = set(d_nodes)
        sink_weight = lambda v: w_d.get(int(v), 0.0)
        source_weight = lambda s: 1.0
    else:  # "DA"
        sources = d_nodes
        targets_set = set(a_nodes)
        sink_weight = lambda v: 1.0 if int(v) in targets_set else 0.0
        source_weight = lambda s: w_d.get(int(s), 1.0)

    deb = defaultdict(float)
    eps = 1e-9
    for s in sources:
        if s not in G_use:
            continue
        dist = defaultdict(lambda: np.inf)
        sigma = defaultdict(float)
        P = defaultdict(list)
        dist[s] = 0.0
        sigma[s] = 1.0
        Q = [(0.0, s)]
        S = []
        while Q:
            d_u, u = heapq.heappop(Q)
            if d_u > dist[u] + 1e-12:
                continue
            S.append(u)
            for v in G_use.successors(u):
                w = G_use[u][v]["weight"]
                alt = d_u + w
                if alt + eps < dist[v]:
                    dist[v] = alt
                    sigma[v] = sigma[u]
                    P[v] = [u]
                    heapq.heappush(Q, (alt, v))
                elif abs(alt - dist[v]) <= eps:
                    sigma[v] += sigma[u]
                    P[v].append(u)

        # mass at targets
        b = defaultdict(float)
        for t in targets_set:
            if dist[t] < np.inf:
                b[t] += sink_weight(t)

        delta = defaultdict(float)
        for wnode in reversed(S):
            coeff = b[wnode] + delta[wnode]
            sig_w = sigma[wnode]
            if sig_w == 0:
                continue
            for v in P[wnode]:
                contrib = (sigma[v] / sig_w) * coeff
                deb[(v, wnode)] += source_weight(s) * contrib
                delta[v] += contrib

    total = float(sum(deb.values()))
    deb_norm = {e: (val / total if total > 0 else 0.0) for e, val in deb.items()}

    # Tabular edge view with mean φ (no geometry needed)
    rows = []
    for (u, v), val in deb_norm.items():
        rows.append((int(u), int(v), float(val), float(phi_uv_map.get((u, v), 0.0))))
    g_edges = pd.DataFrame(rows, columns=["u", "v", "deb_norm", "phi"])
    out = (deb_norm, g_edges)
    cache_save(cpath, out, meta)
    return out

# -----------------------------
# RPE (K-shortest to any access) [CACHE]
# -----------------------------
def k_shortest_to_any_access(G: nx.DiGraph, src: int, access_nodes: list, K: int, weight_key: str = "weight"):
    H = G.copy(); SINK = "__SINK__"; H.add_node(SINK)
    for a in set(access_nodes):
        if a in H:
            H.add_edge(a, SINK, **{weight_key: 0.0})
    try:
        gen = nx.shortest_simple_paths(H, src, SINK, weight=weight_key)
        paths = []
        for _, p in zip(range(K), gen):
            if len(p) >= 2 and p[-1] == SINK:
                paths.append(p[:-1])
        return paths
    except nx.NetworkXNoPath:
        return []

def rpe_for_all_demands(ulmm: dict, veh: str, K: int = 15, eta: float = 1.0,
                        orientation: str = "DA", lambda_override: float | None = None,
                        force_recompute: bool = False):
    city = ulmm["city"]
    meta = {
        "fingerprint": ulmm_fingerprint(ulmm_path_for_city(city)),
        "city": city, "veh": veh, "K": int(K), "eta": float(eta),
        "ori": orientation,
        "lambda": None if lambda_override is None else float(lambda_override),
        "ver": "v2",
    }
    cpath = cache_file("rpe", city=city, veh=veh, K=K, eta=eta, ori=orientation,
                       lam=("none" if lambda_override is None else f"{lambda_override:.6g}"),
                       fp=meta["fingerprint"])
    if not force_recompute:
        cached = cache_load(cpath, meta)
        if cached is not None:
            return cached

    G, _, _ = collapsed_graph(ulmm, veh, lambda_override=lambda_override)
    demand, access = ulmm["demand"], ulmm["access"]
    d_nodes = [int(n) for n in demand["i_node"].tolist()]
    a_nodes = [int(n) for n in access["i_node"].tolist()]

    G_use = G if orientation == "DA" else G.reverse(copy=False)

    rpe_vals, paths_examples = {}, {}
    for d in d_nodes:
        kpaths = k_shortest_to_any_access(G_use, d, a_nodes, K, weight_key="weight")
        if not kpaths:
            rpe_vals[d] = 0.0; paths_examples[d] = []; continue
        lens = []
        for p in kpaths:
            L = 0.0
            for u, v in zip(p[:-1], p[1:]): L += G_use[u][v]["weight"]
            lens.append(L)
        lens  = np.asarray(lens, float)
        probs = np.exp(-eta * lens); Z = probs.sum()
        probs = probs / (Z if Z > 0 else 1.0)
        eps = 1e-12
        H = -float(np.sum(probs * np.log(np.clip(probs, eps, 1.0))))
        rpe_vals[d] = H; paths_examples[d] = kpaths

    out = (rpe_vals, paths_examples, G_use)
    cache_save(cpath, out, meta)
    return out

def rpe_normalized_series(ulmm: dict, veh: str, K: int, eta: float) -> pd.Series:
    rpe_vals, *_ = rpe_for_all_demands(ulmm, veh, K=K, eta=eta, orientation="DA")
    if not rpe_vals:
        return pd.Series(dtype=float)
    H = pd.Series(rpe_vals, dtype=float)
    return H / max(np.log(max(K, 1)), 1e-9)

# -----------------------------
# Metrics
# -----------------------------
def _edge_stress_columns(ulmm: dict, veh: str, g_edges_df: pd.DataFrame, top_q: float = 0.90) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (stress_cont, stress_bin). Uses φ if present, else lanes-scarcity fallback.
    Ensures stress_bin has both 0 and 1 by adaptively lowering the cutoff.
    """
    # 1) source
    if "phi" in g_edges_df and np.isfinite(g_edges_df["phi"]).any():
        src = pd.to_numeric(g_edges_df["phi"], errors="coerce")
    else:
        Gm: nx.MultiDiGraph = ulmm["graph"]
        vals = []
        for u, v in zip(g_edges_df["u"].astype(int), g_edges_df["v"].astype(int)):
            vs = []
            try:
                for _, _, _, d in Gm.edges(int(u), int(v), keys=True, data=True):
                    x = d.get("f_lanes_scarcity")
                    if x is not None and np.isfinite(x): vs.append(float(x))
            except Exception:
                pass
            vals.append(float(np.nanmean(vs)) if vs else np.nan)
        src = pd.Series(vals, index=g_edges_df.index, dtype=float)

    # 2) continuous = ranks
    ranks = src.rank(pct=True, method="average")

    # 3) adaptive binary cut
    q = float(top_q)
    y = None
    for _ in range(6):  # 90,85,80,75,70,65
        thr = np.nanquantile(ranks, q) if np.isfinite(ranks).any() else np.inf
        y_try = (ranks >= thr).astype(int)
        if 0 < y_try.sum() < len(y_try):
            y = y_try; break
        q -= 0.05
    if y is None:
        # fallback: mark top 1% by rank (at least one positive)
        k = max(1, int(0.01 * len(ranks)))
        y = np.zeros(len(ranks), dtype=int)
        y[np.argsort(ranks.to_numpy())[-k:]] = 1
        y = pd.Series(y, index=ranks.index)

    return ranks, y

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman = Pearson on ranks (no SciPy needed)."""
    x = pd.Series(x).rank(method="average").to_numpy()
    y = pd.Series(y).rank(method="average").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xm, ym = x[mask], y[mask]
    xm = (xm - xm.mean()) / (xm.std() + 1e-12)
    ym = (ym - ym.mean()) / (ym.std() + 1e-12)
    return float(np.clip((xm * ym).mean(), -1, 1))

def logit_beta(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, max_iter: int = 100) -> float:
    """
    Single-feature logistic regression (y in {0,1}) using IRLS with L2 ridge.
    Returns slope for x (no SciPy/statsmodels).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    X = np.c_[np.ones(mask.sum()), np.log1p(np.clip(x[mask], 0, None))]
    yv = y[mask].astype(float)
    if len(yv) < 10 or (yv.mean() in (0.0, 1.0)):
        return np.nan
    b = np.zeros(2)
    for _ in range(max_iter):
        z = X @ b
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        W = np.clip(W, 1e-6, None)
        XtW = X.T * W
        H = XtW @ X + l2 * np.eye(X.shape[1])
        g = XtW @ (z + (yv - p) / W)
        try:
            b_new = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            b_new = np.linalg.pinv(H) @ g
        if np.linalg.norm(b_new - b) < 1e-6:
            b = b_new; break
        b = b_new
    return float(b[1])

def H1_and_H4(ulmm: dict, veh: str, lambda_: float, gamma: float) -> Tuple[float, float]:
    """Mean across orientations: (H1 Lift = Spearman rho, H4 = logistic β_DEB)."""
    rhos, betas = [], []  # <-- fixed bug: must initialize both lists
    for ori in ORIS:
        deb_norm, g_edges = deb_pairweighted(ulmm, veh, orientation=ori, lambda_override=lambda_, gamma=gamma)
        x = pd.to_numeric(g_edges["deb_norm"], errors="coerce").to_numpy()
        stress_cont, stress_bin = _edge_stress_columns(ulmm, veh, g_edges)
        rho = spearman_rho(x, stress_cont.to_numpy())
        beta = logit_beta(x, stress_bin.to_numpy())
        rhos.append(rho); betas.append(beta)
    return float(np.nanmean(rhos)), float(np.nanmean(betas))

def H2_deltaAUC(ulmm: dict, veh: str, lambda_: float, kappa: float) -> float:
    """Mean ΔAUC across orientations & TAUS between top-q ServC and median-quantile set."""
    df = compute_servc(ulmm, veh, kappa=kappa, lambda_override=lambda_)
    acc_ids = df["a_node"].astype(int).to_numpy()
    demand = ulmm["demand"]
    d_nodes = demand["i_node"].astype(int).to_numpy()
    w_dem   = demand["w"].astype(float).to_numpy()

    G, _, _ = collapsed_graph(ulmm, veh, lambda_override=lambda_)
    def oriented_graph(ori: str): return G if ori=="AD" else G.reverse(copy=False)

    # Precompute distances from each access, per orientation
    def dist_mat(ori: str):
        Gu = oriented_graph(ori)
        tau = np.full((len(acc_ids), len(d_nodes)), np.inf, dtype=float)
        for i, a in enumerate(acc_ids):
            dist = nx.single_source_dijkstra_path_length(Gu, int(a), weight="weight")
            tau[i, :] = [float(dist.get(int(d), np.inf)) for d in d_nodes]
        return tau

    qs = np.linspace(0.05, 1.0, 20)
    def coverage_for_set(tau_mat: np.ndarray, A_idx: np.ndarray, thr_s: float) -> float:
        tau_min = np.min(tau_mat[A_idx, :], axis=0)
        covered = np.isfinite(tau_min) & (tau_min <= thr_s)
        return float(w_dem[covered].sum()) / float(w_dem.sum())

    dAUCs = []
    for ori in ORIS:
        tau = dist_mat(ori)
        scores = df["servc_AD"].to_numpy() if ori=="AD" else df["servc_DA"].to_numpy()
        idx_sorted = np.argsort(scores)[::-1]
        for thr in TAUS:
            ys_top, ys_med = [], []
            for q in qs:
                k = max(1, int(math.ceil(q * len(acc_ids))))
                A_top = idx_sorted[:k]
                start = max(0, (len(idx_sorted)//2) - (k//2))
                A_med = idx_sorted[start:start+k]
                ys_top.append(coverage_for_set(tau, A_top, thr))
                ys_med.append(coverage_for_set(tau, A_med, thr))
            dAUCs.append(np.trapz(ys_top, qs) - np.trapz(ys_med, qs))
    return float(np.nanmean(dAUCs))

def H3_targeted_medians(ulmm: dict, veh: str, lambda_: float, gamma: float, K: int, eta: float) -> Tuple[float, float]:
    """
    Remove top-pct edges by DEB (targeted), compute conditional median Δτ_d by RPE tertile (AD orientation).
    Return (Low RPE targeted %, High RPE targeted %).
    """
    deb_norm_AD, _ = deb_pairweighted(ulmm, veh, orientation="AD", lambda_override=lambda_, gamma=gamma)
    G, _, _ = collapsed_graph(ulmm, veh, lambda_override=lambda_)
    demand = ulmm["demand"]; access = ulmm["access"]
    d_nodes = demand["i_node"].astype(int).to_numpy()
    a_nodes = access["i_node"].astype(int).to_numpy()

    def _tertile_masks_from_series(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        r = s.rank(pct=True, method="first")  # breaks ties deterministically
        return (r <= 1/3), ((r > 1/3) & (r <= 2/3)), (r > 2/3)

    Hnorm = rpe_normalized_series(ulmm, veh, K=K, eta=eta)
    if Hnorm.empty:
        return (np.nan, np.nan)
    low_m, mid_m, high_m = _tertile_masks_from_series(Hnorm)
    cut1, cut2 = Hnorm.quantile([1/3, 2/3])

    # Orientation graph for AD
    G_use = G
    def min_tau(graph: nx.DiGraph) -> Dict[int, float]:
        tau_min = {int(d): np.inf for d in d_nodes}
        for a in a_nodes:
            dist = nx.single_source_dijkstra_path_length(graph, int(a), weight="weight")
            for d in d_nodes:
                t = dist.get(int(d), np.inf)
                if t < tau_min[d]: tau_min[d] = t
        return tau_min

    base_tau = min_tau(G_use)

    # choose edges to remove
    E = list(deb_norm_AD.keys())
    n_remove = max(1, int(round(len(E) * (BUDGET_PCT / 100.0))))
    E_sorted = sorted(E, key=lambda e: deb_norm_AD[e], reverse=True)
    E_target = E_sorted[:n_remove]

    def remove_edges(graph: nx.DiGraph, edges: List[Tuple[int,int]]) -> nx.DiGraph:
        H = graph.copy()
        for u, v in edges:
            if H.has_edge(u, v): H.remove_edge(u, v)
        return H

    def conditional_median_delta(Hnorm: pd.Series, tau0: Dict[int,float], tau1: Dict[int,float], selector) -> float:
        idx = np.array([int(d) for d in Hnorm.index[selector].tolist()])
        rel = []
        for d in idx:
            t0 = tau0.get(int(d), np.inf); t1 = tau1.get(int(d), np.inf)
            if np.isfinite(t0) and np.isfinite(t1) and t0>0:
                rel.append((t1 - t0) / t0)
        return float(np.median(rel))*100.0 if rel else np.nan

    # targeted
    tau_t = min_tau(remove_edges(G_use, E_target))
    low_T  = conditional_median_delta(Hnorm, base_tau, tau_t, low_m)
    high_T = conditional_median_delta(Hnorm, base_tau, tau_t, high_m)
    return (low_T, high_T)

# -----------------------------
# Runner
# -----------------------------
def run_one_setting(ulmm: dict, veh: str, params: dict) -> dict:
    lambda_ = params.get("lambda_", REF["lambda_"])
    gamma   = params.get("gamma",   REF["gamma"])
    kappa   = params.get("kappa",   REF["kappa"])
    K       = params.get("K",       REF["K"])
    eta     = params.get("eta",     REF["eta"])
    res = dict(lambda_=lambda_, gamma=gamma, kappa=kappa, K=K, eta=eta)

    # H1 & H4 (averaged across orientations)
    h1, h4 = H1_and_H4(ulmm, veh, lambda_=lambda_, gamma=gamma)
    res["H1_Lift"] = h1
    res["H4_betaDEB"] = h4

    # H2 ΔAUC (averaged across orientations and thresholds)
    res["H2_dAUC"] = H2_deltaAUC(ulmm, veh, lambda_=lambda_, kappa=kappa)

    # H3 targeted medians (AD; report Low/High tertiles)
    low_T, high_T = H3_targeted_medians(ulmm, veh, lambda_=lambda_, gamma=gamma, K=K, eta=eta)
    res["H3_LowRPE_T"]  = low_T
    res["H3_HighRPE_T"] = high_T

    return res

def main():
    ulmm = load_ulmm(CITY)
    print(f"Loaded ULMM for {CITY} [{VEHICLE}]")

    # Reference
    ref = run_one_setting(ulmm, VEHICLE, REF)

    # Sweeps (relative to reference)
    rows_rel = []
    for sw in SWEEPS:
        name = sw["name"]
        cand = REF.copy(); cand.update({k: v for k, v in sw.items() if k != "name"})
        # If sweeping lambda, ensure base costs available
        if "lambda_" in sw:
            try:
                _guess_base_cost_key(ulmm["graph"], VEHICLE)
            except KeyError as e:
                warnings.warn(f"Skipping {name} (no base cost): {e}")
                rows_rel.append(dict(Change=name, **{k: np.nan for k in ["H1_Lift","H2_dAUC","H3_LowRPE_T","H3_HighRPE_T","H4_betaDEB"]}))
                continue

        res = run_one_setting(ulmm, VEHICLE, cand)
        # relative changes vs reference
        delta = dict(Change=name)
        for k in ["H1_Lift", "H2_dAUC", "H3_LowRPE_T", "H3_HighRPE_T", "H4_betaDEB"]:
            v0 = ref.get(k, np.nan); v1 = res.get(k, np.nan)
            if np.isnan(v0) or v0 == 0:
                delta[k] = np.nan if np.isnan(v1) else (v1 - v0)
            else:
                delta[k] = (v1 - v0) / abs(v0)
        rows_rel.append(delta)

    # Save CSV with relative changes (for Table~\ref{tab:sensitivity})
    df_rel = pd.DataFrame(rows_rel)[["Change","H1_Lift","H2_dAUC","H3_LowRPE_T","H3_HighRPE_T","H4_betaDEB"]]
    outdir = Path("results"); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"sensitivity_{slugify(CITY)}_{VEHICLE}.csv"
    df_rel.to_csv(csv_path, index=False)
    print(f"\nSaved sensitivity table (relative) to {csv_path}")

    # Print LaTeX rows for the manuscript table
    def fmt(x):
        if pd.isna(x): return r"\textemdash"
        return f"{x:.3f}"
    print("\n% --- paste the following into the LaTeX table body ---")
    for _, r in df_rel.iterrows():
        print(f"{r['Change']} & {fmt(r['H1_Lift'])} & {fmt(r['H2_dAUC'])} & {fmt(r['H3_LowRPE_T'])} & {fmt(r['H3_HighRPE_T'])} & {fmt(r['H4_betaDEB'])} \\\\")
    print("% ------------------------------------------------------\n")

if __name__ == "__main__":
    main()


# In[9]:


    CITY         = "Paris, France"
    VEHICLE      = "van"         # "van" | "cargo_bike"
    main()


# In[10]:


    CITY         = "Amsterdam, Netherlands"
    VEHICLE      = "van"         # "van" | "cargo_bike"
    main()


# In[11]:


    CITY         = "Seattle, Washington, USA"
    VEHICLE      = "van"         # "van" | "cargo_bike"
    main()


# In[12]:


    CITY         = "New York City, New York, USA"
    VEHICLE      = "van"         # "van" | "cargo_bike"
    main()


# In[ ]:




