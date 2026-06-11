#!/usr/bin/env python
# coding: utf-8

# In[6]:


# build_ulmm.py  (OSMnx-version-compatible)
import os, re, math, pickle
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import networkx as nx
import osmnx as ox

# ----------------------------
# Configuration
# ----------------------------
OUTPUT_DIR = Path("ulmm_pickles"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = [
    "New York City, New York, USA",
    "Seattle, Washington, USA",
    "Paris, France",
    "Barcelona, Spain",
    "Amsterdam, Netherlands",
]

GRID_SIZE_M = 500
ALPHA_POP = 0.0
BETA_POI  = 1.0
MU_DI = 0.0
MU_AI = 0.0

VEHICLE_CONFIGS = {
    "van": {
        "theta": {
            "lanes_scarcity": 0.20, "restricted_access": 0.25, "bus_only": 0.10,
            "pedestrian_zone": 0.15, "slope": 0.10, "no_stopping": 0.20,
        },
        "lambda": 1.0,
    },
    "cargo_bike": {
        "theta": {
            "lanes_scarcity": 0.05, "restricted_access": 0.20, "bus_only": 0.05,
            "pedestrian_zone": 0.05, "slope": 0.45, "no_stopping": 0.20,
        },
        "lambda": 0.5,
    },
}

RETAIL_TAGS = {"shop": True, "amenity": ["supermarket","convenience","cafe","fast_food","bar","restaurant","pharmacy"]}
ACCESS_TAGS = {"amenity": ["post_office","parcel_locker"]}

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 180

# ----------------------------
# Version-compat helpers
# ----------------------------
def slugify(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+","-",name.lower()).strip("-")

def ox_geocode_to_gdf(place: str) -> gpd.GeoDataFrame:
    try:
        return ox.geocode_to_gdf(place)
    except AttributeError:
        return ox.gdf_from_place(place)  # very old OSMnx

def ox_project_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        return ox.project_gdf(gdf)
    except AttributeError:
        try:
            from osmnx.projection import project_gdf
            return project_gdf(gdf)
        except Exception:
            return gdf.to_crs(gdf.estimate_utm_crs())  # fallback

def ox_features_from_polygon(poly, tags: dict) -> gpd.GeoDataFrame:
    # OSMnx 2.x
    try:
        return ox.features_from_polygon(poly, tags)
    except AttributeError:
        pass
    # OSMnx 1.x
    try:
        return ox.geometries_from_polygon(poly, tags)
    except AttributeError:
        pass
    # Final fallback: module path import
    try:
        from osmnx.features import features_from_polygon
        return features_from_polygon(poly, tags)
    except Exception as e:
        raise RuntimeError("No compatible features/geometries_from_polygon in this OSMnx version") from e

def ox_add_edge_speeds(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    try:
        return ox.add_edge_speeds(G)
    except AttributeError:
        try:
            from osmnx.speed import add_edge_speeds
            return add_edge_speeds(G)
        except Exception:
            raise

def ox_add_edge_travel_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    try:
        return ox.add_edge_travel_times(G)
    except AttributeError:
        try:
            from osmnx.speed import add_edge_travel_times
            return add_edge_travel_times(G)
        except Exception:
            raise

def ox_nearest_nodes(G, X, Y):
    try:
        return ox.distance.nearest_nodes(G, X=X, Y=Y)
    except AttributeError:
        try:
            return ox.distance.get_nearest_nodes(G, X, Y)  # older API signature
        except Exception:
            return ox.get_nearest_nodes(G, X, Y)          # very old

def gpd_sjoin_within(points_gdf, polys_gdf, point_col="geometry"):
    try:
        return gpd.sjoin(points_gdf[[point_col]], polys_gdf[["geometry"]], predicate="within", how="left")
    except TypeError:
        # geopandas <0.10 used op=
        return gpd.sjoin(points_gdf[[point_col]], polys_gdf[["geometry"]], op="within", how="left")

def centroid_wgs84_safe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute centroids in a projected CRS then return them in WGS84.
    Works for mixed geometries (points/lines/polygons). If already points,
    centroids == original points.
    """
    if gdf.empty:
        return gdf
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    try:
        local_crs = gdf.estimate_utm_crs()  # robust local metric CRS
        gdf_proj = gdf.to_crs(local_crs)
        gdf_proj["geometry"] = gdf_proj.geometry.centroid
        gdf_out = gdf_proj.to_crs("EPSG:4326")
    except Exception:
        # Fallback: representative point in WGS84 (still better than centroid in degrees)
        gdf_out = gdf.copy()
        gdf_out["geometry"] = gdf_out.geometry.representative_point()
    return gdf_out.set_geometry("geometry")

# ----------------------------
# Build steps
# ----------------------------
def get_city_polygon(place_name: str) -> gpd.GeoDataFrame:
    gdf = ox_geocode_to_gdf(place_name)
    geom = unary_union(gdf.geometry.values)
    return gpd.GeoDataFrame({"place":[place_name]}, geometry=[geom], crs="EPSG:4326")

def graph_from_polygon(poly, network_type="drive_service") -> nx.MultiDiGraph:
    G = ox.graph_from_polygon(poly, network_type=network_type, simplify=True)
    G = ox_add_edge_speeds(G)
    G = ox_add_edge_travel_times(G)
    return G

def to_local_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return ox_project_gdf(gdf)

def add_length_if_missing(G: nx.MultiDiGraph) -> None:
    for _,_,_,d in G.edges(keys=True, data=True):
        if "length" not in d:
            if "geometry" in d and d["geometry"] is not None:
                d["length"] = float(d["geometry"].length)
            else:
                d["length"] = 0.0

def extract_retail_pois(poly) -> gpd.GeoDataFrame:
    pois = ox_features_from_polygon(poly, RETAIL_TAGS)
    if pois.empty:
        return pois
    pois = pois.reset_index(drop=True)
    # keep useful attrs, drop empties
    keep_cols = [c for c in pois.columns if c in {"amenity","shop","name"}]
    pois = pois[keep_cols + ["geometry"]]
    # compute centroids in projected CRS, then back to WGS84
    pois = centroid_wgs84_safe(pois)
    # ensure valid points only
    pois = pois[~pois.geometry.is_empty].copy()
    pois = pois.set_geometry("geometry")
    return pois  # already EPSG:4326

def extract_access_points(poly) -> gpd.GeoDataFrame:
    acc = ox_features_from_polygon(poly, ACCESS_TAGS)
    if acc.empty:
        return acc
    acc = acc.reset_index(drop=True)
    keep_cols = [c for c in acc.columns if c in {"amenity","name"}]
    acc = acc[keep_cols + ["geometry"]]
    acc = centroid_wgs84_safe(acc)
    acc = acc[~acc.geometry.is_empty].copy()
    acc = acc.set_geometry("geometry")
    return acc  # already EPSG:4326

def make_square_grid(city_poly_gdf: gpd.GeoDataFrame, cell_size_m: int) -> gpd.GeoDataFrame:
    gdf_proj = to_local_crs(city_poly_gdf)
    poly_proj = gdf_proj.geometry.iloc[0]
    minx, miny, maxx, maxy = poly_proj.bounds
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    polys = []
    for x in xs:
        for y in ys:
            cell = box(x, y, x+cell_size_m, y+cell_size_m)
            if cell.intersects(poly_proj):
                polys.append(cell.intersection(poly_proj))

    grid = gpd.GeoDataFrame({"cell_id": range(len(polys))}, geometry=polys, crs=gdf_proj.crs)
    grid["centroid"] = grid.geometry.centroid
    grid_wgs = grid.to_crs("EPSG:4326")
    grid_wgs["x"] = grid_wgs["centroid"].x
    grid_wgs["y"] = grid_wgs["centroid"].y
    return grid_wgs

def count_points_in_polys(points: gpd.GeoDataFrame,
                          polys: gpd.GeoDataFrame,
                          point_col: str = "geometry") -> np.ndarray:
    """
    Count how many point features fall inside each polygon.
    Robust to NaNs in join results and works across GeoPandas versions.
    Returns a length-|polys| integer array.
    """
    n = len(polys)
    if n == 0 or points.empty:
        return np.zeros(n, dtype=int)

    # ensure both layers share CRS
    pts = points.to_crs(polys.crs)

    # spatial join: left join so we keep all points; 'index_right' may contain NaNs
    joined = gpd_sjoin_within(pts, polys, point_col=point_col)

    # GeoPandas guarantees an 'index_right' column in the join result (for 'how="left"')
    if "index_right" not in joined.columns:
        # very defensive fallback: last column is typically index_right
        idx_series = joined.iloc[:, -1]
    else:
        idx_series = joined["index_right"]

    # drop NaNs (points not within any polygon), cast to int indices
    if idx_series.isna().all():
        return np.zeros(n, dtype=int)

    idx = idx_series.dropna().to_numpy()
    # idx can be float because of NaNs; cast safely to int
    idx = idx.astype(np.int64, copy=False)

    # fast histogram of counts per polygon id
    counts = np.bincount(idx, minlength=n)

    # ensure exact length
    if len(counts) < n:
        counts = np.pad(counts, (0, n - len(counts)), constant_values=0)

    return counts[:n]

def build_demand_weights(grid: gpd.GeoDataFrame, retail_pois: gpd.GeoDataFrame, alpha: float, beta: float) -> np.ndarray:
    poi_counts = count_points_in_polys(retail_pois, grid)
    poi_norm = poi_counts.astype(float)
    poi_norm = poi_norm/poi_norm.max() if poi_norm.max() > 0 else poi_norm
    if "pop" in grid.columns:
        pop = grid["pop"].values.astype(float)
        pop_norm = pop/pop.max() if pop.max() > 0 else np.zeros_like(pop, dtype=float)
    else:
        pop_norm = np.zeros(len(grid), dtype=float)
    return alpha*pop_norm + beta*poi_norm

def parse_lanes(v):
    if v is None:
        return None
    if isinstance(v, (int,float)):
        return int(v)
    if isinstance(v, str):
        parts = re.split(r"[;|,]", v)
        try:
            nums = [int(float(p)) for p in parts if p.strip()!=""]
            return max(nums) if nums else None
        except Exception:
            return None
    return None

def feature_components_for_edge(data: dict) -> Dict[str,float]:
    lanes = parse_lanes(data.get("lanes"))
    lanes_scarcity = 1.0 - min(lanes,3)/3.0 if lanes is not None else (1.0 - 2/3.0)

    access_vals = {str(data.get(k,"")).lower() for k in ["access","motor_vehicle","motorcar","vehicle"]}
    restricted_access = 1.0 if any(v in {"no","private","delivery"} for v in access_vals) else 0.0

    bus_only = 0.0
    for k in ["busway","bus","bus:lanes","bus:lanes:forward","bus:lanes:backward"]:
        if k in data:
            val = str(data.get(k)).lower()
            if any(s in val for s in ["designated","yes","lane","bus_only"]):
                bus_only = 1.0; break

    highway = str(data.get("highway","")).lower()
    ped_like = {"pedestrian","footway","path","living_street","track"}
    pedestrian_zone = 1.0 if highway in ped_like else 0.0

    grade = data.get("grade_abs", None)
    slope = min(abs(float(grade)),0.12)/0.12 if grade is not None else 0.0

    no_stopping = 0.0
    for k in ["parking:lane:both","parking:lane:left","parking:lane:right"]:
        val = str(data.get(k,"")).lower()
        if "no_stopping" in val or "no_parking" in val:
            no_stopping = 1.0; break

    return {
        "lanes_scarcity": float(lanes_scarcity),
        "restricted_access": float(restricted_access),
        "bus_only": float(bus_only),
        "pedestrian_zone": float(pedestrian_zone),
        "slope": float(slope),
        "no_stopping": float(no_stopping),
    }

def compute_friction_and_effective_costs(G: nx.MultiDiGraph) -> None:
    add_length_if_missing(G)
    for _,_,_,d in G.edges(keys=True, data=True):
        c_base = float(d.get("travel_time", np.nan))
        if not np.isfinite(c_base):
            length = float(d.get("length", 0.0))
            speed_kph = float(d.get("speed_kph", 30.0))
            c_base = length / (max(speed_kph,1.0)*1000.0/3600.0)
        d["c_base"] = float(c_base)

        comps = feature_components_for_edge(d)
        for n, val in comps.items():
            d[f"f_{n}"] = val

        for veh, cfg in VEHICLE_CONFIGS.items():
            theta = cfg["theta"]; lam = float(cfg["lambda"])
            phi = sum(float(theta[k])*float(comps[k]) for k in theta.keys())
            d[f"phi_{veh}"]   = float(phi)
            d[f"c_eff_{veh}"] = d["c_base"]*(1.0 + lam*phi)

def anchor_points_to_graph(G: nx.MultiDiGraph, gdf_points: gpd.GeoDataFrame) -> List[int]:
    """
    Return nearest node ids for features in gdf_points.
    If geometry is not Point, convert to safe centroids in a projected CRS.
    """
    if gdf_points.empty:
        return []

    gdf = gdf_points.copy()

    # Ensure we have a valid CRS; default to WGS84 if missing
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)

    # If geometries are not Points, convert to centroids in a projected CRS, then back to WGS84
    if not (gdf.geometry.geom_type == "Point").all():
        try:
            local_crs = gdf.estimate_utm_crs()
            gdf = gdf.to_crs(local_crs)
            gdf["geometry"] = gdf.geometry.centroid
            gdf = gdf.to_crs("EPSG:4326")
        except Exception:
            # Fallback: representative points in WGS84
            gdf["geometry"] = gdf.geometry.representative_point()

    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    nn = ox_nearest_nodes(G, X=xs, Y=ys)
    return [int(n) for n in np.asarray(nn).tolist()]

# ----------------------------
# ULMM builder
# ----------------------------
def build_ulmm_for_city(place_name: str) -> Path:
    city_slug = slugify(place_name)
    print(f"\n=== Building ULMM for: {place_name} ===")

    city_gdf = get_city_polygon(place_name)
    city_poly = city_gdf.geometry.iloc[0]

    G = graph_from_polygon(city_poly, network_type="drive_service")
    compute_friction_and_effective_costs(G)

    gdf_access = extract_access_points(city_poly).copy()
    if gdf_access.empty:
        print("WARNING: No access points found (amenity in {post_office, parcel_locker}).")
    gdf_access["a_id"] = [f"a_{i}" for i in range(len(gdf_access))]
    gdf_access["type"] = gdf_access.get("amenity","").astype(str)

    gdf_grid = make_square_grid(city_gdf, GRID_SIZE_M).copy()
    gdf_grid["d_id"] = [f"d_{i}" for i in range(len(gdf_grid))]

    gdf_retail = extract_retail_pois(city_poly)
    w = build_demand_weights(gdf_grid, gdf_retail, alpha=ALPHA_POP, beta=BETA_POI)
    gdf_grid["w"] = w

    keep = w > 0
    if keep.sum() == 0:
        print("WARNING: All demand weights are zero; keeping a small subset for structure.")
        keep = np.zeros_like(w, dtype=bool); keep[:min(len(w),250)] = True
    gdf_demand = gdf_grid.loc[keep].copy().reset_index(drop=True)

    # Use centroids (Points) as the active geometry for anchoring
    gdf_demand = gdf_demand.set_geometry(gdf_demand["centroid"])

    # Now anchor
    demand_anchor = anchor_points_to_graph(G, gdf_demand)
    gdf_demand["i_node"] = demand_anchor

    if not gdf_access.empty:
        access_anchor = anchor_points_to_graph(G, gdf_access)
        gdf_access["i_node"] = access_anchor
    else:
        gdf_access["i_node"] = []

    ulmm = {
        "city": place_name,
        "graph": G,
        "demand": pd.DataFrame({
            "d_id": gdf_demand["d_id"].values,
            "w": gdf_demand["w"].values,
            "lat": gdf_demand.geometry.y.values,
            "lon": gdf_demand.geometry.x.values,
            "i_node": gdf_demand["i_node"].values,
        }),
        "access": pd.DataFrame({
            "a_id": gdf_access["a_id"].values if not gdf_access.empty else [],
            "atype": gdf_access.get("amenity", pd.Series([], dtype=str)).astype(str).values if not gdf_access.empty else [],
            "lat": gdf_access.geometry.y.values if not gdf_access.empty else [],
            "lon": gdf_access.geometry.x.values if not gdf_access.empty else [],
            "i_node": gdf_access["i_node"].values if not gdf_access.empty else [],
        }),
        "params": {
            "vehicles": VEHICLE_CONFIGS,
            "mu_DI": MU_DI, "mu_AI": MU_AI,
            "grid_size_m": GRID_SIZE_M,
            "alpha_pop": ALPHA_POP, "beta_poi": BETA_POI,
            "access_tags": ACCESS_TAGS, "retail_tags": RETAIL_TAGS,
            "network_type": "drive_service",
            "notes": "Edges carry c_base, f_*, phi_<veh>, c_eff_<veh>.",
        },
        "city_polygon_wkt": city_gdf.geometry.iloc[0].wkt,
    }

    out_path = OUTPUT_DIR / f"ulmm_{city_slug}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(ulmm, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved ULMM pickle: {out_path.resolve()}")
    return out_path

print("Starting ULMM multiplex build...")
built = []
for place in CITIES:
    try:
        pkl = build_ulmm_for_city(place)
        built.append(pkl)
    except Exception as e:
        print(f"[ERROR] {place}: {e}")
print("\nDone. Built pickles:")
for p in built:
    print(" -", p)


# In[7]:


# DESCRIPTIVE
import os
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely import wkt

# ----------------------------
# Config
# ----------------------------
PICKLES_DIR = Path("ulmm_pickles")
VEH_VAN = "van"
VEH_CB  = "cargo_bike"   # matches your build script keys

# Coverage thresholds (seconds) for Table 3
COVERAGE_THRESHOLDS = [300, 600]  # 5, 10 minutes (AD orientation)
COVERAGE_VEHICLE = VEH_VAN        # use van for Table 3 (change to VEH_CB to switch)

# ----------------------------
# Helpers
# ----------------------------
def _edge_attr_array(G: nx.MultiDiGraph, attr: str) -> np.ndarray:
    vals = []
    for _, _, _, d in G.edges(keys=True, data=True):
        v = d.get(attr, np.nan)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(np.nan)
    return np.array(vals, dtype=float)

def _share_positive(vals: np.ndarray) -> float:
    m = np.isfinite(vals)
    if not m.any():
        return np.nan
    return 100.0 * (vals[m] > 0).mean()

def _fmt_pm(mean: float, sd: float, decimals: int = 3) -> str:
    if np.isnan(mean) or np.isnan(sd):
        return "---"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"

def _median(vals: np.ndarray) -> float:
    if vals.size == 0:
        return np.nan
    return float(np.nanmedian(vals))

def _area_km2_from_wkt(wkt_str: str) -> float:
    geom = wkt.loads(wkt_str)
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
    try:
        crs_loc = gdf.estimate_utm_crs()
        gdfp = gdf.to_crs(crs_loc)
        return float(gdfp.geometry.area.iloc[0] / 1e6)
    except Exception:
        # fallback: approximate using Web Mercator
        gdfp = gdf.to_crs(3857)
        return float(gdfp.geometry.area.iloc[0] / 1e6)

def _access_counts(access_df: pd.DataFrame) -> Tuple[int, int, int]:
    if access_df is None or len(access_df) == 0:
        return 0, 0, 0
    at = access_df.get("atype", pd.Series([], dtype=str)).astype(str).str.lower()
    n_po = int((at == "post_office").sum())
    n_lk = int((at == "parcel_locker").sum())
    return n_po, n_lk, int(len(access_df))

def _coverage_ad_percent(ulmm: Dict, veh_key: str, thresholds: List[int]) -> Dict[int, float]:
    """
    Weighted coverage of demand within time thresholds (AD orientation).
    For each demand d, compute min_a tau(a->d) on G with weight=c_eff_<veh>.
    Returns {threshold_seconds: percent_of_total_weight}.
    """
    G: nx.MultiDiGraph = ulmm["graph"]
    demand: pd.DataFrame = ulmm["demand"]
    access: pd.DataFrame = ulmm["access"]
    if len(demand) == 0 or len(access) == 0:
        return {t: 0.0 for t in thresholds}

    weight_attr = f"c_eff_{veh_key}"
    # Build mapping infra-node -> indices of demand rows anchored there
    anch = demand["i_node"].to_numpy()
    idx_by_node: Dict[int, List[int]] = {}
    for i, n in enumerate(anch):
        idx_by_node.setdefault(int(n), []).append(i)

    # Initialize best times per demand row with +inf
    best = np.full(len(demand), np.inf, dtype=float)
    max_cutoff = max(thresholds)

    # Dijkstra from each access anchor with cutoff
    for a in access["i_node"]:
        if pd.isna(a):
            continue
        a = int(a)
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, source=a, weight=weight_attr, cutoff=max_cutoff
            )
        except Exception:
            # fallback to base time if vehicle-specific costs missing
            lengths = nx.single_source_dijkstra_path_length(
                G, source=a, weight="c_base", cutoff=max_cutoff
            )
        # Update best times for demand anchored at reached nodes
        for n, dist in lengths.items():
            if n in idx_by_node:
                for i in idx_by_node[n]:
                    if dist < best[i]:
                        best[i] = dist

    w = demand["w"].to_numpy(dtype=float)
    w_total = float(w.sum()) if len(w) else 0.0
    out = {}
    for t in thresholds:
        if w_total <= 0:
            out[t] = 0.0
        else:
            covered = float(w[(best <= float(t))].sum())
            out[t] = 100.0 * covered / w_total
    return out

def _city_label(ulmm: Dict) -> str:
    # Shorter label if you prefer: takes what's in ulmm["city"] verbatim
    return str(ulmm.get("city", "City"))

# ----------------------------
# Main summarization
# ----------------------------
def summarize_pickles(pickle_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_t1 = []
    rows_t2 = []
    rows_t3 = []

    for pkl_path in sorted(pickle_dir.glob("ulmm_*.pkl")):
        with open(pkl_path, "rb") as f:
            ulmm = pickle.load(f)

        city = _city_label(ulmm)
        G: nx.MultiDiGraph = ulmm["graph"]
        demand: pd.DataFrame = ulmm["demand"]
        access: pd.DataFrame = ulmm["access"]

        # --- Table 1: sizes + medians + area
        nV = int(G.number_of_nodes())
        nE = int(G.number_of_edges())
        nD = int(len(demand))
        nA = int(len(access))

        c_base = _edge_attr_array(G, "c_base")
        c_van  = _edge_attr_array(G, "c_eff_van")
        c_cb   = _edge_attr_array(G, "c_eff_cargo_bike")

        med_c_base = _median(c_base)
        med_c_van  = _median(c_van if np.isfinite(c_van).any() else c_base)
        med_c_cb   = _median(c_cb if np.isfinite(c_cb).any() else c_base)

        area_km2 = _area_km2_from_wkt(ulmm.get("city_polygon_wkt", ""))

        rows_t1.append({
            "City": city,
            "|V_I|": nV,
            "|E_I|": nE,
            "|D|": nD,
            "|A|": nA,
            "med c_e (s)": med_c_base,
            "med \\tilde c^{van}_e (s)": med_c_van,
            "med \\tilde c^{cb}_e (s)": med_c_cb,
            "Area (km^2)": area_km2,
        })

        # --- Table 2: curb flags shares + phi mean/sd
        f_lanes  = _edge_attr_array(G, "f_lanes_scarcity")
        f_rest   = _edge_attr_array(G, "f_restricted_access")
        f_bus    = _edge_attr_array(G, "f_bus_only")
        f_ped    = _edge_attr_array(G, "f_pedestrian_zone")
        f_nostop = _edge_attr_array(G, "f_no_stopping")

        share_lanes = _share_positive(f_lanes)
        share_rest  = _share_positive(f_rest)
        share_bus   = _share_positive(f_bus)
        share_ped   = _share_positive(f_ped)
        share_nstop = _share_positive(f_nostop)

        phi_van = _edge_attr_array(G, "phi_van")
        phi_cb  = _edge_attr_array(G, "phi_cargo_bike")

        phi_van_mean = float(np.nanmean(phi_van)) if phi_van.size else np.nan
        phi_van_sd   = float(np.nanstd(phi_van))  if phi_van.size else np.nan
        phi_cb_mean  = float(np.nanmean(phi_cb))  if phi_cb.size else np.nan
        phi_cb_sd    = float(np.nanstd(phi_cb))   if phi_cb.size else np.nan

        rows_t2.append({
            "City": city,
            "lanes scarce > 0 (%)": share_lanes,
            "restricted (%)": share_rest,
            "bus-only (%)": share_bus,
            "ped-zone (%)": share_ped,
            "no-stopping (%)": share_nstop,
            "phi_van (mean ± sd)": _fmt_pm(phi_van_mean, phi_van_sd, 3),
            "phi_cb (mean ± sd)":  _fmt_pm(phi_cb_mean,  phi_cb_sd,  3),
        })

        # --- Table 3: access inventory + coverage (AD)
        n_po, n_lk, n_acc = _access_counts(access)
        coverage = _coverage_ad_percent(ulmm, COVERAGE_VEHICLE, COVERAGE_THRESHOLDS)

        rows_t3.append({
            "City": city,
            "# Post offices": n_po,
            "# Lockers": n_lk,
            "# Access (|A|)": n_acc,
            "% demand ≤ 5 min": coverage.get(300, 0.0),
            "% demand ≤ 10 min": coverage.get(600, 0.0),
        })

    # Build DataFrames
    t1 = pd.DataFrame(rows_t1).sort_values("City").reset_index(drop=True)
    t2 = pd.DataFrame(rows_t2).sort_values("City").reset_index(drop=True)
    t3 = pd.DataFrame(rows_t3).sort_values("City").reset_index(drop=True)

    # Nice rounding for display
    t1["med c_e (s)"] = t1["med c_e (s)"].round(1)
    t1["med \\tilde c^{van}_e (s)"] = t1["med \\tilde c^{van}_e (s)"].round(1)
    t1["med \\tilde c^{cb}_e (s)"]  = t1["med \\tilde c^{cb}_e (s)"].round(1)
    t1["Area (km^2)"]               = t1["Area (km^2)"].round(2)

    for col in ["lanes scarce > 0 (%)","restricted (%)","bus-only (%)","ped-zone (%)","no-stopping (%)"]:
        t2[col] = t2[col].round(1)

    for col in ["% demand ≤ 5 min","% demand ≤ 10 min"]:
        t3[col] = t3[col].round(1)

    return t1, t2, t3

if not PICKLES_DIR.exists():
    raise SystemExit(f"Directory not found: {PICKLES_DIR.resolve()}")

t1, t2, t3 = summarize_pickles(PICKLES_DIR)

# Print LaTeX fragments you can paste into the paper
print("\n% ===== Table 1: ULMM ingredient summary =====")
print(t1.to_latex(index=False, escape=False))

print("\n% ===== Table 2: Curb/friction summary =====")
print(t2.to_latex(index=False, escape=False))

print("\n% ===== Table 3: Access + coverage (AD, vehicle = {}) =====".format(COVERAGE_VEHICLE))
print(t3.to_latex(index=False, escape=False))

# Optional: also save CSVs
out_dir = PICKLES_DIR / "summaries"
out_dir.mkdir(parents=True, exist_ok=True)
t1.to_csv(out_dir / "table1_ulmm_summary.csv", index=False)
t2.to_csv(out_dir / "table2_friction_summary.csv", index=False)
t3.to_csv(out_dir / "table3_access_coverage_ad.csv", index=False)

print(f"\nSaved CSVs to: {out_dir.resolve()}")


# In[ ]:




