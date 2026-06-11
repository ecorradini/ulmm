"""
Second-domain instantiation (healthcare access), reusing existing street graphs.
For each city: reuse the stored CDAM graph (with effective costs), fetch
healthcare facilities (hospital/clinic/doctors) from OSM as the ACCESS set,
keep the 500 m demand grid with UNIFORM weights (population-agnostic baseline),
anchor access to nearest graph nodes, and save ulmm_<slug>-healthcare.pkl.
"""
import pickle, copy
import numpy as np, pandas as pd, networkx as nx, osmnx as ox
from shapely import wkt
import ablations as ab

CITIES = ["Amsterdam, Netherlands", "Barcelona, Spain", "Paris, France"]
TAGS = {"amenity": ["hospital", "clinic", "doctors"]}

def build(city):
    u = ab.load_ulmm(city)
    G = u["graph"]
    poly = wkt.loads(u["city_polygon_wkt"])
    feats = ox.features_from_polygon(poly, TAGS)
    feats = feats[feats.geometry.notna()].copy()
    # representative point (centroid works for points and building polygons)
    reps = feats.geometry.representative_point()
    lons = reps.x.to_numpy(); lats = reps.y.to_numpy()
    atypes = feats["amenity"].astype(str).to_numpy() if "amenity" in feats else np.array(["facility"]*len(feats))
    # anchor to nearest graph node (graph is EPSG:4326, x=lon y=lat)
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    access = pd.DataFrame({
        "a_id": np.arange(len(nodes)),
        "atype": atypes,
        "lat": lats, "lon": lons,
        "i_node": np.asarray(nodes, dtype=int),
    }).drop_duplicates(subset="i_node").reset_index(drop=True)
    access["a_id"] = np.arange(len(access))
    # demand: reuse grid, UNIFORM weights (population-agnostic)
    demand = u["demand"].copy()
    demand["w"] = 1.0
    new = dict(u)
    new["city"] = city.split(",")[0] + "-Healthcare"
    new["access"] = access
    new["demand"] = demand
    new["params"] = dict(u.get("params", {}))
    new["params"]["domain"] = "healthcare"
    new["params"]["access_tags"] = TAGS
    new["params"]["demand_weighting"] = "uniform"
    slug = ab.slugify(new["city"])
    path = ab.PICKLE_DIR / f"ulmm_{slug}.pkl"
    with open(path, "wb") as f:
        pickle.dump(new, f)
    print(f"[{city}] healthcare: {len(access)} access nodes "
          f"({pd.Series(access['atype']).value_counts().to_dict()}), "
          f"{len(demand)} demand cells -> {path.name}", flush=True)
    return new["city"]

if __name__ == "__main__":
    built = []
    for c in CITIES:
        try:
            built.append(build(c))
        except Exception as e:
            print(f"[{c}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("BUILT:", built, flush=True)
