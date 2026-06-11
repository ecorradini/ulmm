"""
Round-3 Major 6 (shared helper): per-demand-cell population weights from the WorldPop 2020
1 km UNadj rasters (cache_pop/<iso>_ppp_2020_1km_Aggregated_UNadj.tif).

Demand centroids are stored in the ULMM either in WGS84 or in the city's UTM zone (the
columns lat/lon then hold northing/easting); we mirror ablations._resnap_demand's detection
(|lat|max > 1000 -> projected) and transform back to WGS84 with the UTM zone inferred from
the graph nodes. The weight is the raster value at the centroid pixel (people per ~1 km
pixel), clipped at 0 and normalized to [0,1] per city; with 500 m cells this samples the
local population density, which is all the demand weighting uses.
"""
import numpy as np

ISO = {"Amsterdam": "nld", "Barcelona": "esp", "Paris": "fra",
       "Seattle": "usa", "New York City": "usa"}


def centroids_wgs84(u):
    """Demand centroids as (lon, lat) WGS84 arrays, mirroring _resnap_demand's detection."""
    dem = u["demand"]
    lat = dem["lat"].astype(float).to_numpy(); lon = dem["lon"].astype(float).to_numpy()
    if np.abs(lat).max() <= 1000:
        return lon, lat
    import pyproj, ablations as ab
    Gm = u["graph"]
    ns = list(Gm.nodes())[:2000]
    glon = float(np.mean([Gm.nodes[n]["x"] for n in ns])); glat = float(np.mean([Gm.nodes[n]["y"] for n in ns]))
    epsg = ab._utm_epsg(glon, glat)
    tr = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    wlon, wlat = tr.transform(lon, lat)   # lon col holds easting, lat col holds northing
    return np.asarray(wlon), np.asarray(wlat)


def pop_weights(u, city_short):
    """Normalized [0,1] population weight per demand row (order = u['demand'] rows)."""
    import rasterio
    lon, lat = centroids_wgs84(u)
    path = f"cache_pop/{ISO[city_short]}_ppp_2020_1km_Aggregated_UNadj.tif"
    with rasterio.open(path) as r:
        vals = np.array([v[0] for v in r.sample(zip(lon, lat))], float)
        nod = r.nodata
    if nod is not None:
        vals[vals == nod] = 0.0
    vals = np.clip(vals, 0.0, None)
    vals[~np.isfinite(vals)] = 0.0
    mx = vals.max()
    return vals / mx if mx > 0 else vals
