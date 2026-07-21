"""
R10 (panel): population share of ZERO-POI cells (excluded from the demand grid).

The demand grid keeps only 500 m cells with >=1 retail/service POI
(build_ulmm_mod.py: keep = w > 0). This script rebuilds the full grid from the
city_polygon_wkt stored in each ULMM pickle (replicating make_square_grid: osmnx
UTM projection, 500 m boxes, same cell ordering hence same d_id indexing), samples
WorldPop 2020 at every cell centroid, and reports the population share falling in
cells absent from the demand table.

Output: results/pop_coverage.csv
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from osmnx.projection import project_gdf as ox_project_gdf
from shapely import wkt as shapely_wkt
from shapely.geometry import box

import ablations as ab

CITIES = [("Amsterdam, Netherlands", "nld"), ("Barcelona, Spain", "esp"),
          ("Paris, France", "fra"), ("Seattle, Washington, USA", "usa"),
          ("New York City, New York, USA", "usa")]
GRID = 500
os.makedirs("results", exist_ok=True)


def make_square_grid(city_poly_gdf, cell_size_m):
    """Verbatim replica of build_ulmm_mod.make_square_grid (same ordering => same d_id index)."""
    gdf_proj = ox_project_gdf(city_poly_gdf)
    poly_proj = gdf_proj.geometry.iloc[0]
    minx, miny, maxx, maxy = poly_proj.bounds
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)
    polys = []
    for x in xs:
        for y in ys:
            cell = box(x, y, x + cell_size_m, y + cell_size_m)
            if cell.intersects(poly_proj):
                polys.append(cell.intersection(poly_proj))
    grid = gpd.GeoDataFrame({"cell_id": range(len(polys))}, geometry=polys, crs=gdf_proj.crs)
    cent = grid.geometry.centroid
    cent_wgs = gpd.GeoSeries(cent, crs=gdf_proj.crs).to_crs("EPSG:4326")
    grid["x"] = cent_wgs.x.values
    grid["y"] = cent_wgs.y.values
    return grid


def main():
    rows = []
    for city, iso in CITIES:
        short = city.split(",")[0]
        u = ab.load_ulmm(city)
        poly = shapely_wkt.loads(u["city_polygon_wkt"])
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        grid = make_square_grid(gdf, GRID)
        kept_ids = set(int(str(d).split("_")[1]) for d in u["demand"]["d_id"])
        tif = f"cache_pop/{iso}_ppp_2020_1km_Aggregated_UNadj.tif"
        with rasterio.open(tif) as r:
            vals = np.array([v[0] for v in r.sample(zip(grid.x.values, grid.y.values))],
                            dtype=float)
        vals = np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)
        # 1 km raster sampled at 500 m centroids: relative shares are what we need
        in_grid = np.array([i in kept_ids for i in grid.cell_id.values])
        pop_all = vals.sum()
        pop_excl = vals[~in_grid].sum()
        rows.append(dict(city=short, n_cells=len(grid), n_kept=int(in_grid.sum()),
                         n_excluded=int((~in_grid).sum()),
                         pop_share_excluded=round(float(pop_excl / pop_all), 3) if pop_all > 0 else np.nan))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/pop_coverage.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
