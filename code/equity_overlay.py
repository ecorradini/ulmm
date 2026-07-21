"""S2 equity overlay: ACS tract demographics of NYC kappa=1 vs kappa>=2 zones.

Data (keyless, www2.census.gov):
  - ACS 2019-2023 5-year table-based summary files, streamed and filtered to
    the five NYC counties at tract level (sumlevel 140):
      B19013 median household income (E001)
      B01001 sex by age; 65+ = E020-E025 (male) + E044-E049 (female), total E001
  - Cartographic tract boundaries cb_2023_36_tract_500k (EPSG:4269).

Join: demand zone -> snapped street node lon/lat (collapsed-graph cache) ->
point-in-polygon tract. Per orientation and kappa class (1 vs >=2/inf):
demand-weighted median income, weighted-mean 65+ share; Mann-Whitney U on
zone-level values.

Output: results/equity_overlay.csv + printed summary.
"""
import pickle
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
# Layout-agnostic root: in the working tree the scripts sit beside results/;
# in the released repository they live in code/ with results/ one level up.
if not (HERE / "results").exists() and (HERE.parent / "results").exists():
    HERE = HERE.parent
CACHE = HERE / "cache_equity"
CACHE.mkdir(exist_ok=True)

COUNTIES = ("005", "047", "061", "081", "085")
BASE = ("https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
        "table-based-SF/data/5YRData/")
TRACT_ZIP = ("https://www2.census.gov/geo/tiger/GENZ2023/shp/"
             "cb_2023_36_tract_500k.zip")

GRAPH_PKL = (HERE / "cache" /
             "collapsed_graph__city-new-york-city-new-york-usa__"
             "fp-1758788089388532246-105829187__veh-van.pkl")
KAPPA_CSV = HERE / "results" / "reanchor_perdemand.csv"  # primary k=3 labels
OUT = HERE / "results" / "equity_overlay.csv"

GREP_PAT = r"^GEO_ID\|" + r"\|".join([]) or ""  # header handled separately


def fetch_table(table: str) -> pd.DataFrame:
    """Stream a nationwide .dat, keep header + NYC tract rows."""
    cached = CACHE / f"{table}_nyc_tracts.psv"
    if not cached.exists():
        url = f"{BASE}acsdt5y2023-{table}.dat"
        pat = r"^(GEO_ID|1400000US36(005|047|061|081|085))"
        cmd = f"curl -s '{url}' | grep -E '{pat}' > '{cached}.tmp'"
        subprocess.run(["bash", "-c", cmd], check=True)
        Path(f"{cached}.tmp").rename(cached)
    df = pd.read_csv(cached, sep="|")
    df["geoid"] = df["GEO_ID"].str.replace("1400000US", "", regex=False)
    return df


def fetch_tract_geoms() -> gpd.GeoDataFrame:
    zpath = CACHE / "cb_2023_36_tract_500k.zip"
    if not zpath.exists():
        with urllib.request.urlopen(TRACT_ZIP, timeout=300) as r:
            zpath.write_bytes(r.read())
    gdf = gpd.read_file(f"zip://{zpath}")
    gdf = gdf[gdf["COUNTYFP"].isin(COUNTIES)].to_crs(4326)
    return (gdf[["GEOID", "COUNTYFP", "geometry"]]
            .rename(columns={"GEOID": "geoid"}))


def wmedian(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x)
    x, w = x[m], w[m]
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def main():
    inc = fetch_table("b19013")
    inc["med_income"] = pd.to_numeric(inc["B19013_E001"], errors="coerce")
    inc.loc[inc.med_income < 0, "med_income"] = np.nan

    age = fetch_table("b01001")
    m65 = [f"B01001_E{i:03d}" for i in range(20, 26)]
    f65 = [f"B01001_E{i:03d}" for i in range(44, 50)]
    for c in ["B01001_E001"] + m65 + f65:
        age[c] = pd.to_numeric(age[c], errors="coerce")
    age["share65"] = np.where(age.B01001_E001 > 0,
                              age[m65 + f65].sum(axis=1) / age.B01001_E001,
                              np.nan)

    acs = inc[["geoid", "med_income"]].merge(
        age[["geoid", "share65", "B01001_E001"]], on="geoid")
    tracts = fetch_tract_geoms().merge(acs, on="geoid", how="left")
    print(f"tracts: {len(tracts)}, income non-null: "
          f"{tracts.med_income.notna().sum()}")

    with open(GRAPH_PKL, "rb") as f:
        _, _, node_xy = pickle.load(f)["__data__"]
    kap_all = pd.read_csv(KAPPA_CSV)
    nyc = [c for c in kap_all.city.unique() if "York" in c][0]

    rows = []
    joined_by_k = {}
    for kk in (1, 3):
        kap = kap_all[(kap_all.city == nyc) & (kap_all.k == kk)].copy()
        kap["lon"] = kap.d.map(lambda n: node_xy.get(n, (np.nan,) * 2)[0])
        kap["lat"] = kap.d.map(lambda n: node_xy.get(n, (np.nan,) * 2)[1])
        assert kap.lon.notna().all(), "unmapped demand nodes"
        pts = gpd.GeoDataFrame(kap, geometry=gpd.points_from_xy(kap.lon, kap.lat),
                               crs=4326)
        joined_by_k[kk] = gpd.sjoin(pts, tracts, how="left", predicate="within")
    # anchoring sensitivity: the primary k=3 result alongside the k=1 control
    for kk, jj in joined_by_k.items():
        for ori, sub in jj.groupby("orientation"):
            frag, red = sub[sub.kappa == 1], sub[sub.kappa != 1]
            rows.append(dict(
                orientation=f"k{kk}_{ori}", cls="kappa=1", n=len(frag),
                w=frag.w.sum(), med_income_wmed=wmedian(frag.med_income, frag.w),
                share65_wmean=float(np.average(
                    frag.share65.fillna(frag.share65.median()), weights=frag.w))))
            rows.append(dict(
                orientation=f"k{kk}_{ori}", cls="kappa>=2", n=len(red),
                w=red.w.sum(), med_income_wmed=wmedian(red.med_income, red.w),
                share65_wmean=float(np.average(
                    red.share65.fillna(red.share65.median()), weights=red.w))))
            rows.append(dict(
                orientation=f"k{kk}_{ori}", cls="mannwhitney_p", n=len(sub), w=np.nan,
                med_income_wmed=mannwhitneyu(frag.med_income.dropna(),
                                             red.med_income.dropna()).pvalue,
                share65_wmean=mannwhitneyu(frag.share65.dropna(),
                                           red.share65.dropna()).pvalue))

    joined = joined_by_k[3]
    print(f"zones joined (k=3): {joined['geoid'].notna().sum()}/{len(joined)}")

    for ori, sub in joined.groupby("orientation"):
        frag = sub[sub.kappa == 1]
        red = sub[sub.kappa != 1]
        for name, cls in (("kappa=1", frag), ("kappa>=2", red)):
            rows.append(dict(
                orientation=ori, cls=name, n=len(cls), w=cls.w.sum(),
                med_income_wmed=wmedian(cls.med_income, cls.w),
                share65_wmean=float(np.average(
                    cls.share65.fillna(cls.share65.median()), weights=cls.w)),
            ))
        p_inc = mannwhitneyu(frag.med_income.dropna(),
                             red.med_income.dropna()).pvalue
        p_65 = mannwhitneyu(frag.share65.dropna(), red.share65.dropna()).pvalue
        rows.append(dict(orientation=ori, cls="mannwhitney_p", n=len(sub),
                         w=np.nan, med_income_wmed=p_inc, share65_wmean=p_65))
    # borough spread of the fragile class: guards against reading the income
    # difference as a single-neighborhood artifact
    boro = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan",
            "081": "Queens", "085": "StatenIsland"}
    da = joined[joined.orientation == "DA"].copy()
    da["boro"] = da.COUNTYFP.map(boro)
    for b, g in da.groupby("boro"):
        rows.append(dict(orientation="boro", cls=b, n=len(g),
                         w=int((g.kappa == 1).sum()),
                         med_income_wmed=round(100 * (g.kappa == 1).mean(), 1),
                         share65_wmean=np.nan))
    rows.append(dict(orientation="meta", cls="join_coverage",
                     n=int(joined["geoid"].notna().sum()), w=len(joined),
                     med_income_wmed=int(tracts.med_income.notna().sum()),
                     share65_wmean=len(tracts)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
