from pathlib import Path
import geopandas as gpd
import pandas as pd
import glob
from shapely.geometry import box

# 0) Find shapefile (avoid path errors)
candidates = glob.glob("/dkucc/home/zy166/HAB-forcasting/datasets/Lakes/HydroLAKES_polys_v10_shp/**/HydroLAKES_polys_v10.shp", recursive=True)
assert candidates, "找不到 HydroLAKES_polys_v10.shp，请确认解压路径"
src = candidates[0]
print("Using:", src)

# 1) Read files
gdf = gpd.read_file(src)          # CRS=EPSG:4326
print("Total lakes:", len(gdf))
print("Columns:", list(gdf.columns))

# 2) Based on Country, first filter (contains United States)
country = gdf["Country"].fillna("")
is_us = country.str.contains("United States", case=False, regex=False)
gdf_us = gdf[is_us].copy()
print("US lakes (by Country contains):", len(gdf_us))

# 3) Also do a "spatial range"兜底（五大湖经纬度大致包络，更稳健）
#    Longitude: -95 ~ -74, Latitude: 40 ~ 49 (adjustable)
bbox = box(-95, 40, -74, 49)
gdf_bbox = gdf[gdf.intersects(bbox)].copy()
print("Lakes in Great-Lakes bbox:", len(gdf_bbox))

# 4) Combine the two filters (take union), then use names to exactly match the Great Lakes.
candidates_gl = pd.concat([gdf_us, gdf_bbox], ignore_index=True).drop_duplicates(subset="Hylak_id")
names = ["Superior", "Michigan", "Huron", "Erie", "Ontario"]
name_has_gl = candidates_gl["Lake_name"].fillna("").str.contains("|".join(names), case=False, regex=True)
gdf_5 = candidates_gl[name_has_gl].copy()

# 5) Some entries may split the Great Lakes into multiple polygons (islands, blocks).
# gdf_5_diss = gdf_5.dissolve(by="Lake_name", as_index=False)

print("Matched Great Lakes features:", len(gdf_5))
print(gdf_5[["Hylak_id","Lake_name","Country"]].head(10))

# 6) Write to GPKG (ensure the directory exists first)
out_fp = Path("/dkucc/home/zy166/HAB-forcasting/datasets/Lakes/shapes/lakes_greatlakes.gpkg")
out_fp.parent.mkdir(parents=True, exist_ok=True)

gdf_5 = gdf_5.rename(columns={"Hylak_id":"lake_id"})[["lake_id","Lake_name","geometry"]]
gdf_5.to_file(out_fp, driver="GPKG")
print("Saved:", out_fp, "features:", len(gdf_5))