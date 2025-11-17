import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray

def infer_time_label(nc_path, ds, product="monthly"):
    """
    Return a pandas.Timestamp, try to infer from ds or filename.
    product: 'monthly' or 'daily'
    """
    # 1) Directly have time coordinate/variable
    if "time" in ds.coords or "time" in ds.variables:
        try:
            tt = pd.to_datetime(ds["time"].values)
            tt = np.array(tt).reshape(-1)[0]
            return pd.to_datetime(tt)
        except Exception:
            pass

    # 2) Global attributes (common in L3M data)
    start = ds.attrs.get("time_coverage_start") or ds.attrs.get("start_time")
    end   = ds.attrs.get("time_coverage_end")   or ds.attrs.get("end_time")
    if start and end:
        try:
            ts = pd.to_datetime(start)
            te = pd.to_datetime(end)
            if product == "monthly":
                return ts + (te - ts) / 2
            else:
                return ts
        except Exception:
            pass

    # 3) Analyze the filename
    fn = nc_path.split("/")[-1]
    if product == "monthly":
        # ...YYYYMMDD_YYYYMMDD.L3m.MO...
        m = re.search(r"\.(\d{8})_(\d{8})\.L3m\.MO\.", fn)
        if m:
            b, e = m.group(1), m.group(2)
            ts = pd.to_datetime(b, format="%Y%m%d")
            te = pd.to_datetime(e, format="%Y%m%d")
            return ts + (te - ts) / 2
    else:
        # ...YYYYMMDD.L3m.DAY...
        m = re.search(r"\.(\d{8})\.L3m\.DAY\.", fn)
        if m:
            return pd.to_datetime(m.group(1), format="%Y%m%d")

    raise ValueError("Cannot infer time from dataset or filename: " + fn)

def clean_ci(da: xr.DataArray) -> xr.DataArray:
    """
    Filter out values out of physical range and remove near-zero values.
    """
    vmin = float(da.attrs.get("valid_min", np.nan))
    vmax = float(da.attrs.get("valid_max", np.nan))
    if np.isfinite(vmin):
        da = da.where(da >= vmin)
    if np.isfinite(vmax):
        da = da.where(da <= vmax)

    thr = max(vmin, 5e-5) if np.isfinite(vmin) else 5e-5
    da = da.where(da > thr)

    return da

def extract_lakes_from_nc(nc_path: str,
                          lakes_gdf: gpd.GeoDataFrame,
                          lake_id_col: str,
                          product: str) -> pd.DataFrame:
    """
    nc_path: A single NetCDF file (S3B monthly or S3M daily)
    lakes_gdf: A GeoDataFrame containing `lake_id` and `geometry` (EPSG:4326)
    product: 'monthly' | 'daily'
    Returns: One row per lake (timestamp of the file)
    """
    with xr.open_dataset(nc_path, engine="netcdf4", chunks="auto") as ds:
        t  = infer_time_label(nc_path, ds, product=product)
        da = set_spatial_dims_safe(ds["CI_cyano"], ds=ds)
        da = clean_ci(da).rio.write_crs(4326)

        rows = []
        for _, row in lakes_gdf.iterrows():
            lid  = row[lake_id_col]
            geom = [row.geometry]
            try:
                clipped = da.rio.clip(geom, lakes_gdf.crs, drop=True)
                arr     = clipped.data  # dask/np 数组
                arr     = np.asarray(arr)  # 触发 dask 计算
                mask    = np.isfinite(arr)
                n_valid = int(mask.sum())

                if n_valid == 0:
                    mean_val = np.nan
                    p90      = np.nan
                else:
                    vals = arr[mask].ravel()
                    mean_val = float(np.nanmean(vals))
                    p90      = float(np.nanquantile(vals, 0.9))
            except Exception:
                mean_val, p90, n_valid = np.nan, np.nan, 0

            rows.append({
                "lake_id": lid, "time": pd.to_datetime(t), "product": product,
                "CI_mean": mean_val, "CI_p90": p90, "n_valid": n_valid,
                "src": Path(nc_path).name,
            })
    return pd.DataFrame(rows)

def set_spatial_dims_safe(da: xr.DataArray, ds: xr.Dataset | None = None) -> xr.DataArray:
    """
    Make `da` geospatially aware for rioxarray:
    - Prefer real dimension names present on `da` (lon/lat or x/y or variants);
    - Only pass EXISTING dimension names to `rio.set_spatial_dims`;
    - If dataset carries 1D lon/lat arrays matching x/y lengths, bind them as coords;
    - Finally write CRS=EPSG:4326.

    NOTE: If lon/lat are 2D (curvilinear), we DO NOT pass them as dims; we keep x/y dims.
    """
    dims = list(da.dims)

    # normalize common variants
    def _has(*cands):
        return any(c in dims for c in cands)

    # Case A: dims already lon/lat
    if "lon" in dims and "lat" in dims:
        out = da.rio.write_crs(4326)
        out = out.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    elif "longitude" in dims and "latitude" in dims:
        out = da.rename({"longitude": "lon", "latitude": "lat"})
        out = out.rio.write_crs(4326)
        out = out.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

    # Case B: dims are x/y (any case)
    elif (_has("x") and _has("y")) or (_has("X") and _has("Y")):
        # rename upper-case to lower-case to please rioxarray
        rename_map = {}
        if "X" in dims: rename_map["X"] = "x"
        if "Y" in dims: rename_map["Y"] = "y"
        out = da.rename(rename_map) if rename_map else da
        out = out.rio.write_crs(4326)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

        # try binding 1D lon/lat coords from ds if lengths match
        if ds is not None:
            try:
                lon1d = None
                lat1d = None
                for lon_name in ("lon","longitude"):
                    if lon_name in ds.variables and ds[lon_name].ndim == 1 and ds[lon_name].sizes[ds[lon_name].dims[0]] == out.sizes["x"]:
                        lon1d = np.asarray(ds[lon_name].values)
                        break
                for lat_name in ("lat","latitude"):
                    if lat_name in ds.variables and ds[lat_name].ndim == 1 and ds[lat_name].sizes[ds[lat_name].dims[0]] == out.sizes["y"]:
                        lat1d = np.asarray(ds[lat_name].values)
                        break
                if lon1d is not None and lat1d is not None:
                    out = out.assign_coords(x=lon1d, y=lat1d)
            except Exception:
                pass

    # Case C: unknown names → rename the last two dimensions to y/x
    elif len(dims) >= 2:
        ydim, xdim = dims[-2], dims[-1]
        out = da.rename({xdim: "x", ydim: "y"}).rio.write_crs(4326)
        out = out.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    else:
        raise ValueError(f"Cannot determine spatial dims for {da.name!r}; dims={dims}")

    return out

def looks_like_hdf5(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
        return sig.startswith(b"\x89HDF") or sig.startswith(b"CDF")
    except Exception:
        return False

def _extract_one_with_h5netcdf(nc_path: Path,
                               lakes_gdf: gpd.GeoDataFrame,
                               lake_id_col: str,
                               product: str = "daily") -> pd.DataFrame:
    """
    兜底方案：用 h5netcdf 打开并在本函数内完成裁剪与统计。
    只有当 extract_lakes_from_nc 失败时才会被调用。
    """
    rows = []
    # 注意：phony_dims="access" 让 h5netcdf 创建可访问维度名
    with xr.open_dataset(nc_path, engine="h5netcdf", chunks="auto", phony_dims="access") as ds:
        t  = infer_time_label(str(nc_path), ds, product=product)
        da = set_spatial_dims_safe(ds["CI_cyano"], ds=ds)
        da = clean_ci(da).rio.write_crs(4326)

        for _, r in lakes_gdf.iterrows():
            lid  = r[lake_id_col]
            geom = [r.geometry]
            try:
                clipped = da.rio.clip(geom, lakes_gdf.crs, drop=True)
                arr     = np.asarray(clipped.data)
                mask    = np.isfinite(arr)
                n_valid = int(mask.sum())
                if n_valid == 0:
                    mean_val, p90 = np.nan, np.nan
                else:
                    vals = arr[mask].ravel()
                    mean_val = float(np.nanmean(vals))
                    p90      = float(np.nanquantile(vals, 0.9))
            except Exception:
                mean_val, p90, n_valid = np.nan, np.nan, 0

            rows.append({
                "lake_id": lid,
                "date":    pd.to_datetime(t),
                "product": product,
                "CI_mean": mean_val,
                "CI_p90":  p90,
                "n_valid": n_valid,
                "src":     nc_path.name,
                "engine":  "h5netcdf",
            })
    return pd.DataFrame(rows)

def try_open_xarray(fp: Path):
    """Try netcdf4 → h5netcdf → h5py，return (ds or None, engine_used)。"""
    try:
        ds = xr.open_dataset(fp, engine="netcdf4", chunks="auto")
        _ = ds.dims
        return ds, "netcdf4"
    except Exception as e1:
        try:
            ds = xr.open_dataset(fp, engine="h5netcdf", chunks="auto", phony_dims="access")
            _ = ds.dims
            return ds, "h5netcdf"
        except Exception as e2:
            try:
                with h5py.File(fp, "r") as f:
                    pass
                print(f"[WARN] {fp.name} readable by h5py but not by xarray engines (netCDF-4 layout issue?)")
            except Exception as e3:
                print(f"[WARN] {fp.name} not readable even by h5py: {e3}")
            print(f"[SKIP] {fp.name} → netcdf4:{e1} | h5netcdf:{e2}")
            return None, None
        
import glob, numpy as np, pandas as pd, xarray as xr
from pathlib import Path

monthly_dir = Path("/dkucc/home/zy166/HAB-forecasting/datasets/ILW/S3B/2024/CONUS_MO")
out_csv = monthly_dir/"ci_cyano_monthly_mean.csv"

rows = []
for fp in sorted(monthly_dir.glob("S3B_OLCI_EFRNT.*.L3m.MO.ILW_CONUS.V5.all.CONUS.300m.nc")):
    with xr.open_dataset(fp, engine="netcdf4", chunks="auto") as ds:
        da = clean_ci(ds["CI_cyano"])
        arr = np.asarray(da.data)
        mask = np.isfinite(arr)
        m   = float(np.nanmean(arr[mask])) if mask.any() else np.nan
        p90 = float(np.nanquantile(arr[mask], 0.9))    if mask.any() else np.nan
        t   = infer_time_label(str(fp), ds, product="monthly")
        rows.append({"time": pd.to_datetime(t), "CI_mean": m, "CI_p90": p90,
                     "n_valid": int(mask.sum())})

df_mo = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
df_mo.to_csv(out_csv, index=False)
df_mo.head()

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import h5py
import geopandas as gpd

daily_dir = Path("/dkucc/home/zy166/HAB-forecasting/datasets/ILW/Merged/2024/CONUS_DAY")
out_csv = daily_dir / "ci_cyano_daily_mean.csv"

rows = []
for fp in sorted(daily_dir.glob("S3M_OLCI_EFRNT.*.L3m.DAY.ILW_CONUS.V5.all.CONUS.300m.nc")):
    if not looks_like_hdf5(fp):
        print(f"[WARN] Skip (not HDF5 header): {fp.name}")
        continue

    ds, eng = try_open_xarray(fp)
    if ds is None:
        continue

    try:
        da = clean_ci(ds["CI_cyano"])
        arr = np.asarray(da.data)
        mask = np.isfinite(arr)
        m   = float(np.nanmean(arr[mask])) if mask.any() else np.nan
        p90 = float(np.nanquantile(arr[mask], 0.9)) if mask.any() else np.nan

        t = infer_time_label(str(fp), ds, product="daily")

        rows.append({
            "date":   pd.to_datetime(t),
            "CI_mean": m,
            "CI_p90":  p90,
            "n_valid": int(mask.sum()),
            "src":     fp.name,
            "engine":  eng,
        })
    finally:
        ds.close()

df_day = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df_day.to_csv(out_csv, index=False)
print(f"[OK] saved → {out_csv} (rows={len(df_day)})")
df_day.head()