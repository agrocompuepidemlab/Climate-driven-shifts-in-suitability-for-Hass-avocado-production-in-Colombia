# ============================================================
# AUTOMATED DOWNLOAD OF CLIMATIC VARIABLES
# Sources:
# 1. NASA POWER
# 2. TerraClimate
# 3. IDEAM (integration of previously defined stations)
# ============================================================

# =========================
# 1. INSTALLATION & IMPORTS
# =========================
!pip -q install pandas requests xarray netcdf4 rioxarray openpyxl rasterio geopandas shapely pyproj

import os
import io
import json
import math
import time
import requests
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from shapely.geometry import Point

# ============================================================
# 2. GENERAL CONFIGURATION
# ============================================================
# Upload to Colab:
# - a CSV file with points: points_xy.csv
#   with columns: id, lon, lat
# - optionally: list_39_climate_stations_IDEAM_199.txt
# - optionally: climate_stations_ideam_nearby_536_points.txt

POINTS_FILE = "points_xy.csv"

OUTPUT_DIR = "/content/climate_outputs"
NASA_DIR = os.path.join(OUTPUT_DIR, "nasa_power")
TERRA_DIR = os.path.join(OUTPUT_DIR, "terraclimate")
IDEAM_DIR = os.path.join(OUTPUT_DIR, "ideam")

os.makedirs(NASA_DIR, exist_ok=True)
os.makedirs(TERRA_DIR, exist_ok=True)
os.makedirs(IDEAM_DIR, exist_ok=True)

if not os.path.exists(POINTS_FILE):
    raise FileNotFoundError(
        f"File {POINTS_FILE} not found. Upload it to Colab with columns: id, lon, lat"
    )

points_df = pd.read_csv(POINTS_FILE)

required_cols = {"id", "lon", "lat"}
if not required_cols.issubset(points_df.columns):
    raise ValueError("The file points_xy.csv must have columns: id, lon, lat")

print("Points loaded:", points_df.shape[0])
display(points_df.head())


# ============================================================
# 3. VARIABLES ACCORDING TO THE DOCUMENT WORKFLOW
# ============================================================
# The files include variables such as:
# elevation, radiation, temperatures, vapor, wind,
# precipitation, PET, aridity, thermicity, etc.
# Also mentions the use of IDEAM, NASA POWER and TerraClimate.
# Here we download what can be obtained directly from NASA POWER
# and TerraClimate, and leave the IDEAM integration below.
# Documentary reference of variable set and sources:
# ELEV1, mRAD1, mTMAX1, mTMED1, mTMIN1, mVAPR1, mWIND1, sPCP1,
# annualPET1, aridityInd, ... + IDEAM, NASA POWER, TerraClimate.

# ============================================================
# 4. NASA POWER DOWNLOAD
# ============================================================
# Script based on your previous workflow, where queries were made per point
# using coordinates and parameters such as T2M from the POWER API.

NASA_PARAMETERS = [
    "T2M",       # mean temperature at 2m
    "T2M_MAX",   # mean maximum temperature
    "T2M_MIN",   # mean minimum temperature
    "PRECTOT",   # precipitation
    "WS2M",      # wind speed at 2m
    "QV2M",      # specific humidity approx. useful for humidity/atmosphere
    "ALLSKY_SFC_SW_DWN"  # incident solar radiation
]

NASA_COMMUNITY = "AG"
NASA_START = "198101"
NASA_END = "202012"
NASA_FORMAT = "JSON"

def download_nasa_power_point(lat, lon, point_id, parameters=NASA_PARAMETERS,
                              start=NASA_START, end=NASA_END,
                              community=NASA_COMMUNITY, fmt=NASA_FORMAT,
                              out_dir=NASA_DIR):
    """
    Downloads monthly NASA POWER variables per point.
    """
    params_str = ",".join(parameters)

    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters={params_str}"
        f"&community={community}"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start}"
        f"&end={end}"
        f"&format={fmt}"
    )

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()

    param_data = data["properties"]["parameter"]

    # Build monthly table
    all_dates = None
    series_dict = {}

    for p in parameters:
        p_dict = param_data.get(p, {})
        if all_dates is None:
            all_dates = list(p_dict.keys())
        series_dict[p] = p_dict

    out_rows = []
    for ym in all_dates:
        row = {
            "id": point_id,
            "lat": lat,
            "lon": lon,
            "year": int(ym[:4]),
            "month": int(ym[4:6]),
            "date": pd.to_datetime(ym, format="%Y%m")
        }
        for p in parameters:
            row[p] = series_dict[p].get(ym, np.nan)
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows).sort_values("date")

    out_file = os.path.join(out_dir, f"{point_id}_nasa_power.csv")
    out_df.to_csv(out_file, index=False)

    return out_df


# Run for all points
nasa_all = []
for _, row in points_df.iterrows():
    try:
        dfp = download_nasa_power_point(
            lat=row["lat"],
            lon=row["lon"],
            point_id=row["id"]
        )
        nasa_all.append(dfp)
        print(f"NASA POWER OK -> {row['id']}")
        time.sleep(0.5)
    except Exception as e:
        print(f"NASA POWER error at {row['id']}: {e}")

if nasa_all:
    nasa_all_df = pd.concat(nasa_all, ignore_index=True)
    nasa_all_df.to_excel(os.path.join(NASA_DIR, "nasa_power_all_points.xlsx"), index=False)
    print("Consolidated NASA POWER file generated.")


# ============================================================
# 5. VARIABLE DERIVATION FROM NASA POWER
# ============================================================
def summarize_nasa_to_clustering_vars(df):
    """
    Summarizes the monthly series per point to produce variables
    close to those used in clustering.
    """
    g = df.copy()

    # Rename columns to conventions closer to the manuscript
    summary = g.groupby("id").agg(
        lon=("lon", "first"),
        lat=("lat", "first"),
        mRAD1=("ALLSKY_SFC_SW_DWN", "mean"),
        mTMAX1=("T2M_MAX", "mean"),
        mTMED1=("T2M", "mean"),
        mTMIN1=("T2M_MIN", "mean"),
        mWIND1=("WS2M", "mean"),
        sPCP1=("PRECTOT", "sum")
    ).reset_index()

    return summary

if nasa_all:
    nasa_summary = summarize_nasa_to_clustering_vars(nasa_all_df)
    nasa_summary.to_excel(os.path.join(NASA_DIR, "nasa_power_variables_summary.xlsx"), index=False)
    display(nasa_summary.head())


# ============================================================
# 6. TERRACLIMATE DOWNLOAD
# ============================================================
# TerraClimate is typically handled as NetCDF. The workflow below
# reads a monthly file already downloaded or accessible via URL.
# Practical recommendation in Colab:
# 1. Download the .nc files per variable
# 2. Upload them to /content/terraclimate_nc/
# Useful variables:
# ppt  = precipitation
# tmax = maximum temperature
# tmin = minimum temperature
# srad = solar radiation
# vap  = vapor pressure
# ws   = wind
# pet  = potential evapotranspiration

TERRA_NC_DIR = "/content/terraclimate_nc"
os.makedirs(TERRA_NC_DIR, exist_ok=True)

TERRA_VAR_FILES = {
    "ppt":  os.path.join(TERRA_NC_DIR, "TerraClimate_ppt.nc"),
    "tmax": os.path.join(TERRA_NC_DIR, "TerraClimate_tmax.nc"),
    "tmin": os.path.join(TERRA_NC_DIR, "TerraClimate_tmin.nc"),
    "srad": os.path.join(TERRA_NC_DIR, "TerraClimate_srad.nc"),
    "vap":  os.path.join(TERRA_NC_DIR, "TerraClimate_vap.nc"),
    "ws":   os.path.join(TERRA_NC_DIR, "TerraClimate_ws.nc"),
    "pet":  os.path.join(TERRA_NC_DIR, "TerraClimate_pet.nc"),
}

def extract_terraclimate_point_series(nc_file, var_name, points_df, start_date="1981-01-01", end_date="2020-12-31"):
    """
    Extracts time series per point from a TerraClimate NetCDF file.
    """
    ds = xr.open_dataset(nc_file)

    # Temporal slice
    ds = ds.sel(time=slice(start_date, end_date))

    rows = []
    for _, row in points_df.iterrows():
        point_id = row["id"]
        lon = row["lon"]
        lat = row["lat"]

        # TerraClimate usually has lat/lon dims
        series = ds[var_name].sel(lon=lon, lat=lat, method="nearest")

        out_df = series.to_dataframe().reset_index()
        out_df["id"] = point_id
        out_df["lon"] = lon
        out_df["lat"] = lat
        out_df["variable"] = var_name
        rows.append(out_df)

    result = pd.concat(rows, ignore_index=True)
    return result

terra_all = []

for var_name, nc_path in TERRA_VAR_FILES.items():
    if os.path.exists(nc_path):
        try:
            tmp = extract_terraclimate_point_series(nc_path, var_name, points_df)
            terra_all.append(tmp)
            print(f"TerraClimate OK -> {var_name}")
        except Exception as e:
            print(f"TerraClimate error for {var_name}: {e}")
    else:
        print(f"Not found: {nc_path}")

if terra_all:
    terra_df = pd.concat(terra_all, ignore_index=True)
    terra_df.to_excel(os.path.join(TERRA_DIR, "terraclimate_series_points.xlsx"), index=False)
    display(terra_df.head())


# ============================================================
# 7. TERRACLIMATE SUMMARY FOR CLUSTERING VARIABLES
# ============================================================
def summarize_terraclimate(df):
    """
    Summarizes TerraClimate variables per point.
    """
    piv = df.pivot_table(
        index=["id", "lon", "lat", "time"],
        columns="variable",
        values=df.columns[df.columns.isin(["ppt", "tmax", "tmin", "srad", "vap", "ws", "pet"])].tolist()[0]
        if len(df.columns[df.columns.isin(["ppt", "tmax", "tmin", "srad", "vap", "ws", "pet"])]) > 0
        else None,
        aggfunc="mean"
    )
    return piv

# More robust approach to consolidate TerraClimate
if terra_all:
    terra_frames = []
    for var_name, nc_path in TERRA_VAR_FILES.items():
        if os.path.exists(nc_path):
            ds = xr.open_dataset(nc_path).sel(time=slice("1981-01-01", "2020-12-31"))
            rows = []
            for _, row in points_df.iterrows():
                point_id = row["id"]
                lon = row["lon"]
                lat = row["lat"]
                series = ds[var_name].sel(lon=lon, lat=lat, method="nearest").to_series()
                tmp = pd.DataFrame({
                    "time": series.index,
                    "value": series.values
                })
                tmp["id"] = point_id
                tmp["lon"] = lon
                tmp["lat"] = lat
                tmp["variable"] = var_name
                rows.append(tmp)
            terra_frames.append(pd.concat(rows, ignore_index=True))

    terra_long = pd.concat(terra_frames, ignore_index=True)
    terra_wide = terra_long.pivot_table(
        index=["id", "lon", "lat", "time"],
        columns="variable",
        values="value",
        aggfunc="mean"
    ).reset_index()

    # Compute mTMED from tmax/tmin
    if {"tmax", "tmin"}.issubset(terra_wide.columns):
        terra_wide["tmed"] = (terra_wide["tmax"] + terra_wide["tmin"]) / 2

    terra_summary = terra_wide.groupby("id").agg(
        lon=("lon", "first"),
        lat=("lat", "first"),
        mRAD1=("srad", "mean") if "srad" in terra_wide.columns else ("lon", "first"),
        mTMAX1=("tmax", "mean") if "tmax" in terra_wide.columns else ("lon", "first"),
        mTMED1=("tmed", "mean") if "tmed" in terra_wide.columns else ("lon", "first"),
        mTMIN1=("tmin", "mean") if "tmin" in terra_wide.columns else ("lon", "first"),
        mVAPR1=("vap", "mean") if "vap" in terra_wide.columns else ("lon", "first"),
        mWIND1=("ws", "mean") if "ws" in terra_wide.columns else ("lon", "first"),
        sPCP1=("ppt", "sum") if "ppt" in terra_wide.columns else ("lon", "first"),
        annualPET1=("pet", "sum") if "pet" in terra_wide.columns else ("lon", "first"),
    ).reset_index()

    terra_summary.to_excel(os.path.join(TERRA_DIR, "terraclimate_variables_summary.xlsx"), index=False)
    display(terra_summary.head())


# ============================================================
# 8. READING IDEAM STATIONS
# ============================================================
# Your files include station lists with cod_estacion, x, y
# and notes indicating:
# filtering by the station layer, obtaining the mean value of each
# IDEAM station and unification of precipitation + mean temperature.

IDEAM_STATIONS_FILE_1 = "list_39_climate_stations_IDEAM_199.txt"
IDEAM_STATIONS_FILE_2 = "climate_stations_ideam_nearby_536_points.txt"

def read_ideam_station_file(txt_file):
    if not os.path.exists(txt_file):
        return None
    df = pd.read_csv(txt_file, sep="\t")
    return df

ideam_1 = read_ideam_station_file(IDEAM_STATIONS_FILE_1)
ideam_2 = read_ideam_station_file(IDEAM_STATIONS_FILE_2)

if ideam_1 is not None:
    print("IDEAM station list 1")
    display(ideam_1.head())

if ideam_2 is not None:
    print("IDEAM station list 2")
    display(ideam_2.head())

# Save consolidated station lists
if ideam_1 is not None or ideam_2 is not None:
    ideam_stations = pd.concat([d for d in [ideam_1, ideam_2] if d is not None], ignore_index=True).drop_duplicates()
    ideam_stations.to_excel(os.path.join(IDEAM_DIR, "ideam_stations_consolidated.xlsx"), index=False)

    # GeoDataFrame for spatial proximity
    ideam_gdf = gpd.GeoDataFrame(
        ideam_stations,
        geometry=gpd.points_from_xy(ideam_stations["x"], ideam_stations["y"]),
        crs="EPSG:4326"
    )

    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df["lon"], points_df["lat"]),
        crs="EPSG:4326"
    )

    # Metric reprojection
    ideam_m = ideam_gdf.to_crs("EPSG:3116")
    points_m = points_gdf.to_crs("EPSG:3116")

    nearest_rows = []
    for _, prow in points_m.iterrows():
        dists = ideam_m.distance(prow.geometry)
        idx_min = dists.idxmin()
        est = ideam_m.loc[idx_min]

        nearest_rows.append({
            "id": prow["id"],
            "lon": points_df.loc[points_df["id"] == prow["id"], "lon"].values[0],
            "lat": points_df.loc[points_df["id"] == prow["id"], "lat"].values[0],
            "cod_estacion": est["cod_estacion"],
            "dist_m": dists.min()
        })

    nearest_ideam_df = pd.DataFrame(nearest_rows)
    nearest_ideam_df.to_excel(os.path.join(IDEAM_DIR, "points_nearest_ideam_station.xlsx"), index=False)
    display(nearest_ideam_df.head())


# ============================================================
# 9. IDEAM SERIES INTEGRATION
# ============================================================
# Here we assume you have already exported IDEAM series per station,
# for example one CSV per station with:
# cod_estacion, date, precip, tmed
# This section unifies precipitation + mean temperature,
# as it appears in your workflow.

IDEAM_SERIES_DIR = "/content/ideam_series"
os.makedirs(IDEAM_SERIES_DIR, exist_ok=True)

def load_ideam_series_folder(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    if len(files) == 0:
        print("No IDEAM series CSV files found in", folder)
        return None

    frames = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as e:
            print("Error reading", f, e)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return None

ideam_series = load_ideam_series_folder(IDEAM_SERIES_DIR)

if ideam_series is not None:
    print("Consolidated IDEAM series:")
    display(ideam_series.head())

    # Minimal standardization
    ideam_series.columns = [c.lower() for c in ideam_series.columns]

    # Attempt to normalize column names
    rename_map = {}
    if "codigo" in ideam_series.columns:
        rename_map["codigo"] = "cod_estacion"
    if "temperatura_media" in ideam_series.columns:
        rename_map["temperatura_media"] = "tmed"
    if "precipitacion" in ideam_series.columns:
        rename_map["precipitacion"] = "precip"

    ideam_series = ideam_series.rename(columns=rename_map)

    if "fecha" in ideam_series.columns:
        ideam_series["fecha"] = pd.to_datetime(ideam_series["fecha"])

    ideam_series.to_excel(os.path.join(IDEAM_DIR, "ideam_series_consolidated.xlsx"), index=False)

    # Join with nearest station per point
    if "nearest_ideam_df" in locals():
        ideam_join = nearest_ideam_df.merge(ideam_series, on="cod_estacion", how="left")
        ideam_join.to_excel(os.path.join(IDEAM_DIR, "ideam_series_associated_to_points.xlsx"), index=False)
        display(ideam_join.head())


# ============================================================
# 10. FINAL VARIABLE CONSOLIDATION
# ============================================================
# You can combine NASA, TerraClimate and IDEAM to build
# a master table per point.

master = points_df.copy()

if "nasa_summary" in locals():
    nasa_summary = nasa_summary.add_prefix("NASA_")
    nasa_summary = nasa_summary.rename(columns={"NASA_id": "id"})
    master = master.merge(nasa_summary, on="id", how="left")

if "terra_summary" in locals():
    terra_summary = terra_summary.add_prefix("TERRA_")
    terra_summary = terra_summary.rename(columns={"TERRA_id": "id"})
    master = master.merge(terra_summary, on="id", how="left")

master.to_excel(os.path.join(OUTPUT_DIR, "master_table_climatic_variables.xlsx"), index=False)

print("Process completed.")
print("Master file:", os.path.join(OUTPUT_DIR, "master_table_climatic_variables.xlsx"))
display(master.head())
