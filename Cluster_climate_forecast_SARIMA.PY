# ============================================================
# CLIMATE FORECAST BY CLUSTER WITH SARIMA
# Adapted from the SARIMA notebook workflow for Hass avocado
# Monthly series 1981-2019 -> 12-month forecast for 2020
# Variables: TMED and PCP
# Clusters: C1, C2, C3, C4, C5
# ============================================================

# =========================
# 1. INSTALLATION & IMPORTS
# =========================
!pip -q install pmdarima openpyxl

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12


# ==========================================
# 2. FILE LOADING
# ==========================================
# To use Google Drive, uncomment:
# from google.colab import drive
# drive.mount('/content/drive')

file_map = {
    "C1_TMED": "k5_tmed_c1.csv",
    "C2_TMED": "k5_tmed_c2.csv",
    "C3_TMED": "k5_tmed_c3.csv",
    "C4_TMED": "k5_tmed_c4.csv",
    "C5_TMED": "k5_tmed_c5.csv",
    "C1_PCP":  "k5_pcp_c1.csv",
    "C2_PCP":  "k5_pcp_c2.csv",
    "C3_PCP":  "k5_pcp_c3.csv",
    "C4_PCP":  "k5_pcp_c4.csv",
    "C5_PCP":  "k5_pcp_c5.csv",
}

missing = [v for v in file_map.values() if not os.path.exists(v)]
if missing:
    raise FileNotFoundError(
        "The following files are missing in /content:\n" + "\n".join(missing) +
        "\n\nUpload them to the Colab left panel or copy the CSV files to /content."
    )

raw_data = {k: pd.read_csv(v, sep=",") for k, v in file_map.items()}


# ==================================================
# 3. SERIES PREPROCESSING
# ==================================================
def build_monthly_series(df, variable_type="TMED"):
    df = df.copy()

    # Replace negative values with NaN
    df[df < 0] = np.nan

    if variable_type == "TMED":
        sub = df.iloc[:, 3:-6]
    elif variable_type == "PCP":
        sub = df.iloc[:, 3:-1]
    else:
        raise ValueError("variable_type must be 'TMED' or 'PCP'")

    # Average by date
    s = sub.describe().iloc[1:2, :].T.reset_index()
    s.columns = ["date", "mean"]
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)
    s = s.set_index("date")
    s = s.asfreq("MS")
    s["mean"] = s["mean"].interpolate(method="linear").ffill().bfill()

    # Rename column to keep compatibility with downstream code
    s = s.rename(columns={"mean": "prom"})

    return s

series_dict = {}

for key, df in raw_data.items():
    variable_type = "TMED" if "TMED" in key else "PCP"
    series_dict[key] = build_monthly_series(df, variable_type=variable_type)

for k in series_dict:
    print(k, series_dict[k].shape, series_dict[k].index.min(), series_dict[k].index.max())


# ==================================
# 4. HISTORICAL PLOTS BY CLUSTER
# ==================================
cluster_colors = {
    "C1": "red",
    "C2": "green",
    "C3": "orange",
    "C4": "brown",
    "C5": "gray"
}

def plot_historical_all_clusters(series_dict, variable="TMED"):
    plt.figure(figsize=(16, 8))
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        s = series_dict[f"{cluster}_{variable}"]["prom"]
        plt.plot(s.values, label=cluster, color=cluster_colors[cluster])

    plt.xlabel("Number of months in the time series 1981 - 2019", fontsize=16)
    if variable == "TMED":
        plt.ylabel("Average temperature [°C]", fontsize=16)
        plt.title("Historical mean temperature by cluster", fontsize=18)
    else:
        plt.ylabel("Precipitation [mm]", fontsize=16)
        plt.title("Historical precipitation by cluster", fontsize=18)

    plt.legend(fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()

plot_historical_all_clusters(series_dict, variable="TMED")
plot_historical_all_clusters(series_dict, variable="PCP")


# ==========================================
# 5. SARIMA MODELLING FUNCTIONS
# ==========================================
def fit_sarima_and_forecast(series_df, seasonal_period=12, test_size=12):
    """
    Fits SARIMA on the historical series and generates:
    - prediction over the test period
    - 12-month forecast for 2020
    """
    series = series_df["prom"].copy()

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    # Automatic search for the best SARIMA model
    auto_model = pm.auto_arima(
        train,
        seasonal=True,
        m=seasonal_period,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None, D=None,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aic"
    )

    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)

    # Prediction on the test period
    pred_test = fitted_model.forecast(steps=test_size)

    # Metrics
    mae = mean_absolute_error(test, pred_test)
    rmse = math.sqrt(mean_squared_error(test, pred_test))
    aic = fitted_model.aic
    bic = fitted_model.bic

    # Refit on the full series for the final 2020 forecast
    final_model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    future_2020 = final_model.forecast(steps=12)

    return {
        "series": series,
        "train": train,
        "test": test,
        "pred_test": pred_test,
        "future_2020": future_2020,
        "order": order,
        "seasonal_order": seasonal_order,
        "mae": mae,
        "rmse": rmse,
        "aic": aic,
        "bic": bic,
        "model": fitted_model,
        "final_model": final_model
    }


# ==========================================
# 6. TRAINING FOR ALL CLUSTERS
# ==========================================
results = {}

for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        key = f"{cluster}_{variable}"
        print(f"\nFitting SARIMA for {key} ...")
        results[key] = fit_sarima_and_forecast(
            series_dict[key],
            seasonal_period=12,
            test_size=12
        )


# ==========================
# 7. METRICS TABLE
# ==========================
metrics_rows = []

for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        r = results[f"{cluster}_{variable}"]
        metrics_rows.append({
            "Cluster": cluster,
            "Variable": variable,
            "Model": "SARIMA",
            "Order": str(r["order"]),
            "Seasonal_Order": str(r["seasonal_order"]),
            "MAE": r["mae"],
            "RMSE": r["rmse"],
            "AIC": r["aic"],
            "BIC": r["bic"]
        })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.sort_values(["Variable", "Cluster"]).reset_index(drop=True)

print("\n===== SARIMA METRICS =====")
display(metrics_df)

metrics_df.to_excel("metrics_SARIMA_by_cluster.xlsx", index=False)


# ==========================================
# 8. PLOTTING FUNCTIONS
# ==========================================
def plot_train_test_prediction(cluster, variable):
    r = results[f"{cluster}_{variable}"]

    plt.figure(figsize=(15, 6))
    plt.plot(r["train"].index, r["train"].values, label="Train", color="#3c3c3c")
    plt.plot(r["test"].index, r["test"].values, label="Test (observed)", color="green")
    plt.plot(r["pred_test"].index, r["pred_test"].values, label="SARIMA prediction", color="blue")

    if variable == "TMED":
        plt.ylabel("Average temperature [°C]")
        plt.title(f"Average temperature {cluster} - SARIMA")
    else:
        plt.ylabel("Precipitation [mm]")
        plt.title(f"Precipitation {cluster} - SARIMA")

    plt.xlabel("Year")
    plt.legend()
    plt.show()

def plot_2020_forecast(cluster, variable):
    hist = series_dict[f"{cluster}_{variable}"]["prom"].copy()
    pred_2020 = results[f"{cluster}_{variable}"]["future_2020"]

    future_dates = pd.date_range("2020-01-01", periods=12, freq="MS")

    plt.figure(figsize=(16, 6))
    plt.plot(hist.index, hist.values, color="#2f2f2f", label="Historical data 1981-2019")
    plt.plot(future_dates, pred_2020.values, color="blue", marker="o", label="SARIMA 2020")
    plt.axvline(pd.Timestamp("2020-01-01"), color="black", linestyle="--", alpha=0.6)

    if variable == "TMED":
        plt.ylabel("Average temperature [°C]")
        plt.title(f"Average temperature {cluster} - 2020 forecast (SARIMA)")
    else:
        plt.ylabel("Precipitation [mm]")
        plt.title(f"Precipitation {cluster} - 2020 forecast (SARIMA)")

    plt.xlabel("Year")
    plt.legend()
    plt.show()

def plot_2020_all_clusters(variable="TMED"):
    future_dates = pd.date_range("2020-01-01", periods=12, freq="MS")

    plt.figure(figsize=(16, 7))
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        pred = results[f"{cluster}_{variable}"]["future_2020"]
        plt.plot(
            future_dates,
            pred.values,
            marker="o",
            label=cluster,
            color=cluster_colors[cluster]
        )

    plt.xticks(future_dates, [d.strftime("%b") for d in future_dates], rotation=45)

    if variable == "TMED":
        plt.ylabel("Average temperature [°C]")
        plt.title("2020 forecast by cluster - SARIMA - TMED", fontsize=18)
    else:
        plt.ylabel("Precipitation [mm]")
        plt.title("2020 forecast by cluster - SARIMA - PCP", fontsize=18)

    plt.xlabel("Month of 2020")
    plt.legend(fontsize=12)
    plt.show()


# ==========================================
# 9. INDIVIDUAL PLOTS BY CLUSTER
# ==========================================
for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        plot_train_test_prediction(cluster, variable)
        plot_2020_forecast(cluster, variable)


# ==========================================
# 10. COMPARATIVE FIGURES ACROSS CLUSTERS
# ==========================================
plot_2020_all_clusters(variable="TMED")
plot_2020_all_clusters(variable="PCP")


# ==========================================
# 11. 2020 PREDICTION TABLES
# ==========================================
month_names = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

forecast_tables = {}

for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        pred = results[f"{cluster}_{variable}"]["future_2020"].values
        df_pred = pd.DataFrame([pred], index=[2020], columns=month_names)
        forecast_tables[f"{cluster}_{variable}_SARIMA"] = df_pred

        out_name = f"predictions_{variable}_SARIMA_{cluster}_2020.xlsx"
        df_pred.to_excel(out_name, sheet_name="2020 Predictions")

print("\n===== PREDICTION TABLE EXAMPLE =====")
display(forecast_tables["C1_TMED_SARIMA"])


# ==========================================
# 12. GENERAL SUMMARY IN EXCEL
# ==========================================
with pd.ExcelWriter("summary_predictions_2020_SARIMA.xlsx", engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    for k, dfp in forecast_tables.items():
        sheet = k[:31]
        dfp.to_excel(writer, sheet_name=sheet)

print("\nExported files:")
print("- metrics_SARIMA_by_cluster.xlsx")
print("- summary_predictions_2020_SARIMA.xlsx")
print("- and one individual Excel file per cluster/variable")
