# ============================================================
# PRONÓSTICO CLIMÁTICO POR CLUSTER CON SARIMA
# Adaptado al flujo del cuaderno SARIMA de aguacate Hass
# Series mensuales 1981-2019 -> pronóstico 12 meses para 2020
# Variables: TMED y PCP
# Clusters: C1, C2, C3, C4, C5
# ============================================================

# =========================
# 1. INSTALACIÓN E IMPORTS
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
# 2. CARGA DE ARCHIVOS
# ==========================================
# Si quieres usar Google Drive, descomenta:
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
        "Faltan estos archivos en /content:\n" + "\n".join(missing) +
        "\n\nSúbelos al panel izquierdo de Colab o copia los CSV a /content."
    )

raw_data = {k: pd.read_csv(v, sep=",") for k, v in file_map.items()}


# ==================================================
# 3. PREPROCESAMIENTO DE SERIES
# ==================================================
def build_monthly_series(df, variable_type="TMED"):
    df = df.copy()

    # Reemplazar negativos por NaN
    df[df < 0] = np.nan

    if variable_type == "TMED":
        sub = df.iloc[:, 3:-6]
    elif variable_type == "PCP":
        sub = df.iloc[:, 3:-1]
    else:
        raise ValueError("variable_type debe ser 'TMED' o 'PCP'")

    # Promedio por fecha
    s = sub.describe().iloc[1:2, :].T.reset_index()
    s.columns = ["fecha", "prom"]
    s["fecha"] = pd.to_datetime(s["fecha"])
    s = s.sort_values("fecha").reset_index(drop=True)
    s = s.set_index("fecha")
    s = s.asfreq("MS")
    s["prom"] = s["prom"].interpolate(method="linear").ffill().bfill()

    return s

series_dict = {}

for key, df in raw_data.items():
    variable_type = "TMED" if "TMED" in key else "PCP"
    series_dict[key] = build_monthly_series(df, variable_type=variable_type)

for k in series_dict:
    print(k, series_dict[k].shape, series_dict[k].index.min(), series_dict[k].index.max())


# ==================================
# 4. GRAFICAS HISTÓRICAS POR CLUSTER
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
# 5. FUNCIONES DE MODELAMIENTO SARIMA
# ==========================================
def fit_sarima_and_forecast(series_df, seasonal_period=12, test_size=12):
    """
    Ajusta SARIMA sobre la serie histórica y genera:
    - predicción sobre el período de prueba
    - pronóstico 12 meses para 2020
    """
    series = series_df["prom"].copy()

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    # Búsqueda automática del mejor modelo SARIMA
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

    # Predicción del período test
    pred_test = fitted_model.forecast(steps=test_size)

    # Métricas
    mae = mean_absolute_error(test, pred_test)
    rmse = math.sqrt(mean_squared_error(test, pred_test))
    aic = fitted_model.aic
    bic = fitted_model.bic

    # Reajustar sobre toda la serie para pronóstico final 2020
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
# 6. ENTRENAMIENTO PARA TODOS LOS CLUSTERS
# ==========================================
results = {}

for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        key = f"{cluster}_{variable}"
        print(f"\nAjustando SARIMA para {key} ...")
        results[key] = fit_sarima_and_forecast(
            series_dict[key],
            seasonal_period=12,
            test_size=12
        )


# ==========================
# 7. TABLA DE MÉTRICAS
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

print("\n===== MÉTRICAS SARIMA =====")
display(metrics_df)

metrics_df.to_excel("metricas_SARIMA_por_cluster.xlsx", index=False)


# ==========================================
# 8. FUNCIONES DE GRAFICACIÓN
# ==========================================
def plot_train_test_prediction(cluster, variable):
    r = results[f"{cluster}_{variable}"]

    plt.figure(figsize=(15, 6))
    plt.plot(r["train"].index, r["train"].values, label="Train", color="#3c3c3c")
    plt.plot(r["test"].index, r["test"].values, label="Test real", color="green")
    plt.plot(r["pred_test"].index, r["pred_test"].values, label="Predicción SARIMA", color="blue")

    if variable == "TMED":
        plt.ylabel("Temperatura promedio [°C]")
        plt.title(f"Temperatura promedio {cluster} - SARIMA")
    else:
        plt.ylabel("Precipitación [mm]")
        plt.title(f"Precipitación {cluster} - SARIMA")

    plt.xlabel("Año")
    plt.legend()
    plt.show()

def plot_2020_forecast(cluster, variable):
    hist = series_dict[f"{cluster}_{variable}"]["prom"].copy()
    pred_2020 = results[f"{cluster}_{variable}"]["future_2020"]

    future_dates = pd.date_range("2020-01-01", periods=12, freq="MS")

    plt.figure(figsize=(16, 6))
    plt.plot(hist.index, hist.values, color="#2f2f2f", label="Datos históricos 1981-2019")
    plt.plot(future_dates, pred_2020.values, color="blue", marker="o", label="SARIMA 2020")
    plt.axvline(pd.Timestamp("2020-01-01"), color="black", linestyle="--", alpha=0.6)

    if variable == "TMED":
        plt.ylabel("Temperatura promedio [°C]")
        plt.title(f"Temperatura promedio {cluster} - Pronóstico 2020 (SARIMA)")
    else:
        plt.ylabel("Precipitación [mm]")
        plt.title(f"Precipitación {cluster} - Pronóstico 2020 (SARIMA)")

    plt.xlabel("Año")
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
# 9. GRAFICAS INDIVIDUALES POR CLUSTER
# ==========================================
for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        plot_train_test_prediction(cluster, variable)
        plot_2020_forecast(cluster, variable)


# ==========================================
# 10. FIGURAS COMPARATIVAS ENTRE CLUSTERS
# ==========================================
plot_2020_all_clusters(variable="TMED")
plot_2020_all_clusters(variable="PCP")


# ==========================================
# 11. TABLAS DE PREDICCIONES 2020
# ==========================================
month_names = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
]

forecast_tables = {}

for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        pred = results[f"{cluster}_{variable}"]["future_2020"].values
        df_pred = pd.DataFrame([pred], index=[2020], columns=month_names)
        forecast_tables[f"{cluster}_{variable}_SARIMA"] = df_pred

        out_name = f"predicciones_{variable}_SARIMA_{cluster}_2020.xlsx"
        df_pred.to_excel(out_name, sheet_name="Predicciones 2020")

print("\n===== EJEMPLO DE TABLA DE PREDICCIÓN =====")
display(forecast_tables["C1_TMED_SARIMA"])


# ==========================================
# 12. RESUMEN GENERAL EN EXCEL
# ==========================================
with pd.ExcelWriter("resumen_predicciones_2020_SARIMA.xlsx", engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="Metricas", index=False)

    for k, dfp in forecast_tables.items():
        sheet = k[:31]
        dfp.to_excel(writer, sheet_name=sheet)

print("\nArchivos exportados:")
print("- metricas_SARIMA_por_cluster.xlsx")
print("- resumen_predicciones_2020_SARIMA.xlsx")
print("- y un Excel individual por cluster/variable")