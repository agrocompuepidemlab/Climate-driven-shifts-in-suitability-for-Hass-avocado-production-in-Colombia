# ============================================================
# PRONÓSTICO CLIMÁTICO POR CLUSTER CON LSTM Y BI-LSTM
# Adaptado al flujo del cuaderno SARIMA de aguacate Hass
# Series mensuales 1981-2019 -> pronóstico 12 meses para 2020
# Variables: TMED y PCP
# Clusters: C1, C2, C3, C4, C5
# ============================================================

# =========================
# 1. INSTALACIÓN E IMPORTS
# =========================
!pip -q install openpyxl

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12


# ==========================================
# 2. CARGA DE ARCHIVOS (MISMA LÓGICA TUYA)
# ==========================================
# Si los CSV ya están en /content, no hace falta montar Drive.
# Si quieres usar Drive, descomenta:
# from google.colab import drive
# drive.mount('/content/drive')

# Nombres esperados según tu notebook
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

# Verificación
missing = [v for v in file_map.values() if not os.path.exists(v)]
if missing:
    raise FileNotFoundError(
        "Faltan estos archivos en /content:\n" + "\n".join(missing) +
        "\n\nSúbelos al panel izquierdo de Colab o copia los CSV a /content."
    )

raw_data = {k: pd.read_csv(v, sep=",") for k, v in file_map.items()}


# ==================================================
# 3. PREPROCESAMIENTO IGUAL AL ENFOQUE DE TU CUADERNO
# ==================================================
# En tu notebook:
# TMED -> iloc[:,3:-6]
# PCP  -> iloc[:,3:-1]
# y luego promedio por fecha

def build_monthly_series(df, variable_type="TMED"):
    df = df.copy()

    # Reemplazar negativos por NaN, como en tu flujo
    df[df < 0] = np.nan

    if variable_type == "TMED":
        sub = df.iloc[:, 3:-6]
    elif variable_type == "PCP":
        sub = df.iloc[:, 3:-1]
    else:
        raise ValueError("variable_type debe ser 'TMED' o 'PCP'")

    # Promedio por columna/fecha
    s = sub.describe().iloc[1:2, :].T.reset_index()
    s.columns = ["fecha", "prom"]
    s["fecha"] = pd.to_datetime(s["fecha"])
    s = s.sort_values("fecha").reset_index(drop=True)
    s = s.set_index("fecha")
    s = s.asfreq("MS")  # Monthly Start
    s["prom"] = s["prom"].interpolate(method="linear").ffill().bfill()

    return s

series_dict = {}

for key, df in raw_data.items():
    variable_type = "TMED" if "TMED" in key else "PCP"
    series_dict[key] = build_monthly_series(df, variable_type=variable_type)

# Revisión rápida
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


# ===============================
# 5. FUNCIONES DE MODELAMIENTO DL
# ===============================
def create_sequences(values, look_back=12):
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i:i + look_back])
        y.append(values[i + look_back])
    return np.array(X), np.array(y)

def prepare_data_for_dl(series, look_back=12, train_ratio=0.80):
    values = series.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = create_sequences(scaled, look_back=look_back)

    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM espera [muestras, pasos, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, scaler, scaled

def build_lstm_model(look_back=12):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_bilstm_model(look_back=12):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(look_back, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def recursive_forecast(model, scaled_series, scaler, look_back=12, n_future=12):
    """
    scaled_series: serie completa ya escalada (n,1)
    Toma los últimos 12 meses y predice 12 hacia adelante recursivamente.
    """
    last_window = scaled_series[-look_back:].reshape(1, look_back, 1)
    preds_scaled = []

    for _ in range(n_future):
        pred = model.predict(last_window, verbose=0)[0, 0]
        preds_scaled.append(pred)

        new_window = np.append(last_window[:, 1:, :], [[[pred]]], axis=1)
        last_window = new_window

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds

def inverse_transform_1d(arr, scaler):
    arr = np.array(arr).reshape(-1, 1)
    return scaler.inverse_transform(arr).flatten()

def evaluate_model(model, X_test, y_test, scaler):
    pred_test = model.predict(X_test, verbose=0)
    y_test_inv = inverse_transform_1d(y_test, scaler)
    pred_inv   = inverse_transform_1d(pred_test, scaler)

    mae = mean_absolute_error(y_test_inv, pred_inv)
    rmse = math.sqrt(mean_squared_error(y_test_inv, pred_inv))

    return y_test_inv, pred_inv, mae, rmse

def train_and_forecast(series_df, model_type="LSTM", look_back=12, epochs=500, batch_size=16):
    series = series_df["prom"].copy()

    X_train, X_test, y_train, y_test, scaler, scaled = prepare_data_for_dl(
        series, look_back=look_back, train_ratio=0.80
    )

    if model_type.upper() == "LSTM":
        model = build_lstm_model(look_back=look_back)
    elif model_type.upper() == "BILSTM":
        model = build_bilstm_model(look_back=look_back)
    else:
        raise ValueError("model_type debe ser 'LSTM' o 'BiLSTM'")

    es = EarlyStopping(
        monitor="val_loss",
        patience=40,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es]
    )

    y_test_inv, pred_inv, mae, rmse = evaluate_model(model, X_test, y_test, scaler)

    future_2020 = recursive_forecast(
        model=model,
        scaled_series=scaled,
        scaler=scaler,
        look_back=look_back,
        n_future=12
    )

    return {
        "model": model,
        "history": history,
        "mae": mae,
        "rmse": rmse,
        "y_test_inv": y_test_inv,
        "pred_test_inv": pred_inv,
        "future_2020": future_2020,
        "series": series
    }


# ==========================================
# 6. ENTRENAMIENTO PARA TODOS LOS CLUSTERS
# ==========================================
results = {}

for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        key = f"{cluster}_{variable}"
        print(f"\nEntrenando {key} con LSTM...")
        results[f"{key}_LSTM"] = train_and_forecast(
            series_dict[key],
            model_type="LSTM",
            look_back=12,
            epochs=500,
            batch_size=16
        )

        print(f"Entrenando {key} con BiLSTM...")
        results[f"{key}_BiLSTM"] = train_and_forecast(
            series_dict[key],
            model_type="BiLSTM",
            look_back=12,
            epochs=500,
            batch_size=16
        )


# ==========================
# 7. TABLA DE MÉTRICAS
# ==========================
metrics_rows = []
for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        for model_name in ["LSTM", "BiLSTM"]:
            r = results[f"{cluster}_{variable}_{model_name}"]
            metrics_rows.append({
                "Cluster": cluster,
                "Variable": variable,
                "Model": model_name,
                "MAE": r["mae"],
                "RMSE": r["rmse"]
            })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.sort_values(["Variable", "Cluster", "Model"]).reset_index(drop=True)
print("\n===== MÉTRICAS =====")
display(metrics_df)

metrics_df.to_excel("metricas_LSTM_BiLSTM_por_cluster.xlsx", index=False)


# ==========================================
# 8. FUNCIONES DE GRAFICACIÓN TIPO DOCUMENTO
# ==========================================
def get_test_dates(series, look_back=12, train_ratio=0.80):
    values = series.values
    n_sequences = len(values) - look_back
    split = int(n_sequences * train_ratio)

    # Fechas objetivo de y
    target_dates = series.index[look_back:]
    test_dates = target_dates[split:]
    return test_dates

def plot_train_test_prediction(cluster, variable, model_name):
    r = results[f"{cluster}_{variable}_{model_name}"]
    series = r["series"]

    test_dates = get_test_dates(series, look_back=12, train_ratio=0.80)

    plt.figure(figsize=(15, 6))
    plt.plot(series.index, series.values, color="#3c3c3c", label="Serie histórica")
    plt.plot(test_dates, r["pred_test_inv"], color="purple", label=f"Predicción test ({model_name})")

    if variable == "TMED":
        plt.ylabel("Temperatura promedio [°C]")
        plt.title(f"Temperatura promedio {cluster} - {model_name}")
    else:
        plt.ylabel("Precipitación [mm]")
        plt.title(f"Precipitación {cluster} - {model_name}")

    plt.xlabel("Año")
    plt.legend()
    plt.show()

def plot_2020_forecast(cluster, variable):
    """
    Gráfica parecida a la del documento:
    serie histórica + pronóstico 2020 para LSTM y BiLSTM
    """
    hist = series_dict[f"{cluster}_{variable}"]["prom"].copy()

    future_dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    lstm_pred = results[f"{cluster}_{variable}_LSTM"]["future_2020"]
    bilstm_pred = results[f"{cluster}_{variable}_BiLSTM"]["future_2020"]

    plt.figure(figsize=(16, 6))
    plt.plot(hist.index, hist.values, color="#2f2f2f", label="Datos históricos 1981-2019")
    plt.plot(future_dates, lstm_pred, color="blue", marker="o", label="LSTM 2020")
    plt.plot(future_dates, bilstm_pred, color="darkorange", marker="s", label="BiLSTM 2020")

    plt.axvline(pd.Timestamp("2020-01-01"), color="black", linestyle="--", alpha=0.6)

    if variable == "TMED":
        plt.ylabel("Temperatura promedio [°C]")
        plt.title(f"Temperatura promedio {cluster} - Pronóstico 2020")
    else:
        plt.ylabel("Precipitación [mm]")
        plt.title(f"Precipitación {cluster} - Pronóstico 2020")

    plt.xlabel("Año")
    plt.legend()
    plt.show()

def plot_2020_all_clusters(variable="TMED", model_name="BiLSTM"):
    future_dates = pd.date_range("2020-01-01", periods=12, freq="MS")

    plt.figure(figsize=(16, 7))
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        pred = results[f"{cluster}_{variable}_{model_name}"]["future_2020"]
        plt.plot(
            future_dates,
            pred,
            marker="o",
            label=cluster,
            color=cluster_colors[cluster]
        )

    plt.xticks(future_dates, [d.strftime("%b") for d in future_dates], rotation=45)

    if variable == "TMED":
        plt.ylabel("Average temperature [°C]")
        plt.title(f"2020 forecast by cluster - {model_name} - TMED", fontsize=18)
    else:
        plt.ylabel("Precipitation [mm]")
        plt.title(f"2020 forecast by cluster - {model_name} - PCP", fontsize=18)

    plt.xlabel("Month of 2020")
    plt.legend(fontsize=12)
    plt.show()


# ==========================================
# 9. GRAFICAS INDIVIDUALES POR CLUSTER
# ==========================================
for variable in ["TMED", "PCP"]:
    for cluster in ["C1", "C2", "C3", "C4", "C5"]:
        plot_train_test_prediction(cluster, variable, "LSTM")
        plot_train_test_prediction(cluster, variable, "BiLSTM")
        plot_2020_forecast(cluster, variable)


# ==========================================
# 10. FIGURAS COMPARATIVAS ENTRE CLUSTERS
# ==========================================
plot_2020_all_clusters(variable="TMED", model_name="LSTM")
plot_2020_all_clusters(variable="TMED", model_name="BiLSTM")
plot_2020_all_clusters(variable="PCP", model_name="LSTM")
plot_2020_all_clusters(variable="PCP", model_name="BiLSTM")


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
        for model_name in ["LSTM", "BiLSTM"]:
            pred = results[f"{cluster}_{variable}_{model_name}"]["future_2020"]
            df_pred = pd.DataFrame([pred], index=[2020], columns=month_names)
            forecast_tables[f"{cluster}_{variable}_{model_name}"] = df_pred

            out_name = f"predicciones_{variable}_{model_name}_{cluster}_2020.xlsx"
            df_pred.to_excel(out_name, sheet_name="Predicciones 2020")

# Mostrar ejemplo
print("\n===== EJEMPLO DE TABLA DE PREDICCIÓN =====")
display(forecast_tables["C1_TMED_LSTM"])


# ==========================================
# 12. RESUMEN GENERAL EN EXCEL
# ==========================================
with pd.ExcelWriter("resumen_predicciones_2020_LSTM_BiLSTM.xlsx", engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="Metricas", index=False)

    for k, dfp in forecast_tables.items():
        sheet = k[:31]  # límite de Excel
        dfp.to_excel(writer, sheet_name=sheet)

print("\nArchivos exportados:")
print("- metricas_LSTM_BiLSTM_por_cluster.xlsx")
print("- resumen_predicciones_2020_LSTM_BiLSTM.xlsx")
print("- y un Excel individual por cluster/variable/modelo")