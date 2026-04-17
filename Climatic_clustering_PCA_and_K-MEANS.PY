# ============================================================
# CLUSTERIZACIÓN CLIMÁTICA CON PCA + K-MEANS
# Método del codo, correlación de Spearman, métricas, boxplots
# Adaptado al flujo metodológico del manuscrito de aguacate Hass
# ============================================================

# =========================
# 1. INSTALACIÓN E IMPORTS
# =========================
!pip -q install openpyxl

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 12

# ============================================================
# 2. CARGA DEL ARCHIVO BASE
# ============================================================
# OPCIÓN A:
# Sube un archivo CSV al panel izquierdo de Colab y déjalo en /content
# por ejemplo: datos_clusters_aguacate.csv

# OPCIÓN B:
# Usa Drive
# from google.colab import drive
# drive.mount('/content/drive')

# ------------------------------------------------------------
# AJUSTA AQUÍ EL NOMBRE DE TU ARCHIVO
# ------------------------------------------------------------
input_file = "datos_clusters_aguacate.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"No se encontró el archivo '{input_file}' en /content.\n"
        "Súbelo a Colab y vuelve a ejecutar."
    )

df = pd.read_csv(input_file)
print("Dimensión del dataset:", df.shape)
display(df.head())


# ============================================================
# 3. DEFINICIÓN DE VARIABLES
# ============================================================
# Aquí defines las columnas ambientales que entran al análisis.
# Puedes dejarlas manuales o permitir que el código las detecte.

# ---- OPCIÓN MANUAL (RECOMENDADA) ----
candidate_vars = [
    "ELEV1",
    "mRAD1",
    "mTMAX1",
    "mTMED1",
    "mTMIN1",
    "mVAPR1",
    "mWIND1",
    "sPCP1",
    "annualPET1",
    "aridityInd",
    "climaticMo",
    "continenta",
    "maxTempCol",
    "minTempWar",
    "PETseasona",
    "Thermicity"
]

# Si además tienes coordenadas:
possible_xy = ["X", "Y", "x", "y", "lon", "lat", "Longitude", "Latitude"]

# Filtrar las que sí existan
candidate_vars = [c for c in candidate_vars if c in df.columns]

if len(candidate_vars) == 0:
    raise ValueError(
        "Ninguna de las variables definidas en candidate_vars existe en el archivo.\n"
        "Revisa los nombres de columnas."
    )

xy_cols = [c for c in possible_xy if c in df.columns]

print("Variables candidatas encontradas:", candidate_vars)
print("Columnas espaciales detectadas:", xy_cols)


# ============================================================
# 4. LIMPIEZA Y PREPARACIÓN
# ============================================================
data = df[candidate_vars].copy()

# Convertir a numérico por seguridad
for c in data.columns:
    data[c] = pd.to_numeric(data[c], errors="coerce")

# Quitar registros con demasiados nulos
data = data.dropna(axis=0, how="any").reset_index(drop=True)

# Sincronizar df original con registros válidos
df_valid = df.loc[data.index].reset_index(drop=True)

print("Dimensión luego de limpieza:", data.shape)


# ============================================================
# 5. MATRIZ DE CORRELACIÓN DE SPEARMAN
# ============================================================
corr_spearman = data.corr(method="spearman")

plt.figure(figsize=(14, 11))
sns.heatmap(
    corr_spearman,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5
)
plt.title("Spearman correlation matrix")
plt.tight_layout()
plt.show()

corr_spearman.to_excel("01_matriz_correlacion_spearman.xlsx")


# ============================================================
# 6. FILTRADO DE VARIABLES ALTAMENTE CORRELACIONADAS
#    criterio: |rho| > 0.80
# ============================================================
threshold = 0.80
corr_abs = corr_spearman.abs()

upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

selected_vars = [c for c in data.columns if c not in to_drop]
filtered_data = data[selected_vars].copy()

print("Variables eliminadas por alta correlación:", to_drop)
print("Variables retenidas:", selected_vars)
print("Número final de variables:", len(selected_vars))

pd.DataFrame({
    "variables_iniciales": pd.Series(data.columns),
    "variables_eliminadas": pd.Series(to_drop),
    "variables_retenidas": pd.Series(selected_vars)
}).to_excel("02_resumen_filtrado_correlacion.xlsx", index=False)


# ============================================================
# 7. ESTANDARIZACIÓN
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_data)

X_scaled_df = pd.DataFrame(X_scaled, columns=selected_vars)
X_scaled_df.to_excel("03_variables_estandarizadas.xlsx", index=False)


# ============================================================
# 8. PCA COMPLETO
# ============================================================
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

explained_var = pca_full.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var)

pca_summary = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(explained_var))],
    "Explained_variance_ratio": explained_var,
    "Cumulative_explained_variance": cum_explained_var
})

print("\nResumen PCA:")
display(pca_summary)

pca_summary.to_excel("04_resumen_pca.xlsx", index=False)

# Gráfica de varianza explicada
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(explained_var)+1), explained_var, alpha=0.8, label="Individual")
plt.plot(range(1, len(cum_explained_var)+1), cum_explained_var, marker="o", label="Cumulative")
plt.xticks(range(1, len(explained_var)+1))
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained variance by principal component")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. CARGAS DE LAS VARIABLES EN PCA
# ============================================================
loadings = pd.DataFrame(
    pca_full.components_.T,
    columns=[f"PC{i+1}" for i in range(len(selected_vars))],
    index=selected_vars
)

print("\nCargas de las variables:")
display(loadings)

loadings.to_excel("05_cargas_pca.xlsx")

# Variables más relevantes para PC1 y PC2
important_pc1 = loadings["PC1"].abs().sort_values(ascending=False)
important_pc2 = loadings["PC2"].abs().sort_values(ascending=False)

important_vars_df = pd.DataFrame({
    "PC1_abs_loading": important_pc1,
    "PC2_abs_loading": loadings["PC2"].abs().reindex(important_pc1.index)
})

important_vars_df.to_excel("06_variables_importantes_pc1_pc2.xlsx")

# Barras PC1 y PC2
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

important_pc1.sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Variable importance in PC1")
axes[0].set_xlabel("Absolute loading")

important_pc2.sort_values().plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("Variable importance in PC2")
axes[1].set_xlabel("Absolute loading")

plt.tight_layout()
plt.show()


# ============================================================
# 10. PCA EN 2 COMPONENTES PARA VISUALIZACIÓN
# ============================================================
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

pca_2_df = pd.DataFrame(X_pca_2, columns=["PC1", "PC2"])

print("Varianza explicada PC1:", pca_2.explained_variance_ratio_[0])
print("Varianza explicada PC2:", pca_2.explained_variance_ratio_[1])
print("Varianza acumulada PC1+PC2:", pca_2.explained_variance_ratio_.sum())

# Scatter sin clusters
plt.figure(figsize=(10, 8))
plt.scatter(pca_2_df["PC1"], pca_2_df["PC2"], s=20, alpha=0.6)
plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("PCA scatter plot")
plt.tight_layout()
plt.show()


# ============================================================
# 11. BIPLOT PCA
# ============================================================
plt.figure(figsize=(12, 10))
plt.scatter(pca_2_df["PC1"], pca_2_df["PC2"], alpha=0.35, s=15)

for i, var in enumerate(selected_vars):
    plt.arrow(
        0, 0,
        loadings.loc[var, "PC1"] * 4,
        loadings.loc[var, "PC2"] * 4,
        color="red",
        alpha=0.8,
        head_width=0.05
    )
    plt.text(
        loadings.loc[var, "PC1"] * 4.2,
        loadings.loc[var, "PC2"] * 4.2,
        var,
        color="darkred",
        fontsize=10
    )

plt.axhline(0, color="gray", lw=1)
plt.axvline(0, color="gray", lw=1)
plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("PCA biplot")
plt.tight_layout()
plt.show()


# ============================================================
# 12. MÉTODO DEL CODO + MÉTRICAS DE CLUSTERING
# ============================================================
k_range = range(2, 11)
inertias = []
silhouettes = []
calinskis = []
davies = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    calinskis.append(calinski_harabasz_score(X_scaled, labels))
    davies.append(davies_bouldin_score(X_scaled, labels))

metrics_k_df = pd.DataFrame({
    "k": list(k_range),
    "inertia": inertias,
    "silhouette_score": silhouettes,
    "calinski_harabasz": calinskis,
    "davies_bouldin": davies
})

print("\nMétricas por número de clúster:")
display(metrics_k_df)

metrics_k_df.to_excel("07_metricas_numero_clusters.xlsx", index=False)

# Método del codo
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), inertias, marker="o")
plt.xticks(list(k_range))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.tight_layout()
plt.show()

# Silhouette
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), silhouettes, marker="o")
plt.xticks(list(k_range))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score by k")
plt.tight_layout()
plt.show()

# Calinski-Harabasz
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), calinskis, marker="o")
plt.xticks(list(k_range))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Calinski-Harabasz score")
plt.title("Calinski-Harabasz by k")
plt.tight_layout()
plt.show()

# Davies-Bouldin
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), davies, marker="o")
plt.xticks(list(k_range))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Davies-Bouldin index")
plt.title("Davies-Bouldin by k (lower is better)")
plt.tight_layout()
plt.show()


# ============================================================
# 13. MODELO FINAL K-MEANS CON 5 CLÚSTERES
# ============================================================
# En el manuscrito, el resultado final fue k = 5
n_clusters_final = 5

kmeans_final = KMeans(n_clusters=n_clusters_final, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_scaled)

result_df = df_valid.copy()
result_df = result_df.reset_index(drop=True)
result_df["cluster"] = cluster_labels + 1  # para dejar C1..C5 como 1..5

# Métricas finales
final_silhouette = silhouette_score(X_scaled, cluster_labels)
final_calinski = calinski_harabasz_score(X_scaled, cluster_labels)
final_davies = davies_bouldin_score(X_scaled, cluster_labels)
final_inertia = kmeans_final.inertia_

final_metrics = pd.DataFrame({
    "metric": ["n_clusters", "inertia", "silhouette_score", "calinski_harabasz", "davies_bouldin"],
    "value": [n_clusters_final, final_inertia, final_silhouette, final_calinski, final_davies]
})

print("\nMétricas finales del modelo:")
display(final_metrics)

final_metrics.to_excel("08_metricas_modelo_final_k5.xlsx", index=False)


# ============================================================
# 14. SCATTER PCA CON CLÚSTERES FINALES
# ============================================================
plot_df = pca_2_df.copy()
plot_df["cluster"] = cluster_labels + 1

plt.figure(figsize=(11, 9))
sns.scatterplot(
    data=plot_df,
    x="PC1", y="PC2",
    hue="cluster",
    palette="Set1",
    s=40,
    alpha=0.8
)
plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("PCA scatter plot by cluster")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()


# ============================================================
# 15. CENTROIDES EN EL ESPACIO ORIGINAL Y PCA
# ============================================================
centroids_scaled = kmeans_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

centroids_df = pd.DataFrame(centroids_original, columns=selected_vars)
centroids_df.index = [f"C{i}" for i in range(1, n_clusters_final + 1)]

print("\nCentroides en unidades originales:")
display(centroids_df)

centroids_df.to_excel("09_centroides_clusters_unidades_originales.xlsx")

centroids_pca = pca_2.transform(centroids_scaled)
centroids_pca_df = pd.DataFrame(centroids_pca, columns=["PC1", "PC2"])
centroids_pca_df["cluster"] = [f"C{i}" for i in range(1, n_clusters_final + 1)]

# Scatter PCA con centroides
plt.figure(figsize=(11, 9))
sns.scatterplot(
    data=plot_df,
    x="PC1", y="PC2",
    hue="cluster",
    palette="Set1",
    s=35,
    alpha=0.6
)
plt.scatter(
    centroids_pca_df["PC1"],
    centroids_pca_df["PC2"],
    marker="X",
    s=250,
    c="black",
    label="Centroids"
)
for _, row in centroids_pca_df.iterrows():
    plt.text(row["PC1"] + 0.05, row["PC2"] + 0.05, row["cluster"], fontsize=11, weight="bold")

plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("Final clusters and centroids in PCA space")
plt.tight_layout()
plt.show()


# ============================================================
# 16. TABLA DE ESTADÍSTICOS POR CLÚSTER
#     Media y desviación estándar como en la tabla del artículo
# ============================================================
summary_mean = result_df.groupby("cluster")[selected_vars].mean().T
summary_std = result_df.groupby("cluster")[selected_vars].std().T

summary_combined = pd.DataFrame(index=selected_vars)

for c in range(1, n_clusters_final + 1):
    summary_combined[f"C{c}_Mean"] = summary_mean[c]
    summary_combined[f"C{c}_Std"] = summary_std[c]

print("\nResumen estadístico por clúster:")
display(summary_combined)

summary_combined.to_excel("10_resumen_media_std_por_cluster.xlsx")


# ============================================================
# 17. BOXPLOTS POR VARIABLE Y CLÚSTER
# ============================================================
n_vars = len(selected_vars)
ncols = 2
nrows = int(np.ceil(n_vars / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
axes = axes.flatten()

for i, var in enumerate(selected_vars):
    sns.boxplot(data=result_df, x="cluster", y=var, ax=axes[i], palette="Set2")
    axes[i].set_title(f"{var} by cluster")
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel(var)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# ============================================================
# 18. SI EXISTEN COORDENADAS, GRAFICAR DISTRIBUCIÓN ESPACIAL
# ============================================================
xy_found = None
if "X" in result_df.columns and "Y" in result_df.columns:
    xy_found = ("X", "Y")
elif "x" in result_df.columns and "y" in result_df.columns:
    xy_found = ("x", "y")
elif "lon" in result_df.columns and "lat" in result_df.columns:
    xy_found = ("lon", "lat")
elif "Longitude" in result_df.columns and "Latitude" in result_df.columns:
    xy_found = ("Longitude", "Latitude")

if xy_found is not None:
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        data=result_df,
        x=xy_found[0],
        y=xy_found[1],
        hue="cluster",
        palette="Set1",
        s=18,
        alpha=0.8
    )
    plt.title("Spatial distribution of climatic clusters")
    plt.xlabel(xy_found[0])
    plt.ylabel(xy_found[1])
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()


# ============================================================
# 19. EXPORTACIÓN FINAL DE RESULTADOS
# ============================================================
result_df.to_excel("11_dataset_con_clusters.xlsx", index=False)
plot_df.to_excel("12_pca_scores_con_clusters.xlsx", index=False)

with pd.ExcelWriter("00_resumen_general_clusterizacion.xlsx", engine="openpyxl") as writer:
    corr_spearman.to_excel(writer, sheet_name="Spearman_corr")
    pd.DataFrame({"selected_vars": selected_vars}).to_excel(writer, sheet_name="Variables_retenidas", index=False)
    pca_summary.to_excel(writer, sheet_name="PCA_summary", index=False)
    loadings.to_excel(writer, sheet_name="PCA_loadings")
    metrics_k_df.to_excel(writer, sheet_name="Metricas_k", index=False)
    final_metrics.to_excel(writer, sheet_name="Metricas_finales", index=False)
    centroids_df.to_excel(writer, sheet_name="Centroides")
    summary_combined.to_excel(writer, sheet_name="Resumen_cluster")
    result_df.to_excel(writer, sheet_name="Datos_con_cluster", index=False)
    plot_df.to_excel(writer, sheet_name="PCA_scores", index=False)

print("\nArchivos exportados:")
print("- 00_resumen_general_clusterizacion.xlsx")
print("- 01_matriz_correlacion_spearman.xlsx")
print("- 02_resumen_filtrado_correlacion.xlsx")
print("- 04_resumen_pca.xlsx")
print("- 05_cargas_pca.xlsx")
print("- 07_metricas_numero_clusters.xlsx")
print("- 08_metricas_modelo_final_k5.xlsx")
print("- 09_centroides_clusters_unidades_originales.xlsx")
print("- 10_resumen_media_std_por_cluster.xlsx")
print("- 11_dataset_con_clusters.xlsx")
print("- 12_pca_scores_con_clusters.xlsx")