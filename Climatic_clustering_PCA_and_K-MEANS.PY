# ============================================================
# CLIMATIC CLUSTERING WITH PCA + K-MEANS
# Elbow method, Spearman correlation, metrics, boxplots
# Adapted from the methodological workflow of the Hass avocado manuscript
# ============================================================

# =========================
# 1. INSTALLATION & IMPORTS
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
# 2. LOADING THE BASE FILE
# ============================================================
# OPTION A:
# Upload a CSV to the Colab left panel and leave it in /content
# for example: avocado_cluster_data.csv

# OPTION B:
# Use Drive
# from google.colab import drive
# drive.mount('/content/drive')

# ------------------------------------------------------------
# ADJUST THE FILE NAME HERE
# ------------------------------------------------------------
input_file = "avocado_cluster_data.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"File '{input_file}' not found in /content.\n"
        "Upload it to Colab and run again."
    )

df = pd.read_csv(input_file)
print("Dataset dimensions:", df.shape)
display(df.head())


# ============================================================
# 3. VARIABLE DEFINITION
# ============================================================
# Define here the environmental columns that enter the analysis.
# You can leave them manual or let the code detect them.

# ---- MANUAL OPTION (RECOMMENDED) ----
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

# If you also have coordinates:
possible_xy = ["X", "Y", "x", "y", "lon", "lat", "Longitude", "Latitude"]

# Filter only those that exist
candidate_vars = [c for c in candidate_vars if c in df.columns]

if len(candidate_vars) == 0:
    raise ValueError(
        "None of the variables defined in candidate_vars exist in the file.\n"
        "Check the column names."
    )

xy_cols = [c for c in possible_xy if c in df.columns]

print("Candidate variables found:", candidate_vars)
print("Spatial columns detected:", xy_cols)


# ============================================================
# 4. CLEANING AND PREPARATION
# ============================================================
data = df[candidate_vars].copy()

# Convert to numeric for safety
for c in data.columns:
    data[c] = pd.to_numeric(data[c], errors="coerce")

# Remove records with too many nulls
data = data.dropna(axis=0, how="any").reset_index(drop=True)

# Sync original df with valid records
df_valid = df.loc[data.index].reset_index(drop=True)

print("Dimensions after cleaning:", data.shape)


# ============================================================
# 5. SPEARMAN CORRELATION MATRIX
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

corr_spearman.to_excel("01_spearman_correlation_matrix.xlsx")


# ============================================================
# 6. FILTERING HIGHLY CORRELATED VARIABLES
#    criterion: |rho| > 0.80
# ============================================================
threshold = 0.80
corr_abs = corr_spearman.abs()

upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

selected_vars = [c for c in data.columns if c not in to_drop]
filtered_data = data[selected_vars].copy()

print("Variables removed due to high correlation:", to_drop)
print("Retained variables:", selected_vars)
print("Final number of variables:", len(selected_vars))

pd.DataFrame({
    "initial_variables": pd.Series(data.columns),
    "removed_variables": pd.Series(to_drop),
    "retained_variables": pd.Series(selected_vars)
}).to_excel("02_correlation_filtering_summary.xlsx", index=False)


# ============================================================
# 7. STANDARDIZATION
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_data)

X_scaled_df = pd.DataFrame(X_scaled, columns=selected_vars)
X_scaled_df.to_excel("03_standardized_variables.xlsx", index=False)


# ============================================================
# 8. FULL PCA
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

print("\nPCA summary:")
display(pca_summary)

pca_summary.to_excel("04_pca_summary.xlsx", index=False)

# Explained variance plot
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
# 9. VARIABLE LOADINGS IN PCA
# ============================================================
loadings = pd.DataFrame(
    pca_full.components_.T,
    columns=[f"PC{i+1}" for i in range(len(selected_vars))],
    index=selected_vars
)

print("\nVariable loadings:")
display(loadings)

loadings.to_excel("05_pca_loadings.xlsx")

# Most relevant variables for PC1 and PC2
important_pc1 = loadings["PC1"].abs().sort_values(ascending=False)
important_pc2 = loadings["PC2"].abs().sort_values(ascending=False)

important_vars_df = pd.DataFrame({
    "PC1_abs_loading": important_pc1,
    "PC2_abs_loading": loadings["PC2"].abs().reindex(important_pc1.index)
})

important_vars_df.to_excel("06_important_variables_pc1_pc2.xlsx")

# Bar charts PC1 and PC2
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
# 10. PCA WITH 2 COMPONENTS FOR VISUALIZATION
# ============================================================
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

pca_2_df = pd.DataFrame(X_pca_2, columns=["PC1", "PC2"])

print("Explained variance PC1:", pca_2.explained_variance_ratio_[0])
print("Explained variance PC2:", pca_2.explained_variance_ratio_[1])
print("Cumulative variance PC1+PC2:", pca_2.explained_variance_ratio_.sum())

# Scatter without clusters
plt.figure(figsize=(10, 8))
plt.scatter(pca_2_df["PC1"], pca_2_df["PC2"], s=20, alpha=0.6)
plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("PCA scatter plot")
plt.tight_layout()
plt.show()


# ============================================================
# 11. PCA BIPLOT
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
# 12. ELBOW METHOD + CLUSTERING METRICS
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

print("\nMetrics by number of clusters:")
display(metrics_k_df)

metrics_k_df.to_excel("07_cluster_number_metrics.xlsx", index=False)

# Elbow method
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
# 13. FINAL K-MEANS MODEL WITH 5 CLUSTERS
# ============================================================
# In the manuscript, the final result was k = 5
n_clusters_final = 5

kmeans_final = KMeans(n_clusters=n_clusters_final, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_scaled)

result_df = df_valid.copy()
result_df = result_df.reset_index(drop=True)
result_df["cluster"] = cluster_labels + 1  # to label C1..C5 as 1..5

# Final metrics
final_silhouette = silhouette_score(X_scaled, cluster_labels)
final_calinski = calinski_harabasz_score(X_scaled, cluster_labels)
final_davies = davies_bouldin_score(X_scaled, cluster_labels)
final_inertia = kmeans_final.inertia_

final_metrics = pd.DataFrame({
    "metric": ["n_clusters", "inertia", "silhouette_score", "calinski_harabasz", "davies_bouldin"],
    "value": [n_clusters_final, final_inertia, final_silhouette, final_calinski, final_davies]
})

print("\nFinal model metrics:")
display(final_metrics)

final_metrics.to_excel("08_final_model_metrics_k5.xlsx", index=False)


# ============================================================
# 14. PCA SCATTER WITH FINAL CLUSTERS
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
# 15. CENTROIDS IN ORIGINAL AND PCA SPACE
# ============================================================
centroids_scaled = kmeans_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

centroids_df = pd.DataFrame(centroids_original, columns=selected_vars)
centroids_df.index = [f"C{i}" for i in range(1, n_clusters_final + 1)]

print("\nCentroids in original units:")
display(centroids_df)

centroids_df.to_excel("09_cluster_centroids_original_units.xlsx")

centroids_pca = pca_2.transform(centroids_scaled)
centroids_pca_df = pd.DataFrame(centroids_pca, columns=["PC1", "PC2"])
centroids_pca_df["cluster"] = [f"C{i}" for i in range(1, n_clusters_final + 1)]

# PCA scatter with centroids
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
# 16. STATISTICS TABLE BY CLUSTER
#     Mean and standard deviation as in the article table
# ============================================================
summary_mean = result_df.groupby("cluster")[selected_vars].mean().T
summary_std = result_df.groupby("cluster")[selected_vars].std().T

summary_combined = pd.DataFrame(index=selected_vars)

for c in range(1, n_clusters_final + 1):
    summary_combined[f"C{c}_Mean"] = summary_mean[c]
    summary_combined[f"C{c}_Std"] = summary_std[c]

print("\nStatistical summary by cluster:")
display(summary_combined)

summary_combined.to_excel("10_mean_std_summary_by_cluster.xlsx")


# ============================================================
# 17. BOXPLOTS BY VARIABLE AND CLUSTER
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
# 18. IF COORDINATES EXIST, PLOT SPATIAL DISTRIBUTION
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
# 19. FINAL EXPORT OF RESULTS
# ============================================================
result_df.to_excel("11_dataset_with_clusters.xlsx", index=False)
plot_df.to_excel("12_pca_scores_with_clusters.xlsx", index=False)

with pd.ExcelWriter("00_general_clustering_summary.xlsx", engine="openpyxl") as writer:
    corr_spearman.to_excel(writer, sheet_name="Spearman_corr")
    pd.DataFrame({"selected_vars": selected_vars}).to_excel(writer, sheet_name="Retained_variables", index=False)
    pca_summary.to_excel(writer, sheet_name="PCA_summary", index=False)
    loadings.to_excel(writer, sheet_name="PCA_loadings")
    metrics_k_df.to_excel(writer, sheet_name="Metrics_k", index=False)
    final_metrics.to_excel(writer, sheet_name="Final_metrics", index=False)
    centroids_df.to_excel(writer, sheet_name="Centroids")
    summary_combined.to_excel(writer, sheet_name="Cluster_summary")
    result_df.to_excel(writer, sheet_name="Data_with_cluster", index=False)
    plot_df.to_excel(writer, sheet_name="PCA_scores", index=False)

print("\nExported files:")
print("- 00_general_clustering_summary.xlsx")
print("- 01_spearman_correlation_matrix.xlsx")
print("- 02_correlation_filtering_summary.xlsx")
print("- 04_pca_summary.xlsx")
print("- 05_pca_loadings.xlsx")
print("- 07_cluster_number_metrics.xlsx")
print("- 08_final_model_metrics_k5.xlsx")
print("- 09_cluster_centroids_original_units.xlsx")
print("- 10_mean_std_summary_by_cluster.xlsx")
print("- 11_dataset_with_clusters.xlsx")
print("- 12_pca_scores_with_clusters.xlsx")
