import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=" * 60)
print("TASK 6 - SCALING + PCA")
print("=" * 60)

# --------------------------------------------------
# Load correlation-filtered embeddings
# --------------------------------------------------

df = pd.read_csv(
    "output/correlation_filtered.csv"
)

print(
    "Correlation Filtered Data Loaded Successfully"
)

print(
    "Input Shape:",
    df.shape
)

# --------------------------------------------------
# Keep numerical features
# --------------------------------------------------

features = df.select_dtypes(
    include="number"
)

print(
    "Numerical Features:",
    features.shape
)

# --------------------------------------------------
# Standard Scaling
# --------------------------------------------------

print("\nApplying StandardScaler...")

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    features
)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=features.columns
)

# Save scaled embeddings
scaled_df.to_csv(
    "output/scaled_embeddings.csv",
    index=False
)

print(
    "Scaling Completed Successfully"
)

print(
    "Scaled Shape:",
    scaled_df.shape
)

# --------------------------------------------------
# PCA
# Keep 95% variance
# --------------------------------------------------

print("\nApplying PCA...")

pca = PCA(
    n_components=0.95
)

pca_data = pca.fit_transform(
    scaled_data
)

# PCA column names
pca_columns = [
    f"PC{i + 1}"
    for i in range(
        pca_data.shape[1]
    )
]

pca_df = pd.DataFrame(
    pca_data,
    columns=pca_columns
)

# Save PCA embeddings
pca_df.to_csv(
    "output/pca_embeddings.csv",
    index=False
)

print(
    "PCA Embeddings Saved Successfully"
)

print(
    "PCA Shape:",
    pca_df.shape
)

print(
    "Explained Variance:",
    pca.explained_variance_ratio_.sum()
)

print("=" * 60)
print("SCALING + PCA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles Created:")

print(
    "1. output/scaled_embeddings.csv"
)

print(
    "2. output/pca_embeddings.csv"
)

print("=" * 60)