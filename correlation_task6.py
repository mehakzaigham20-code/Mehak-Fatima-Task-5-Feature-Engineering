import pandas as pd
import numpy as np

print("=" * 60)
print("TASK 6 - CORRELATION ANALYSIS")
print("=" * 60)

# Load Task 6 embeddings
df = pd.read_csv(
    "output/task6_embeddings.csv"
)

print(
    "Task 6 Embeddings Loaded Successfully"
)

print(
    "Original Shape:",
    df.shape
)

# --------------------------------------------------
# Remove duplicate columns
# --------------------------------------------------

print("\nChecking duplicate columns...")

duplicate_columns = df.T.duplicated()

duplicate_count = duplicate_columns.sum()

df = df.loc[
    :,
    ~duplicate_columns
]

print(
    "Duplicate Columns Removed:",
    duplicate_count
)

print(
    "Shape After Duplicate Removal:",
    df.shape
)

# --------------------------------------------------
# Create correlation matrix
# --------------------------------------------------

print("\nCreating correlation matrix...")

correlation_matrix = df.corr()

correlation_matrix.to_csv(
    "output/correlation_matrix.csv"
)

print(
    "Correlation Matrix Saved Successfully"
)

# --------------------------------------------------
# Find correlation > 0.95
# Absolute correlation is used
# --------------------------------------------------

print(
    "\nChecking absolute correlation > 0.95..."
)

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(
            correlation_matrix.shape
        ),
        k=1
    ).astype(bool)
)

columns_to_remove = [
    column
    for column in upper_triangle.columns
    if any(
        upper_triangle[column].abs() > 0.95
    )
]

print(
    "Highly Correlated Columns Found:",
    len(columns_to_remove)
)

# --------------------------------------------------
# Remove highly correlated columns
# --------------------------------------------------

df_filtered = df.drop(
    columns=columns_to_remove
)

# --------------------------------------------------
# Save filtered embeddings
# --------------------------------------------------

df_filtered.to_csv(
    "output/correlation_filtered.csv",
    index=False
)

print("=" * 60)
print("CORRELATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "Original Columns:",
    correlation_matrix.shape[0]
)

print(
    "Duplicate Columns Removed:",
    duplicate_count
)

print(
    "Highly Correlated Columns Removed:",
    len(columns_to_remove)
)

print(
    "Remaining Columns:",
    df_filtered.shape[1]
)

print(
    "Rows:",
    df_filtered.shape[0]
)

print("\nFiles Created:")

print(
    "1. output/correlation_matrix.csv"
)

print(
    "2. output/correlation_filtered.csv"
)

print("=" * 60)