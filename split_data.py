import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("TASK 6 - TRAIN / TEST SPLIT")
print("=" * 60)

# --------------------------------------------------
# Load PCA features
# --------------------------------------------------

X = pd.read_csv(
    "output/pca_embeddings.csv"
)

print(
    "PCA Data Loaded Successfully"
)

print(
    "PCA Shape:",
    X.shape
)

# --------------------------------------------------
# Load Feature Engineering dataset
# --------------------------------------------------

original_df = pd.read_csv(
    "output/feature_engineered_articles.csv"
)

print(
    "Feature Engineering Data Shape:",
    original_df.shape
)

# --------------------------------------------------
# Check target
# --------------------------------------------------

if "no_of_clicks" not in original_df.columns:
    raise ValueError(
        "no_of_clicks column not found!"
    )

# Target
y = original_df["no_of_clicks"]

print(
    "Target Shape:",
    y.shape
)

# --------------------------------------------------
# Check rows
# --------------------------------------------------

if len(X) != len(y):
    raise ValueError(
        f"Row mismatch! "
        f"PCA rows = {len(X)}, "
        f"Target rows = {len(y)}"
    )

# --------------------------------------------------
# 80% Train / 20% Test
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# Create train dataset
# --------------------------------------------------

train_df = X_train.copy()

train_df["no_of_clicks"] = y_train.values

# --------------------------------------------------
# Create test dataset
# --------------------------------------------------

test_df = X_test.copy()

test_df["no_of_clicks"] = y_test.values

# --------------------------------------------------
# Save files
# --------------------------------------------------

train_df.to_csv(
    "output/train.csv",
    index=False
)

test_df.to_csv(
    "output/test.csv",
    index=False
)

# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 60)
print("TRAIN / TEST SPLIT COMPLETED")
print("=" * 60)

print(
    "Total Rows:",
    len(X)
)

print(
    "Training Rows:",
    len(X_train)
)

print(
    "Testing Rows:",
    len(X_test)
)

print(
    "Training Percentage:",
    round(
        len(X_train) / len(X) * 100,
        2
    ),
    "%"
)

print(
    "Testing Percentage:",
    round(
        len(X_test) / len(X) * 100,
        2
    ),
    "%"
)

print(
    "Training Shape:",
    train_df.shape
)

print(
    "Testing Shape:",
    test_df.shape
)

print("\nFiles Created:")

print(
    "1. output/train.csv"
)

print(
    "2. output/test.csv"
)

print("=" * 60)