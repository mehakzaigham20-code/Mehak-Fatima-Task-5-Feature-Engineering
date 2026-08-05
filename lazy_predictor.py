import pandas as pd
from lazypredict.Supervised import LazyRegressor

print("=" * 60)
print("TASK 6 - LAZY PREDICTOR")
print("=" * 60)

# --------------------------------------------------
# Load train and test data
# --------------------------------------------------

train_df = pd.read_csv(
    "output/train.csv"
)

test_df = pd.read_csv(
    "output/test.csv"
)

print(
    "Train Data Shape:",
    train_df.shape
)

print(
    "Test Data Shape:",
    test_df.shape
)

# --------------------------------------------------
# Check target
# --------------------------------------------------

if "no_of_clicks" not in train_df.columns:
    raise ValueError(
        "no_of_clicks not found in training data!"
    )

if "no_of_clicks" not in test_df.columns:
    raise ValueError(
        "no_of_clicks not found in testing data!"
    )

# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

X_train = train_df.drop(
    columns=["no_of_clicks"]
)

y_train = train_df["no_of_clicks"]

X_test = test_df.drop(
    columns=["no_of_clicks"]
)

y_test = test_df["no_of_clicks"]

print("\nX_train:", X_train.shape)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)

# --------------------------------------------------
# Lazy Regression
# --------------------------------------------------

print("\nRunning LazyRegressor...")
print("Please wait...")

reg = LazyRegressor(
    verbose=0,
    ignore_warnings=True,
    custom_metric=None
)

models, predictions = reg.fit(
    X_train,
    X_test,
    y_train,
    y_test
)

# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 60)
print("LAZY PREDICTOR RESULTS")
print("=" * 60)

print(models)

# --------------------------------------------------
# Save results
# --------------------------------------------------

models.to_csv(
    "output/lazy_predictor_results.csv"
)

print(
    "\nResults Saved Successfully"
)

print(
    "File: output/lazy_predictor_results.csv"
)

# --------------------------------------------------
# Best Model
# --------------------------------------------------

best_model_name = models[
    "R-Squared"
].idxmax()

best_r2 = models.loc[
    best_model_name,
    "R-Squared"
]

best_rmse = models.loc[
    best_model_name,
    "RMSE"
]

print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Model:",
    best_model_name
)

print(
    "R-Squared:",
    best_r2
)

print(
    "RMSE:",
    best_rmse
)

print("=" * 60)