import pandas as pd
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("TASK 6 - EMBEDDING")
print("=" * 60)

# Load Feature Engineering dataset
df = pd.read_csv(
    "output/feature_engineered_articles.csv"
)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# Check title column
if "title" not in df.columns:
    raise ValueError("title column not found!")

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Generating title embeddings...")

# Generate embeddings using ONLY title
embeddings = model.encode(
    df["title"]
    .fillna("")
    .astype(str)
    .tolist(),
    show_progress_bar=True
)

# Convert embeddings to DataFrame
embedding_df = pd.DataFrame(embeddings)

# Save Task 6 embeddings
embedding_df.to_csv(
    "output/task6_embeddings.csv",
    index=False
)

print("=" * 60)
print("EMBEDDING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "Embedding Shape:",
    embedding_df.shape
)

print(
    "Saved: output/task6_embeddings.csv"
)