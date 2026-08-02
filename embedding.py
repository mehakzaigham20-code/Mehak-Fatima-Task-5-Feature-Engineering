import pandas as pd
from sentence_transformers import SentenceTransformer

# Load cleaned dataset
df = pd.read_csv("output/cleaned_articles.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# Combine title and description
df["text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("")
)

# Load pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nGenerating Embeddings...")

# Generate embeddings
embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True
)

print("Embedding Shape:", embeddings.shape)

# Convert embeddings to DataFrame
embedding_df = pd.DataFrame(embeddings)

# Save embeddings
embedding_df.to_csv(
    "output/article_embeddings.csv",
    index=False
)

print("\nEmbeddings Generated Successfully!")
print("Saved File: output/article_embeddings.csv")