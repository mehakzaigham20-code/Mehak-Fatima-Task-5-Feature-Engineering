import pandas as pd
import re

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("output/final_articles.csv")

print("Original Shape:", df.shape)

# ==========================
# Remove Duplicate Rows
# ==========================
df.drop_duplicates(inplace=True)

# ==========================
# Fill Missing Values
# ==========================
df["description"] = df["description"].fillna("")
df["published"] = df["published"].fillna("Unknown")

# ==========================
# Text Cleaning Function
# ==========================
def clean_text(text):

    text = str(text)

    # Remove HTML Tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove Special Characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Convert to Lowercase
    text = text.lower()

    # Remove Extra Spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ==========================
# Clean Title
# ==========================
df["title"] = df["title"].apply(clean_text)

# ==========================
# Clean Description
# ==========================
df["description"] = df["description"].apply(clean_text)

# ==========================
# Save Clean Dataset
# ==========================
df.to_csv("output/cleaned_articles.csv", index=False)

print("\nPreprocessing Completed Successfully")
print("Final Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nSaved As: output/cleaned_articles.csv")

# ==========================
# DATA CLEANING REPORT
# ==========================
print("\n==============================")
print("      DATA CLEANING REPORT")
print("==============================")

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# First 5 Records
print("\nFirst 5 Cleaned Records:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe(include="all"))