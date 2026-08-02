import pandas as pd
import random
import re
import math
from collections import Counter

# ==========================================
# LOAD PREPROCESSED DATASET
# ==========================================

input_file = "output/cleaned_articles.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("PREPROCESSED DATASET LOADED")
print("=" * 60)
print("Original Shape:", df.shape)


# ==========================================
# KEEP TITLE COLUMN
# ==========================================

feature_df = df[["title"]].copy()


# ==========================================
# RANDOM NUMBER OF CLICKS
# 5,000 - 100,000
# ==========================================

random.seed(42)

feature_df["no_of_clicks"] = [
    random.randint(5000, 100000)
    for _ in range(len(feature_df))
]


# ==========================================
# BASIC TITLE FEATURES
# ==========================================

# Title Length
feature_df["title_length"] = feature_df["title"].apply(
    lambda x: len(str(x))
)

# Word Count
feature_df["word_count"] = feature_df["title"].apply(
    lambda x: len(str(x).split())
)


# Average Word Length
def average_word_length(title):

    words = str(title).split()

    if not words:
        return 0

    return round(
        sum(len(word) for word in words) / len(words),
        2
    )


feature_df["avg_word_length"] = feature_df["title"].apply(
    average_word_length
)


# Uppercase Count
feature_df["uppercase_count"] = feature_df["title"].apply(
    lambda x: sum(
        1 for char in str(x)
        if char.isupper()
    )
)


# Digit Count
feature_df["digit_count"] = feature_df["title"].apply(
    lambda x: sum(
        1 for char in str(x)
        if char.isdigit()
    )
)


# Special Character Count
feature_df["special_char_count"] = feature_df["title"].apply(
    lambda x: len(
        re.findall(
            r"[^a-zA-Z0-9\s]",
            str(x)
        )
    )
)


# ==========================================
# NEW FEATURE 1
# UNIQUE WORD COUNT
# ==========================================

def unique_word_count(title):

    words = str(title).lower().split()

    return len(set(words))


feature_df["unique_word_count"] = feature_df["title"].apply(
    unique_word_count
)


# ==========================================
# NEW FEATURE 2
# LEXICAL DIVERSITY
# ==========================================

def lexical_diversity(title):

    words = str(title).lower().split()

    if not words:
        return 0

    return round(len(set(words)) / len(words), 3)


feature_df["lexical_diversity"] = feature_df["title"].apply(
    lexical_diversity
)


# ==========================================
# NEW FEATURE 3
# LONG WORD COUNT
# Words containing 8+ characters
# ==========================================

def long_word_count(title):

    words = str(title).split()

    return sum(
        1 for word in words
        if len(word) >= 8
    )


feature_df["long_word_count"] = feature_df["title"].apply(
    long_word_count
)


# ==========================================
# NEW FEATURE 4
# SHORT WORD COUNT
# Words containing 3 or fewer characters
# ==========================================

def short_word_count(title):

    words = str(title).split()

    return sum(
        1 for word in words
        if len(word) <= 3
    )


feature_df["short_word_count"] = feature_df["title"].apply(
    short_word_count
)


# ==========================================
# NEW FEATURE 5
# REPEATED WORD COUNT
# ==========================================

def repeated_word_count(title):

    words = str(title).lower().split()

    counts = Counter(words)

    return sum(
        1 for word, count in counts.items()
        if count > 1
    )


feature_df["repeated_word_count"] = feature_df["title"].apply(
    repeated_word_count
)


# ==========================================
# STOP WORD LIST
# ==========================================

stop_words = {
    "the", "a", "an", "and", "or",
    "but", "if", "in", "on", "at",
    "to", "for", "of", "with",
    "from", "by", "is", "are",
    "was", "were", "this", "that",
    "as", "it", "be", "has",
    "have", "had"
}


# ==========================================
# NEW FEATURE 6
# STOP WORD COUNT
# ==========================================

def stopword_count(title):

    words = str(title).lower().split()

    return sum(
        1 for word in words
        if word in stop_words
    )


feature_df["stopword_count"] = feature_df["title"].apply(
    stopword_count
)


# ==========================================
# NEW FEATURE 7
# STOP WORD RATIO
# ==========================================

def stopword_ratio(title):

    words = str(title).lower().split()

    if not words:
        return 0

    count = sum(
        1 for word in words
        if word in stop_words
    )

    return round(count / len(words), 3)


feature_df["stopword_ratio"] = feature_df["title"].apply(
    stopword_ratio
)


# ==========================================
# NEW FEATURE 8
# VOWEL COUNT
# ==========================================

def vowel_count(title):

    return sum(
        1 for char in str(title).lower()
        if char in "aeiou"
    )


feature_df["vowel_count"] = feature_df["title"].apply(
    vowel_count
)


# ==========================================
# NEW FEATURE 9
# VOWEL RATIO
# ==========================================

def vowel_ratio(title):

    text = str(title).lower()

    letters = [
        char for char in text
        if char.isalpha()
    ]

    if not letters:
        return 0

    vowels = sum(
        1 for char in letters
        if char in "aeiou"
    )

    return round(vowels / len(letters), 3)


feature_df["vowel_ratio"] = feature_df["title"].apply(
    vowel_ratio
)


# ==========================================
# NEW FEATURE 10
# CONSONANT COUNT
# ==========================================

def consonant_count(title):

    text = str(title).lower()

    return sum(
        1 for char in text
        if char.isalpha() and char not in "aeiou"
    )


feature_df["consonant_count"] = feature_df["title"].apply(
    consonant_count
)


# ==========================================
# NEW FEATURE 11
# CONSONANT RATIO
# ==========================================

def consonant_ratio(title):

    text = str(title).lower()

    letters = [
        char for char in text
        if char.isalpha()
    ]

    if not letters:
        return 0

    consonants = sum(
        1 for char in letters
        if char not in "aeiou"
    )

    return round(consonants / len(letters), 3)


feature_df["consonant_ratio"] = feature_df["title"].apply(
    consonant_ratio
)


# ==========================================
# NEW FEATURE 12
# QUESTION MARK COUNT
# ==========================================

feature_df["question_mark_count"] = feature_df["title"].apply(
    lambda x: str(x).count("?")
)


# ==========================================
# NEW FEATURE 13
# EXCLAMATION MARK COUNT
# ==========================================

feature_df["exclamation_mark_count"] = feature_df["title"].apply(
    lambda x: str(x).count("!")
)


# ==========================================
# NEW FEATURE 14
# TITLE ENTROPY
# Measures character diversity
# ==========================================

def title_entropy(title):

    text = str(title).lower()

    if not text:
        return 0

    counts = Counter(text)
    total = len(text)

    entropy = 0

    for count in counts.values():

        probability = count / total

        entropy -= probability * math.log2(probability)

    return round(entropy, 3)


feature_df["title_entropy"] = feature_df["title"].apply(
    title_entropy
)


# ==========================================
# NEW FEATURE 15
# UNIQUE CHARACTER COUNT
# ==========================================

feature_df["unique_character_count"] = feature_df["title"].apply(
    lambda x: len(set(str(x)))
)


# ==========================================
# NEW FEATURE 16
# DIGIT RATIO
# ==========================================

def digit_ratio(title):

    text = str(title)

    if not text:
        return 0

    digits = sum(
        1 for char in text
        if char.isdigit()
    )

    return round(digits / len(text), 3)


feature_df["digit_ratio"] = feature_df["title"].apply(
    digit_ratio
)


# ==========================================
# NEW FEATURE 17
# WORD LENGTH VARIATION
# ==========================================

def word_length_variation(title):

    words = str(title).split()

    if len(words) <= 1:
        return 0

    lengths = [len(word) for word in words]

    mean_length = sum(lengths) / len(lengths)

    variance = sum(
        (length - mean_length) ** 2
        for length in lengths
    ) / len(lengths)

    return round(math.sqrt(variance), 2)


feature_df["word_length_variation"] = feature_df["title"].apply(
    word_length_variation
)


# ==========================================
# SAVE FEATURE ENGINEERED DATASET
# ==========================================

output_file = "output/feature_engineered_articles.csv"

feature_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# REPORT
# ==========================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Final Shape:", feature_df.shape)

print("\nFeatures Created:")

for column in feature_df.columns:
    print("-", column)

print("\nFirst 5 Records:")
print(feature_df.head())

print("\nSaved As:")
print(output_file)

print("=" * 60)