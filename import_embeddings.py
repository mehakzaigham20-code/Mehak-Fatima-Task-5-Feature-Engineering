import pandas as pd
import psycopg2

# ==========================================
# LOAD EMBEDDING CSV
# ==========================================

df = pd.read_csv("output/article_embeddings.csv")

print("=" * 50)
print("Embeddings CSV Loaded Successfully")
print("Shape:", df.shape)
print("=" * 50)


# ==========================================
# POSTGRESQL CONNECTION
# ==========================================

try:
    conn = psycopg2.connect(
        host="localhost",
        database="akc_scraper",
        user="postgres",
        password="5432",
        port="5432"
    )

    print("Connected to PostgreSQL Successfully!")

except Exception as e:
    print("PostgreSQL Connection Failed!")
    print(e)
    exit()


# ==========================================
# RENAME COLUMNS
# ==========================================

df.columns = [
    f"embedding_{i}"
    for i in range(df.shape[1])
]


# ==========================================
# CREATE TABLE
# ==========================================

cursor = conn.cursor()

columns = ", ".join(
    f'"{column}" DOUBLE PRECISION'
    for column in df.columns
)

create_table_query = f"""
CREATE TABLE IF NOT EXISTS article_embeddings (
    id SERIAL PRIMARY KEY,
    {columns}
);
"""

cursor.execute(create_table_query)

conn.commit()

print("Table Created Successfully!")


# ==========================================
# INSERT DATA
# ==========================================

print("Importing embeddings...")

column_names = ", ".join(
    f'"{column}"'
    for column in df.columns
)

placeholders = ", ".join(
    ["%s"] * len(df.columns)
)

insert_query = f"""
INSERT INTO article_embeddings
({column_names})
VALUES ({placeholders})
"""

data = [
    tuple(row)
    for row in df.itertuples(index=False, name=None)
]

cursor.executemany(
    insert_query,
    data
)

conn.commit()

print("Embeddings Imported Successfully!")


# ==========================================
# CLOSE CONNECTION
# ==========================================

cursor.close()
conn.close()


# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 50)
print("IMPORT COMPLETED SUCCESSFULLY")
print("=" * 50)

print("Database :", "akc_scraper")
print("Table    : article_embeddings")
print("Rows     :", len(df))
print("Columns  :", len(df.columns))
print("=" * 50)