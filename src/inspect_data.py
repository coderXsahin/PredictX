
from pathlib import Path

import pandas as pd

# Locate the CSV relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

df = pd.read_csv(DATA_PATH)

print("\n--- DATASET SHAPE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- MACHINE FAILURE DISTRIBUTION ---")
print(df["Machine failure"].value_counts())
print("\nFailure percentages:")
print(
    (df["Machine failure"].value_counts(normalize=True) * 100).round(2)
)
