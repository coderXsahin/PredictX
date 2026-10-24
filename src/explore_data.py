
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

# 1. Machine failure distribution
failure_counts = df["Machine failure"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
sns.barplot(
    x=["Normal", "Failure"],
    y=[failure_counts.get(0, 0), failure_counts.get(1, 0)],
)
plt.title("Machine Failure Distribution")
plt.ylabel("Number of Records")
plt.xlabel("Machine Status")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "failure_distribution.png", dpi=150)
plt.close()

# 2. Compare sensor measurements by failure class
sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, column in enumerate(sensor_columns):
    sns.boxplot(
        data=df,
        x="Machine failure",
        y=column,
        ax=axes[i],
    )
    axes[i].set_title(column)
    axes[i].set_xlabel("Machine failure (0 = No, 1 = Yes)")

axes[-1].axis("off")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "sensor_comparison.png", dpi=150)
plt.close()

# 3. Numerical feature correlations
numeric_columns = sensor_columns + [
    "Machine failure"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
)
plt.title("Sensor and Failure Correlations")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

# 4. Compare failure rates by machine type
type_failure_rate = (
    df.groupby("Type")["Machine failure"]
    .agg(["sum", "count", "mean"])
    .rename(
        columns={
            "sum": "failures",
            "count": "total_records",
            "mean": "failure_rate",
        }
    )
)

type_failure_rate["failure_rate_percent"] = (
    type_failure_rate["failure_rate"] * 100
).round(2)

print("\n--- FAILURE RATE BY MACHINE TYPE ---")
print(type_failure_rate)

print("\n--- SENSOR STATISTICS BY FAILURE CLASS ---")
print(
    df.groupby("Machine failure")[sensor_columns]
    .mean()
    .round(2)
)

print("\n--- GENERATED REPORTS ---")
for filename in [
    "failure_distribution.png",
    "sensor_comparison.png",
    "correlation_heatmap.png",
]:
    print(REPORTS_DIR / filename)

print("\nEDA completed successfully.")
