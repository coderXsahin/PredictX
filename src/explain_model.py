
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "predictx_model.joblib"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

print("\n--- TOP 10 FEATURE IMPORTANCES ---")
print(importance_df.head(10).to_string(index=False))

csv_path = REPORTS_DIR / "feature_importance.csv"
importance_df.to_csv(csv_path, index=False)

top_features = importance_df.head(10).sort_values("importance")

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Random Forest feature importance")
plt.title("PredictX — Top 10 Important Features")
plt.tight_layout()

chart_path = REPORTS_DIR / "feature_importance.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print("\nSaved:", csv_path)
print("Saved:", chart_path)
