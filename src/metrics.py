import pandas as pd
import json
import os

METRICS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics"
)
METRICS_FILE = os.path.join(METRICS_DIR, "all_metrics.json")

if not os.path.exists(METRICS_FILE):
    print(f"Nema fajla {METRICS_FILE}. Pokreni modele prvo.")
    exit(1)

with open(METRICS_FILE, "r") as f:
    all_metrics = json.load(f)

rows = []
for model_name, m in all_metrics.items():
    rows.append(
        {
            "Model": model_name,
            "MAE": round(m["mae"], 2),
            "RMSE": round(m["rmse"], 2),
            "R\u00b2": round(m["r2"], 4),
        }
    )

df = pd.DataFrame(rows)
df = df.sort_values("R\u00b2", ascending=False)

# Čuvanje tabele kao CSV
csv_path = os.path.join(METRICS_DIR, "all_metrics.csv")
df.to_csv(csv_path, index=False)
