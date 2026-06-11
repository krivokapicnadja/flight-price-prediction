import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from preprocess import X_test_f as X_test
from preprocess import X_train_f as X_train
from preprocess import X_val_f as X_val
from preprocess import y_test_f as y_test
from preprocess import y_val_f as y_val
from preprocess import y_train_f as y_train

# optimalne parametre sa decision treeja cu preslikati ovde, i kroz for petlju samo ispitati vrednost
# optimalnog n estimators

n_estimators = [50, 100, 150, 200]
greska = {}

# zasto ne radim grid search:
# Svaki fit je __Random Forest__ koji gradi `n_estimators` stabala. U najgorem slučaju (`n_estimators=300`) to je 300 stabala po fitovanju. Prosečno ~160 stabala po fitu × 400 fitova = __~64.000 stabala__ ukupno. To traje satima.
# Poređenja radi, Decision Tree GridSearch je imao samo 16 kombinacija × 5 CV = 80 fitova (i to sa jednim stablom po fitu).

for br_stabala in n_estimators:
    randtree = RandomForestRegressor(
        n_estimators=br_stabala, min_samples_leaf=2, min_samples_split=20
    )
    randtree.fit(X_train, y_train)
    y_pred = randtree.predict(X_val)

    # koristim rmse za optimizaciju sused parametra
    greska[br_stabala] = root_mean_squared_error(y_val, y_pred)

# Najbolji broj estimatora
najbolji_estimator = min(greska, key=greska.get)
print(greska[najbolji_estimator], najbolji_estimator)

# 2783.787646741996 150 --> RMSE sa najboljim brojem estimatora 150
# znam da nije usao u underfitting jer mu je najbolje rmse sa hiperparametrom koji nije poslednja vrednost iteracije
randtree = RandomForestRegressor(
    n_estimators=najbolji_estimator, min_samples_leaf=2, min_samples_split=20
)
randtree.fit(X_train, y_train)
y_pred = randtree.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mae, r2, rmse)

# Predicted vs Actual Prices plot
FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures"
)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="white")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Savrsena predikcija",
)
plt.xlabel("Stvarna cena")
plt.ylabel("Predvidjena cena")
plt.title(
    f"Random forest: Odnos predvidjene i stvarne cene\nR² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}"
)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "random_forest_predicted_vs_actual.png"))
plt.close()

# --- Čuvanje metrika u zajednički JSON ---
METRICS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics"
)
os.makedirs(METRICS_DIR, exist_ok=True)
METRICS_FILE = os.path.join(METRICS_DIR, "all_metrics.json")

if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r") as f:
        all_metrics = json.load(f)
else:
    all_metrics = {}

all_metrics["Random Forest"] = {"mae": mae, "rmse": rmse, "r2": r2}

with open(METRICS_FILE, "w") as f:
    json.dump(all_metrics, f, indent=2)
