import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from preprocess import X_test_f as X_test
from preprocess import X_train_f as X_train
from preprocess import X_val_f as X_val
from preprocess import y_test_f as y_test
from preprocess import y_val_f as y_val
from preprocess import y_train_f as y_train

# lr i n_est su obrnuto proporcionalni:
#   mali lr → mali udeo prethodnog koraka → treba više stabala
#   veliki lr → veći udeo → treba manje stabala

# pokusaj 1 - dao je najbolje vrednosti za poslednju iteraciju - probati sa vecim vrednostima i vecom dubinom
# n_estimators = [100, 150, 200, 300]
# learning_rate = [0.05, 0.1, 0.2]

# pokusaj 2 - opet underfittuje, ali bolji rezultat
# n_estimators = [300, 400, 500]
# learning_rate = [0.2, 0.3, 0.5] + max_depth = 4

# fiksiracu lr na 0.5 (jako grubo, ali daje najbolje rez), u pokusati sa razlicitim vrednostima estimatora i dubina

# pokusaj 3 - opet je poslednja iteracija najbolja, sa RMSE 2851.. pokusacu jos jednom cisto jer je opet poslednja iteracija najbolja
# max_depth = [5, 6]
# n_estimators = [500, 600, 700]

# max_depth = [7, 8]
# n_estimators = [700, 800, 900]
# #ovo je napokon dalo najbolju kombinaciju parametara - 7 depth, 700 estimatora i RMSE 2808.18
# greska = {}

# # GB koristi PLITKA stabla (max_depth=3 je difolt i standard)
# # min_samples_split=2 i min_samples_leaf=1 su slabi learners — svako stablo hvata mali deo signala - u kombinaciji sa max depth daje plitka stabla
# # subsample=0.8 → stohastički GB, svako stablo vidi 80% nasumičnih redova (regularizacija)
# for dubina in max_depth:
#     for br_stabala in n_estimators:
#         gradboost = GradientBoostingRegressor(
#             n_estimators=br_stabala,
#             learning_rate=0.5,
#             max_depth=dubina,
#             min_samples_split=2,
#             min_samples_leaf=1,
#             subsample=0.8,
#             random_state=42,
#         )
#         gradboost.fit(X_train, y_train)
#         y_pred = gradboost.predict(X_val)

#         greska[(dubina, br_stabala)] = root_mean_squared_error(y_val, y_pred)

# # --- Ispis svih kombinacija ---
# print("\nPretraga hiperparametara za Gradient Boosting")
# print("-" * 55)
# print(f"{'max_depth':<15} {'n_estimators':<15} {'RMSE (val)':<15}")
# print("-" * 55)

# for (depth, n), rmse_val in sorted(greska.items(), key=lambda x: x[1]):
#     print(f"{depth:<15} {n:<15} {rmse_val:<15.2f}")

# print("-" * 55)
# best_hparams = min(greska, key=greska.get)
# print(
#     f"Najbolji:  max_depth={best_hparams[0]},  n_estimators={best_hparams[1]},  "
#     f"RMSE={greska[best_hparams]:.2f}"
# )
# print("-" * 55 + "\n")

# --- Finalni model na testu (odkomentariši kad si zadovoljna hiperparametrima) ---
gradboost = GradientBoostingRegressor(
    n_estimators=700,  # best_params[1]
    learning_rate=0.5,
    max_depth=7,  # best_params[0]
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,
    random_state=42,
)
gradboost.fit(X_train, y_train)
y_pred = gradboost.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test: MAE={mae:.2f}, R²={r2:.4f}, RMSE={rmse:.2f}")

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
    f"Gradient Boosting: Odnos predvidjene i stvarne cene\nR² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}"
)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "gradient_boosting_predicted_vs_actual.png"))
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

all_metrics["Gradient Boosting"] = {"mae": mae, "rmse": rmse, "r2": r2}

with open(METRICS_FILE, "w") as f:
    json.dump(all_metrics, f, indent=2)
