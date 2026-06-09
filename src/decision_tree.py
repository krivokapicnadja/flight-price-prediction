import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from preprocess import X_test_f as X_test
from preprocess import X_train_f as X_train
from preprocess import X_val_f as X_val
from preprocess import y_test_f as y_test
from preprocess import y_val_f as y_val
from preprocess import y_train_f as y_train

# sa difoltnim vrednostima hiperparametara -  rmse 3625
params = {
    "max_depth": [None, 3, 5, 10, 20],  # maksimalna dubina stabla
    "min_samples_split": [
        2,
        5,
        10,
        20,
    ],  # koliko uzoraka minimalno treba biti u cvoru da bi se dalje delio
    "min_samples_leaf": [
        1,
        2,
        5,
        10,
    ],  # minimalan broj uzoraka u listu - poslednjoj iteraciji
    "max_features": [None, "sqrt", "log2"],  # max broj razmatranih featura pri podeli
}

X_train_full = np.vstack(
    [X_train, X_val]
)  # rez = n_train+n_val, n_features; vstack radi nad 2D nizovima
y_train_full = np.concatenate([y_train, y_val])  # n_train+n_val
# jer grid search u sebi ima implementiranu validaiciju - prosirujem scope podataka
# dtree = DecisionTreeRegressor()

# grid_search = GridSearchCV(  # difoltno radi 5 cross fold validaciju
#     dtree,
#     params,
#     scoring="neg_root_mean_squared_error",  # zato sto scikit learn radi po principu sto veci broj, to bolji
#     verbose=1,  # ispis progressa
# )
# grid_search.fit(X_train_full, y_train_full)

# print("Najbolji parametri:", grid_search.best_params_)
# print("Najbolji RMSE:", -grid_search.best_score_)

# grid searchom je dobijeno sledece:
# Najbolji parametri: {'max_depth': None, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split':20}
# Najbolji RMSE: 2948.0825033462365...... sa tim parametrima pozivamo model nad testnim skupom
dtree = DecisionTreeRegressor(min_samples_leaf=2, min_samples_split=20)
dtree.fit(X_train_full, y_train_full)
y_pred = dtree.predict(X_test)

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
    f"Stablo odlucivanja - regresija: Odnos predvidjene i stvarne cene\nR² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}"
)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "decision_tree_predicted_vs_actual.png"))
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

all_metrics["Decision Tree"] = {"mae": mae, "rmse": rmse, "r2": r2}

with open(METRICS_FILE, "w") as f:
    json.dump(all_metrics, f, indent=2)
