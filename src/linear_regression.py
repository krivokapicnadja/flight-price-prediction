import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from preprocess import X_test_lin_scaled as X_test
from preprocess import X_train_lin_scaled as X_train
from preprocess import y_test_lin as y_test
from preprocess import y_train_lin as y_train


regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

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
    f"Linearna regresija: Odnos predvidjene i stvarne cene\nR² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}"
)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "linear_regression_predicted_vs_actual.png"))
plt.close()
