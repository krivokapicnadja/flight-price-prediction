import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from preprocess import X_test_lin_scaled as X_test
from preprocess import X_train_lin_scaled as X_train
from preprocess import X_val_lin_scaled as X_val
from preprocess import y_test_lin as y_test
from preprocess import y_val_lin as y_val
from preprocess import y_train_lin as y_train

# Generiše 50 vrednosti ekvidisstantnte u opsegu 10na -5 do 10na5
opseg_alfa = np.logspace(-5, 5, 50)

greska = {}


for alfa in opseg_alfa:
    ridge = Ridge(alpha=alfa)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_val)

    # koristim rmse za optimizaciju alfa parametra
    greska[alfa] = root_mean_squared_error(y_val, y_pred)

# Ispis 10 najlošijih alfa (najveća greška)
najlosije_alfa = sorted(greska.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n10 najlosijih alfa (najveca RMSE greska):")
for alfa, rmse_val in najlosije_alfa:
    print(f"  alfa = {alfa:.6f}  =>  RMSE = {rmse_val:.4f}")

# Najbolja alfa
najbolje_alfa = min(greska, key=greska.get)
ridge = Ridge(alpha=najbolje_alfa)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

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
    f"Ridge Regresija: Odnos predvidjene i stvarne cene\nR² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}"
)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ridge_predicted_vs_actual.png"))
plt.close()

# zakljucak - zato sto nijedan par mojih atributa nema jaku multikolinearnost - ne zavise jedni od drugih,
# nikad necemo dobiti prevelike koeficijente, i situaciju da se model dvoumi koji koeficijent da iskoristi - ovaj dobijen po jednom ili drugom atributu.
# + imam dovoljno podataka, ridge regresija ne pravi nikakvo poboljsanje u odnosu na baseline model

# vece alfa ce nam samo pogorsati predikciju - implementiram ispis 10 najlosijih alfa i njihov rmse kroz terminal
