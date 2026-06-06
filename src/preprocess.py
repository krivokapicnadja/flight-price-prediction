import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "raw", "Clean_Dataset.csv")
CACHE_DIR = os.path.join(BASE_DIR, "dataset", "processed")
CACHE_NPZ = os.path.join(CACHE_DIR, "preprocessed_data.npz")
CACHE_SCALER = os.path.join(CACHE_DIR, "scaler.pkl")
os.makedirs(CACHE_DIR, exist_ok=True)


def _load_from_cache_or_compute():
    """
    Load preprocessed data from a single compressed .npz file if it exists.
    Otherwise, run the full preprocessing pipeline and save to cache.
    Returns a dict with all exported variables.
    """
    if os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_SCALER):
        # Učitaj iz keša (gotovo trenutno)
        cached = np.load(CACHE_NPZ, allow_pickle=True)
        data = {key: cached[key] for key in cached.files}
        data["scaler"] = joblib.load(CACHE_SCALER)
        return data

    # ---- Prvi put: izvrši ceo preprocessing ----
    dataset = pd.read_csv(DATA_PATH)
    y = dataset["price"]
    X = dataset.drop(columns=["price", "Unnamed: 0", "flight"])

    # OneHotEncoder za linearne modele
    transformer_linear = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(sparse_output=False),
                [
                    "airline",
                    "source_city",
                    "departure_time",
                    "stops",
                    "arrival_time",
                    "destination_city",
                    "class",
                ],
            )
        ],
        remainder="passthrough",
    )
    X_linear = transformer_linear.fit_transform(X)

    # OrdinalEncoder za forest modele
    transformer_forest = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(),
                [
                    "airline",
                    "source_city",
                    "departure_time",
                    "stops",
                    "arrival_time",
                    "destination_city",
                    "class",
                ],
            )
        ],
        remainder="passthrough",
    )
    X_forest = transformer_forest.fit_transform(X)

    # Train / validation / test split
    X_train_lin, X_temp_lin, y_train_lin, y_temp_lin = train_test_split(
        X_linear, y, test_size=0.3, random_state=42
    )
    X_val_lin, X_test_lin, y_val_lin, y_test_lin = train_test_split(
        X_temp_lin, y_temp_lin, test_size=0.5, random_state=42
    )

    X_train_f, X_temp_f, y_train_f, y_temp_f = train_test_split(
        X_forest, y, test_size=0.3, random_state=42
    )
    X_val_f, X_test_f, y_val_f, y_test_f = train_test_split(
        X_temp_f, y_temp_f, test_size=0.5, random_state=42
    )

    # Skaliranje za linearne modele (samo poslednje 2 numeričke kolone: duration, days_left)
    scaler = StandardScaler()

    X_train_lin_scaled = X_train_lin.copy()
    X_train_lin_scaled[:, -2:] = scaler.fit_transform(X_train_lin[:, -2:])

    X_val_lin_scaled = X_val_lin.copy()
    X_val_lin_scaled[:, -2:] = scaler.transform(X_val_lin[:, -2:])

    X_test_lin_scaled = X_test_lin.copy()
    X_test_lin_scaled[:, -2:] = scaler.transform(X_test_lin[:, -2:])

    # Build result dict
    data = {
        "X_train_lin_scaled": X_train_lin_scaled,
        "X_val_lin_scaled": X_val_lin_scaled,
        "X_test_lin_scaled": X_test_lin_scaled,
        "y_train_lin": y_train_lin.to_numpy(),
        "y_val_lin": y_val_lin.to_numpy(),
        "y_test_lin": y_test_lin.to_numpy(),
        "X_train_f": X_train_f,
        "X_val_f": X_val_f,
        "X_test_f": X_test_f,
        "y_train_f": y_train_f.to_numpy(),
        "y_val_f": y_val_f.to_numpy(),
        "y_test_f": y_test_f.to_numpy(),
        "scaler": scaler,
    }

    # Sačuvaj sve u jedan .npz fajl (bez kompresije — brže učitavanje)
    save_data = {k: v for k, v in data.items() if k != "scaler"}
    np.savez(CACHE_NPZ, **save_data)
    joblib.dump(scaler, CACHE_SCALER)

    return data


# Učitaj (ili preračunaj) sve podatke
_data = _load_from_cache_or_compute()

# Eksportuj promenljive na nivou modula
X_train_lin_scaled = _data["X_train_lin_scaled"]
X_val_lin_scaled = _data["X_val_lin_scaled"]
X_test_lin_scaled = _data["X_test_lin_scaled"]
y_train_lin = _data["y_train_lin"]
y_val_lin = _data["y_val_lin"]
y_test_lin = _data["y_test_lin"]
X_train_f = _data["X_train_f"]
X_val_f = _data["X_val_f"]
X_test_f = _data["X_test_f"]
y_train_f = _data["y_train_f"]
y_val_f = _data["y_val_f"]
y_test_f = _data["y_test_f"]
scaler = _data["scaler"]
