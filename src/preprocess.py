import math
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Clean_Dataset.csv")
dataset = pd.read_csv(DATA_PATH)

y = dataset["price"]
X = dataset.drop(columns=["price", "Unnamed: 0", "flight"])

# transformisanjhe kategorijskih u numericke vrednosti

# 1. Definišemo ColumnTransformer
# Parametar 'remainder=passthrough' znači: ostale kolone (Godine, Plata) ne diraj!
transformer_linear = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(),
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

X_train_lin, X_temp_lin, y_train_lin, y_temp_lin = train_test_split(
    X_linear, y, test_size=0.3
)
X_val_lin, X_test_lin, y_val_lin, y_test_lin = train_test_split(
    X_temp_lin, y_temp_lin, test_size=0.5
)

X_train_f, X_temp_f, y_train_f, y_temp_f = train_test_split(X_forest, y, test_size=0.3)
X_val_f, X_test_f, y_val_f, y_test_f = train_test_split(
    X_temp_f, y_temp_f, test_size=0.5
)

# skaliranje samo za skupove za linearne modele
scaler = StandardScaler()

# Poslednje 2 kolone su duration i days_left
X_train_lin_scaled = X_train_lin.copy()
X_train_lin_scaled[:, -2:] = scaler.fit_transform(X_train_lin[:, -2:])

X_val_lin_scaled = X_val_lin.copy()
X_val_lin_scaled[:, -2:] = scaler.transform(X_val_lin[:, -2:])  # transform samo

X_test_lin_scaled = X_test_lin.copy()
X_test_lin_scaled[:, -2:] = scaler.transform(X_test_lin[:, -2:])  # transform samo
