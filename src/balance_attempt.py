# iz important_features.py je zakljuceno da class ima najveci uticaj na cenu.
# buduci da bussines ima duplo manje od economy klase, probacemo da ih izbalansiramo
# i tako probamo istrenirati model

# probacu sa svim tree modelima jer daju slicne metrike
import pandas as pd
import os
import joblib
from sklearn.utils import resample
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "raw", "Clean_Dataset.csv")

dataset = pd.read_csv(DATA_PATH)

# Odvoji Economy i Business - ovo su delovi dataseta po klasama odvojeni, ali citavi, sa svim atributima
dataset_economy = dataset[dataset["class"] == "Economy"]  # ~66% uzoraka
dataset_business = dataset[dataset["class"] == "Business"]  # ~33% uzoraka

# Oversample Business na broj Economy uzoraka (balansira samo Business)
dataset_business_upsampled = resample(
    dataset_business, replace=True, n_samples=len(dataset_economy), random_state=42
)
# Spoji ponovo
dataset_balanced_classes = pd.concat([dataset_economy, dataset_business_upsampled])

X = dataset_balanced_classes.drop(columns=["price", "Unnamed: 0", "flight"])
y = dataset_balanced_classes["price"]

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

# Train / test split - necu raditi validaciju jer su mi svi hiperparametri vec podeseni
X_train, X_test, y_train, y_test = train_test_split(
    X_forest, y, test_size=0.3, random_state=42
)

models = {
    "Decision Tree": DecisionTreeRegressor(min_samples_leaf=2, min_samples_split=20),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,  # najbolji_estimator
        min_samples_leaf=2,
        min_samples_split=20,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=700,  # best_params[1]
        learning_rate=0.5,
        max_depth=7,  # best_params[0]
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42,
    ),
}

trained_models = {}
results = []

for model_name, regressor in models.items():
    print("-------------")
    print(model_name)
    print("-------------")

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    trained_models[model_name] = regressor

    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("RMSE: ", root_mean_squared_error(y_test, y_pred))
    print("R2 score: ", r2_score(y_test, y_pred))
    results.append(
        {
            "model": model_name,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred),
            "r2 score": r2_score(y_test, y_pred),
        }
    )
results_data = pd.DataFrame(results)
results_data = results_data.sort_values("r2 score", ascending=False)

# --- Čuvanje metrika u CSV ---
METRICS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics"
)
os.makedirs(METRICS_DIR, exist_ok=True)
METRICS_FILE = os.path.join(METRICS_DIR, "balanced_metrics.csv")

results_data.to_csv(METRICS_FILE, index=False)

# nakon prvog pokretanja, vidim da je sa ovako izbalansiranim klasama bolji rezultat nego bez
# najbolji rmse ima gradient boosting, to uzimam kao merodavnu metriku jer je veca od mae - znaci da taj model najbolje koriguje velike greske tj outliere
# cuvacu gradient boosting kao najbolji model za deployment

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(
    trained_models["Gradient Boosting"], os.path.join(MODELS_DIR, "deploy.joblib")
)
joblib.dump(
    transformer_forest, os.path.join(MODELS_DIR, "encoder.joblib")
)  # cuvam encoder zbog apija
