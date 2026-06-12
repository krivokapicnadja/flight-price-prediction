# __Feature importance jednog stabla__ = 
# ukupno (sumarno) smanjenje impurity-a koje je taj feature doneo kroz __sve splitove__ u celom stablu,
# __normalizovano__ tako da zbir svih importanci bude 1.

#racuna se tako sto se nadje MSE za svaki cvor u stablu gde se dati feature
#koristi za split, pomnozi se sa brojem uzoraka koji su prosli kroz cvor
#saberu se sve te vrednosti i normalizuju na max 1

#zakljucak ove skriptee: ubedljivo najveci importance ima class atribut


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DTREE_PATH = MODELS_DIR / "decision_tree.joblib"
RFOREST_PATH = MODELS_DIR / "random_forest.joblib"
GRADBOOST_PATH = MODELS_DIR / "gradboost.joblib"

FEATURE_NAMES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
]

dtree = joblib.load(DTREE_PATH)
rforest = joblib.load(RFOREST_PATH)
gradboost = joblib.load(GRADBOOST_PATH)

dtree_importances = dtree.feature_importances_
rforest_importances = rforest.feature_importances_
gradboost_importances = gradboost.feature_importances_


# sortiran prikaz i bar-plot
def display_and_plot(importances, model_name, filename):

    # Sortiraj opadajuće
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n{'=' * 60}")
    print(f"  FEATURE IMPORTANCE — {model_name}")
    print(f"{'=' * 60}")
    print(f"{'Feature':<22s} {'Importance':>10s}")
    print(f"{'-' * 32}")
    for idx in sorted_idx:
        print(f"  {FEATURE_NAMES[idx]:<20s} {importances[idx]:>10.4f}")

    # Bar-plot
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(FEATURE_NAMES))
    ax.barh(
        y_pos,
        importances[sorted_idx],
        color="steelblue",
        edgecolor="white",
        height=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx])
    ax.invert_yaxis()  # najvažniji na vrhu
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}", fontweight="bold")
    plt.tight_layout()
    filepath = FIG_DIR / filename
    plt.savefig(filepath, dpi=200)
    plt.close()

display_and_plot(
    dtree_importances, "Decision Tree", "feature_importance_decision_tree.png"
)
display_and_plot(
    rforest_importances, "Random Forest", "feature_importance_random_forest.png"
)
display_and_plot(
    gradboost_importances, "Gradient Boosting", "feature_importance_grad_boosting.png"
)