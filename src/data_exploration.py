"""
=============================================================================
DATA EXPLORATION — Faza 1: Inicijalno upoznavanje sa skupom podataka
=============================================================================
Cilj:  Učitavanje, osnovna statistika, distribucija ciljne promenljive (Price),
       analiza kategorijskih i numeričkih varijabli, korelaciona analiza
       i detekcija outlier-a.  Sve slike se čuvaju u results/figures/.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # ne otvara prozor — samo čuva fajlove
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ==========================================================================
#  PODEŠAVANJA
# ==========================================================================
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "raw", "Clean_Dataset.csv")
FIG_DIR = os.path.join(BASE_DIR, "results", "exploration")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = "Set2"


def save_fig(name: str, dpi: int = 200) -> str:
    """Sačuvaj trenutni matplotlib figure, zatvori ga i vrati putanju."""
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


# ==========================================================================
#  UČITAVANJE PODATAKA
# ==========================================================================
print("=" * 72)
print(" UČITAVANJE PODATAKA")
print("=" * 72)

df = pd.read_csv(DATA_PATH)

# Uklanjanje kolone sa starim indeksom ako postoji
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
    # inplace - radi direktno nad mojim podacima, ne pravi kopiju

# Standardizacija imena kolona (strip + lower)
df.columns = df.columns.str.strip().str.lower()

print(f"Oblik skupa:    {df.shape[0]} redova × {df.shape[1]} kolona")
print(f"Duplikati:      {df.duplicated().sum()} reda")


# ==========================================================================
# 0. OSNOVNE INFORMACIJE O KOLONAMA
# ==========================================================================
print("\n" + "=" * 72)
print(" OSNOVNE INFORMACIJE")
print("=" * 72)

print("\n--- dtypes i broj ne-NaN vrednosti ---")
df.info()

print("\n--- Nedostajuće vrednosti ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("  (nema nedostajućih vrednosti)")
else:
    print(missing)

print("\n--- Broj jedinstvenih vrednosti po koloni ---")
for col in df.columns:
    n = df[col].nunique()
    info = ""
    if n <= 20:
        info = f"  →  {sorted(df[col].dropna().unique())}"
    print(f"  {col:<25s} {n:>5d} {info}")
    # ako je kolona numericka, kao price, nece ispisati sve cene
    # vec samo broj jedinstvenih. za kategoricke ispisuje sve


# ==========================================================================
# 1. DISTRIBUCIJA CILJNE PROMENLJIVE — PRICE
# ==========================================================================
print("\n" + "=" * 72)
print("  1. CILJNA PROMENLJIVA — PRICE")
print("=" * 72)

p = df["price"]
print(f"Min       : {p.min():>12,.0f}")
print(f"Max       : {p.max():>12,.0f}")
print(f"Mean      : {p.mean():>12,.2f}")
print(f"Median    : {p.median():>12,.0f}")
print(f"Std       : {p.std():>12,.2f}")

# Cena po klasama
print("\n--- Cena po klasi ---")
print(df.groupby("class")["price"].describe().round(2).to_string())

plt.figure(figsize=(9, 5.2))

sns.histplot(p, kde=True, bins=50, color=sns.color_palette(PALETTE)[0])

plt.title("Histogram cene", fontweight="bold")
print(f"\n→ Grafik:  {save_fig('01_price_distribution.png')}")

# ==========================================================================
# 2. KATEGORIJSKE VARIJABLE
# ==========================================================================
print("\n" + "=" * 72)
print("  2.  KATEGORIJSKE VARIJABLE")
print("=" * 72)

cat_cols = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]

fig, axes = plt.subplots(4, 2, figsize=(16, 22))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    counts = df[col].value_counts()  # koliko koje vrednosti ima u svakoj koloni

    sns.countplot(data=df, x=col, order=counts.index, palette=PALETTE, ax=axes[i])
    axes[i].set_title(f"Raspodela: {col}", fontweight="bold")
    axes[i].tick_params(axis="x", rotation=30)

# Sakrij prazan subplot (7 kolona, 4×2 = 8 slotova, jedan viška)
axes[-1].set_visible(False)

print(f"\n→ Grafik:  {save_fig('02_categorical_distributions.png')}")

# ==========================================================================
# 3. KATEGORIJSKE VARIJABLE vs CENA (boxplot)
# ==========================================================================
print("\n" + "=" * 72)
print("  3.  KATEGORIJSKE VARIJABLE vs CENA")
print("=" * 72)

fig, axes = plt.subplots(4, 2, figsize=(16, 22))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.boxplot(data=df, x=col, y="price", palette=PALETTE, ax=axes[i])
    axes[i].set_title(f"Cena po: {col}", fontweight="bold")
    axes[i].tick_params(axis="x", rotation=30)

axes[-1].set_visible(False)

print(f"\n→ Grafik:  {save_fig('03_categorical_vs_price.png')}")

# ==========================================================================
# 4. NUMERIČKE VARIJABLE
# ==========================================================================
print("\n" + "=" * 72)
print("  4.  NUMERIČKE VARIJABLE")
print("=" * 72)

num_cols = ["duration", "days_left"]

for col in num_cols:
    s = df[col]
    print(f"\n--- {col} ---")
    print(
        f"  Min={s.min():.2f}  Max={s.max():.2f}  Mean={s.mean():.2f}  "
        f"Median={s.median():.2f}  Std={s.std():.2f}  "
        # f"Skew={s.skew():.4f}  Kurt={s.kurtosis():.4f}"
    )

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for i, col in enumerate(num_cols):
    # Histogram
    sns.histplot(
        df[col], kde=True, bins=40, color=sns.color_palette(PALETTE)[i], ax=axes[i]
    )
    axes


print(f"\n→ Grafik:  {save_fig('04_numerical_analysis.png')}")

# ==========================================================================
# 5. NUMERIČKE VARIJABLE vs CENA
# ==========================================================================
print("\n" + "=" * 72)
print("  5.  NUMERIČKE VARIJABLE vs CENA")
print("=" * 72)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, col in enumerate(num_cols):
    sns.scatterplot(data=df, x=col, y="price", alpha=0.4, ax=axes[i])
    sns.regplot(
        data=df,
        x=col,
        y="price",
        scatter=False,
        color="red",
        line_kws={"linewidth": 2},
        ax=axes[i],
    )
    axes[i].set_title(f"Cena vs {col}", fontweight="bold")

    corr_val = df[col].corr(df["price"])
    axes[i].text(
        0.95,
        0.95,
        f"r = {corr_val:.3f}",  # koef korelacije
        transform=axes[i].transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

print(f"\n→ Grafik:  {save_fig('05_numerical_vs_price.png')}")

# ==========================================================================
# 6. KORELACIONA MATRICA (svi atributi)
# ==========================================================================
print("\n" + "=" * 72)
print("  6.  KORELACIONA MATRICA (svi atributi)")
print("=" * 72)

# Enkodiranje kategoričkih kolona da bi mogle u korelacionu matricu
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = pd.factorize(df_encoded[col])[0]

corr_matrix = df_encoded.corr()

plt.figure(figsize=(14, 11))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".3f",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=1,
)
plt.title("Korelaciona matrica (svi atributi)", fontweight="bold")
print(f"\n→ Grafik:  {save_fig('06_correlation_heatmap.png')}")
