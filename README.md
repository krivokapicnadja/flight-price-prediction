# Flight Price Prediction

Aplikacija za predikciju cena avionskih karata korišćenjem mašinskog učenja. Projekat obuhvata celokupan pipeline — od eksploracije podataka i treniranja modela, do deploy-menta putem FastAPI-ja i Streamlit korisničkog interfejsa.

## Tehnologije

- **Python 3.10+**
- **scikit-learn** — treniranje modela (linearna regresija, ridge, random forest, decision tree, gradient boosting, KNR)
- **pandas / numpy / scipy** — obrada podataka
- **matplotlib / seaborn** — vizuelizacija
- **FastAPI** — REST API
- **Streamlit** — web korisnički interfejs
- **uvicorn** — ASGI server
- **joblib** — serijalizacija modela

## Instalacija i pokretanje

### 1. Kloniraj repozitorijum

```bash
git clone https://github.com/krivokapicnadja/flight-price-prediction.git
cd flight-price-prediction
```

### 2. Kreiraj virtuelno okruženje i instaliraj zavisnosti

Sa `pip`:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

Ili sa `uv`:

```bash
uv sync
```

### 3. Postavi dataset

Postavi `Clean_Dataset.csv` u direktorijum `dataset/raw/`.

### 4. Treniraj modele

Pokreni redom:

```bash
uv run src/data_exploration.py  # Eksploratorna analiza
uv run src/preprocess.py        # Preprocesiranje podataka
uv run src/linear_regression.py # Linearna regresija
uv run src/ridge.py             # Ridge regresija
uv run src/random_forest.py     # Random Forest
uv run src/decision_tree.py     # Decision Tree
uv run src/grad_boosting.py     # Gradient Boosting
uv run src/knr.py               # KNR
uv run src/metrics.py           # Evaluacija svih modela
uv run src/important_features.py   #izdvajanje najbitnijih atributa
uv run src/balance_attempt.py #treniranje forest modela nakon balansiranja bitnih atributa
```

### 5. Pokreni API (FastAPI)

```bash
uvicorn app.api:app --reload
```

API će biti dostupan na: `http://127.0.0.1:8000`

Swagger dokumentacija: `http://127.0.0.1:8000/docs`

**POST /predict** — prima JSON sa atributima leta, vraća predikciju cene.

### 6. Pokreni web aplikaciju (Streamlit)

```bash
streamlit run app/ui.py
```

Aplikacija će se otvoriti na: `http://localhost:8501`

Najbolji model se serijalizuje kao `models/deploy.joblib` i koristi se u produkciji, zajedno sa encoderom podataka potrebnim za taj model na `models/encoder.joblib`.
