# AI FIFA World Cup Predictor

A machine learning-based football match outcome prediction system. The application predicts match outcomes (Home Win, Draw, Away Win) using historical international match data, Elo ratings, rolling team form, and match conditions.

---

## Features

- Multi-class match outcome probability prediction (Home Win, Draw, Away Win).
- Dynamic feature modeling based on Elo ratings, team momentum, and goal differentials.
- Contextual match adjustment for neutral venues and match competition level.
- RESTful API developed with FastAPI.
- Interactive web-based dashboard with probability visualization.

---

## Performance & Benchmark Metrics

| Metric | Specification | Details |
|---|---|---|
| **Model Accuracy** | 58.17% | Evaluated on 7,411 test fixtures (70/30 split) across 3 outcomes. Random chance baseline is 33.3%. |
| **Inference Latency** | ~0.40 ms | Per-request feature transformation and inference execution time. |
| **Training Time** | ~2.50 s | Full model training on 17,290 samples using the Scikit-Learn L-BFGS solver. |
| **Dataset Size** | 24,701 rows | Curated feature-engineered dataset (`data/CombinedData.csv`, 24 columns). |
| **Raw Match Records** | 49,287 rows | Historical international match records (`data/results.csv`, 1872–present). |
| **Elo Rankings** | 17,200 rows | Historical international rankings (`data/ranking_soccer_1901-2023.csv`). |
| **Teams Indexed** | 222 teams | Distinct national football teams represented in the system. |
| **Documents Indexed** | N/A | Structured tabular ML system; not an Information Retrieval (IR) / RAG pipeline. |
| **Retrieval Accuracy** | N/A | Exact-match deterministic DataFrame lookup for team stats; no vector search used. |
| **Deployment Profile** | Single-tenant | Local REST service hosted with FastAPI and Uvicorn with asynchronous concurrency. |

### Classification Report

```text
              precision    recall  f1-score   support

    Away Win       0.52      0.60      0.56      2018
        Draw       0.35      0.01      0.02      1782
    Home Win       0.61      0.86      0.72      3611

    accuracy                           0.58      7411
   macro avg       0.50      0.49      0.43      7411
weighted avg       0.53      0.58      0.51      7411
```

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, Tailwind CSS, JavaScript, Chart.js

---

## Requirements

- Python 3.10 or higher
- Modern web browser

Key Python dependencies specified in `requirements.txt`:
- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

---

## Project Structure

```text
AI-FIFA-WORLD-CUP-PREDICTOR/
├── data/
│   ├── CombinedData.csv
│   ├── ranking_soccer_1901-2023.csv
│   └── results.csv
├── frontend/
│   └── index.html
├── notebooks/
│   └── feature_engineering.ipynb
├── src/
│   ├── app.py
│   ├── main.py
│   ├── CombinedData.csv
│   ├── football_model.pkl
│   └── scaler.pkl
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JayPatel1309/AI-FIFA-World-Cup-Predictor.git
cd AI-FIFA-World-Cup-Predictor
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model (Optional)

Pre-trained model artifacts are included in `src/`. To retrain the model:

```bash
python src/main.py
```

### 2. Run the Backend API

Start the FastAPI application with Uvicorn:

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

The API service will be accessible at `http://127.0.0.1:8000`.

### 3. Run the Frontend Dashboard

Open `frontend/index.html` directly in a browser, or serve it locally:

```bash
cd frontend
python -m http.server 3000
```

Access the dashboard at `http://localhost:3000`.

---

## API Reference

Interactive API documentation is accessible at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service health status |
| `GET` | `/teams` | List of available national teams |
| `POST` | `/predict` | Predict match outcome probabilities |

### Sample Prediction Request

**Endpoint:** `POST /predict`

```json
{
  "home_team": "Brazil",
  "away_team": "Argentina",
  "neutral": true,
  "friendly": false
}
```

### Sample Prediction Response

```json
{
  "home_team": "Brazil",
  "away_team": "Argentina",
  "features": {
    "elo_diff": -126.0,
    "home_form": 7,
    "away_form": 12,
    "home_goal_diff_form": 1.0,
    "away_goal_diff_form": 1.8
  },
  "probabilities": {
    "away_win": 35.17,
    "draw": 30.61,
    "home_win": 34.22
  }
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
