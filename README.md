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
