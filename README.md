# ⚽ AI FIFA World Cup Predictor

An end-to-end Machine Learning powered football match prediction system that predicts football match outcomes using historical match data, Elo ratings, recent form, and advanced feature engineering.

The project combines:
- 🤖 Machine Learning
- ⚡ FastAPI Backend
- 🎨 Interactive Frontend
- 📊 Football Analytics

---

# 🚀 Features

✅ Predict Home Win / Draw / Away Win probabilities  
✅ Elo rating based prediction system  
✅ Team form & goal difference analysis  
✅ Home advantage support  
✅ Friendly vs Competitive match support  
✅ Real-time prediction API  
✅ Interactive analytics dashboard  
✅ Modern glassmorphism inspired UI  
✅ Probability visualization using Chart.js  

---

# 🧠 Machine Learning Model

The prediction engine uses:

- **Multinomial Logistic Regression**
- **StandardScaler**
- Custom football feature engineering

### Features Used

| Feature | Description |
|---|---|
| `elo_diff` | Elo rating difference between teams |
| `home_form` | Recent form of home team |
| `away_form` | Recent form of away team |
| `home_advantage` | Whether match is played at home |
| `is_friendly` | Friendly or competitive match |
| `home_goal_diff_form` | Goal difference trend of home team |
| `away_goal_diff_form` | Goal difference trend of away team |

---

# 🛠️ Tech Stack

## Backend
- Python
- FastAPI
- Pandas
- Scikit-learn
- Joblib

## Frontend
- HTML5
- Tailwind CSS
- JavaScript
- Chart.js

## Machine Learning
- Logistic Regression
- Feature Engineering
- Data Scaling

---

# 📂 Project Structure

```bash
AI-FIFA-WORLD-CUP-PREDICTOR/
│
├── data/
│   ├── CombinedData.csv
│   ├── ranking_soccer.csv
│   └── results.csv
│
├── frontend/
│   └── index.html
│
├── notebooks/
│
├── src/
│   ├── app.py
│   ├── main.py
│   ├── CombinedData.csv
│   ├── football_model.pkl
│   └── scaler.pkl
│
├── .gitignore
├── LICENSE
└── README.md
