# ⚽ AI FIFA World Cup Predictor

An end-to-end Machine Learning powered football match prediction system that predicts the probability of a **Home Win**, **Draw**, or **Away Win** using historical football analytics and Elo-based performance metrics.

Built with:
- 🤖 Machine Learning
- ⚡ FastAPI
- 🎨 Modern Frontend UI
- 📊 Football Analytics

---

## 🚀 Features

- Predict football match outcomes in real time
- Probability prediction for:
  - Home Win
  - Draw
  - Away Win
- Elo rating based analysis
- Team form & goal difference evaluation
- Home advantage support
- Friendly vs Competitive match support
- Interactive charts and analytics dashboard
- Glassmorphism inspired modern UI

---

## 🧠 Machine Learning Model

The project uses:

- **Multinomial Logistic Regression**
- **StandardScaler**
- Feature engineered football analytics

### Features Used

| Feature | Description |
|---|---|
| elo_diff | Elo rating difference |
| home_form | Recent home team form |
| away_form | Recent away team form |
| home_advantage | Venue advantage |
| is_friendly | Match type |
| home_goal_diff_form | Home goal trend |
| away_goal_diff_form | Away goal trend |

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI
- Pandas
- Scikit-learn
- Joblib

### Frontend
- HTML
- Tailwind CSS
- JavaScript
- Chart.js

### ML Techniques
- Logistic Regression
- Feature Engineering
- Data Scaling

---

## 📂 Project Structure

```bash
AI-FIFA-World-Cup-Predictor/
│
├── app.py
├── main.py
├── football_model.pkl
├── scaler.pkl
├── CombinedData.csv
├── feature_engineering.ipynb
├── index.html
│
└── README.md
