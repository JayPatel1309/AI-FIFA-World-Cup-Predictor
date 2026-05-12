# ⚽ AI FIFA World Cup Predictor

An AI-powered football match prediction system that uses **Elo Ratings, team form, goal statistics, and home advantage** to predict international football match outcomes and simulate FIFA World Cup scenarios.

Built using **Python, Pandas, Scikit-learn/XGBoost, and Streamlit**, this project combines sports analytics with machine learning to create a realistic football prediction engine.

---

## 🚀 Features

* 📊 **Elo Rating Integration**
  Uses international team Elo ratings to measure team strength dynamically.

* 🔥 **Recent Form Analysis**
  Calculates performance based on the last 5 matches.

* ⚽ **Goal Statistics**
  Tracks average goals scored and conceded.

* 🏠 **Home Advantage Detection**
  Includes home/away/neutral venue impact.

* 🤖 **Machine Learning Predictions**
  Predicts:

  * Win
  * Draw
  * Loss

* 🌍 **World Cup Match Simulation**
  Simulate tournament outcomes using trained models.

* 📈 **Data Visualization Dashboard**
  Interactive charts and analytics using Streamlit.

---

# 🧠 Tech Stack

### Languages & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Streamlit

### Data Sources

* International Football Match Results
* Elo Ratings Dataset

---

# 📂 Project Structure

```bash
worldcup-predictor/
│
├── data/
│   ├── raw_matches.csv
│   ├── elo_ratings.csv
│   └── final_dataset.csv
│
├── notebooks/
│   └── feature_engineering.ipynb
│
├── src/
│   ├── scraper.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── worldcup_model.pkl
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Features

| Feature            | Description            |
| ------------------ | ---------------------- |
| elo_team           | Team Elo rating        |
| elo_opp            | Opponent Elo rating    |
| elo_diff           | Elo difference         |
| recent_form        | Last 5 matches form    |
| avg_goals_scored   | Average goals scored   |
| avg_goals_conceded | Average goals conceded |
| venue              | Home/Away/Neutral      |
| win_streak         | Consecutive wins       |

---

# 🎯 Machine Learning Goal

Predict the probability of:

* ✅ Team Win
* 🤝 Draw
* ❌ Team Loss

using historical international football data.

---

# 🛠️ Installation

```bash
git clone https://github.com/yourusername/worldcup-predictor.git

cd worldcup-predictor

pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Train the Model

```bash
python src/train.py
```

## Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

# 📈 Future Improvements

* Deep Learning Match Predictor
* Live API Integration
* Player-level statistics
* Injury & lineup analysis
* Betting odds comparison
* Tournament bracket simulation
* Real-time prediction dashboard

---

# 🌟 Why This Project?

This project demonstrates:

* Machine Learning
* Feature Engineering
* Data Cleaning
* Sports Analytics
* Model Deployment
* End-to-End AI Workflow

Perfect for:

* AI/ML portfolios
* GitHub showcases
* Resume projects
* Hackathons
* Sports analytics research

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
