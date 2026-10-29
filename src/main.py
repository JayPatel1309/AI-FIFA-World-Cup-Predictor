import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

data_path = ROOT_DIR / "data" / "CombinedData.csv"
if not data_path.exists():
    data_path = BASE_DIR / "CombinedData.csv"

df = pd.read_csv(data_path)
features = [
    "elo_diff",
    "home_form",
    "away_form",
    "home_advantage",
    "is_friendly",
    "home_goal_diff_form",
    "away_goal_diff_form",
]
X = df[features]
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)
model.fit(X_train_scaled, y_train)

# Save models to src directory
joblib.dump(model, BASE_DIR / "football_model.pkl")
joblib.dump(scaler, BASE_DIR / "scaler.pkl")

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
