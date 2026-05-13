import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib


df=pd.read_csv(r"data\CombinedData.csv")
features=[ "elo_diff",
    "home_form",
    "away_form",
    "home_advantage",
    "is_friendly",
    "home_goal_diff_form",
    "away_goal_diff_form",]
X=df[features]
y=df['result']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model=LogisticRegression(
    solver="lbfgs",
    max_iter=1000)
model.fit(X_train_scaled,y_train)
joblib.dump(model, "football_model.pkl")
joblib.dump(scaler, "scaler.pkl")
y_pred=model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
