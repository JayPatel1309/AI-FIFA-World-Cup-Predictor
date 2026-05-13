from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import joblib

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(r"CombinedData.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

# LOAD MODEL + SCALER

model = joblib.load(r"football_model.pkl")

scaler = joblib.load(r"scaler.pkl")

# TEAM LIST

teams = sorted(list(set(
    df["home_team"]
).union(set(
    df["away_team"]
))))

# REQUEST MODEL


class MatchRequest(BaseModel):

    home_team: str

    away_team: str

    neutral: bool

    friendly: bool

# HOME

@app.get("/")
def home():

    return {
        "message": "World Cup Predictor AI Running"
    }

# GET TEAMS

@app.get("/teams")
def get_teams():

    return teams
# GET LATEST TEAM DATA

def get_latest_team_data(team):

    matches = df[
        (df["home_team"] == team) |
        (df["away_team"] == team)
    ].sort_values("date")

    latest = matches.iloc[-1]

    # Team was home
    if latest["home_team"] == team:

        return {
            "elo": latest["elo_home"],
            "form": latest["home_form"],
            "goal_diff_form":
                latest["home_goal_diff_form"]
        }

    # Team was away
    else:

        return {
            "elo": latest["elo_opp"],
            "form": latest["away_form"],
            "goal_diff_form":
                latest["away_goal_diff_form"]
        }

# PREDICT
@app.post("/predict")

def predict(request: MatchRequest):

    # Validation
    if request.home_team == request.away_team:

        return {
            "error":
                "Please select different teams"
        }

    # Latest team data
    home = get_latest_team_data(
        request.home_team
    )

    away = get_latest_team_data(
        request.away_team
    )
    # FEATURE ENGINEERING

    elo_diff = (
        home["elo"] -
        away["elo"]
    )

    home_form = home["form"]

    away_form = away["form"]

    home_goal_diff_form = (
        home["goal_diff_form"]
    )

    away_goal_diff_form = (
        away["goal_diff_form"]
    )

    # Neutral venue
    if request.neutral:
        home_advantage = 0
    else:
        home_advantage = 1

    # Friendly match
    if request.friendly:
        is_friendly = 1
    else:
        is_friendly = 0
    # FINAL FEATURES

    features = [[
        elo_diff,
        home_form,
        away_form,
        home_advantage,
        is_friendly,
        home_goal_diff_form,
        away_goal_diff_form
    ]]
    # SCALE

    scaled = scaler.transform(
        features
    )
    # PREDICT

    probabilities = (
        model.predict_proba(
            scaled
        )[0]
    )

    return {

        "home_team":
            request.home_team,

        "away_team":
            request.away_team,

        "features": {

            "elo_diff":
                round(elo_diff, 2),

            "home_form":
                int(home_form),

            "away_form":
                int(away_form),

            "home_goal_diff_form":
                round(
                    home_goal_diff_form,
                    2
                ),

            "away_goal_diff_form":
                round(
                    away_goal_diff_form,
                    2
                )
        },

        "probabilities": {

            "away_win":
                round(
                    probabilities[0] * 100,
                    2
                ),

            "draw":
                round(
                    probabilities[1] * 100,
                    2
                ),

            "home_win":
                round(
                    probabilities[2] * 100,
                    2
                )
        }
    }