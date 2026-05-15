from fastapi import FastAPI
from matplotlib.pylab import log
import pandas as pd
from datetime import datetime

app = FastAPI()

@app.get("/")
def home():

    return {
        "message":
        "Smart Elevator RL API"
    }

@app.get("/predict")
def predict(current_floor: int,
            target_floor: int):

    if current_floor < target_floor:
        action = "UP"

    elif current_floor > target_floor:
        action = "DOWN"

    else:
        action = "STAY"
    log = pd.DataFrame([{

    "timestamp": str(datetime.now()),

    "current_floor": current_floor,

    "target_floor": target_floor,

    "action": action

    }])

    log.to_csv(

        "logs/prediction_logs.csv",

        mode="a",

        header=False,

        index=False
    )

    return {

        "current_floor": current_floor,

        "target_floor": target_floor,

        "recommended_action": action

    }