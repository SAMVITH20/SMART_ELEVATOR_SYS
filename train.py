import yaml
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mlflow

from sim.elevator_env import ElevatorEnv

from models.qlearning import train_qlearning
from models.sarsa import train_sarsa

# --------------------------------------
# LOAD CONFIG
# --------------------------------------

config_path = sys.argv[1]

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

algorithm = config.get(
    "algorithm",
    "qlearning"
)

NUM_FLOORS = config["num_floors"]

ALPHA = config["learning_rate"]

GAMMA = config["discount_factor"]

EPSILON = config["epsilon"]

EPISODES = config["episodes"]

# --------------------------------------
# ENVIRONMENT
# --------------------------------------

env = ElevatorEnv(NUM_FLOORS)

# --------------------------------------
# TRAIN MODEL
# --------------------------------------

if algorithm == "sarsa":

    q_table, rewards = train_sarsa(
        env,
        EPISODES,
        ALPHA,
        GAMMA,
        EPSILON
    )

    EXP_NAME = "sarsa"

else:

    q_table, rewards = train_qlearning(
        env,
        EPISODES,
        ALPHA,
        GAMMA,
        EPSILON
    )

    if "v1" in config_path:
        EXP_NAME = "exp1"
    else:
        EXP_NAME = "exp2"

# --------------------------------------
# SAVE RESULTS
# --------------------------------------

results = pd.DataFrame({

    "episode": list(range(EPISODES)),
    "reward": rewards

})

results.to_csv(
    f"experiments/{EXP_NAME}_results.csv",
    index=False
)

# --------------------------------------
# SAVE SUMMARY
# --------------------------------------

summary = pd.DataFrame([{

    "run_id": EXP_NAME,

    "algorithm": algorithm,

    "episodes": EPISODES,

    "average_reward": np.mean(rewards),

    "learning_rate": ALPHA,

    "epsilon": EPSILON

}])

summary_file = "experiments/summary.csv"

summary.to_csv(

    summary_file,

    mode="a",

    header=not os.path.exists(summary_file),

    index=False
)

# --------------------------------------
# MLFLOW TRACKING
# --------------------------------------

mlflow.set_experiment(
    "Smart Elevator RL"
)

with mlflow.start_run():

    mlflow.log_param(
        "algorithm",
        algorithm
    )

    mlflow.log_param(
        "learning_rate",
        ALPHA
    )

    mlflow.log_param(
        "epsilon",
        EPSILON
    )

    mlflow.log_metric(
        "average_reward",
        np.mean(rewards)
    )

# --------------------------------------
# PLOT GRAPH
# --------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(rewards)

plt.xlabel("Episodes")

plt.ylabel("Reward")

plt.title(f"{EXP_NAME} Performance")

plt.grid(True)

plt.savefig(
    f"plots/{EXP_NAME}_plot.png"
)

plt.close()

print(f"{EXP_NAME} completed!")