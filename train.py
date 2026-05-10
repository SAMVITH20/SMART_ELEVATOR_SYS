import numpy as np
import random
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from sim.elevator_env import ElevatorEnv

# ---------------------------------------
# LOAD CONFIG
# ---------------------------------------

config_path = sys.argv[1]

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

NUM_FLOORS = config["num_floors"]

ALPHA = config["learning_rate"]

GAMMA = config["discount_factor"]

EPSILON = config["epsilon"]

EPISODES = config["episodes"]

# ---------------------------------------
# EXPERIMENT NAME
# ---------------------------------------

if "v1" in config_path:
    EXP_NAME = "exp1"
else:
    EXP_NAME = "exp2"

# ---------------------------------------
# CREATE ENVIRONMENT
# ---------------------------------------

env = ElevatorEnv(NUM_FLOORS)

actions = [0, 1, 2]

q_table = np.zeros((NUM_FLOORS, len(actions)))

episode_rewards = []

# ---------------------------------------
# TRAINING
# ---------------------------------------

for episode in range(EPISODES):

    state = env.reset()

    done = False

    total_reward = 0

    while not done:

        # Epsilon Greedy

        if random.uniform(0, 1) < EPSILON:
            action = random.choice(actions)

        else:
            action = np.argmax(q_table[state])

        next_state, reward, done = env.step(action)

        # Q-Learning Update

        old_value = q_table[state, action]

        next_max = np.max(q_table[next_state])

        new_value = old_value + ALPHA * (
            reward +
            GAMMA * next_max -
            old_value
        )

        q_table[state, action] = new_value

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

# ---------------------------------------
# SAVE RESULTS
# ---------------------------------------

results = pd.DataFrame({

    "episode": list(range(EPISODES)),
    "reward": episode_rewards

})

results.to_csv(
    f"experiments/{EXP_NAME}_results.csv",
    index=False
)

# ---------------------------------------
# SAVE SUMMARY
# ---------------------------------------

summary = pd.DataFrame([{

    "run_id": EXP_NAME,
    "episodes": EPISODES,
    "average_reward": np.mean(episode_rewards),
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

# ---------------------------------------
# PLOT GRAPH
# ---------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(episode_rewards)

plt.xlabel("Episodes")

plt.ylabel("Reward")

plt.title(
    f"Q-Learning Performance ({EXP_NAME})"
)

plt.grid(True)

plt.savefig(
    f"plots/{EXP_NAME}_plot.png"
)

plt.show()

# ---------------------------------------
# FINAL OUTPUT
# ---------------------------------------

print(f"\n{EXP_NAME} completed!")

print("\nAverage Reward:")

print(np.mean(episode_rewards))

print("\nQ Table:\n")

print(q_table)