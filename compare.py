import pandas as pd
import matplotlib.pyplot as plt

exp1 = pd.read_csv(
    "experiments/exp1_results.csv"
)

exp2 = pd.read_csv(
    "experiments/exp2_results.csv"
)

sarsa = pd.read_csv(
    "experiments/sarsa_results.csv"
)

plt.figure(figsize=(10, 5))

plt.plot(exp1["reward"], label="QLearning Exp1")

plt.plot(exp2["reward"], label="QLearning Exp2")

plt.plot(sarsa["reward"], label="SARSA")

plt.xlabel("Episodes")

plt.ylabel("Reward")

plt.title("Algorithm Comparison")

plt.legend()

plt.grid(True)

plt.savefig(
    "plots/comparison_plot.png"
)

plt.show()