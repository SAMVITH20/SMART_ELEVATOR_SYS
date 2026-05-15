import numpy as np
import random

def train_qlearning(env,
                    episodes,
                    alpha,
                    gamma,
                    epsilon):

    actions = [0, 1, 2]

    q_table = np.zeros(
        (env.num_floors, len(actions))
    )

    rewards = []

    for episode in range(episodes):

        state = env.reset()

        done = False

        total_reward = 0

        while not done:

            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = np.argmax(
                    q_table[state]
                )

            next_state, reward, done = env.step(action)

            old_value = q_table[state, action]

            next_max = np.max(
                q_table[next_state]
            )

            new_value = old_value + alpha * (
                reward +
                gamma * next_max -
                old_value
            )

            q_table[state, action] = new_value

            state = next_state

            total_reward += reward

        rewards.append(total_reward)

    return q_table, rewards