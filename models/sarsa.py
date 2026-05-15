import numpy as np
import random

def train_sarsa(env,
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

        # Initial Action

        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[state])

        done = False

        total_reward = 0

        while not done:

            next_state, reward, done = env.step(action)

            # Next Action

            if random.uniform(0, 1) < epsilon:
                next_action = random.choice(actions)
            else:
                next_action = np.argmax(
                    q_table[next_state]
                )

            # SARSA Update

            old_value = q_table[state, action]

            next_q = q_table[
                next_state,
                next_action
            ]

            new_value = old_value + alpha * (
                reward +
                gamma * next_q -
                old_value
            )

            q_table[state, action] = new_value

            state = next_state
            action = next_action

            total_reward += reward

        rewards.append(total_reward)

    return q_table, rewards