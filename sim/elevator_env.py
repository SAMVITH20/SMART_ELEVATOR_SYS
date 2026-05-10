import random

class ElevatorEnv:

    def __init__(self, num_floors):

        self.num_floors = num_floors

        self.reset()

    def reset(self):

        self.current_floor = random.randint(
            0,
            self.num_floors - 1
        )

        self.target_floor = random.randint(
            0,
            self.num_floors - 1
        )

        return self.current_floor

    def step(self, action):

        # Actions
        # 0 -> UP
        # 1 -> DOWN
        # 2 -> STAY

        if action == 0:

            if self.current_floor < self.num_floors - 1:
                self.current_floor += 1

        elif action == 1:

            if self.current_floor > 0:
                self.current_floor -= 1

        elif action == 2:
            pass

        # Reward Function

        if self.current_floor == self.target_floor:

            reward = 100
            done = True

        else:

            reward = -abs(
                self.target_floor - self.current_floor
            )

            done = False

        return self.current_floor, reward, done