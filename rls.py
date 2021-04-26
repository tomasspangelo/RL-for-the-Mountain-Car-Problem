import numpy as np
from simworld import SimWorld
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from datetime import datetime


class ReinforcementLearningSystem:
    """
    Class for the reinforcement learning system.
    """

    def __init__(self, actor, tc, gamma, episodes):
        timestamp = datetime.now().strftime("%d_%H")
        self.actor = actor
        self.tc = tc
        self.max_actions = 1000
        self.gamma = gamma  # discount factor
        self.episodes = episodes
        self.filename = timestamp
        self.n = 3

    def learn(self, save_interval):
        """
        This method will be the main SARSA learning algorithm.
        Let S be a vector, A is -1, 1 og 0.
        """
        x = None
        velocity = None
        progress = []

        self.actor.save_policy(0, self.filename)
        # FOR EACH EPISODE
        for episode in tqdm(range(self.episodes)):
            self.actor.reset_eligibilities()
            # INITIALIZE STATE S: x is randomly chosen in range [-0.6, -0.4] and velocity set to zero.
            x = np.random.uniform(-0.6, -0.4, 1)[0]
            velocity = 0
            env = SimWorld(x, velocity)

            # CHOOSE ACTION A FROM S USING Q
            state_vector = self.tc.get_encoding(x, velocity)

            action = self.actor.get_action(state_vector)

            num_actions = 1
            finished = False
            # FOR EACH STEP IN EPISODE
            while num_actions < self.max_actions and not finished:
                if (episode + 1) % save_interval == 0:
                    env.render()
                    time.sleep(0)

                # TAKE ACTION A OBSERVE R, S'
                x, velocity, reward, finished = env.perform_action(action)
                if num_actions % self.n != 0:
                    # CHOOSE ACTION A' FROM S' USING Q
                    next_state_vector = self.tc.get_encoding(x, velocity)
                    next_action = self.actor.get_action(next_state_vector)
                    # UPDATE Q
                    next_q = self.actor.get_q(next_state_vector, next_action)
                    target = reward + self.gamma * next_q
                    td_error = target - self.actor.get_q(state_vector, action)
                    self.actor.update_policy(
                        state_vector, action, target, td_error)

                    # READY FOR NEXT STEP: S = S', A = A'
                    state_vector = next_state_vector
                    action = next_action
                num_actions += 1

                if finished:
                    print("MADE IT!")
                elif num_actions >= self.max_actions:
                    print("Timeout :(")

            progress.append(num_actions)
            if (episode + 1) % save_interval == 0:
                self.actor.save_policy(episode + 1, self.filename)

            self.actor.update_epsilon()

        self.actor.save_policy(self.episodes, self.filename)

        data = pd.DataFrame({"Steps": progress})
        data["Episodes"] = [i for i in range(self.episodes)]
        file = open(f"./data/{self.filename}_data.pickle", "wb")
        pickle.dump(data, file)
        data.plot.scatter(x="Episodes", y="Steps")
        plt.show()

    def test_actor(self):
        self.actor.epsilon = 0.1
        n = 3
        x = np.random.uniform(-0.6, -0.4, 1)[0]
        velocity = 0
        env = SimWorld(x, velocity)

        state_vector = self.tc.get_encoding(x, velocity)
        action = self.actor.get_action(state_vector)

        finished = False
        num_actions = 1
        while num_actions < self.max_actions and not finished:
            env.render()
            time.sleep(0)

            x, velocity, reward, finished = env.perform_action(action)
            next_state_vector = self.tc.get_encoding(x, velocity)
            next_action = self.actor.get_action(next_state_vector)

            state_vector = next_state_vector
            if n % 3 != 0:
                action = next_action
            num_actions += 1
