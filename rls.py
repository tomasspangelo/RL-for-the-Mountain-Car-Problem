import numpy as np
from simworld import SimWorld
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd


class ReinforcementLearningSystem:
    """
    Class for the reinforcement learning system.
    """

    def __init__(self, actor, tc, gamma, episodes):
        self.actor = actor
        self.tc = tc
        self.max_actions = 1000
        self.gamma = gamma                      # discount factor
        self.episodes = episodes

    def learn(self):
        """
        This method will be the main SARSA learning algorithm.
        Let S be a vector, A is -1, 1 og 0.
        """
        x = None
        velocity = None
        progress = []

        # FOR EACH EPISODE
        for episode in tqdm(range(self.episodes)):
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
                if episode == self.episodes-1:
                    env.render()
                    time.sleep(0)

                # TAKE ACTION A OBSERVE R, S'
                x, velocity, reward, finished = env.perform_action(action)

                # CHOOSE ACTION A' FROM S' USING Q
                next_state_vector = self.tc.get_encoding(x, velocity)
                next_action = self.actor.get_action(next_state_vector)

                # UPDATE Q
                next_q = self.actor.get_q(next_state_vector, next_action)
                target = reward + self.gamma*next_q
                self.actor.update_policy(state_vector, action, target)

                # READY FOR NEXT STEP: S = S', A = A'
                state_vector = next_state_vector
                action = next_action
                num_actions += 1
            progress.append(num_actions)
        data = pd.DataFrame({"Steps": progress})
        data["Episodes"] = [i for i in range(self.episodes)]
        data.plot.scatter(x="Episodes", y="Steps")
        plt.show()
