import random


class Actor:
    """
    Class for an actor.
    """

    def __init__(self, keras_model, epsilon, epsilon_decay):
        self.policy = keras_model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_q(self, tiled_state, action):
        state_action_pair = tiled_state.copy().append(action)
        state_action_pair.append(action)
        return self.policy(state_action_pair)

    def get_action(self, tiled_state):
        legal_actions = [-1, 0, 1]
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        best_action = None
        best_value = float('-inf')
        for a in legal_actions:
            q = self.get_q(tiled_state, a)
            if q > best_value:
                best_value = q
                best_action = a
        return best_action

    def update_policy(self, tiled_state, action, target):
        state_action_pair = tiled_state.copy()
        state_action_pair.append(action)
        self.policy.fit(state_action_pair, target, epochs=1, verbosity=0)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
