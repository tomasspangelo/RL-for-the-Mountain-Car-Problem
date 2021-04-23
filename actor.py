import random
import tensorflow as tf
import numpy as np


class Actor:
    """
    Class for an actor.
    """

    def __init__(self, keras_model, epsilon, epsilon_decay):
        self.policy = keras_model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_q(self, tiled_state, action):
        sa = np.concatenate((tiled_state, [action]), axis=None)
        return self.policy(sa.reshape((1,)+sa.shape)).numpy()[0, 0]

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
        sa = np.concatenate((tiled_state, [action]), axis=None)
        sa = sa.reshape((1,) + sa.shape)
        target = np.array([target])
        target = target.reshape((1,) + target.shape)
        # print("Feature:",sa)
        # print("Target:",target)
        self.policy.fit(sa, target, epochs=1, verbose=0)
        #print("Output", self.policy(sa).numpy())

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def save_policy(self, episode):
        self.policy.save("./anets/{episode}".format(episode=episode))


if __name__ == "__main__":
    keras_model = tf.keras.models.Sequential()
    keras_model.add(tf.keras.Input(shape=(5,)))
    keras_model.add(tf.keras.layers.Dense(units=1, activation='tanh'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    keras_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[
                            tf.keras.metrics.MeanSquaredError()
                        ])
    actor = Actor(keras_model, 0.1, 1)
    print(actor.get_q(np.array([1, 1, 1, 1]), 1))
    actor.update_policy(np.array([1, 1, 1, 1]), 1, 2.2)
