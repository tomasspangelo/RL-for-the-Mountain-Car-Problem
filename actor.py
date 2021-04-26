import random
import tensorflow as tf
import numpy as np
from splitgd import SplitGD


class Actor(SplitGD):
    """
    Class for an actor.
    """

    def __init__(self, keras_model, epsilon, epsilon_decay, decay_factor, gamma):
        super().__init__(keras_model)
        self.policy = keras_model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.eligibilities = []
        self.td_error = 0
        self.decay_factor = decay_factor
        self.discount_factor = gamma

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

    def update_policy(self, tiled_state, action, target, td_error):
        self.td_error = td_error
        sa = np.concatenate((tiled_state, [action]), axis=None)
        sa = sa.reshape((1,) + sa.shape)
        target = np.array([target])
        target = target.reshape((1,) + target.shape)
        # print("Feature:",sa)
        # print("Target:",target)
        self.fit(sa, target, epochs=3, verbosity=0)
        #print("Output", self.policy(sa).numpy())

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def save_policy(self, episode, filename):
        self.policy.save("./anets/{filename}/{episode}".format(episode=episode, filename=filename))

    def modify_gradients(self, gradients):
        """
        Modifies the gradient according to the eligibilities to fit the RL actor-critic algorithm.
        :param gradients: The gradients given by the keras model
        :return: Array of updated gradients
        """
        if not self.eligibilities:
            for gradient_layer in gradients:
                self.eligibilities.append(np.zeros(gradient_layer.shape))

        for i in range(len(gradients)):
            gradient_layer = gradients[i]
            v_grad = 0 * gradient_layer if self.td_error == 0 else 1 / self.td_error * gradient_layer
            self.eligibilities[i] = self.discount_factor * self.decay_factor * self.eligibilities[i] + v_grad

        return [self.td_error * eligibility for eligibility in self.eligibilities]


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
