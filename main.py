from configparser import ConfigParser
import tensorflow as tf
from rls import ReinforcementLearningSystem

from actor import Actor


def init_actor(actor_config, input_shape):
    epsilon = float(actor_config['epsilon'])
    epsilon_decay_factor = float(actor_config['epsilon_decay_factor'])
    learning_rate = float(actor_config['learning_rate'])

    keras_model = tf.keras.models.Sequential()
    keras_model.add(tf.keras.Input(shape=input_shape))
    keras_model.add(tf.keras.layers.Dense(units=1, activation='linear'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    keras_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[
                            tf.keras.metrics.MeanSquaredError()
                        ])
    actor = Actor(keras_model, epsilon, epsilon_decay_factor)

    return actor

def init_rls(actor, coarse,rls_config):
    gamma = float(rls_config["gamma"])
    alpha = float(rls_config["alpha"])
    episodes = int(rls_config["episodes"])
    rls = ReinforcementLearningSystem(actor, coarse,gamma, alpha, episodes)

    return rls


def main():
    config = ConfigParser()

    config.read("./config.ini")

    #Must set up
    coarse = init_coarse(config['coarse'])

    input_shape = coarse.partitions**2 * coarse.tilings + 1
    actor = init_actor(config['actor'], input_shape)

    # Must set up
    learner = init_rls(actor, coarse, config["rls"])

    learner.learn()


if __name__ == "__main__":
    main()
