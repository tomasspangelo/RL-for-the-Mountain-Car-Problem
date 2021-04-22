from configparser import ConfigParser
import tensorflow as tf

from coarse import TileCoding
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


def init_rls(actor, tc, rls_config):
    gamma = float(rls_config["gamma"])
    alpha = float(rls_config["alpha"])
    episodes = int(rls_config["episodes"])
    rls = ReinforcementLearningSystem(actor, tc, gamma, alpha, episodes)

    return rls


def init_tc(tc_config):
    num_tilings = int(tc_config["num_tilings"])
    partitions = int(tc_config["partitions"])
    x_range = int(tc_config["x_range"])
    y_range = int(tc_config["y_range"])
    extra_lengths = int(tc_config["extra_lengths"])
    offset_percent = int(tc_config["offset_percent"])
    tc = TileCoding(num_tilings, partitions, x_range, y_range, extra_lengths, offset_percent)
    return tc


def main():
    config = ConfigParser()

    config.read("./config.ini")

    tc = init_tc(config['tile_coding'])

    input_shape = tc.partitions ** 2 * tc.num_tilings + 1
    actor = init_actor(config['actor'], input_shape)

    learner = init_rls(actor, tc, config["rls"])

    learner.learn()


if __name__ == "__main__":
    main()
