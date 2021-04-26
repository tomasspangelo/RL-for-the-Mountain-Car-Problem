from configparser import ConfigParser
import tensorflow as tf
import pickle

from coarse import TileCoding
from rls import ReinforcementLearningSystem
from actor import Actor
import json
import sys

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def init_actor(actor_config, input_shape, gamma):
    epsilon = float(actor_config['epsilon'])
    epsilon_decay_factor = float(actor_config['epsilon_decay_factor'])
    learning_rate = float(actor_config['learning_rate'])
    decay_factor = float(actor_config['decay_factor'])

    keras_model = tf.keras.models.Sequential()
    keras_model.add(tf.keras.Input(shape=input_shape))
    keras_model.add(tf.keras.layers.Dense(units=1, activation='linear'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    keras_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.MeanSquaredError())
    actor = Actor(keras_model, epsilon,
                  epsilon_decay_factor, decay_factor, gamma)

    return actor


def init_rls(actor, tc, rls_config):
    gamma = float(rls_config["gamma"])
    episodes = int(rls_config["episodes"])
    rls = ReinforcementLearningSystem(actor, tc, gamma, episodes)

    return rls


def init_tc(tc_config):
    num_tilings = int(tc_config["num_tilings"])
    partitions = int(tc_config["partitions"])
    x_range = json.loads(tc_config["x_range"])
    y_range = json.loads(tc_config["y_range"])
    extra_lengths = json.loads(tc_config["extra_lengths"])
    offset_percent = float(tc_config["offset_percent"])
    tc = TileCoding(num_tilings, partitions, x_range,
                    y_range, extra_lengths, offset_percent)
    file = open("./actor/tc.pickle", "wb")
    pickle.dump(tc, file)

    return tc


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
        return

    config = ConfigParser()

    config.read("./config.ini")

    tc = init_tc(config['tile_coding'])

    input_shape = tc.partitions ** 2 * tc.num_tilings + 1
    gamma = float(config.get("rls", "gamma"))
    actor = init_actor(config['actor'], input_shape, gamma)

    learner = init_rls(actor, tc, config["rls"])

    save_interval = int(config.get("rls", "save_interval"))
    learner.learn(save_interval)


def test():
    config = ConfigParser()

    config.read("./config.ini")
    file = open("./actor/tc.pickle", "rb")
    tc = pickle.load(file)

    input_shape = tc.partitions ** 2 * tc.num_tilings + 1
    gamma = float(config.get("rls", "gamma"))
    actor = init_actor(config['actor'], input_shape, gamma)

    path_list = [f.path for f in os.scandir("./actor") if f.is_dir()]
    if len(path_list) != 1:
        raise ValueError(
            "There are no or more than one actor in the ./actor directory.")
    path = path_list[-1]
    model = tf.keras.models.load_model(path, compile=True)

    actor.policy = model

    learner = init_rls(actor, tc, config["rls"])
    learner.test_actor()


if __name__ == "__main__":
    main()
