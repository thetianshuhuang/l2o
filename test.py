import os
import json
import pprint

import tensorflow as tf

import l2o


def load(folder, target):

    print("Loading {}/{}".format(folder, target))

    with open(os.path.join(folder, "config.json"), 'w') as f:
        config = json.load(f)
    print("config & hyperparameters:")
    pprint.pprint(config)

    network = getattr(
        l2o.networks, config["constructor"] + "Optimizer")(**config["network"])
    network.load_weights(target)

    learner = network.architecture(network, **config["loss_args"])

    return learner


def get_model(info, conv=True):
    if conv:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 5, activation=tf.nn.relu,
                input_shape=info.features['image'].shape),
            tf.keras.layers.Conv2D(
                32, 3, strides=(2, 2), activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
    else:
        return tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=info.features['image'].shape),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation="softmax")
        ])


def test_classify(folder, target, conv=True):

    opt = load(folder, target)

    dataset, info = l2o.problems.load_images("mnist")
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model = get_model(info, conv=conv)
    print(model.summary())
    model.compile(opt, loss)
    model.fit(dataset.batch(32), epochs=2)

    model = get_model(info, conv=conv)
    model.compile(tf.keras.optimizers.Adam(), loss)
    model.fit(dataset.batch(32), epochs=2)


if __name__ == '__main__':

    import sys
    folder = sys.argv[1]
    target = sys.argv[2]

    test_classify(folder, target)
