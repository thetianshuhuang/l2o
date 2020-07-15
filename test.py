import l2o
import tensorflow as tf
import numpy as np


quad = [l2o.problems.ProblemSpec(
    l2o.problems.Quadratic, [20], {}
)]
mlp = [l2o.problems.ProblemSpec(
    l2o.problems.mlp_classifier, [],
    {"layers": [128, ], "dataset": "mnist", "activation": tf.nn.relu}
)]
conv = [l2o.problems.ProblemSpec(
    l2o.problems.conv_classifier, [],
    {"layers": [(5, 32, 2), ], "dataset": "mnist", "activation": tf.nn.relu}
)]


def create():
    opt = l2o.optimizer.CoordinateWiseOptimizer(
        l2o.networks.RNNPropOptimizer())
    opt.save("testopt")


def load():
    return l2o.optimizer.CoordinateWiseOptimizer(
        l2o.networks.RNNPropOptimizer(), weights_file="testopt")


def train_meta(problems, repeat=1, epochs=1):
    opt = load()
    opt.train(
        problems, tf.keras.optimizers.Adam(), repeat=repeat, epochs=epochs)
    opt.save("testopt")


def train_imitation(problems, repeat=1, epochs=1):
    opt = load()
    opt.train(
        problems, tf.keras.optimizers.Adam(), repeat=repeat, epochs=epochs,
        teacher=tf.keras.optimizers.Adam())
    opt.save("testopt")


def test_quadratic(opt=None):
    if opt is None:
        opt = load()
    problem = l2o.problems.Quadratic(20, persistent=True)
    start = problem.test_objective(None)
    for _ in range(100):
        opt.minimize(
            lambda: problem.test_objective(None), problem.trainable_variables)
    print("{} -> {}".format(start, problem.test_objective(None)))
    difference = start - problem.test_objective(None)
    return difference


def validate_quadratic():
    differences = [test_quadratic() for _ in range(100)]
    print("mean:", np.mean(differences))
    print("min:", np.min(differences))
    print("max:", np.max(differences))
    return differences


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


def test_classify(opt=None, conv=True):

    if opt is None:
        opt = load()

    dataset, info = l2o.problems.load_images("mnist")

    model = get_model(info, conv=conv)
    print(model.summary())
    model.compile(
        opt,
        tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(dataset.batch(32), epochs=2)

    model = get_model(info, conv=conv)
    model.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(dataset.batch(32), epochs=2)
