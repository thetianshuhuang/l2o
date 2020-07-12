import l2o
import tensorflow as tf


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

    net = l2o.networks.DMOptimizer()
    opt = l2o.optimizer.CoordinateWiseOptimizer(net)

    opt.save("dmoptimizer")


def load():
    return l2o.optimizer.CoordinateWiseOptimizer(
        l2o.networks.DMOptimizer(), weights_file="dmoptimizer")


def train(problems, repeat=1, epochs=1):

    opt = load()
    mgr = l2o.optimizer.MetaOptimizerMgr(
        tf.keras.optimizers.Adam(), opt, noise_stddev=0.0, unroll=20)
    mgr.train(problems, repeat=repeat, epochs=epochs)
    opt.save("dmoptimizer")


def test_quadratic(opt):
    problem = l2o.problems.Quadratic(20, persistent=True)
    start = problem.test_objective(None)
    for _ in range(100):
        opt.minimize(
            lambda: problem.test_objective(None), problem.trainable_variables)
    print("{} -> {}".format(start, problem.test_objective(None)))


def test_classify(opt, conv=True):

    dataset, info = l2o.problems.load_images("mnist")

    if conv:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 5, activation=tf.nn.relu,
                input_shape=info.features['image'].shape),
            tf.keras.layers.Conv2D(
                32, 3, strides=(2, 2), activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=info.features['image'].shape),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

    print(model.summary())

    model.compile(
        opt,
        tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(dataset.batch(32), epochs=5)

    model.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.SparseCategoricalCrossentropy())
    model.fit(dataset.batch(32), epochs=5)
