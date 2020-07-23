import os
import functools

import tensorflow as tf
import numpy as np
import pandas as pd


def makedir(path, assert_empty=False):
    if os.path.isdir(path):
        if assert_empty:
            raise Exception(
                path, "Directory {} already exists; please rename or delete "
                "the directory.".format(path))
    else:
        os.mkdir(path)


class SimpleStrategy:

    def __init__(
            self, learner, loss_args={}, problems=[],
            optimizer=tf.keras.optimizers.Adam(),
            period_size=10, num_periods=100,
            directory="weights",
            unroll_distribution=lambda: np.random.geometric(0.05),
            annealing_schedule=lambda i: np.exp(i * -0.5),
            validation_unroll=50):

        self.learner = learner

        self.problems = problems
        self.optimizer = optimizer
        self.loss_args = loss_args

        self.period_size = period_size
        self.num_periods = num_periods
        self.directory = directory

        self.unroll_distribution = unroll_distribution
        self.annealing_schedule = annealing_schedule

        makedir(self.directory)
        try:
            self.summary = pd.read_csv(
                os.path.join(self.directory, "summary.csv"))
            self.period = self.summary["period"].max()
            self._load_network(self.period)
            self.period += 1
        except FileNotFoundError:
            self.summary = pd.DataFrame({
                "period": pd.Series([], dtype='int'),
                "training_loss": pd.Series([], dtype='float'),
                "validation_loss": pd.Series([], dtype='float')
            })
            self.period = 0

    def _load_network(self, period):
        path = os.path.join(self.directory, "period_{}".format(period))
        self.learner.network.load_weights(path)
        print("Loaded weights: {}".format(path))

    def _mean_loss(self, results):
        """Helper function to compute mean loss."""
        return np.mean([
            np.mean([np.mean(loss_array) for loss_array in result.loss])
            for result in results
        ])

    def learning_period(self):
        train_func = functools.partial(
            self.learner.train,
            self.problems, self.optimizer, **self.loss_args)

        training_loss = []
        for i in range(self.period_size):
            print("Training [epoch {}/{}]".format(
                i + 1, self.period_size))
            results = train_func(
                unroll_len=self.unroll_distribution,
                p_teacher=self.annealing_schedule(), validation=False)
            training_loss.append(self._mean_loss(results))
        training_loss = np.mean(training_loss)

        print("Validating")
        validation_loss = self._mean_loss(train_func(
            unroll_len=lambda: self.validation_unroll,
            p_teacher=0, validation=True))

        print("training_loss: {} | validation_loss: {}".format(
            training_loss, validation_loss))

        self.summary = self.summary.append(
            pd.DataFrame({
                "period": [self.period],
                "training_loss": [training_loss],
                "validation_loss": [validation_loss]
            }), ignore_index=True)
        self.summary.to_csv(
            os.path.join(self.directory, "summary.csv"), index=True)

        self.period += 1

    def train(self):

        while self.period < self.num_periods:
            self.learning_period()
