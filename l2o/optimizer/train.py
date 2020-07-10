import tensorflow as tf
import time


def weights_sum(n):
    return tf.ones([n])


def weights_mean(n):
    return tf.ones([n]) / tf.cast(n, tf.float32)


class MetaOptimizerMgr:

    def __init__(
            self, optimizer, learner,
            noise_stddev=0.0, unroll=20, unroll_weights=weights_mean):

        self.optimizer = optimizer
        self.learner = learner

        self.noise_stddev = noise_stddev
        self.unroll = unroll
        self.unroll_weights = unroll_weights

    def _prepare_cf(
            self, problem, weights, data=None, is_batched=False):

        params = problem.get_parameters()
        states = [self.learner._initialize_state(p) for p in params]

        cf = self.learner.meta_loss.get_concrete_function(
            problem, weights, self.unroll, params, states,
            data=data, is_batched=is_batched, noise_stddev=self.noise_stddev)

        return cf, params, states

    def _step(
            self, concrete_loss,
            problem, weights, params, states, is_batched, data=None):

        trainable = self.learner.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(trainable)
            loss, params, states = concrete_loss(
                problem, weights, self.unroll, params, states,
                data=data, is_batched=is_batched,
                noise_stddev=self.noise_stddev)
        grads = tape.gradient(loss, trainable)
        self.optimizer.apply_gradients(zip(grads, trainable))

        return loss, params, states

    def _train_meta_full(self, spec, repeat=1):

        pbar = tf.keras.utils.Progbar(repeat, unit_name='step')
        problem = spec.build()

        concrete_loss = None

        for _ in range(repeat):

            weights = self.unroll_weights(self.unroll)
            internal = problem.get_internal()

            if concrete_loss is None:
                concrete_loss = self._prepare_cf(
                    problem, weights, data=internal, is_batched=False)

            loss, _, _ = self._step(problem, weights)

            pbar.add(1, values=[("loss", loss)])

    # def _train_meta_batch(self, spec, epochs=1):

    #     problem = spec.build()
    #     dataset = problem.get_dataset(self.unroll)
    #     weights = self.unroll_weights(self.unroll)

    #     params = None
    #     states = None

    #     for i in range(epochs):

    #         print("Epoch {}".format(i + 1))
    #         pbar = tf.keras.utils.Progbar(
    #             problem.size(self.unroll), unit_name='step')

    #         for batch in dataset:
    #             batch_stacked = [
    #                 tf.stack(tf.split(dim, num_or_size_splits=self.unroll))
    #                 for dim in batch]

    #             loss, params, states = self._step(
    #                 problem, weights,
    #                 params=params, states=states, data=batch_stacked)

    #             pbar.add(1, values=[("loss", loss)])

    def train(self, problems, epochs=1, repeat=1, teacher=None):

        start = time.time()
        for itr, spec in enumerate(problems):
            spec.print(itr)

            if hasattr(spec, "get_dataset"):
                self._train_meta_batch(spec, epochs=epochs)
            else:
                self._train_meta_full(spec, repeat=repeat)

        return time.time() - start
