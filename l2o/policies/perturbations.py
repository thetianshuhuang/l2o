"""Parameter perturbations."""

import tensorflow as tf
from l2o.optimizer.tf_utils import _var_key


class BasePerturbation:
    """Perturbation Base Class."""

    adversarial_attack_steps = 0

    def add(self, param):
        """Add noise."""
        return param

    def build(self, trainable_variables):
        """Prepare perturbation variables."""
        pass

    def reset(self, train=True):
        """Reset noise variables."""
        pass

    def apply_gradients(self, params_and_grads):
        """Apply adversarial gradients."""
        pass


class RandomPerturbation(BasePerturbation):
    """Random gaussian noise.

    Keyword Args
    ------------
    noise_stddev : float
        IID gaussian stddev.
    relative : bool
        Whether noise stddev is specified relative to the weight l2 magnitude.
    epsilon : float
        Epsilon value for sqrt.
    """

    def __init__(self, noise_stddev=0.01, relative=False, epsilon=1e-8):
        self.noise_stddev = noise_stddev
        self.relative = relative
        self.epsilon = epsilon

    def add(self, param):
        """Add noise to parameter."""
        if self.relative:
            noise = (
                self.noise_stddev * tf.math.sqrt(
                    tf.reduce_sum(tf.math.square(param) + self.epsilon)
                ) / tf.cast(tf.size(param), tf.float32))
        else:
            noise = self.noise_stddev

        return param + tf.random.normal(param.shape, mean=0.0, stddev=noise)


class PersistentRandomPerturbation(BasePerturbation):
    """Random gaussian noise which persists across iterations within an unroll.

    Keyword Args
    ------------
    noise_stddev : float
        IID gaussian stddev.
    """

    def __init__(self, noise_stddev=0.01):
        self.noise_stddev = noise_stddev

    def build(self, trainable_variables):
        """Prepare perturbation variables."""
        self.perturbable_variables = [
            tf.Variable(tf.zeros_like(x), trainable=False)
            for x in trainable_variables]
        self._perturbable_variables = {
            _var_key(t): p
            for t, p in zip(trainable_variables, self.perturbable_variables)
        }

    def reset(self, train=True):
        """Reset noise variables."""
        if train:
            for v in self.perturbable_variables:
                v.assign(tf.random.normal(
                    v.shape, mean=0.0, stddev=self.noise_stddev))
        else:
            for v in self.perturbable_variables:
                v.assign(tf.zeros_like(v))

    def add(self, param):
        """Add noise to parameter."""
        return param + self._perturbable_variables[_var_key(param)]


class FGSMPerturbation(BasePerturbation):
    """Adversarial perturbation using fast gradient sign method.

    Keyword Args
    ------------
    step_size : float
        FGSM step size.
    """

    adversarial_attack_steps = 1

    def __init__(self, step_size=0.01):
        assert(step_size > 0)
        self.step_size = step_size

    def add(self, param):
        """Add noise to parameter."""
        return param + self._perturbable_variables[_var_key(param)]

    def build(self, trainable_variables):
        """Prepare perturbation variables."""
        self.perturbable_variables = [
            tf.Variable(tf.zeros_like(x), trainable=False)
            for x in trainable_variables]
        self._perturbable_variables = {
            _var_key(t): p
            for t, p in zip(trainable_variables, self.perturbable_variables)
        }

    def reset(self, train=True):
        """Reset noise variables."""
        for v in self.perturbable_variables:
            v.assign(tf.zeros_like(v))

    def apply_gradients(self, params_and_grads):
        """Apply adversarial gradients."""
        for _, param, grad in params_and_grads:
            param.assign_add(tf.math.sign(grad) * self.step_size)


class PGDPerturbation(FGSMPerturbation):
    """Adversarial perturbation using projected gradient descent.

    Keyword Args
    ------------
    magnitude : float
        Magnitude of perturbation.
    norm : str
        Norm type for PGD attack. Options:
        - l2: ordinary l2 norm
        - scale: l2 norm, with magnitude scaled by # of parameters
        - inf: l-infinity norm
    learning_rate : float
        Learning rate for PGD attack.
    steps : int
        Number of steps to take.
    """

    def __init__(
            self, steps=1, magnitude=0.01, norm="l2", learning_rate=0.005):

        self.adversarial_attack_steps = steps
        self.magnitude = magnitude
        self.norm = norm
        self.learning_rate = learning_rate
        self.relative = relative

    def apply_gradients(self, params_and_grads):
        """Apply adversarial gradients."""
        for _, param, grad in params_and_grads:
            param_new = param + grad * self.learning_rate

            if self.norm == "l2":
                param.assign(tf.clip_by_norm(param_new, self.magnitude))
            elif self.norm == "scale":
                param.assign(tf.clip_by_norm(
                    param_new, self.magnitude * tf.size(param)))
            elif self.norm == "inf":
                param.assign(tf.clip_by_value(
                    param_new,
                    clip_value_min=-self.magnitude,
                    clip_value_max=self.magnitude))
            else:
                raise ValueError("Invalid norm: {}".format(self.norm))


class CGDPerturbation(FGSMPerturbation):
    """Adversarial perturbation using clipped gradient descent.

    Keyword Args
    ------------
    magnitude : float
        Magnitude of perturbation per step, relative to parameter magnitude.
    steps : int
        Number of steps to take.
    """

    def __init__(self, steps=1, magnitude=0.005):

        self.adversarial_attack_steps = steps
        self.magnitude = magnitude

    def apply_gradients(self, params_and_grads):
        """Apply adversarial gradients."""
        for parent, param, grad in params_and_grads:
            param.assign(
                param + tf.clip_by_norm(
                    grad, tf.norm(parent) * self.magnitude))
