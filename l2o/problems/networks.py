import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


from .problem import Problem


class Classifier(Problem):
    """Generic classifier problem

    Parameters
    ----------
    model : object
        Core model (i.e. classifier or regression). Must have
        ``get_parameters`` and ``call`` methods.
    loss : callable (tf.Tensor, tf.Tensor -> tf.Tensor)
        Computes loss from model output and ground truth. Dataset should have
        input as the first array and output as the second.
    dataset : tf.data.Dataset
        Tensorflow dataset to use

    Keyword Args
    ------------
    shuffle_buffer : int
        Shuffle buffer size for dataset shuffling (see tf.data.Dataset). If
        None, then the dataset is not shuffled. Does nothing if no dataset
        is associated with this problem.
    batch_size : int
        Batch size for dataset
    size : int
        Number of elements in this dataset, if known.
    """

    def __init__(
            self, model, dataset, loss, persistent=False,
            shuffle_buffer=None, batch_size=32, size=None, **kwargs):

        self.loss = loss
        self.dataset = dataset

        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self._size = size

        super().__init__(persistent=persistent)

    def size(self, unroll):
        return math.floor(self._size / (unroll * self.batch_size))

    def get_dataset(self, unroll):
        if self.shuffle_buffer is not None:
            self.dataset = self.dataset.shuffle(self.shuffle_buffer)
        return self.dataset.batch(
            self.batch_size * unroll, drop_remainder=True)

    def get_parameters(self):
        return self.model.get_parameters()

    def objective(self, params, data):
        x, y = data
        return self.loss(y, self.model.call(params, x))


class MLPClassifier:

    def __init__(
            self, in_size, out_size, layers=[128, ],
            activation=tf.nn.relu,
            kernel_init=tf.keras.initializers.GlorotUniform,
            bias_init=tf.keras.initializers.Zeros):

        self.kernel_init = kernel_init()
        self.bias_init = bias_init()
        self.activation = activation

        self.layer_shapes = []
        for i, layer in enumerate(layers):
            out = self.out_size if i + 1 == len(layers) else layers[i + 1]
            self.layer_shapes.append({
                "kernel": [layer, out],
                "bias": [out]
            })

        self.layer_indices = np.cumsum([len(s) for s in self.layer_shapes])

    def get_parameters(self):
        self.parameters = []
        for i, shapes in enumerate(self.layer_shapes):
            self.parameters.append(
                self.kernel_init(shape=shapes["kernel"], dtype=tf.float32))
            self.parameters.append(
                self.kernel_init(shape=shapes["bias"], dtype=tf.float32))

    def call(self, params, data):

        x = data

        prev = 0
        for idx, layer in enumerate(self.layer_indices):
            kernel, bias = params[prev, layer]
            prev = layer

            x = tf.matmul(kernel, x) + bias

            if idx == len(self.layer_indices) - 1:
                x = tf.nn.softmax(x)
            else:
                x = self.activation(x)

        return x


def _make_tfds(network, dataset="mnist", **kwargs):
    """Helper function to create training problem using tensorflow_datasets"""

    dataset, info = tfds.load(
        dataset, split="train", shuffle_files=True,
        with_info=True, as_supervised=True)

    try:
        input_shape = info.features['image'].shape
        labels = info.features['label'].num_classes
    except KeyError:
        raise TypeError("Dataset must have 'image' and 'label' features.")
    except AttributeError:
        raise TypeError(
            "'image' feature must have a shape, and 'label' feature must have"
            + "a number of classes num_classes.")
    if None in input_shape:
        raise TypeError("Dataset does not have fixed input dimension.")

    return Classifier(
        network(input_shape, labels),
        tf.keras.losses.SparseCategoricalCrossentropy(), dataset,
        size=info.splits['train'].num_examples, **kwargs)


def mlp_classifier(
        dataset="mnist", layers=[128, ], activation=tf.nn.relu, **kwargs):
    """Create MLP classifier training problem.

    Keyword Args
    ------------
    dataset : str
        Dataset from tdfs catalog. Must have fixed input dimension and
        output labels.
    layers : int[]
        Array of hidden layer sizes for MLP.
    activation : str
        Keras activation type
    **kwargs : dict
        Passed on to Classifier()

    Returns
    -------
    problem.Problem
        Created problem

    Raises
    ------
    KeyError
        Selected dataset does not have an image or label.
    AttributeError
        Image does not specify shape or label does not specify num_classes.
    TypeError
        Dataset does not have a fixed input dimension.
    """

    def _network(input_shape, labels):
        return MLPClassifier(
            input_shape, labels, layers=layers, activation=activation)

    return _make_tfds(_network, dataset=dataset, **kwargs)
