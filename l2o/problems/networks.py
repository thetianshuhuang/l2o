"""NN based training problems."""

import math
import tensorflow as tf
import tensorflow_datasets as tfds

from .problem import Problem
from .stateless_keras import Dense, Sequential, Conv2D


class Classifier(Problem):
    """Generic classifier problem.

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

    Attributes
    ----------
    batch_size : int
        Replica-adjusted batch size (# samples per replica)
    """

    def __init__(
            self, model, loss, dataset, persistent=False,
            shuffle_buffer=None, batch_size=32, size=None, distribute=None):

        self.dataset = dataset
        self.model = model
        self.loss = loss

        self.shuffle_buffer = shuffle_buffer
        self._size = size

        super().__init__(persistent=persistent, distribute=distribute)

        if batch_size % self.distribute.num_replicas_in_sync != 0:
            raise ValueError(
                "Number of replicas must divide batch size. Received "
                "batch_size={}, num_replicas_in_sync={}".format(
                    batch_size, distribute.num_replicas_in_sync))
        self.batch_size = int(
            batch_size / self.distribute.num_replicas_in_sync)

    def size(self, unroll):
        """Get number of batches for this unroll duration."""
        return math.floor(self._size / (unroll * self.batch_size))

    def get_dataset(self, unroll, seed=None, distribute=None):
        """Get problem dataset."""
        dataset = self.dataset
        if self.shuffle_buffer is not None:
            dataset = self.dataset.shuffle(self.shuffle_buffer, seed=seed)
        return self.distribute.experimental_distribute_dataset(
            dataset.batch(
                self.batch_size * unroll, drop_remainder=True
            ).prefetch(tf.data.experimental.AUTOTUNE))

    def get_parameters(self, seed=None):
        """Make variables corresponding to this problem."""
        return self.model.get_parameters(seed=seed)

    def objective(self, parameters, data):
        """Objective function."""
        x, y = data
        return self.loss(y, self.model.call(parameters, x))


def load_images(dataset, split="train"):
    """Load images and cast to float between 0 and 1.

    Note: shuffle_files MUST be false, since shuffling with seeds occurs later
    in the pipeline.

    Suggested Datasets:

    dataset       | shape     | k   | description
    --------------+-----------+-----+------------------------------------------
    cifar10       | 32x32x3   | 10  | images
    emnist        | 28x28     | 10  | handwritten digits
    fashion_mnist | 28x28     | 10  | images (clothing)
    kmnist        | 28x28     | 10  | handwritten Japanese characters
    mnist         | 28x28     | 10  | handwritten digits
    cifar100      | 32x32x3   | 100 | images
    omniglot      | 105x105x3 | 50  | handwritten characters
    stl10         | 96x96x3   | 10  | images

    Parameters
    ----------
    dataset : str
        Dataset name in tfds.
    split : str
        "train" or "test".

    Returns
    -------
    [tf.data.Dataset, tfds.core.DatasetInfo]
        [0] Loaded dataset
        [1] tfds info object.
    """
    dataset, info = tfds.load(
        dataset, split=split, shuffle_files=False,
        with_info=True, as_supervised=True)

    def _cast(x, y):
        return tf.cast(x, tf.float32) / 255., y

    return dataset.map(_cast), info


def _make_tfds(network, dataset="mnist", **kwargs):
    """Create training problem using tensorflow_datasets."""
    dataset, info = load_images(dataset)

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
        tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE),
        dataset, size=info.splits['train'].num_examples, **kwargs)


def mlp_classifier(
        dataset="mnist", layers=[128, ], activation="relu", **kwargs):
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
    if isinstance(activation, str):
        activation = tf.keras.activations.get(activation)

    def _network(input_shape, labels):
        return Sequential(
            [Dense(u, activation=activation) for u in layers]
            + [Dense(labels, activation=tf.nn.softmax)], input_shape)

    return _make_tfds(_network, dataset=dataset, **kwargs)


def conv_classifier(
        dataset="mnist",
        layers=[(5, 32, 2), ], activation=tf.nn.relu, **kwargs):
    """Create Convolutional classifier training problem.

    Keyword Args
    ------------
    dataset : str
        Dataset from tdfs catalog. Must have fixed input dimension and
        output labels.
    layers : int[][3]
        List of (kernel size, num_filters, stride) for convolutional layers
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
    if isinstance(activation, str):
        activation = tf.keras.activations.get(activation)

    def _preprocess(img):
        shape = img.shape.as_list()
        return tf.cast(tf.reshape(img, shape[:-1] + []), tf.float32) / 255.

    def _network(input_shape, labels):
        return Sequential(
            [Conv2D(f, k, stride=s, activation=activation)
             for f, k, s in layers]
            + [Dense(labels, activation=tf.nn.softmax)], input_shape)

    return _make_tfds(_network, dataset=dataset, **kwargs)
