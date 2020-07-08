import math
import tensorflow as tf
import tensorflow_datasets as tfds


from .problem import Problem


class Classifier(Problem):
    """Generic classifier problem

    Parameters
    ----------
    model : tf.keras.Model
        Core model (i.e. classifier or regression)
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
            self, model, loss, dataset,
            shuffle_buffer=None, batch_size=32, size=None, **kwargs):

        # Optimizer params
        self.model = model
        self.loss = loss
        self.trainable_variables = model.trainable_variables
        self.dataset = dataset

        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self._size = size

    def size(self, unroll):
        return math.floor(self._size / (unroll * self.batch_size))

    def clone_problem(self):
        return Classifier(
            tf.keras.models.clone_model(self.model), self.loss, None)

    def objective(self, data):
        x, y = data
        return self.loss(y, self.model(x))

    def get_dataset(self, unroll):
        if self.shuffle_buffer is not None:
            self.dataset = self.dataset.shuffle(self.shuffle_buffer)
        return self.dataset.batch(
            self.batch_size * unroll, drop_remainder=True)


def _make_tdfs(network, dataset="mnist", **kwargs):
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

    def _network(input_shape, labels):
        return tf.keras.Sequential(
            [tf.keras.layers.Flatten(input_shape=input_shape)]
            + [tf.keras.layers.Dense(d, activation=activation) for d in layers]
            + [tf.keras.layers.Dense(labels, activation="softmax")]
        )

    return _make_tdfs(_network, dataset=dataset, **kwargs)


def conv_classifier(
        dataset="mnist", layers=[(16, 3), ], activation="relu", **kwargs):
    """Create Convolutional classifier training problem. All layers are
    convolutional, except for the last layer which is fully connected.

    Keyword Args
    ------------
    dataset : str
        Dataset from tdfs catalog. Must have fixed input dimension and
        output labels.
    layers : int[][2]
        Array of (num_filters, kernel_size) for convolutional layers.
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
        return tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=input_shape)]
            + [tf.keras.layers.Conv2D(
                n, kernel_size=k, activation=activation) for n, k in layers]
            + [tf.keras.layers.Flatten()]
            + [tf.keras.layers.Dense(labels, activation="softmax")])

    return _make_tdfs(_network, dataset=dataset, **kwargs)
