
import tensorflow as tf
import tensorflow_datasets as tfds


from .problem import Problem


def reset_recursive(layer):
    """Recursively reset variables for a keras model or layer.

    Will reset ``.kernel`` using ``.kernel_initializer``, ``.bias`` with
    ``.bias_initializer``, and any layers held by ``.layers``.

    Parameters
    ----------
    layer : tf.keras.Model | tf.keras.Layer
        Model or layer to reset.
    """
    if hasattr(layer, 'kernel'):
        layer.kernel.assign(layer.kernel_initializer(shape=layer.kernel.shape))
    if hasattr(layer, 'bias'):
        layer.bias.assign(layer.bias_initializer(shape=layer.bias.shape))
    if hasattr(layer, 'layers'):
        for x in layer.layers:
            reset_recursive(x)


class Classifier(Problem):
    """Abstract classifier problem

    Parameters
    ----------
    model : tf.keras.Model
        Core model (i.e. classifier or regression)
    loss : callable (tf.Tensor, tf.Tensor -> tf.Tensor)
        Computes loss from model output and ground truth. Dataset should have
        input as the first array and output as the second.
    dataset : tf.data.Dataset
        Tensorflow dataset to use
    """

    def __init__(self, model, loss, dataset, **kwargs):

        super().__init__(**kwargs)

        self.model = model
        self.loss = loss
        self.dataset = dataset

        self.trainable_variables = model.trainable_variables
        self.initializers = []

    def clone_problem(self):
        return Classifier(
            tf.keras.models.clone_model(self.model), self.dataset)

    def objective(self, data):
        x, y = data
        return self.loss(y, self.model(x))

    def reset(self, *args, **kwargs):

        # Manually reinitialize since keras.Model doesn't have a clean way
        # of doing so
        reset_recursive(self.model)

        super().reset(*args, **kwargs)


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

    cce = tf.keras.losses.CategoricalCrossentropy()

    def loss(y_hat, y):
        return cce(tf.one_hot(y), y_hat)

    return Classifier(
        network(input_shape, labels), loss, dataset, **kwargs)


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
