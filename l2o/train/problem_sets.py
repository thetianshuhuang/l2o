"""
Datasets

dataset       | shape     | k   | description
--------------+-----------+-----+---------------------------------------------
cifar10       | 32x32x3   | 10  | images
emnist        | 28x28     | 10  | handwritten digits
fashion_mnist | 28x28     | 10  | images (clothing)
kmnist        | 28x28     | 10  | handwritten Japanese characters
mnist         | 28x28     | 10  | handwritten digits
cifar100      | 32x32x3   | 100 | images
omniglot      | 105x105x3 | 50  | handwritten characters
stl10         | 96x96x3   | 10  | images
"""


from ..problems import ProblemSpec, Quadratic, mlp_classifier, conv_classifier


# Quadratic
simple_train = [
    ProblemSpec(Quadratic, [dim], {})
    for dim in [10, 15, 20, 25, 30, 40, 50]
]


# 18 problems
mlp_train = [
    ProblemSpec(
        mlp_classifier, [],
        {
            "layers": [128 for _ in range(depth)],
            "activation": activation,
            "dataset": dataset
        })
    for depth in [1, 2, 3]
    for dataset in ["emnist", "mnist", "fashion_mnist"]
    for activation in ["relu", "sigmoid"]
]


# 18 problems
conv_train = [
    ProblemSpec(
        conv_classifier, [],
        {
            "layers": [(16, kernel) for _ in range(depth)],
            "activation": "relu",
            "dataset": dataset
        })
    for depth in [1, 2, 3]
    for kernel in [3, 5]
    for dataset in ["emnist", "mnist", "fashion_mnist"]
]
