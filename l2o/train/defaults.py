

meta_learning = {
    "training": {
        "unroll_weights": "mean",
        "teachers": [],
        "imitation_optimizer": None,
        "strategy": "mean",
        "p_teacher": 0,
        "epochs": 1,
        "repeat": 1,
        "persistent": False
    },
    "curriculum": {
        "schedule": lambda i: 50 * (2**i),
        "min_periods": 100,
        "epochs_per_period": 10,
        "max_stages": 0,
        "directory": "weights",
    },
    "problems": [
        ("mlp_classifier", [],
         {"layers": [20], "dataset": "mnist", "activation": "sigmoid",
          "shuffle_buffer": 1024, "batch_size": 32}),
    ],
    "loss": {
        "use_log_objective": True,
        "scale_objective": False,
        "obj_train_max_multiplier": -1,
        "use_numerator_epsilon": True,
        "epsilon": 1e-6
    },
    # Optimizer settings.
    # If this is a tf.keras.optimizers.Optimizer instead, it will be passed
    # through directly.
    "optimizer": {
        "class_name": "Adam",
        "config": {
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
        }
    }
}


scale_hierarchical_args = {
    # Scale network args
    "param_units": 10,
    "tensor_units": 5,
    "global_units": 5,
    "init_lr": (1e-6, 1e-2),
    "timescales": 5,
    "epsilon": 1e-10,
    "name": "ScaleHierarchicalOptimizer",
    # GRUCell args
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
}


scale_basic_args = {
    # Scale network args
    "layers": (20, 20),
    "init_lr": (1., 1.),
    "name": "ScaleBasicOptimizer",
    # LSTMCell Args
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
    "unit_forget_bias": True,
}


rnnprop_args = {
    # RNNProp
    "layers": (20, 20),
    "beta_1": 0.9,
    "beta_2": 0.9,
    "alpha": 0.1,
    "epsilon": 1e-10,
    "name": "RNNPropOptimizer",
    # LSTMCell Args
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
    "unit_forget_bias": True,
}


dmoptimizer_args = {
    # DMOptimizer
    "layers": (20, 20),
    "name": "DMOptimizer",
    # LSTMCell Args
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
    "unit_forget_bias": True,
}
