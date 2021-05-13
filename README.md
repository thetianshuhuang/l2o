# Learn To Optimize
Framework for L2O extending ```tf.keras.optimizers.Optimizer```.

## Dependencies

| Library | Known Working | Known Not Working |
| - | - | - |
| tensorflow | 2.3.0, 2.4.1 | <= 2.2 |
| tensorflow_datasets | 3.1.0, 4.2.0 | n/a |
| pandas | 0.24.1, 1.2.4 | n/a |
| numpy | 1.18.5, 1.19.2 | n/a |
| scipy | 1.4.1, 1.6.2 | n/a |

Note that tensorflow 2.2 or earlier is not compatible due to a bug in parsing nested structures in ```get_concrete_function```.

## Common Errors

- Some variation of
```OperatorNotAllowedInGraphError: iterating over `tf.Tensor` is not allowed```: see https://github.com/tensorflow/tensorflow/issues/44146; caused by tensorflow dependency version mismatch.

- GPUs not showing up: make sure the ```tensorflow-gpu``` conda package is installed, not just ```tensorflow```.

## Training Support

### Supported
- Training on multiple GPUs with MirroredStrategy
- Training on multiple devices with MirroredStrategy should work in theory, but is not tested.

### Not Supported
- Sparse training
- Training with model split between different GPUs