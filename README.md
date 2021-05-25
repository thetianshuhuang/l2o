# Learn To Optimize
Gradient-based Learning to Optimize Framework for extending ```tf.keras.optimizers.Optimizer```.

## Description of Modules

- ```l2o.deserialize```: utilities used to deserialize json and command line arguments; used by ```train.py```, ```evaluation.py```, etc
- ```l2o.evaluate```: evaluation methods and optimizee prototypes for evaluation
- ```l2o.optimizer```: ```tf.keras.optimizers.Optimizer``` extension back end
- ```l2o.policies```: policy descriptions
- ```l2o.problems```: optimizees used in training; dataset management
- ```l2o.strategy```: training strategy (i.e. Curriculum Learning)
- ```l2o.train```: truncated backpropagation implementation

## Dependencies

| Library | Known Working | Known Not Working |
| - | - | - |
| tensorflow | 2.3.0, 2.4.1 | <= 2.2 |
| tensorflow_datasets | 3.1.0, 4.2.0 | n/a |
| pandas | 0.24.1, 1.2.4 | n/a |
| numpy | 1.18.5, 1.19.2 | >=1.20 |
| scipy | 1.4.1, 1.6.2 | n/a |

## Common Errors

- Nested structure issue: tensorflow 2.2 or earlier have a bug in parsing nested structures in ```get_concrete_function```. Solution: upgrade tensorflow to >=2.3.0.

- ```OperatorNotAllowedInGraphError: iterating over `tf.Tensor` is not allowed```: see [issue here](https://github.com/tensorflow/tensorflow/issues/44146); caused by tensorflow dependency ```gast``` version mismatch. Solution: ```pip install gast==0.3.3```.

- ```NotImplementedError: Cannot convert a symbolic Tensor (Size:0) to a numpy array.```: see [question here](https://stackoverflow.com/questions/66207609/notimplementederror-cannot-convert-a-symbolic-tensor-lstm-2-strided-slice0-t/66207610); caused by ```numpy``` API version mismatch. Solution: downgrade numpy to <1.20 (Tested: 1.19.2, 1.18.5)

- GPUs not showing up: make sure the ```tensorflow-gpu``` conda package is installed, not just ```tensorflow```.

## Training Support

### Supported
- Training on multiple GPUs with MirroredStrategy
- Training on multiple devices with MirroredStrategy should work in theory, but is not tested.

### Not Supported
- Sparse training
- Training with model split between different GPUs

## Known Problems
- Some systems may be up to 2x slower than others, even with identical GPUs, sufficient RAM, and roughly equivalent CPUs. I believe this is due to some kernel launch inefficiency or CUDA/TF configuration problem.
- Sometimes, training will "NaN" out, and turn all optimizer weights to NaN. There is supposed to be a guard preventing NaN gradient updates from being committed, but it doesn't seem to be fully working.

## Todos
- Add optimizer export, quick loading. Optimizer export should export the final checkpoint and a file with just optimizer config; other information (i.e. training metadata) should be stored in a separate json. Config needed: ```policy/*```, ```strategy/validation_warmup * strategy/validation_unroll```, ```strategy/validaton_warmup_rate```.
