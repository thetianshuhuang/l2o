# Learn To Optimize
Framework for L2O extending ```tf.keras.optimizers.Optimizer```.


## Load trained network

```
import l2o

opt = l2o.load(
    l2o.policies.RNNPropOptimizer,
    directory="rnnprop-20x25CAR-repeat/period_99.0")
```