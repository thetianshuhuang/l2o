# Learn To Optimize
Framework for L2O extending ```tf.keras.optimizers.Optimizer```.

## Dependencies

- Tensorflow >= 2.3.0. *NOT COMPATIBLE* with tf 2.2 or earlier due to a bug that occurs when tf.function parses nested structures in ```get_concrete_function```.
- Numpy, any version that works with tensorflow
- Pandas, any version

## Training Support

### Supported
- Training on multiple GPUs with MirroredStrategy
- Training on multiple devices with MirroredStrategy should work in theory, but is not tested.

### Not Supported
- Sparse training
- Training with model split between different GPUs

## Tmp

### Load trained network

```
import l2o

opt = l2o.load(
    l2o.policies.RNNPropOptimizer,
    directory="rnnprop-20x25CAR-repeat/period_99.0")
```


### Prepare environment
```
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
```

### Naming
```
policy-{unroll}x{depth}{problem}{teachers}-mods
```

Codes:
- Unroll=20,50,100
- Depth=25
- Problem=M,C
    - M=MLP
    - C=Conv
- Teachers=A,R,AR (alphabetical)
    - A=Adam
    - R=RMSProp
    - D=Radam
    - S=SGD

Examples:
```
choice-20x25M
scale-20x25MAR-slow
scale-20x25MAR-fast
scale-20x25MACRS-slow
```

### Batch Template
```
sbatch -p gtx -N 1 -n 1 -o choice-20x25C.log -J CH20x25C -t 10:00:00 -A Senior-Design_UT-ECE choice-20x25C.sh
```
