# Optimizer Distillation

Instructions for running Optimizer Distillation experiments.

See ```README.md``` for installation and dependency requirements.

### Mean, min-max distillation

Training with min-max distillation, rnnprop as target, small pool, convolutional network for training:
```
python train.py \
    --presets=conv_train,adam,rmsprop,il_more \
    --strategy=curriculum \
    --policy=rnnprop \
    --directory=results/rnnprop/min-max/1
```

Evaluation:
```
python evaluate.py \
    --problem=conv_train \
    --directory=results/rnnprop/min-max/1 \
    --repeat=10
```

Min-max distillation is the default setting. To use mean distillation, add the ```reduce_mean``` preset.

### Choice distillation

Train the choice policy:
```
python train.py \
    --presets=conv_train,cl_fixed \
    --strategy=repeat \
    --policy=less_choice \
    --directory=results/less-choice/base/1
```

Train for the final distillation step:
```
python train.py \
    --presets=conv_train,less_choice,il_more \
    --strategy=curriculum \
    --policy=rnnprop \
    --directory=results/rnnprop/choice2/1
```

Evaluation:
```
python evaluate.py \
    --problem=conv_train \
    --directory=results/rnnprop/choice2/1 \
    --repeat=10
```

### Stability-Aware Optimizer Distillation

FGSM, PGD, Adaptive PGD, Gaussian, and Adaptive Gaussian perturbations are implemented.
| Perturbation | Description | Preset Name | Magnitude Parameter |
| - | - | - | - |
| FGSM | Fast Gradient Sign Method | ```fgsm``` | ```step_size``` |
| PGD | Projected Gradient Descent | ```pgd``` | ```magnitude``` |
| Adaptive PGD | Adaptive PGD / "Clipped" GD | ```cgd``` | ```magnitude``` |
| Random | Random Gaussian | ```gaussian``` | ```noise_stddev``` |
| Adaptive Random | Random Gaussian, Adaptive Magnitude | ```gaussian_rel``` | ```noise_stddev``` |

Modify the magnitude of noise by passing
```
--policy/perturbation/config/[Magnitude Parameter]=[Desired Magnitude].
```

For PGD variants, the number of adversarial attack steps can also be modified:
```
--policy/perturbation/config/steps=[Desired Steps]
```
