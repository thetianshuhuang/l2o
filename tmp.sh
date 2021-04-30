#!/bin/sh
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/more-choice/base/1 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/more-choice/base/2 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/more-choice/base/3 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/more-choice/base/4 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/more-choice/base/5 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/more-choice/base/6 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/more-choice/base/7 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/more-choice/base/8 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/less-choice/base/1 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/less-choice/base/2 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/less-choice/base/3 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/less-choice/base/4 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/less-choice/base/5 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/less-choice/base/6 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/less-choice/base/7 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/less-choice/base/8 \
    --repeat=10


#!/bin/sh
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
python3 train.py \
    --presets=conv_train,less_choice,il_more \
    --policy=rnnprop \
    --strategy=curriculum \
    --directory=results/rnnprop/choice2/5
python3 evaluate.py \
    --problem=conv_train \
    --directory=results/rnnprop/choice2/5 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice2/1 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice2/2 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice2/3 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice2/4 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice2/5 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice2/6 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice2/7 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice2/8 \
    --repeat=10


#!/bin/sh
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
python3 train.py \
    --presets=conv_train,more_choice,il_more \
    --policy=rnnprop \
    --strategy=curriculum \
    --directory=results/rnnprop/choice6/6
python3 evaluate.py \
    --problem=conv_train \
    --directory=results/rnnprop/choice6/6 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice6/1 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice6/2 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice6/3 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice6/4 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice6/5 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice6/6 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice6/7 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/choice6/8 \
    --repeat=10


#!/bin/sh
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
python3 train.py \
    --presets=conv_train,more_choice,il_more \
    --policy=rnnprop \
    --strategy=curriculum \
    --directory=results/rnnprop/os-gs/7
python3 evaluate.py \
    --problem=conv_train \
    --directory=results/rnnprop/os-gs/7 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/os-gs/1 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/os-gs/2 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/os-gs/3 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/os-gs/4 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/os-gs/5 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/os-gs/6 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/os-gs/7 \
    --repeat=10
python3 evaluate.py \
    --problem=conv_deeper_pool,conv_cifar10_pool \
    --directory=results/rnnprop/os-gs/8 \
    --repeat=10



