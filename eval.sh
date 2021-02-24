#!/bin/sh
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
python3 evaluate.py \
    --problem=conv_train,conv_kmnist,conv_avg,conv_wider \
    --directory=results/choice/long1 \
    --repeat=10 \
    --periods=49
python3 evaluate.py \
    --problem=conv_train,conv_kmnist,conv_avg,conv_wider \
    --directory=results/choice/long2 \
    --repeat=10 \
    --periods=49
python3 evaluate.py \
    --problem=conv_train,conv_kmnist,conv_avg,conv_wider \
    --directory=results/choice/long3 \
    --repeat=10 \
    --periods=49
python3 evaluate.py \
    --problem=conv_train,conv_kmnist,conv_avg,conv_wider \
    --directory=results/choice/long4 \
    --repeat=10 \
    --periods=49
python3 evaluate.py \
    --problem=conv_train,conv_kmnist,conv_avg,conv_wider \
    --directory=results/choice_1/newpipe-1 \
    --repeat=10 \
    --periods=49
python3 evaluate.py \
    --problem=conv_train,conv_kmnist,conv_avg,conv_wider \
    --directory=results/choice_1/newpipe-2 \
    --repeat=10 \
    --periods=49