#!/bin/sh
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
python3 evaluate.py \\
    --problem=conv_kmnist,conv_avg,conv_wider \\
    --directory=results/choice/base,results/choice/fconv,results/choice/fconv-wup2,results/choice/wup2,results/choice_ext/base,results/choice_ext/fconv,results/choice_ext/fconv-wup2,results/choice_ext/wup2 \\
    --repeat=10 --periods=49
