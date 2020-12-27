sbatch -p gtx -N 1 -n 1 -o log.txt -e err.txt -J TestTrain -t 00:10:00 -A Exploration-into-Aut test.sh

module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6

sbatch -p gtx -N 1 -n 1 -o eval_sgd.txt -J EvalSGD -t 00:02:00 -A Exploration-into-Aut test.sh


```
sbatch -p gtx -N 1 -n 1 -o eval_mult.txt -J EvalMult -t 02:00:00 -A Exploration-into-Aut --dependency=afterany:129442:129441:129460:129452 run_eval.sh
```
