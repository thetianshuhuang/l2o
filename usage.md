sbatch -p gtx -N 1 -n 1 -o log.txt -e err.txt -J TestTrain -t 00:10:00 -A Exploration-into-Aut test.sh

module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6

sbatch -p gtx -N 1 -n 1 -o eval_sgd.txt -J EvalSGD -t 00:02:00 -A Exploration-into-Aut test.sh


```
sbatch -p gtx -N 1 -n 1 -o eval_mult.txt -J EvalMult -t 02:00:00 -A Exploration-into-Aut --dependency=afterany:129442:129441:129460:129452 run_eval.sh
```




sbatch -p gtx -N 1 -n 1 -o choice-20x25C.log -J CH20x25C -t 10:00:00 -A Senior-Design_UT-ECE choice-20x25C.sh
sbatch -p gtx -N 1 -n 1 -o choice-20x25M.log -J CH20x25M -t 10:00:00 -A Senior-Design_UT-ECE choice-20x25M.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25C.log -J RP20x25C -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25C.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25M.log -J RP20x25M -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25M.sh


sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25CR.log -J 20x25CR -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25CR.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25CA.log -J 20x25CA -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25CA.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25CAR.log -J 20x25CAR -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25CAR.sh


sbatch -p gtx -N 1 -n 1 -o choice-20x50C.log -J CH20x50C -t 20:00:00 -A Senior-Design_UT-ECE choice-20x50C.sh
sbatch -p gtx -N 1 -n 1 -o choice-20x50M.log -J CH20x50M -t 20:00:00 -A Senior-Design_UT-ECE choice-20x50M.sh

sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25CAR-repeat.log -J RepeatC -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25CAR-repeat.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25MAR-repeat.log -J RepeatM -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25MAR-repeat.sh


sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25CARm-repeat.log -J RepeatC -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25CARm-repeat.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25MARm-repeat.log -J RepeatM -t 10:00:00 -A Senior-Design_UT-ECE rnnprop-20x25MARm-repeat.sh


sbatch -p gtx -N 1 -n 1 -o choice-20x25C-noscale.log -J CH20x25C -t 10:00:00 -A Senior-Design_UT-ECE choice-20x25C-noscale.sh
sbatch -p gtx -N 1 -n 1 -o choice-20x25M-noscale.log -J CH20x25M -t 10:00:00 -A Senior-Design_UT-ECE choice-20x25M-noscale.sh

sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25CAR-noscale.log -J NoScaleC -t 12:00:00 -A Senior-Design_UT-ECE rnnprop-20x25CAR-noscale.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25MAR-noscale.log -J NoScaleM -t 12:00:00 -A Senior-Design_UT-ECE rnnprop-20x25MAR-noscale.sh


sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25CARs-repeat.log -J ConvSlow -t 12:00:00 -A Senior-Design_UT-ECE rnnprop-20x25CARs-repeat.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25MARs-repeat.log -J MLPSlow -t 12:00:00 -A Senior-Design_UT-ECE rnnprop-20x25MARs-repeat.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25CARc-repeat.log -J ConvConst -t 12:00:00 -A Senior-Design_UT-ECE rnnprop-20x25CARc-repeat.sh
sbatch -p gtx -N 1 -n 1 -o rnnprop-20x25MARc-repeat.log -J MLPConst -t 12:00:00 -A Senior-Design_UT-ECE rnnprop-20x25MARc-repeat.sh
