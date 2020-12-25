sbatch -p gtx -N 1 -n 1 -o log.txt -e err.txt -J TestTrain -t 00:10:00 -A Exploration-into-Aut test.sh

module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
