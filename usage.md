sbatch -p gtx -N 1 -n 1 -o log.txt -e err.txt -J TestTrain -t 00:10:00 -A Exploration-into-Aut test.sh
