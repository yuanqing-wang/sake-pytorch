#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
###BSUB -R V100
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 23:59
#BSUB -n 1

python run.py --property U --lr 1e-5
# python run_baseline.py --property U --lr 5e-4


