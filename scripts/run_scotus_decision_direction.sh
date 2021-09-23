#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=8000M
# we run on the gpu partition
#SBATCH -p gpu --gres=gpu:titanrtx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=24:00:00
#SBATCH -o logs/scotus_decision_direction_%A-%a.log
#SBATCH --array=1-3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairlex 

RAND_NUM=123
PARAM=""
ATTRIBUTE=decisionDirection

if [[ $SLURM_ARRAY_TASK_ID == 1 ]];
then
    ALGO=groupDRO
fi
if [[ $SLURM_ARRAY_TASK_ID == 2 ]];
then
    ALGO=ERM
fi
if [[ $SLURM_ARRAY_TASK_ID == 3 ]];
then
    ALGO=deepCORAL
    PARAM="--groupby_fields decisionDirection"
fi
nvidia-smi
python3 -c "import torch; print('cuda is available = ',torch.cuda.is_available())"

echo 'ALGO='$ALGO
echo 'RAND_NUM='$RAND_NUM

COMMAND="run_expt.py --dataset scotus --algorithm $ALGO $PARAM --root_dir data \
--log_dir logs/scotus/decision_direction/$ALGO/official \
--save_best --seed $RAND_NUM --split_scheme official \
--dataset_kwargs protected_attribute=decisionDirection"
echo ''
echo "SPLIT_SCHEME=official"
echo $COMMAND
PYTHONPATH=src python $COMMAND 2>&1 

# COMMAND="run_expt.py --dataset scotus --algorithm $ALGO $PARAM --root_dir data \
# --log_dir logs/scotus/decision_direction/$ALGO/temporal \
# --save_best --seed $RAND_NUM --split_scheme temporal --dataset_kwargs protected_attribute=decisionDirection"
# echo ""
# echo "SPLIT_SCHEME=temporal"
# echo $COMMAND
# PYTHONPATH=src python $COMMAND 2>&1 

# COMMAND="run_expt.py --dataset scotus --algorithm $ALGO $PARAM --root_dir data \
# --log_dir logs/scotus/decision_direction/$ALGO/uniform \
# --save_best --seed $RAND_NUM --split_scheme uniform --dataset_kwargs protected_attribute=decisionDirection"
# echo ""
# echo "SPLIT_SCHEME=uniform"
# echo $COMMAND
# PYTHONPATH=src python $COMMAND 2>&1 