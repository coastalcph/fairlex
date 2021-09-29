#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=8000M
# we run on the gpu partition
#SBATCH -p gpu --gres=gpu:titanrtx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=24:00:00
#SBATCH -o logs/scotus_v04/scotus_issue_area_%A-%a.log
#SBATCH --array=1-5
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairlex 

PARAM=""
ATTRIBUTE=respondent
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
    ALGO=adversarialRemoval
fi
if [[ $SLURM_ARRAY_TASK_ID == 4 ]];
then
    ALGO=IRM
fi
if [[ $SLURM_ARRAY_TASK_ID == 5 ]];
then
    ALGO=REx
fi

nvidia-smi
python3 -c "import torch; print('cuda is available = ',torch.cuda.is_available())"


for RAND_NUM in 1 2 3 4 5
do
    echo 'ALGO='$ALGO
    echo 'RAND_NUM='$RAND_NUM
    COMMAND="run_expt.py --dataset scotus \
    --algorithm $ALGO $PARAM \
    --root_dir data/datasets \
    --log_dir logs_final/scotus/$ALGO/$ATTRIBUTE/seed_$RAND_NUM/ \
    --split_scheme official \
    --seed $RAND_NUM \
    --groupby_fields $ATTRIBUTE \
    --lr 1e-5 \
    --train_transform bert \
    --eval_transform bert \
    --model scotus-mini-hier-bert \
    --save_best \
    --fp16 True \
    --n_groups_per_batch 4 \
    --batch_size 12"
    echo ''
    echo $COMMAND
    PYTHONPATH=src python $COMMAND 
done
