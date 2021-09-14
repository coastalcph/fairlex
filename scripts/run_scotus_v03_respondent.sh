#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=8000M
# we run on the gpu partition
#SBATCH -p gpu --gres=gpu:titanrtx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=24:00:00
#SBATCH -o logs/scotus_v03/scotus_issue_area_%A-%a.log
#SBATCH --array=1-4
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

nvidia-smi
python3 -c "import torch; print('cuda is available = ',torch.cuda.is_available())"


for RAND_NUM in 123 42 1028 91272 627
do
    echo 'ALGO='$ALGO
    echo 'RAND_NUM='$RAND_NUM
    COMMAND="run_expt.py --dataset scotus --algorithm $ALGO $PARAM --root_dir data \
    --log_dir logs/scotus_v03/$ATTRIBUTE/$ALGO/$RAND/temporal --split_scheme temporal \
    --dataset_kwargs protected_attribute=$ATTRIBUTE --seed $RAND_NUM\
    --save_best"
    echo ''
    echo $COMMAND
    PYTHONPATH=src python $COMMAND 
done
