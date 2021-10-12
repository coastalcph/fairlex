#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=8000M
# we run on the gpu partition
#SBATCH -p gpu --gres=gpu:titanx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=24:00:00
#SBATCH -o logs/scotus_v04/scotus_issue_area_tfidf_interpreter_decision_direction=liberal.log
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairlex 

PARAM=""
ATTRIBUTE=decisionDirection
ALGO=ERM

nvidia-smi
python3 -c "import torch; print('cuda is available = ',torch.cuda.is_available())"


RAND_NUM=1
echo 'ALGO='$ALGO
echo 'RAND_NUM='$RAND_NUM
COMMAND="run_expt.py --dataset scotus --algorithm $ALGO $PARAM --root_dir data/linear_interpreter_datasets_decision-direction=liberal_seed_1/ \
--log_dir logs_final_tfidf_interpreter/scotus/$ALGO/$ATTRIBUTE=liberal/seed_$RAND_NUM/ 
--split_scheme official \
--resume True \
--n_epochs 40 \
--early_stopping_patience 40 \
--seed $RAND_NUM \
--groupby_fields $ATTRIBUTE \
--save_best \
--lr 3e-3 \
--batch_size 12 \
--n_groups_per_batch 2 \
--model regressor \
--train_transform tfidf \
--eval_transform tfidf \
--model_kwargs tfidf_vectorizer_path=data/datasets/scotus_v0.4/tfidf_tokenizer_3grams_10000.pkl \
--model_path logs_final_tfidf_regressor/scotus/ERM/decisionDirection/seed_1/
"
echo ''
echo $COMMAND
PYTHONPATH=src python $COMMAND 