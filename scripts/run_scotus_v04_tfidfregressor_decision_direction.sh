#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=8000M
# we run on the gpu partition
#SBATCH -p gpu --gres=gpu:titanx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=24:00:00
#SBATCH -o logs/scotus_v04/scotus_issue_area_tfidf_regressor_decision_direction.log
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairlex 

PARAM=""
ATTRIBUTE=decisionDirection
ALGO=ERM

nvidia-smi
python3 -c "import torch; print('cuda is available = ',torch.cuda.is_available())"


for RAND_NUM in 1 2 3 4 5
do
    echo 'ALGO='$ALGO
    echo 'RAND_NUM='$RAND_NUM
    COMMAND="run_expt.py --dataset scotus --algorithm $ALGO --root_dir data/datasets \
    --log_dir logs_final_tfidf_regressor/scotus/$ALGO/$ATTRIBUTE/seed_$RAND_NUM/ --split_scheme official \
    --seed $RAND_NUM \
    --groupby_fields $ATTRIBUTE \
    --save_best \
    --lr 3e-3 \
    --batch_size 12 \
    --n_groups_per_batch 2 \
    --model regressor \
    --train_transform tfidf \
    --eval_transform tfidf \
    --model_kwargs tfidf_vectorizer_path=/home/npf290/dev/fairlex-wilds/data/datasets/scotus_v0.4/tfidf_tokenizer_3grams_10000.pkl"
    echo ''
    echo $COMMAND
    PYTHONPATH=src python $COMMAND 
done