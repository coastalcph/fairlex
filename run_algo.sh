#!/bin/bash

ALGORITHM='ERM'
DATASET='ecthr'
GROUP_FIELD=applicant_gender
BATCH_SIZE=16
WANDB_PROJECT=fairlex-wilds
LOG_DIR=logs
N_GROUPS=3
LOSS_FUNCTION=cross_entropy
FOLDER_NAME='ERM'

python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_1" --seed 1 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_2" --seed 2 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_3" --seed 3 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_4" --seed 4 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_5" --seed 5 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
