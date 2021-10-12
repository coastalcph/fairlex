#!/bin/bash

ALGORITHM='adversarialRemoval'
GPU_NUMBER=7
DATASET='ecthr'
GROUP_FIELD=gender
BATCH_SIZE=18
WANDB_PROJECT=fairlex-wilds
LOG_DIR=hier_final_logs
N_GROUPS=3
LOSS_FUNCTION=binary_cross_entropy
FOLDER_NAME='adversarialRemoval'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_1" --seed 1 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_2" --seed 2 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_3" --seed 3 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_4" --seed 4 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --loss_function ${LOSS_FUNCTION} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./${LOG_DIR}/${DATASET}/${FOLDER_NAME}/${GROUP_FIELD}/seed_5" --seed 5 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
