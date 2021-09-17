#!/bin/bash

ALGORITHM='REx'
GPU_NUMBER=2
DATASET='fscs'
GROUP_FIELD=legal_area
BATCH_SIZE=12
WANDB_PROJECT=fairlex-wilds
N_GROUPS=6

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./final_logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_1" --seed 1 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./final_logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_2" --seed 2 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./final_logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_3" --seed 3 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./final_logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_4" --seed 4 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./final_logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_5" --seed 5 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
