#!/bin/bash

ALGORITHM='REx'
GPU_NUMBER=0
DATASET='ecthr'
GROUP_FIELD=age
BATCH_SIZE=12
WANDB_PROJECT=fairlex-wilds
N_GROUPS=3

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb False --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./logs2/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_1" --fp16 True --seed 1 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
# CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_2" --fp16 True --seed 2 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
# CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_3" --fp16 True --seed 3 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
# CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_4" --fp16 True --seed 4 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
# CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --use_wandb True --dataset ${DATASET} --algorithm ${ALGORITHM} --batch_size ${BATCH_SIZE} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_5" --fp16 True --seed 5 --groupby_fields ${GROUP_FIELD} --n_groups_per_batch ${N_GROUPS}
