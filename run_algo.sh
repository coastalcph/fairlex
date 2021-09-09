#!/bin/bash

ALGORITHM='ERM'
GPU_NUMBER=5
DATASET='ecthr'
GROUP_FIELD=age

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${ALGORITHM} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_1" --fp16 True --seed 1 --groupby_fields ${GROUP_FIELD}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${ALGORITHM} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_2" --fp16 True --seed 2 --groupby_fields ${GROUP_FIELD}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${ALGORITHM} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_3" --fp16 True --seed 3 --groupby_fields ${GROUP_FIELD}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${ALGORITHM} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_4" --fp16 True --seed 4 --groupby_fields ${GROUP_FIELD}
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${ALGORITHM} --root_dir data/datasets --log_dir "./logs/${DATASET}/${ALGORITHM}/${GROUP_FIELD}/seed_5" --fp16 True --seed 5 --groupby_fields ${GROUP_FIELD}