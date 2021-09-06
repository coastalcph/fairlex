#!/bin/bash

algorithm='ERM'
GPU_NUMBER=7
DATASET='ecthr'
GROUP_FIELD='age'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${algorithm} --root_dir data/datasets --log_dir "./logs/ecthr/${algorithm}_1" --fp16 True --seed 1 --groupby_fields [${GROUP_FIELD}]
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${algorithm} --root_dir data/datasets --log_dir "./logs/ecthr/${algorithm}_2" --fp16 True --seed 2 --groupby_fields [${GROUP_FIELD}]
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${algorithm} --root_dir data/datasets --log_dir "./logs/ecthr/${algorithm}_3" --fp16 True --seed 3 --groupby_fields [${GROUP_FIELD}]
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${algorithm} --root_dir data/datasets --log_dir "./logs/ecthr/${algorithm}_4" --fp16 True --seed 4 --groupby_fields [${GROUP_FIELD}]
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${algorithm} --root_dir data/datasets --log_dir "./logs/ecthr/${algorithm}_5" --fp16 True --seed 5 --groupby_fields [${GROUP_FIELD}]