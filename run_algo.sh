#!/bin/bash

algorithm='ERM'
GPU_NUMBER=7
DATASET='ecthr'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${algorithm} --root_dir data/datasets --log_dir "./logs/ecthr/${algorithm}_1" --fp16 True
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${algorithm} --root_dir data/datasets --log_dir "./logs/ecthr/${algorithm}_2" --fp16 True
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python run_expt.py --dataset ${DATASET} --algorithm ${algorithm} --root_dir data/datasets --log_dir "./logs/ecthr/${algorithm}_3" --fp16 True