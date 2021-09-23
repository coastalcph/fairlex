MODEL_NAME='mini-longformer'
DATASET='ecthr'
BATCH_SIZE=2
CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/iliasc/fairlex-wilds/data/models/run_mlm.py \
    --model_name_or_path /home/iliasc/fairlex-wilds/data/models/${MODEL_NAME} \
    --train_file /home/iliasc/fairlex-wilds/data/datasets/${DATASET}_v1.0/${DATASET}_dump.txt \
    --max_seq_length 4096  \
    --line_by_line true \
    --do_train true \
    --do_eval true  \
    --overwrite_output_dir false  \
    --evaluation_strategy steps  \
    --save_strategy steps  \
    --save_total_limit 2  \
    --num_train_epochs 10  \
    --learning_rate 1e-4  \
    --per_device_train_batch_size ${BATCH_SIZE}  \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --adam_eps 1e-6 \
    --weight_decay 0.01 \
    --warmup_steps 4000 \
    --fp16 true \
    --output_dir /home/iliasc/fairlex-wilds/data/models/${DATASET}-${MODEL_NAME}-v3 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps=100 \
    --eval_steps=1000 \
    --save_strategy steps \
    --save_steps=1000
