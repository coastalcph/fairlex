CONFIG_NAME='lex-longformer'
BATCH_SIZE = 32

deepspeed --num_gpus=8 python models/pre-training/run_mlm.py \
    --config_name ./data/models/${CONFIG_NAME} \
    --train_file ./data/datasets/mlm_dump.txt \
    --max_seq_length 4096  \
    --do_train  \
    --do_eval  \
    --overwrite_output_dir  \
    --evaluation_strategy epoch  \
    --save_strategy epoch  \
    --save_total_limit 5  \
    --num_train_epochs 20  \
    --learning_rate 1e-4  \
    --per_device_train_batch_size ${BATCH_SIZE}  \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --adam_eps 1e-06 \
    --fp16 \
    --deepspeed ../../ds_config.json \
    --output_dir ./data/models/${CONFIG_NAME}