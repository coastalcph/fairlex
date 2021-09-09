CONFIG_NAME='mini-longformer'
DATASET = 'ecthr'
BATCH_SIZE = 32

deepspeed --num_gpus=1 python ./data/models/run_mlm.py \
    --config_name ./data/models/${CONFIG_NAME} \
    --train_file ./data/datasets/${DATASET}_v1.0/text.raw \
    --max_seq_length 4096  \
    --do_train  \
    --do_eval  \
    --overwrite_output_dir  \
    --evaluation_strategy epoch  \
    --save_strategy epoch  \
    --save_total_limit 5  \
    --num_train_epochs 20  \
    --learning_rate 1e-5  \
    --per_device_train_batch_size ${BATCH_SIZE}  \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --adam_eps 1e-06 \
    --fp16 \
    --deepspeed ../../ds_config.json \
    --output_dir ./data/models/${DATASET}-${CONFIG_NAME}