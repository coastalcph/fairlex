MODEL_NAME='fairlex-cail-minilm'
DATASET='cail'
BATCH_SIZE=8

python language_modelling/run_mlm.py \
    --model_name_or_path microsoft/MiniLM-L12-H384-uncased \
    --train_file data/datasets/${DATASET}_v1.0/${DATASET}_dump.txt \
    --max_seq_length 128  \
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
    --warmup_steps 24000 \
    --fp16 true \
    --output_dir /home/iliasc/fairlex-wilds/data/models/${DATASET}-${MODEL_NAME} \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps=500 \
    --eval_steps=5000 \
    --save_strategy steps \
    --save_steps=5000
