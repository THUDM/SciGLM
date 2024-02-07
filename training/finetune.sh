LR=3e-6
WR=0.0
LST=linear
epoch=2
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --include="localhost:0,1,2,3" --master_port $MASTER_PORT main.py \
    --deepspeed /path/to/deepspeed.json \
    --do_train \
    --train_file /path/to/SciInstruct.json \
    --eval_steps 200 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /path/to/ChatGLM3-6b-Base \
    --output_dir ./output/SciGLM-LR$LR-WR$WR-$LST-epoch$epoch \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --num_train_epochs $epoch \
    --logging_steps 20 \
    --save_strategy epoch \
    --learning_rate $LR \
    --lr_scheduler_type $LST \
    --warmup_ratio $WR \
    --fp16