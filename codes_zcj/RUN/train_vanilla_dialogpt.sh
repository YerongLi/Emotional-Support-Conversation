CUDA_VISIBLE_DEVICES=6,7 python train.py \
    --config_name vanilla_dialogpt \
    --inputter_name vanilla \
    --eval_input_file ./_reformat/valid.txt \
    --seed 13 \
    --max_input_length 180 \
    --max_decoder_input_length 45 \
    --train_batch_size 1\
    --gradient_accumulation_steps 1 \
    --eval_batch_size 1 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_steps 100 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true