CUDA_VISIBLE_DEVICES=2,3 python train.py \
    --config_name vanilla \
    --inputter_name vanilla \
    --eval_input_file ./_reformat/valid.txt \
    --seed 13 \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 2 \
    --warmup_steps 100 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true