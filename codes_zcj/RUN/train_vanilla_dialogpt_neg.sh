CUDA_VISIBLE_DEVICES=2,3 python train_cls.py \
    --config_name vanilla_dialogpt_neg \
    --inputter_name vanilla \
    --eval_input_file ./_reformat/valid.txt \
    --seed 13 \
    --max_input_length 180 \
    --max_decoder_input_length 45 \
    --train_batch_size 6\
    --gradient_accumulation_steps 1 \
    --eval_batch_size 6 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --warmup_steps 100 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true