#!/bin/bash

WEIGHT_VERSION=$1

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    chatbind/train/train_imagebind_select_mem.py \
    --model_name_or_path ./checkpoints/vicuna-7b \
    --version $WEIGHT_VERSION \
    --data_path /path/to/blip_laion_cc_sbu_558k.json \
    --image_folder /path/to/blip_laion_cc_sbu_558k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/chatbind-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

python scripts/extract_mm_projector.py \
    --model_name_or_path ./checkpoints/chatbind-7b-pretrain \
    --output ./checkpoints/mm_projector/chatbind-7b-pretrain.bin

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    chatbind/train/train_imagebind_select_mem.py \
    --model_name_or_path ./checkpoints/vicuna-7b \
    --version $WEIGHT_VERSION \
    --data_path /path/to/llava_instruct_80k.json \
    --image_folder /path/to/coco/train2014 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/chatbind-7b-pretrain.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/chatbind-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb