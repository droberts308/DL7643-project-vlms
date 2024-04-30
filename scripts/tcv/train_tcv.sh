#!/bin/bash

torchrun --standalone --nnodes=1 --nproc-per-node=4 ./modules/ahmad_conditional_visual/train.py \
    --llm_model_name 'microsoft/Phi-3-mini-128k-instruct' \
    --vit_to_llm_projector_type 'mlp2x_gelu' \
    --tcv_vit_name "openai/clip-vit-large-patch14-336" \
    --tcv_vit_select_layer -2 \
    --tcv_vit_select_feature "patch" \
    --tcv_text_encoder_name "google-bert/bert-base-uncased" \
    --tcv_text_select_layer -2 \
    --tcv_text_select_feature "all" \
    --tcv_text_to_vit_projector_type 'mlp2x_gelu' \
    --data_path "./modules/ahmad_conditional_visual/train-data-shard/coco_only_364k_new.json" \
    --lazy_preprocess True \
    --is_multimodal True \
    --image_folder "./modules/ahmad_conditional_visual/train-data-shard/train2017" \
    --image_aspect_ratio "pad" \
    --llm_dora_enable True \
    --llm_lora_r 128 \
    --llm_lora_alpha 256 \
    --vit_dora_enable True \
    --vit_lora_r 128 \
    --vit_lora_alpha 128 \
    --group_by_modality_length True \
    --model_max_length 2048 \
    --bf16 True \
    --output_dir "./modules/ahmad_conditional_visual/ckpts/phi-3b-dora" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.25 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --projectors_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing  False \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory  True \
    --dataloader_persistent_workers True \
1> ./modules/ahmad_conditional_visual/logs/out \
2> ./modules/ahmad_conditional_visual/logs/err