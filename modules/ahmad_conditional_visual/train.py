import transformers
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from PIL import Image
from typing import Dict, Optional, Sequence
from peft import LoraConfig
import torch

import transformers
import tokenizers


import conversation as conversation_lib
from tcv_trainer import TCVTrainer
from model_arch import TCVForCausalLM
from model_configs import TCVForCausalLMConfig
from data_args_utils import find_all_linear_names, LazySupervisedDataset, DataCollatorForSupervisedDataset, ModelArguments, TrainingArguments, DataArguments




model_args = ModelArguments()
data_args = DataArguments()
training_args = TrainingArguments(output_dir = "./checkpoints/llava-v1.5-13b-lora" )

# model_args 
model_args.model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'

model_args.tcv_vit_name = "openai/clip-vit-large-patch14-336"
model_args.tcv_vit_select_layer = -2
model_args.tcv_vit_select_feature = "patch"

model_args.tcv_text_encoder_name = "google-bert/bert-base-uncased"
model_args.tcv_text_select_layer = -2 # Shapiro
model_args.tcv_text_select_feature = "all"

model_args.tcv_text_to_vit_projector_type = 'mlp2x_gelu'
model_args.vit_to_llm_projector_type = 'mlp2x_gelu'



# data_args
data_args.data_path = "/scratch/mt/new-structure/experiments/ashapiro/LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
data_args.lazy_preprocess = True
data_args.is_multimodal = True
data_args.image_folder = "/scratch/mt/new-structure/experiments/ashapiro/LLaVA/playground/data/LLaVA-Pretrain/images"
data_args.image_aspect_ratio = "pad"



# training_args
training_args.llm_dora_enable = True
training_args.llm_lora_r = 128
training_args.llm_lora_alpha = 256 

training_args.vit_dora_enable = True
training_args.vit_lora_r = 128
training_args.vit_lora_alpha = 128 

    
training_args.group_by_modality_length = True 
training_args.model_max_length = 2048 
training_args.bf16 = True 
training_args.output_dir = "./checkpoints/llava-v1.5-13b-lora" 
training_args.num_train_epochs = 1 
training_args.per_device_train_batch_size = 3 
training_args.per_device_eval_batch_size = 4 
training_args.gradient_accumulation_steps = 1 
training_args.evaluation_strategy = "no" 
training_args.save_strategy = "steps" 
training_args.save_steps = 50000 
training_args.save_total_limit = 1 
training_args.learning_rate = 2e-4 
training_args.projector_lr = 2e-5 
training_args.weight_decay = 0. 
training_args.warmup_ratio = 0.03 
training_args.lr_scheduler_type = "cosine" 
training_args.logging_steps = 1 
training_args.tf32 = True 
training_args.gradient_checkpointing =  False 
training_args.dataloader_num_workers = 4 
training_args.report_to = []

training_args.dataloader_num_workers = 1
training_args.dataloader_pin_memory  = True
training_args.dataloader_persistent_workers = True


config = TCVForCausalLMConfig()
model = TCVForCausalLM(config)
model.to(device = torch.device("cuda:0"), dtype=torch.bfloat16)



llm_lora_config = LoraConfig(
    use_dora = True,
    r=training_args.llm_lora_r,
    lora_alpha=training_args.llm_lora_alpha,
    target_modules=find_all_linear_names(model.llm),
    lora_dropout=training_args.llm_lora_dropout,
    bias=training_args.llm_lora_bias,
    task_type="CAUSAL_LM",
)

vit_lora_config = LoraConfig(
    use_dora = True,
    r=16,
    lora_alpha=16,
    target_modules=find_all_linear_names(model.tcv.vision_model),
    lora_dropout=0.1,
    bias="none"
)

model.wrap_peft(
    
    llm_lora_config= llm_lora_config,
    vit_lora_config= vit_lora_config
)


llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
    model.config.llm_config._name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=False,
)
if not llm_tokenizer.pad_token:
    llm_tokenizer.pad_token = llm_tokenizer.unk_token


vit_tokenizer = transformers.AutoTokenizer.from_pretrained(
    model.config.tcv_config.text_config._name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=False,
)

if not vit_tokenizer.pad_token:
    vit_tokenizer.pad_token = vit_tokenizer.unk_token





data_args.image_processor = model.tcv.image_processor
data_args.is_multimodal = True



model.config.tokenizer_padding_side = llm_tokenizer.padding_side
model.config.tokenizer_model_max_length = llm_tokenizer.model_max_length

model.config.tcv_vit_select_layer = model_args.tcv_vit_select_layer
model.config.tcv_vit_select_feature = model_args.tcv_vit_select_feature

model.config.tcv_text_select_layer = model_args.tcv_text_select_layer
model.config.tcv_text_select_feature = model_args.tcv_text_select_feature

model.config.text_select_feature = model_args.text_select_feature
model.config.vit_select_layer = model_args.mm_vision_select_layer
model.config.vit_select_layer = model_args.mm_vision_select_layer




train_dataset = LazySupervisedDataset(  tokenizer=llm_tokenizer,
                                        data_path=data_args.data_path,
                                        data_args=data_args,
                                        vit_text_tokenizer= vit_tokenizer
                                    )

data_collator = DataCollatorForSupervisedDataset(tokenizer=llm_tokenizer,
                                                 vit_text_tokenizer= vit_tokenizer,
                                                 device= model.device)

trainer = TCVTrainer(   model = model,
                        tokenizer = llm_tokenizer,
                        args = training_args,
                        train_dataset = train_dataset,
                        eval_dataset = None,
                        data_collator = data_collator)

trainer.train()