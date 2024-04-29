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


def main():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    config = TCVForCausalLMConfig(  llm_model_name = model_args.llm_model_name,
                                    vit_to_llm_projector_name = model_args.vit_to_llm_projector_type ,
                                    tcv_vit_model_name = model_args.tcv_vit_name,
                                    tcv_text_model_name = model_args.tcv_text_encoder_name,
                                    tcv_text_to_vit_projector_name = model_args.tcv_text_to_vit_projector_type,
                                    tcv_vit_select_layer = model_args.tcv_vit_select_layer,
                                    tcv_vit_select_feature = model_args.tcv_vit_select_feature,
                                    tcv_text_select_layer = model_args.tcv_text_select_layer,
                                    tcv_text_select_feature = model_args.tcv_text_select_feature,
                                    tokenizer_padding_side = "right"
                                    
                                    )
    model = TCVForCausalLM(config)
    model.to(device = torch.device(f"cuda:{local_rank}"), dtype=torch.bfloat16)



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
        use_fast=False,
    )
    if not llm_tokenizer.pad_token:
        llm_tokenizer.pad_token = llm_tokenizer.unk_token


    vit_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.config.tcv_config.text_config._name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
    )

    if not vit_tokenizer.pad_token:
        vit_tokenizer.pad_token = vit_tokenizer.unk_token





    data_args.image_processor = model.tcv.image_processor
    data_args.is_multimodal = True



    model.config.tokenizer_padding_side = llm_tokenizer.padding_side
    model.config.tokenizer_model_max_length = llm_tokenizer.model_max_length






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
    
    
if __name__ == "__main__":
    main()