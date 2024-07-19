from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
import argparse
# from loguru import logger
import os
from os.path import join
import torch
from peft import LoraConfig, get_peft_model 
from dataclasses import dataclass, field
from typing import Optional
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_from_disk, load_dataset, concatenate_datasets
import bitsandbytes as bnb

import numpy as np
from typing import Any, Dict, List, Optional, Union


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "Input the max length"})
    data_path: str = field(metadata={"help": "training set"})
    model_path: str = field(metadata={"help": "pre-trained weight path"})
    lora_r : int = 64
    lora_alpha : int = 16
    training_embedding: bool = False

def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='/dev/train/sft.json', help="")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # train_args_file = 'train_args/finetune.json'
    # 读取训练的参数配置
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['conversation'])):
        text = ""
        for data in example['conversation'][i]:
            text += f"{BOS_TOKEN}user:\n{data['human']}{EOS_TOKEN}{BOS_TOKEN}assistant:\n{data['assistant']}{EOS_TOKEN}"
        output_texts.append(text)
    return output_texts

def init_components(args, training_args):
    """
    初始化各个组件
    """

    print('Initializing components...')
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    training_args.ddp_find_unused_parameters = False if ddp else None

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch.bfloat16,
        trust_remote_code = True,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
        ),
    )
    
    # Lora_config
    if args.training_embedding:
        modules_to_save = ["wte", "lm_head"]
    else:
        modules_to_save = None
    target_modules = find_all_linear_names(model)
    lora_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules =  target_modules,
        fan_in_fan_out = False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side = 'right',
        pad_token = '<|endoftext|>',
        bos_token = '<|im_start|>',
        eos_token = '<|im_end|>',
    )
    # 部分tokenizer没有pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.unk_token_id is not None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise Exception('pad_token_id should not be equal to eos_token_id')
    
    global BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
    BOS_TOKEN = tokenizer.bos_token
    EOS_TOKEN = tokenizer.eos_token
    PAD_TOKEN = tokenizer.pad_token
    
    
    lm_datasets = []
    files = os.listdir(args.data_path)

    for idx, file in enumerate(files):
        data_file = os.path.join(args.data_path, file)
        raw_dataset = load_dataset('json', data_files = data_file, keep_in_memory = False, split = 'train')
        
        if idx == 0:
            lm_datasets = raw_dataset
        else:
            lm_datasets = concatenate_datasets([lm_datasets, raw_dataset])

    lm_datasets = lm_datasets.shuffle()
    train_data = lm_datasets
    print(f"Size of the train set: {len(train_data)}.")
    
    instruction_template = f"{BOS_TOKEN}user:\n"
    response_template = f"{BOS_TOKEN}assistant:\n"
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template = instruction_template,
        response_template = response_template,
        tokenizer = tokenizer,
        mlm = False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        # eval_dataset=valid_data,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    print_trainable_parameters(trainer.model)

    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    
    # 加载各种组件
    trainer = init_components(args, training_args)
    
    # 开始训练
    print("*** starting training ***")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path) 
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
