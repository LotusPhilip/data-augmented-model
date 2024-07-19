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
from dataclasses import dataclass, field
from typing import Optional
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_from_disk, load_dataset, concatenate_datasets
import bitsandbytes as bnb

import numpy as np
from typing import Any, Dict, List, Optional, Union
import torch.nn as nn



@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "Input the max length"})
    data_path: str = field(metadata={"help": "training set"})
    model_path: str = field(metadata={"help": "pre-trained weight path"})

def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='./train_args/pt.json', help="")
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
    # logger.info('Initializing components...')
    print('Initializing components...')
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    training_args.ddp_find_unused_parameters = False if ddp else None

    # 初始化model
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype = torch.bfloat16,
        trust_remote_code = True,
    )
    
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side = 'right',
        pad_token = '<|endoftext|>',
        bos_token = '<|im_start|>',
        eos_token = '<|im_end|>',
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # 部分tokenizer没有pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    # 部分tokenizer的pad_token_id与eos_token_id相同，如InternLM，会导致无法计算eos_token_id的loss。将pad_token_id设为unk_token_id
    if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.unk_token_id is not None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    # 如果两者相同，模型训练时不会计算eos_token_id的loss
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise Exception('pad_token_id should not be equal to eos_token_id')
    
    global BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
    BOS_TOKEN = tokenizer.bos_token
    EOS_TOKEN = tokenizer.eos_token
    PAD_TOKEN = tokenizer.pad_token
    
    # 数据集加载
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
    # print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    # print_trainable_parameters(trainer.model)
    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    # logger.info("*** starting training ***")
    print("*** starting training ***")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
