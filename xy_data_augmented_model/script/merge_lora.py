from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""

def merge_lora_to_base_model():
    # model_name_or_path = '/llm_data/renyi/Model/qwen_base_14B/'
    model_name_or_path = '/llm_data/lora_merge_checkpoint/qwen_chat_14B-merge-final-pt'
    adapter_name_or_path = '/llm_data/checkpoints_qwen_chat_14B_sft_2/final'
    save_path = '/llm_data/lora_merge_checkpoint/qwen_chat_14B-sft_2'

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code = True,
        padding_side = "right",
        pad_token = '<|endoftext|>',
        bos_token = '<|im_start|>',
        eos_token = '<|im_end|>',
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code = True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        # device_map={'': 'cpu'}
    )
    # model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map="auto")
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
