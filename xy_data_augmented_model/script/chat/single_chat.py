from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
"""
单轮对话，不具有对话历史的记忆功能
"""

def main():
    model_name = '/llm_data/lora_merge_checkpoint/qwen_chat_14B-merge-final-sft'

    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    device = 'cuda:0'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0'
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True,
        padding_side = "right",
        local_files_only = True,
    )


    text = input('User：')
    while True:
        text = text.strip()
        text = f"{tokenizer.bos_token}user:\n{text}{tokenizer.eos_token}{tokenizer.bos_token}assistant:\n"
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
        print("Qwen-14B:{}".format(response))
        text = input('User：')


if __name__ == '__main__':
    main()
