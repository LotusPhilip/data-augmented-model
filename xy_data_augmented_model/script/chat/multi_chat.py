from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
import torch


def main():
    model_name = '/llm_data/lora_merge_checkpoint/qwen_chat_14B-sft_2'

    device = 'cuda:0'
    max_new_tokens = 500    # 每轮对话最多生成多少个token
    history_max_len = 1500  # 模型记忆的最大token长度
    top_p = 0.85
    top_k = 5
    temperature = 0.5
    repetition_penalty = 1.1
    device = 'cuda:0'

    # 加载模型
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     trust_remote_code=True,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map='cuda:0'
    # ).to(device).eval()
    model = load_model_on_gpus(model_name, num_gpus=2).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    streamer = TextStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True)

    # 记录所有历史记录
    history_token_ids = torch.tensor([[]], dtype=torch.long)

    # 开始对话
    user_input = input('User：')
    while True:
        user_input = user_input.strip()
        user_input = f"{tokenizer.bos_token}user:\n{user_input}{tokenizer.eos_token}{tokenizer.bos_token}assistant:\n"
        user_input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        model_input_ids = history_token_ids[:, -history_max_len:].to(device)
        print("Model-14B:")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, top_k = top_k, temperature=temperature, repetition_penalty=repetition_penalty,
                streamer = streamer, 
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
                bos_token_id = tokenizer.bos_token_id,
            )
        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
        response = tokenizer.batch_decode(response_ids)
        # print("Qwen-14B：" + response[0].strip().replace(tokenizer.eos_token, ""))
        user_input = input('User：')


if __name__ == '__main__':
    main()

