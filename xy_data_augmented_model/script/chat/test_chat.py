from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_name = '/llm_data/lora_merge_checkpoint/qwen_chat_14B-merge-final-sft'

device = 'cuda:0'
max_new_tokens = 500    # 每轮对话最多生成多少个token
history_max_len = 1500  # 模型记忆的最大token长度
top_p = 0.85
top_k = 5
temperature = 0.5
repetition_penalty = 1.1
device = 'cuda:0'

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)

input_text = "Yesterday has gone，"
inputs = tokenizer([input_text], return_tensors="pt", add_special_tokens=False).to(device)
streamer = TextIteratorStreamer(tokenizer)

# Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
generated_text = ""
for new_text in streamer:
    generated_text += new_text
    
