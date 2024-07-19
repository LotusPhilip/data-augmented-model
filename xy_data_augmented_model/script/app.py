from flask import Flask, request, Response, stream_with_context
import json
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from dataclasses import dataclass
from threading import Thread
import argparse
# from utils import load_model_on_gpus
from datetime import datetime
from vllm import LLM, SamplingParams
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

@dataclass
class GenerationArguments:
    max_new_tokens: int = 500    # 每轮对话最多生成多少个token
    history_max_len: int = 900  # 模型记忆的最大token长度
    top_p: float = 0.85
    top_k: int = 5
    temperature: float = 0.1
    repetition_penalty: float = 1.1
    do_sample: bool = False
    device: str = 'cuda:0'

@app.route('/chat', methods=['POST'])
def ds_llm():
    # 从请求中获取数据
    data = request.get_json()

    system = "You are an assistant."
    text = ""
    text += f"{tokenizer.bos_token}system:\n{system}{tokenizer.eos_token}"
    for item in data:
        if item['type'] == 'user':
            text += f"{tokenizer.bos_token}user:\n{item['content'].strip()}{tokenizer.eos_token}"
        else:
            text += f"{tokenizer.bos_token}assistant:\n{item['content'].strip()}{tokenizer.eos_token}"
    text += f"{tokenizer.bos_token}assistant:\n"

    inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(GenerationArguments.device)
    inputs.input_ids = inputs.input_ids[:, -GenerationArguments.history_max_len:]
    # input_ids = tokenizer([text], return_tensors="pt", add_special_tokens=False).input_ids
    # input_ids = input_ids[:, -GenerationArguments.history_max_len:].to(device)

    # 生成流式结果
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    # 在新线程中运行模型生成
    generation_kwargs = dict(inputs,
                             temperature=GenerationArguments.temperature,
                             streamer=streamer,
                             top_p=GenerationArguments.top_p,
                            #  top_k=GenerationArguments.top_k,
                             repetition_penalty=GenerationArguments.repetition_penalty,
                             max_new_tokens=GenerationArguments.max_new_tokens,
                             do_sample=GenerationArguments.do_sample,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id = tokenizer.pad_token_id,
                             bos_token_id = tokenizer.bos_token_id,
                            )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 生成器函数，每次生成一段文本
    def generate_text():
        for new_text in streamer:
            print(new_text)
            yield new_text

    # 返回结果
    return Response(stream_with_context(generate_text()), mimetype='text/event-stream')
    # return Response(generate_text(), mimetype='text/event-stream')


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/data/Model/qwen_chat_14B-merge-final-sft_2/', help="")
    parser.add_argument("--log_file", type=str, default='./service_history.txt', help="")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="")
    parser.add_argument("--port", type=int, default=8877, help="")
    args = parser.parse_args()

    logger.info(f"Starting to load the model {args.model_path,} into memory")

    # # 加载model和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=GenerationArguments.device,
    ).eval()
    # model = load_model_on_gpus(args.model_path, num_gpus=2).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # model = LLM(model = args.model_path,
    #             tokenizer_mode = "auto",
    #             trust_remote_code = True,
    #             tensor_parallel_size = 2,
    #             quantization = 'awq'
    #         )
    # sampling_params = SamplingParams(temperature = GenerationArguments.temperature,
    #                                  top_p = GenerationArguments.top_p,
    #                                  top_k = GenerationArguments.top_k,
    #                                  length_penalty = GenerationArguments.repetition_penalty,
    #                                  max_tokens = GenerationArguments.max_new_tokens,
    #                                 )


    logger.info(f"Successfully loaded the model {args.model_path,} into memory")

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    print("Total model params: %.2fM" % (total / 1e6))


    app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
    app.run(host=args.host, debug=False, port = args.port)