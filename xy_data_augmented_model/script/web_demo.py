# -*- coding: utf-8 -*-
import argparse
import os
from threading import Thread

import gradio as gr
import mdtex2html
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)

@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.85,
        top_k = 5,
        repetition_penalty=1.1,
        context_len=2048
):
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    yield from streamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="/llm_data/lora_merge_checkpoint/qwen_chat_14B-merge-final-sft", type=str, required=True)
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    args = parser.parse_args()
    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    def postprocess(self, y):
        if y is None:
            return []
        for i, (message, response) in enumerate(y):
            y[i] = (
                None if message is None else mdtex2html.convert((message)),
                None if response is None else mdtex2html.convert(response),
            )
        return y

    gr.Chatbot.postprocess = postprocess

    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    ).to(device).eval()
    
    try:
        model.generation_config = GenerationConfig.from_pretrained(args.model_name, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    
    if device == torch.device('cpu'):
        model.float()

    def reset_user_input():
        return gr.update(value='')

    def reset_state():
        return [], []

    history = []

    def predict(
            input,
            chatbot,
            history,
            max_new_tokens,
            temperature,
            top_p,
            top_k
    ):
        now_input = input
        chatbot.append((input, ""))
        history = history or []
        history.append([now_input, ''])

        convs = []
        sys = "You are now a physician:"
        system_prompt = tokenizer.bos_token + sys + tokenizer.eos_token
        for turn_idx, [user_query, bot_resp] in enumerate(history):
            if turn_idx == 0:
                convs.append(system_prompt + tokenizer.bos_token + user_query + tokenizer.eos_token)
                convs.append(tokenizer.bos_token + bot_resp + tokenizer.eos_token)
            else:
                convs.append(tokenizer.bos_token + user_query + tokenizer.eos_token)
                convs.append(tokenizer.bos_token + bot_resp + tokenizer.eos_token)
                
        response = ""
        for new_text in stream_generate_answer(
                model,
                tokenizer,
                convs,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
        ):
            response += new_text
            new_history = history + [(now_input, response)]
            chatbot[-1] = (now_input, response)
            yield chatbot, new_history

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">MedicalGPT</h1>""")
        gr.Markdown(
            "> To facilitate the open research on medical LLM, this project open-source the model.")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(
                    0, 4096, value=512, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                                  label="Top P", interactive=True)
                temperature = gr.Slider(
                    0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)
        history = gr.State([])
        submitBtn.click(predict, [user_input, chatbot, history, max_length, temperature, top_p], [chatbot, history],
                        show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
    demo.queue().launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=8082)


if __name__ == '__main__':
    main()
