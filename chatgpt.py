# @Time : 2023/12/21 18:24
# @Author : Li Jiaqi
# @Description :
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI as ChatOpenAI
from langchain_openai import OpenAI as LangchainOpenAI
from langchain.chains import create_citation_fuzzy_match_chain
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch

from configs import openai_apikey

api_key = openai_apikey
client = OpenAI(api_key=api_key)
lora_model = None


class ModelEnums:
    NONE = "none"
    GPT3T = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4T = 'gpt-4-1106-preview'
    ORION_RAG_QA_14B = "orion-RAG-qa-14b"
    LOCAL = "local"
    LLAMA2_CHAT_7B = 'llama2c-7b'
    VICUNA_7B = 'vicuna-7b'
    SELF_RAG = 'self-rag'
    COGNITION = 'TrustworthyLLM_Cognition_Finetuning_Model'
    COGNITION_QA = 'TrustworthyLLM_Cognition_QA_Finetuning_Model'
    PROMPT_CENTERED = 'TrustworthyLLM_Ablation_PSQA_Finetuning_Model'
    PROMPT_CENTERED_QA_COGNITION = "TrustworthyLLM_Cognition_PSQA_Finetuning_Model_2"
    QA_MODEL = "TrustworthyLLM_Ablation_QA_Finetuning_Model"
    PROMPT_CENTERED_QA = "trustworthy_prompt_qa_model"
    TEMP= "temp"


def call_lora(prompt, model):
    global lora_model
    if lora_model is None:
        lora_model = LLM(model="/mnt/f/Models/llama-2-7b-chat-hf", enable_lora=True, gpu_memory_utilization=0.93, max_model_len=2048,)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512
    )
    outputs = lora_model.generate(
        prompt,
        sampling_params,
        lora_request=LoRARequest("augmentation_adapter", 1, f"LLaMA-Factory/models/{model}")
    )
    return outputs[0].outputs[0].text


def call_transformers(prompt, model=ModelEnums.ORION_RAG_QA_14B, **kwargs):
    if model == ModelEnums.ORION_RAG_QA_14B:
        tokenizer = AutoTokenizer.from_pretrained("OrionStarAI/Orion-14B", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("OrionStarAI/Orion-14B", device_map="auto",
                                                     torch_dtype=torch.bfloat16, trust_remote_code=True)

        model.generation_config = GenerationConfig.from_pretrained("OrionStarAI/Orion-14B")
        messages = [{"role": "user", "content": prompt}]
        response = model.chat(tokenizer, messages, streaming=False)
        return response


def call_vllm(prompt, modelName):
    openai.api_base = "http://localhost:9091/v1"
    openai.model = modelName
    openai.api_key = "Empty"
    langchain_model = LangchainOpenAI(
        model=openai.model,
        openai_api_key=openai.api_key,
        temperature=0,
        max_tokens=256,
        openai_api_base=openai.api_base,
        max_retries=2
    )
    return langchain_model.predict(prompt)


def ask_gpt(query, model=ModelEnums.GPT3T, logit=0, **kwargs):
    openai.api_base = None
    openai.api_key = api_key
    logit = logit if type(logit) == int else 0
    if logit != 0:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You need to accomplish the provided task without outputting any other content."},
                {"role": "user", "content": query}
            ],
            temperature=0,
            logprobs=logit != 0,
            top_logprobs=logit,
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You need to accomplish the provided task without outputting any other content."},
                {"role": "user", "content": query}
            ],
            temperature=0,
        )
    if logit != 0:
        return completion.choices[0].message.content, completion.choices[0].logprobs.content
    return completion.choices[0].message.content


def langchain_citation(query, context, model=ModelEnums.GPT4T):
    llm = ChatOpenAI(temperature=0, model=model, openai_api_key=api_key)
    chain = create_citation_fuzzy_match_chain(llm)
    result = chain.run(question=query, context=context)
    return result


def call_llm(query, model, **kwargs):
    if model == ModelEnums.GPT4T or model == ModelEnums.GPT3T or model == ModelEnums.GPT4:
        return ask_gpt(query, model, **kwargs)
    if model == ModelEnums.ORION_RAG_QA_14B:
        return call_transformers(query, model)
    if model in [ModelEnums.COGNITION, ModelEnums.PROMPT_CENTERED, ModelEnums.COGNITION_QA,
                 ModelEnums.PROMPT_CENTERED_QA_COGNITION, ModelEnums.QA_MODEL,
                 ModelEnums.PROMPT_CENTERED_QA]:
        return call_lora(query, model)
    return call_vllm(query, **kwargs)


def autodl_speedup():
    import subprocess
    import os
    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True,
                            text=True)
    output = result.stdout
    print('speeding up', output)
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value
