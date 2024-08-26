# @Time : 2023/12/21 18:24
# @Author : Li Jiaqi
# @Description :
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI as ChatOpenAI
from langchain_openai import OpenAI as LangchainOpenAI
from langchain.chains import create_citation_fuzzy_match_chain
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import configs
from configs import openai_apikey

api_key = openai_apikey
client = OpenAI(api_key=api_key)
lora_model = None
tf_model=None
tf_tokenizer=None

import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True,
                        text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

class ModelEnums:
    NONE = "none"
    GPT3T = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4T = 'gpt-4-1106-preview'
    ORION_RAG_QA_14B = "OrionStarAI/Orion-14B"
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

    SAMPLING_LLAMA= "sampling_llama"
    VALIDATION_LLAMA= "validation_llama"

    MISTRAL= "mistral"
    MISTRAL_COGNITION="TrustworthyLLM_Cognition_Finetuning_Model_Mistral"
    MISTRAL_COGNITION_QA= "TrustworthyLLM_Cognition_QA_Finetuning_Model_Mistral"
    MISTRAL_PSQA = "TrustworthyLLM_Cognition_PSQA_Finetuning_Model_Mistral"

    MAMBA="tiiuae/falcon-mamba-7b-instruct"

    GEMMA="Gemma-2-9b-it"
    GEMMA_COGNITION = 'TrustworthyLLM_Cognition_Finetuning_Model_Gemma'
    GEMMA_COGNITION_QA = 'TrustworthyLLM_Cognition_QA_Finetuning_Model_Gemma'
    GEMMA_PSQA = 'TrustworthyLLM_Cognition_PSQA_Finetuning_Model_Gemma'
    LLAMA3 = 'llama3'
    LLAMA3_COGNITION = 'TrustworthyLLM_Cognition_Finetuning_Model_Llama3'
    LLAMA3_COGNITION_QA= 'TrustworthyLLM_Cognition_QA_Finetuning_Model_Llama3'
    LLAMA3_PSQA= 'TrustworthyLLM_Cognition_PSQA_Finetuning_Model_Llama3_0'


def call_lora(prompt, model):
    global lora_model
    if lora_model is None:
        lora_model = LLM(model="/mnt/f/Models/llama-2-7b-chat-hf", enable_lora=True, gpu_memory_utilization=0.94, max_model_len=1600,)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200
    )
    outputs = lora_model.generate(
        prompt,
        sampling_params,
        lora_request=LoRARequest("augmentation_adapter", 1, f"LLaMA-Factory/models/{model}")
    )
    return outputs[0].outputs[0].text

def call_lora_mistral(prompt, model):
    global lora_model
    base_path="/mnt/f/Models/Mistral-7B-Instruct-v0.2"
    if model==ModelEnums.MISTRAL_PSQA or model==ModelEnums.MISTRAL_COGNITION_QA:
        base_path = "LLaMA-Factory/models/TrustworthyLLM_Cognition_Finetuning_Model_Mistral_Merged"
    if lora_model is None:
        lora_model = LLM(model=base_path, enable_lora=True, gpu_memory_utilization=0.90, max_model_len=1600,)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200
    )
    outputs = lora_model.generate(
        prompt,
        sampling_params,
        lora_request=LoRARequest("augmentation_adapter", 1, f"LLaMA-Factory/models/{model}")
    )
    return outputs[0].outputs[0].text

def call_lora_llama3(prompt, model):
    global lora_model
    base_path="/mnt/f/Models/Llama3.1-8B-Instruct"
    if model==ModelEnums.LLAMA3_COGNITION_QA or model==ModelEnums.LLAMA3_PSQA:
        base_path = "LLaMA-Factory/models/TrustworthyLLM_Cognition_Finetuning_Model_Llama3_Merged"
    if lora_model is None:
        lora_model = LLM(model=base_path, enable_lora=True, gpu_memory_utilization=0.90, max_model_len=1600,)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200
    )
    outputs = lora_model.generate(
        prompt,
        sampling_params,
        lora_request=LoRARequest("augmentation_adapter", 1, f"LLaMA-Factory/models/{model}")
    )
    return outputs[0].outputs[0].text

def call_lora_gemma(prompt, model):
    global lora_model
    base_path="/root/autodl-tmp/model/gemma-2-9b-it"
    if model==ModelEnums.GEMMA_COGNITION_QA or model==ModelEnums.GEMMA_PSQA:
        base_path = "/root/autodl-tmp/model/TrustworthyLLM_Cognition_Finetuning_Model_Gemma_Merged"
    if lora_model is None:
        lora_model = LLM(model=base_path, enable_lora=True, gpu_memory_utilization=0.98, max_model_len=1400,)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200
    )
    outputs = lora_model.generate(
        prompt,
        sampling_params,
        lora_request=LoRARequest("augmentation_adapter", 1, f"/root/autodl-tmp/model/{model}")
    )
    return outputs[0].outputs[0].text


def call_transformers(prompt, model=ModelEnums.ORION_RAG_QA_14B, **kwargs):
    global tf_model, tf_tokenizer
    if tf_model is None:
        from huggingface_hub import login
        login(token=configs.huggingface_token)
        tf_tokenizer = AutoTokenizer.from_pretrained(model)
        tf_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto",torch_dtype="auto")

    if model==ModelEnums.ORION_RAG_QA_14B:
        # model.generation_config = GenerationConfig.from_pretrained(model)
        messages = [{"role": "user", "content": prompt}]
        response = tf_model.chat(tf_tokenizer, messages, streaming=False)
    else:
        messages = [
            {"role": "user", "content": prompt},
        ]

        input_text = tf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tf_tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

        response = tf_model.generate(input_ids, max_new_tokens=200)
        response=tf_tokenizer.decode(response[0], skip_special_tokens=True)
        response =response.split('ANSWER')[-1]
    return response


def call_vllm(prompt, modelName):
    openai.api_base = "http://localhost:9091/v1"
    openai.model = modelName
    openai.api_key = "Empty"
    langchain_model = LangchainOpenAI(
        model=openai.model,
        openai_api_key=openai.api_key,
        temperature=0,
        max_tokens=200,
        openai_api_base=openai.api_base,
        max_retries=2
    )
    return langchain_model.invoke(prompt)

def call_vllm_sampling(prompt, modelName):
    openai.api_base = "http://localhost:9091/v1"
    openai.model = modelName
    openai.api_key = "Empty"
    langchain_model = LangchainOpenAI(
        model=openai.model,
        openai_api_key=openai.api_key,
        temperature=0.6,
        max_tokens=200,
        openai_api_base=openai.api_base,
        max_retries=2
    )
    return langchain_model.invoke(prompt)


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
    if model == ModelEnums.ORION_RAG_QA_14B or model == ModelEnums.MAMBA:
        return call_transformers(query, model)
    if model in [ModelEnums.COGNITION, ModelEnums.PROMPT_CENTERED, ModelEnums.COGNITION_QA,
                 ModelEnums.PROMPT_CENTERED_QA_COGNITION, ModelEnums.QA_MODEL,
                 ModelEnums.PROMPT_CENTERED_QA]:
        return call_lora(query, model)
    if model==ModelEnums.MISTRAL_PSQA or model==ModelEnums.MISTRAL_COGNITION or model==ModelEnums.MISTRAL_COGNITION_QA:
        return call_lora_mistral(query, model)
    if model==ModelEnums.LLAMA3_COGNITION or model==ModelEnums.LLAMA3_COGNITION_QA or model==ModelEnums.LLAMA3_PSQA:
        return call_lora_llama3(query, model)
    if model==ModelEnums.GEMMA_COGNITION or model==ModelEnums.GEMMA_COGNITION_QA or model==ModelEnums.GEMMA_PSQA:
        return call_lora_gemma(query, model)
    if model == ModelEnums.SAMPLING_LLAMA:
        return call_vllm_sampling(query, **kwargs)
    return call_vllm(query, **kwargs)


def autodl_speedup():
    import subprocess
    import os
    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
    output = result.stdout
    print('speeding up', output)
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value
