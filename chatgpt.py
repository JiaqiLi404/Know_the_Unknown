# @Time : 2023/12/21 18:24
# @Author : Li Jiaqi
# @Description :
from dataclasses import dataclass, field

import openai
import transformers
from openai import OpenAI
from langchain_openai import ChatOpenAI as ChatOpenAI
from langchain_openai import OpenAI as LangchainOpenAI
from langchain.chains import create_citation_fuzzy_match_chain
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import configs
from configs import openai_apikey
import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from huggingface_hub import file_exists as hf_file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from peft.utils.other import infer_device
from dataclasses import dataclass, asdict as dataclassasdict
import dataclasses
from datasets.formatting.formatting import LazyRow

api_key = openai_apikey
client = OpenAI(api_key=api_key, base_url="https://api.zhizengzeng.com/v1")
lora_model = None
tf_model = None
tf_tokenizer = None
peft_model = None

import subprocess
import os
from peft import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PEFT_TYPE_TO_CONFIG_MAPPING,
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    TaskType,
)
import sys

LLAMA_3_SYS_PROMPT = "You are an expert who responds with concise, correct answers. Directly state the answer without phrases like 'the correct answer is'"


@dataclass
class LMText:
    context: str
    prompt: str = ""
    target_prompt: str = ""
    target: str = ""

    ## Misc.
    output: str = None
    query_label: int = None

    def __str__(self):
        return (
                self.prompt + self.context + self.target_prompt + " " + self.target
        ).strip()

    def to_pydict(self):
        return {k: v for k, v in dataclassasdict(self).items() if v is not None}

    @staticmethod
    def field_names():
        return [f.name for f in dataclasses.fields(LMText)]

    @staticmethod
    def from_(instance):
        if isinstance(instance, LMText):
            return instance

        if isinstance(instance, LazyRow):
            instance = {k: v for k, v in zip(instance.keys(), instance.values())}

        assert isinstance(
            instance, dict
        ), f"Could not convert instance to dict. Found {type(instance)}"

        instance = {
            k: v
            for k, v in instance.items()
            if k in set(f.name for f in dataclasses.fields(LMText))
        }
        return LMText(**instance)


IGNORE_LABEL = -100


@dataclass
class LabeledStringDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    target_name: str = "target"

    @staticmethod
    def get_tokenizer_args(tokenizer):
        return dict(
            padding=True,
            truncation=True,
            max_length=(
                tokenizer.model_max_length
                if hasattr(tokenizer, "model_max_length")
                else None
            ),
            return_tensors="pt",
            return_length=True,
        )

    def __call__(self, instances):
        tokenizer_args = self.get_tokenizer_args(self.tokenizer)

        prompts = [str(LMText.from_(instance)) for instance in instances]

        if (
                self.tokenizer.name_or_path
                and ("Llama-3" in self.tokenizer.name_or_path)
                and ("Instruct" in self.tokenizer.name_or_path)
        ):
            msgs = [
                [
                    {"role": "system", "content": LLAMA_3_SYS_PROMPT},
                    {"role": "user", "content": p},
                ]
                for p in prompts
            ]

            prompts = [
                self.tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in msgs
            ]

        inputs = self.tokenizer(prompts, **tokenizer_args)
        input_lengths = inputs.pop("length")

        if self.target_name in instances[0]:
            ## inputs without targets for labeling lengths.
            un_inputs = self.tokenizer(
                [
                    str(
                        LMText.from_(
                            {k: v for k, v in instance.items() if k != self.target_name}
                        )
                    )
                    for instance in instances
                ],
                **tokenizer_args,
            )
            un_input_lengths = un_inputs.pop("length")

            labels = inputs.get("input_ids").clone()
            for i, l in enumerate(input_lengths - un_input_lengths):
                labels[i, :-l] = IGNORE_LABEL
            inputs["labels"] = labels

        return inputs


@dataclass
class CalibratedLoraConfig(LoraConfig):
    task_type: str = field(
        default="CALIBRATED_CAUSAL_LM", metadata={"help": "Task type"}
    )
    query_format: int = field(default="roman_choice", metadata={"help": "Query format"})
    use_temperature: bool = field(
        default=False, metadata={"help": "Temperature-scaled query probabilities"}
    )


class PeftModelForCalibratedCausalLM(PeftModelForCausalLM):
    TEMPERATURE_WEIGHTS_NAME = "temperature_model.pt"
    SAFETENSORS_TEMPERATURE_WEIGHTS_NAME = "temperature_model.safetensors"

    class TemperatureScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temperature = nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            return inputs / self.log_temperature.exp()

    def _get_token_vec(self, tokenizer):
        query_format = self.active_peft_config.query_format

        vocab = tokenizer.get_vocab()

        def _create_vec(raw_list):
            for t in raw_list:
                assert t in vocab, f"Cannot handle {t} as a single token."

            return torch.tensor([tokenizer(t).input_ids[-1] for t in raw_list])

        if query_format == "roman_choice":
            raw_strings = ["i", "ii"]
        else:
            raise NotImplementedError(f'Format "{self.format}" not supported.')

        return _create_vec(raw_strings)

    def _prepare_uncertainty_query(self, contexts, predictions):
        query_format = self.active_peft_config.query_format

        def _format_query_text(c, p):
            if query_format == "roman_choice":
                query_text = "\n".join(
                    [
                        c + p,
                        "\nIs the proposed answer correct?",
                        "Choices:",
                        "(i): no",
                        "(ii): yes",
                        "Answer:",
                    ]
                )
            else:
                raise NotImplementedError(f'Format "{query_format}" not supported.')

            return query_text

        query_inputs = [_format_query_text(c, p) for c, p in zip(contexts, predictions)]

        return query_inputs

    def generate(
            self, *args, tokenizer=None, collate_fn=None, use_temperature=False, **kwargs
    ):
        use_temperature = use_temperature or self.active_peft_config.use_temperature

        with self.disable_adapter():
            outputs = super().generate(*args, **kwargs)

        input_ids = kwargs.get("input_ids")

        str_inputs = tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        str_outputs = tokenizer.batch_decode(
            outputs[:, input_ids.size(-1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        q_token_vec = self._get_token_vec(tokenizer)
        q_str_inputs = self._prepare_uncertainty_query(str_inputs, str_outputs)

        q_str_inputs = [{"context": s} for s in q_str_inputs]
        q_inputs = {
            k: v.cuda(input_ids.device) for k, v in collate_fn(q_str_inputs).items()
        }

        q_logits = self(**q_inputs).logits[..., -1, q_token_vec]
        if use_temperature:
            q_logits = self.temperature_scale[self.active_adapter](q_logits)

        p_correct = q_logits.softmax(dim=-1)[:, 1]

        return outputs, p_correct

    def load_temperature_adapter(self, model_id, device=None, **hf_hub_download_kwargs):
        path = (
            os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else model_id
        )

        if device is None:
            device = infer_device()

        if os.path.exists(
                os.path.join(path, self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME)
        ):
            filename = os.path.join(path, self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME)
            use_safetensors = True
        elif os.path.exists(os.path.join(path, self.TEMPERATURE_WEIGHTS_NAME)):
            filename = os.path.join(path, self.TEMPERATURE_WEIGHTS_NAME)
            use_safetensors = False
        else:
            token = hf_hub_download_kwargs.get("token", None)
            if token is None:
                token = hf_hub_download_kwargs.get("use_auth_token", None)

            hub_filename = (
                os.path.join(
                    hf_hub_download_kwargs["subfolder"],
                    self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME,
                )
                if hf_hub_download_kwargs.get("subfolder", None) is not None
                else self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME
            )
            has_remote_safetensors_file = hf_file_exists(
                repo_id=model_id,
                filename=hub_filename,
                revision=hf_hub_download_kwargs.get("revision", None),
                repo_type=hf_hub_download_kwargs.get("repo_type", None),
                token=token,
            )

            use_safetensors = has_remote_safetensors_file

            if has_remote_safetensors_file:
                filename = hf_hub_download(
                    model_id,
                    self.SAFETENSORS_TEMPERATURE_WEIGHTS_NAME,
                    **hf_hub_download_kwargs,
                )
            else:
                try:
                    filename = hf_hub_download(
                        model_id,
                        self.TEMPERATURE_WEIGHTS_NAME,
                        **hf_hub_download_kwargs,
                    )
                except EntryNotFoundError:
                    filename = None

        if filename:
            if use_safetensors:
                if hasattr(torch.backends, "mps") and (device == torch.device("mps")):
                    temperature_weights = safe_load_file(filename, device="cpu")
                else:
                    temperature_weights = safe_load_file(filename, device=device)
            else:
                temperature_weights = torch.load(
                    filename, map_location=torch.device(device)
                )

            return temperature_weights

    def load_adapter(self, model_id, adapter_name, **kwargs):
        load_result = super().load_adapter(model_id, adapter_name, ignore_mismatched_sizes=True, **kwargs)

        hf_hub_download_kwargs, _ = self._split_kwargs(kwargs)

        temperature_weights = self.load_temperature_adapter(
            model_id, **hf_hub_download_kwargs
        )

        if not hasattr(self, "temperature_scale"):
            self.temperature_scale = dict()

        self.temperature_scale[adapter_name] = self.TemperatureScale()
        if temperature_weights:
            self.temperature_scale[adapter_name].load_state_dict(temperature_weights)

        return load_result


## Hot patch config/model mapping.
PEFT_TYPE_TO_CONFIG_MAPPING["CALIBRATED_LORA"] = CalibratedLoraConfig
MODEL_TYPE_TO_PEFT_MODEL_MAPPING["CALIBRATED_CAUSAL_LM"] = (
    PeftModelForCalibratedCausalLM
)

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
    PROMPT_CENTERED_QA_COGNITION = "TrustworthyLLM_Cognition_PSQA_Finetuning_Model_0"
    QA_MODEL = "TrustworthyLLM_Ablation_QA_Finetuning_Model"
    PROMPT_CENTERED_QA = "trustworthy_prompt_qa_model"
    TEMP = "temp"

    SAMPLING_LLAMA = "sampling_llama"
    VALIDATION_LLAMA = "validation_llama"

    MISTRAL = "mistral"
    MISTRAL_COGNITION = "TrustworthyLLM_Cognition_Finetuning_Model_Mistral"
    MISTRAL_COGNITION_QA = "TrustworthyLLM_Cognition_QA_Finetuning_Model_Mistral"
    MISTRAL_PSQA = "TrustworthyLLM_Cognition_PSQA_Finetuning_Model_Mistral"

    MAMBA = "tiiuae/falcon-mamba-7b-instruct"

    GEMMA = "Gemma-2-9b-it"
    GEMMA_COGNITION = 'TrustworthyLLM_Cognition_Finetuning_Model_Gemma'
    GEMMA_COGNITION_QA = 'TrustworthyLLM_Cognition_QA_Finetuning_Model_Gemma'
    GEMMA_PSQA = 'TrustworthyLLM_Cognition_PSQA_Finetuning_Model_Gemma'
    LLAMA3 = 'llama3'
    LLAMA3_COGNITION = 'TrustworthyLLM_Cognition_Finetuning_Model_Llama3'
    LLAMA3_COGNITION_QA = 'TrustworthyLLM_Cognition_QA_Finetuning_Model_Llama3'
    LLAMA3_PSQA = 'TrustworthyLLM_Cognition_PSQA_Finetuning_Model_Llama3_0'

    CALIBRATION = "Llama-2-7b-chat-hf-ct-oe"
    HONESTY = "confucius-confidence-verb"
    HONESTY_COGNITION = 'Honesty_Cognition_Finetuning_Model'

    CONTEXT_DPO= "Context-Faithful-LLaMA-2-7b-chat-hf"


def handle_logits(logits, traces=('no', 'not'), strict=True, **kwargs):
    max_logit = -100
    traces = set([trace.lower().strip() for trace in traces])
    for logi in logits:
        for logprob in logi.values():
            token: str = logprob.decoded_token
            token = token.lower().strip()
            logit = logprob.logprob
            if strict:
                if token in traces:
                    max_logit = max(max_logit, logit)
            else:
                for trace in traces:
                    if trace.startswith(token) and len(token) >= 2:
                        max_logit = max(max_logit, logit)
    return max_logit


def call_lora(prompt, model, trace_logit=False, traces=('no', 'not'), strict=True,
              modelName="/mnt/e/OneDrive - wqa/Models/llama-2-7b-chat-hf", **kwargs):
    global lora_model
    if lora_model is None:
        lora_model = LLM(model=modelName, enable_lora=True,
                         gpu_memory_utilization=0.65, max_model_len=1500, max_lora_rank=64)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=150,
        logprobs=5 if trace_logit else None
    )

    outputs = lora_model.generate(
        prompt,
        sampling_params,
        lora_request=LoRARequest("augmentation_adapter", 1, f"LLaMA-Factory/models/{model}")
    )
    # outputs the logits
    if trace_logit:
        return outputs[0].outputs[0].text, handle_logits(outputs[0].outputs[0].logprobs, traces=traces, strict=strict)
    return outputs[0].outputs[0].text


def call_lora_mistral(prompt, model, **kwargs):
    global lora_model
    base_path = "/mnt/e/One_Drive/\"OneDrive - wqa\"/Models/Mistral-7B-Instruct-v0.2"
    if model == ModelEnums.MISTRAL_PSQA or model == ModelEnums.MISTRAL_COGNITION_QA:
        base_path = "LLaMA-Factory/models/TrustworthyLLM_Cognition_Finetuning_Model_Mistral_Merged"
    if lora_model is None:
        lora_model = LLM(model=base_path, enable_lora=True, gpu_memory_utilization=0.95, max_model_len=1600, )
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


def call_lora_llama3(prompt, model, **kwargs):
    global lora_model
    base_path = "/mnt/f/Models/Llama3.1-8B-Instruct"
    if model == ModelEnums.LLAMA3_COGNITION_QA or model == ModelEnums.LLAMA3_PSQA:
        base_path = "LLaMA-Factory/models/TrustworthyLLM_Cognition_Finetuning_Model_Llama3_Merged"
    if lora_model is None:
        lora_model = LLM(model=base_path, enable_lora=True, gpu_memory_utilization=0.90, max_model_len=1600, )
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


def call_lora_gemma(prompt, model, **kwargs):
    global lora_model
    base_path = "/mnt/e/One_Drive/\"OneDrive - wqa\"/Models/Gemma-2-9b-it"
    if model == ModelEnums.GEMMA_COGNITION_QA or model == ModelEnums.GEMMA_PSQA:
        base_path = "/root/autodl-tmp/model/TrustworthyLLM_Cognition_Finetuning_Model_Gemma_Merged"
    if lora_model is None:
        lora_model = LLM(model=base_path, enable_lora=True, gpu_memory_utilization=0.98, max_model_len=1400, )
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


def call_Calibration(prompt, model, **kwargs):
    global tf_model, tf_tokenizer, peft_model
    if tf_model is None:
        tf_tokenizer = AutoTokenizer.from_pretrained(f"/dcs/large/u5590030/Models/llama-2-7b-chat-hf")
        tf_tokenizer.pad_token = tf_tokenizer.eos_token
        tf_tokenizer.model_max_length = 2048
        tf_model = AutoModelForCausalLM.from_pretrained(f"/dcs/large/u5590030/Models/llama-2-7b-chat-hf",
                                                        device_map="auto", torch_dtype="auto")
        peft_model = PeftModel.from_pretrained(
            tf_model,
            f"/dcs/large/u5590030/Models/{model}",
            adapter_name="query",
        )
        peft_model.peft_config["query"].use_temperature = False
        peft_model.eval()

    generation_config = GenerationConfig(
        pad_token_id=tf_tokenizer.pad_token_id,
        bos_token_id=tf_tokenizer.bos_token_id,
        eos_token_id=tf_tokenizer.eos_token_id,
        max_new_tokens=200,
        do_sample=False,
    )

    collate_fn = LabeledStringDataCollator(tf_tokenizer)
    inputs = {k: v.cuda() for k, v in collate_fn([{"context": prompt}]).items()}

    response, P = peft_model.generate(**inputs, generation_config=generation_config, tokenizer=tf_tokenizer,
                                      collate_fn=collate_fn)
    P = float(P[0].item())
    print("P: ", P, flush=True, file=sys.stdout)

    response = tf_tokenizer.batch_decode(
        response[:, inputs.get("input_ids").size(-1):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    response = response[0]
    if P < 0.5:
        response = "Not Provided"

    if kwargs.get('trace_logit', False):
        return response, 0
    return response


def call_transformers(prompt, model=ModelEnums.ORION_RAG_QA_14B, **kwargs):
    global tf_model, tf_tokenizer
    if tf_model is None:
        from huggingface_hub import login
        login(token=configs.huggingface_token)
        tf_tokenizer = AutoTokenizer.from_pretrained(model)
        tf_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype="auto")

    if model == ModelEnums.ORION_RAG_QA_14B:
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
        response = tf_tokenizer.decode(response[0], skip_special_tokens=True)
        response = response.split('ANSWER')[-1]

    if kwargs.get('trace_logit', False):
        return response, 0
    return response


def call_Honesty(prompt, model=ModelEnums.CALIBRATION, **kwargs):
    global tf_model, tf_tokenizer
    if tf_model is None:
        tf_tokenizer = AutoTokenizer.from_pretrained(f"/dcs/large/u5590030/Models/{model}")
        tf_model = AutoModelForCausalLM.from_pretrained(f"/dcs/large/u5590030/Models/{model}", device_map="auto",
                                                        torch_dtype="auto")
    # LlamaForCausalLM
    response = tf_model.generate(
        tf_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda"),
        max_new_tokens=200
    )
    response = tf_tokenizer.decode(response[0], skip_special_tokens=True)
    ans = response.split('ANSWER')[-1]

    context = response.split('QUESTION')[-1]

    PROMPT = f"""\
Is the proposed answer to the given question correct? Please reply with "Yes" or "No".
Question: {context}

Output: """

    check = tf_model.generate(
        tf_tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda"),
        max_new_tokens=200
    )
    check = tf_tokenizer.decode(check[0], skip_special_tokens=True)
    check = check.split('Output')[-1]

    if "no" in check.lower():
        ans = "Not Provided"

    if kwargs.get('trace_logit', False):
        return ans, 0
    return ans


def call_vllm(prompt, modelName, api="http://localhost:9091/v1", trace_logit=False, traces=('no', 'not'), strict=True,
              **kwargs):
    if api:
        openai.api_base = api
        openai.model = modelName
        openai.api_key = "Empty"
        langchain_model = ChatOpenAI(
            model=openai.model,
            openai_api_key=openai.api_key,
            temperature=0,
            max_tokens=200,
            openai_api_base=openai.api_base,
            max_retries=2
        ).bind(logprobs=True,
               top_logprobs=5 if trace_logit else None)
    else:
        global lora_model
        if lora_model is None:
            lora_model = LLM(model=modelName, gpu_memory_utilization=0.65, max_model_len=1500)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=200,
            logprobs=5 if trace_logit else None
        )
        outputs = lora_model.generate(
            prompt,
            sampling_params,
        )
        if trace_logit:
            return outputs[0].outputs[0].text, handle_logits(outputs[0].outputs[0].logprobs, traces=traces,
                                                             strict=strict)
        return outputs[0].outputs[0].text
    msg = langchain_model.invoke(prompt, )  # AIMessage
    res = msg.content
    if trace_logit:
        res = msg.response_metadata["logprobs"]
    return res


def call_vllm_sampling(prompt, modelName, **kwargs):
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
    openai.api_base = "https://api.zhizengzeng.com/v1"
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
    print(completion)
    return completion.choices[0].message.content


def langchain_citation(query, context, model=ModelEnums.GPT4T):
    llm = ChatOpenAI(temperature=0, model=model, openai_api_key=api_key)
    chain = create_citation_fuzzy_match_chain(llm)
    result = chain.run(question=query, context=context)
    return result


def call_llm(query, model, **kwargs):
    if model == ModelEnums.GPT4T or model == ModelEnums.GPT3T or model == ModelEnums.GPT4:
        return ask_gpt(query, model, **kwargs)
    if model in [ModelEnums.ORION_RAG_QA_14B, ModelEnums.MAMBA]:
        return call_transformers(query, model)
    if model in [ModelEnums.HONESTY]:
        return call_Honesty(query, model, **kwargs)
    if model == ModelEnums.CALIBRATION:
        return call_Calibration(query, model, **kwargs)
    if model in [ModelEnums.COGNITION, ModelEnums.PROMPT_CENTERED, ModelEnums.COGNITION_QA,
                 ModelEnums.PROMPT_CENTERED_QA_COGNITION, ModelEnums.QA_MODEL,
                 ModelEnums.PROMPT_CENTERED_QA, ModelEnums.HONESTY_COGNITION,ModelEnums.CONTEXT_DPO]:
        return call_lora(query, model, **kwargs)
    if model == ModelEnums.MISTRAL_PSQA or model == ModelEnums.MISTRAL_COGNITION or model == ModelEnums.MISTRAL_COGNITION_QA:
        return call_lora_mistral(query, model)
    if model == ModelEnums.LLAMA3_COGNITION or model == ModelEnums.LLAMA3_COGNITION_QA or model == ModelEnums.LLAMA3_PSQA:
        return call_lora_llama3(query, model)
    if model == ModelEnums.GEMMA_COGNITION or model == ModelEnums.GEMMA_COGNITION_QA or model == ModelEnums.GEMMA_PSQA:
        return call_lora_gemma(query, model)
    if model == ModelEnums.SAMPLING_LLAMA:
        return call_vllm_sampling(query, **kwargs)
    if model == ModelEnums.MISTRAL:
        return call_vllm(query, modelName="/dcs/large/u5590030/Models/Mistral-7B-Instruct-v0.2", api=None)
    if model == ModelEnums.GEMMA:
        return call_vllm(query, modelName="/dcs/large/u5590030/Models/Gemma-2-9b-it", api=None)
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
