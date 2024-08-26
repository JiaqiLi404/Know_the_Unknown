import json
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from load_data import load_jsonl_dataset


def generate_trustworthy_augmentation_model_dataset(dataset):
    llm = LLM(model="/mnt/f/Models/llama-2-7b-chat-hf", enable_lora=True, gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2048
    )
    for sample in dataset:
        prompt = json.dumps(sample)
        outputs = llm.generate(
            prompt,
            sampling_params,
            lora_request=LoRARequest("augmentation_adapter", 1, "/mnt/f/Models/augmentation_sft_model")
        )
        print(outputs)


dataset = load_jsonl_dataset(os.path.join("qa_datasets", "json_analyzer_files", "simplified-nq-train.jsonl"))
for d in dataset:
    query, context, answer = d["question_text"], d["document_text"], d['annotations'][0]['short_answers']
    answer_text = "Not Provided"
    if len(answer)>0:
        answer_text = " ".join(context.split(" ")[answer[0]['start_token']:answer[0]['end_token']])
    print(query, "|||||",answer_text)
generate_trustworthy_augmentation_model_dataset(dataset)
