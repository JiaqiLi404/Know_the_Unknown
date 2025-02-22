import os
import json
import random

"""
It is used to generate the data for hallucination fine-tuning.
"""
choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_question(input_list):
    prompt = input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], input_list[j + 1])
    prompt += "\nAnswer:"
    return prompt


def format_shots(prompt_data):
    prompt = ""
    for data in prompt_data:
        prompt += data[0]
        k = len(data) - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], data[j + 1])
        prompt += "\nAnswer:"
        prompt += data[k + 1] + "\n\n"

    return prompt


def generate_MMLU_finetuning_dataset():
    file_path = os.path.join("data", "MMLU", "MMLU_ID_train.json")
    example_path = os.path.join("data", "MMLU", "MMLU_ID_prompt.json")
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.readlines()[0]
        data = json.loads(content)
    with open(example_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    for category in data:
        example = examples[category]
        for sample in data[category]:
            [question, opt1, opt2, opt3, opt4, answer] = sample
            instruction = "The following are multiple choice questions (with answers) about {}.\n\n".format(
                format_subject(category)
            )
            instruction += format_shots(example)
            instruction += format_question(sample)
            res = {
                "instruction": instruction,
                "input": "",
                "output": answer
            }
            results.append(res)

    random.shuffle(results)

    with open(os.path.join('data', "MMLU_ID_Train_LLaMA_Factory.json"), 'w') as file:
        json.dump(results, file, indent=2)


def generate_PARAREL_finetuning_dataset():
    file_path = os.path.join("data", "PARAREL", "training_data.json")
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.readlines()[0]
        data = json.loads(content)
    for sample in data:
        [question, answer, category] = sample
        instruction = f"Question: {question} Answer: "
        res = {
            "instruction": instruction,
            "input": "",
            "output": answer
        }
        results.append(res)

    random.shuffle(results)

    with open(os.path.join('data', "PARAREL_ID_Train_LLaMA_Factory.json"), 'w') as file:
        json.dump(results, file, indent=2)


generate_PARAREL_finetuning_dataset()
# generate_MMLU_finetuning_dataset()
