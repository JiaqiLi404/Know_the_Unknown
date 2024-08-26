from typing import List, Dict
import os
from chatgpt import ModelEnums, call_llm, autodl_speedup
import prompts
import copy
import json

from load_data import save_xls_dataset, load_xls_dataset

neg_words = ['not mentioned', 'not provided', 'not given', "'t mentioned", "'t provided", "'t given", 'not provide',
             "'t provide", "'t mention", "not mention"]


def generate_trustworthy_reward_model_dataset_xlsx(first=True):
    """
    Generate the xlsx formatted dataset for trustworthy reward model training
    :param first: Whether exits 'trustworthy_reward' in xlsx dataset
    :return: a xlsx formatted dataset, sheet named 'trustworthy_reward' records the training data of trustworthy reward model
    """
    positive_model = ModelEnums.GPT4T
    positive_prompt_flag = prompts.model_prompt_map.get(positive_model, None)
    negative_model = ModelEnums.NONE
    negative_prompt_flag = prompts.model_prompt_map.get(negative_model, None)
    benchmark_dataset, xls_file = load_xls_dataset(os.path.join('qa_datasets', 'TrustworthyLLM Benchmark.xlsx'))
    if first:
        benchmark_cleaned: List = benchmark_dataset['benchmark']
    else:
        benchmark_cleaned: List = benchmark_dataset['trustworthy_reward']
    benchmark_trustworthy_reward = copy.deepcopy(benchmark_cleaned)

    pos_correct_num = 0
    neg_correct_num = 0
    # for i in range(1):
    for i in range(len(benchmark_trustworthy_reward)):
        pos_correct, neg_correct = False, False
        [id, context, pos_query, pos_answer, neg_query] = benchmark_trustworthy_reward[i][:5]
        while len(benchmark_trustworthy_reward[i]) < 9:
            benchmark_trustworthy_reward[i].append('none')
        print("query:", neg_query)

        if positive_prompt_flag:
            positive_prompt = prompts.get_citation_prompt(neg_query, context, version=positive_prompt_flag)
            positive_answer = call_llm(query=positive_prompt, model=positive_model, modelName="InvestLM2-dpo-awq")
            print("pos_answer:", positive_answer)
            positive_answer_lower = positive_answer.lower()
            for word in neg_words:
                if positive_answer_lower.find(word) != -1:
                    pos_correct = True
                    pos_correct_num += 1
                    break
            benchmark_trustworthy_reward[i][5] = positive_answer
            benchmark_trustworthy_reward[i][6] = pos_correct

        if negative_prompt_flag:
            negative_prompt = prompts.get_citation_prompt(neg_query, context, version=negative_prompt_flag)
            negative_answer = call_llm(query=negative_prompt, model=negative_model, modelName="InvestLM2-dpo-awq")
            print("neg_answer:", negative_answer)
            negative_answer_lower = negative_answer.lower()
            for word in neg_words:
                if negative_answer_lower.find(word) != -1:
                    neg_correct = True
                    neg_correct_num += 1
                    break
            benchmark_trustworthy_reward[i][7] = negative_answer
            benchmark_trustworthy_reward[i][8] = neg_correct

        print(f'correct: {pos_correct} {neg_correct}')
        print(f'correct rate: {100 * pos_correct_num / (i + 1):.2f} {100 * neg_correct_num / (i + 1):.2f}')
        print('-' * 10, f" {i + 1}/{len(benchmark_trustworthy_reward)} ", '-' * 10)

    benchmark_dataset['trustworthy_reward'] = benchmark_trustworthy_reward
    xls_file.close()
    save_xls_dataset(benchmark_dataset, os.path.join('qa_datasets', f'TrustworthyLLM Benchmark.xlsx'),
                     ['id', 'context', 'pos_query', 'pos_answer', 'neg_query', 'pos_answer',
                      'pos_answer_correctness', 'neg_answer', 'neg_answer_correctness'])


def convert_xlsx_to_json():
    """
    {
    "instruction": "What are the three primary colors?",
    "input": "",
    "output": [
      "The three primary colors are red, blue, and yellow. These colors are called primary because they cannot be created by mixing other colors and all other colors can be made by combining them in various proportions. In the additive color system, used for light, the primary colors are red, green, and blue (RGB).",
      "Red, Yellow, and Green."
    ]
    }
    """
    results = []
    benchmark_dataset, xls_file = load_xls_dataset(os.path.join('qa_datasets', 'TrustworthyLLM Benchmark.xlsx'))
    trustworthy_reward_dataset: List = benchmark_dataset['trustworthy_reward']
    for sample in trustworthy_reward_dataset:
        [id, context, pos_query, pos_answer, neg_query, pos_answer, pos_answer_correctness, neg_answer,
         neg_answer_correctness] = sample
        res = {
            "instruction": neg_query,
            "input": context,
            "output": [pos_answer, neg_answer]
        }
        results.append(res)
    with open(os.path.join('qa_datasets', 'TrustworthyLLM_Reward_Model.json'), 'w') as file:
        json.dump(results, file, indent=2)

    return results


def generate_augmentation_dataset():
    results = []
    benchmark_dataset, xls_file = load_xls_dataset(os.path.join('qa_datasets', 'TrustworthyLLM Benchmark.xlsx'))
    trustworthy_reward_dataset: List = benchmark_dataset['trustworthy_reward']
    res = []
    for sample in trustworthy_reward_dataset:
        [id, context, pos_query, pos_answer, neg_query, pos_answer, pos_answer_correctness, neg_answer,
         neg_answer_correctness] = sample
        if pos_answer_correctness == "True":
            res = {
                "instruction": prompts.get_task_prompt(context, pos_query),
                "input": "",
                "output": neg_query
            }
            results.append(res)
    with open(os.path.join('qa_datasets', 'TrustworthyLLM_Augmentation_Model.json'), 'w') as file:
        json.dump(results, file, indent=2)

    return results


if __name__ == '__main__':
    # autodl_speedup()
    # generate_trustworthy_reward_model_dataset_xlsx(first=False)
    # convert_xlsx_to_json()
    generate_augmentation_dataset()
