from typing import List, Dict
import os
import json
import random

import prompts
from chatgpt import ModelEnums, call_llm
from load_data import save_xls_dataset, load_xls_dataset


def generate_cognition_finetuning_dataset(with_qa=False):
    """
    Generate the xlsx formatted dataset for trustworthy reward model training
    :param first: Whether exits 'trustworthy_reward' in xlsx dataset
    :return: a xlsx formatted dataset, sheet named 'trustworthy_reward' records the training data of trustworthy reward model
    """
    results = []
    benchmark_dataset, xls_file = load_xls_dataset(os.path.join('datasets', 'TrustworthyLLM Benchmark.xlsx'))
    benchmark_cleaned: List = benchmark_dataset['benchmark']
    if with_qa:
        qa_dataset = []
        qa_data = json.loads(open(os.path.join("datasets", 'original', "hotpot_train_v1.1.json")).read())
        qa_i = 0
    for sample in benchmark_cleaned:
        [id, context, pos_query, pos_answer, neg_query] = sample
        res = {
            "instruction": prompts.get_task_prompt(context, pos_query, prompts.PromptEnums.TASK_COGNITION),
            "input": "",
            "output": "Sufficient"
        }
        results.append(res)
        res = {
            "instruction": prompts.get_task_prompt(context, neg_query, prompts.PromptEnums.TASK_COGNITION),
            "input": "",
            "output": "Insufficient"
        }
        results.append(res)
        if with_qa:
            d = qa_data[qa_i]
            question = d['question']
            context = ["".join(c[1]) for c in d['context']]
            context = "\n".join(context)
            answer = d['answer']
            results.append(
                {"instruction": prompts.get_task_prompt(context, question, version=prompts.PromptEnums.TASK_QA),
                 "input": "", "output": answer})
            qa_dataset.append(results[-1])
            qa_i += 1
    with open(os.path.join('datasets',
                           'TrustworthyLLM_Cognition_Finetuning_Dataset.json' if not with_qa else "TrustworthyLLM_Cognition_QA_Finetuning_Dataset.json"),
              'w') as file:
        json.dump(results, file, indent=2)

    return results


def generate_prompt_sensitive_finetuning_dataset(st=0, end=300):
    '''
    Before running this function, you should have a prompt_check_dataset.xlsx file,
    with the instructions sheet recording the additional instructions
    '''
    model = ModelEnums.GPT4T
    prompt_flag = prompts.PromptEnums.TASK_PROMPT

    # results = []
    # dataset, xls_file = load_xls_dataset(os.path.join('datasets', 'prompt_sensitive_dataset.xlsx'))
    #
    # hotpot_dataset = json.loads(
    #     open(os.path.join('datasets', 'original', "hotpot_train_v1.1.json")).read())
    # instructions = [i[0] for i in dataset['instructions']]
    #
    # for i, d in enumerate(hotpot_dataset):
    #     question = d['question']
    #     context = ["".join(c[1]) for c in d['context']]
    #     context = "\n".join(context)
    #     answer = d['answer']
    #     task = instructions[random.randint(0, len(instructions) - 1)]
    #     results.append([task, question, context, answer])
    # dataset['dataset'] = results
    # save_xls_dataset(dataset, os.path.join('datasets', f'prompt_sensitive_dataset.xlsx'),
    #                  ['task', 'query', 'context', 'answer'])
    #
    # dataset, xls_file = load_xls_dataset(os.path.join('datasets', 'prompt_sensitive_dataset.xlsx'))
    # prompt_check_dataset = dataset['dataset']
    # for i in range(st, end):
    #     [task, query, context, answer] = prompt_check_dataset[i][:4]
    #     print("context:", context)
    #     print("query:", query)
    #     print("task:", task)
    #     prompt = prompts.get_task_prompt(context, query, version=prompt_flag, specific_instruction=task)
    #     answer = call_llm(query=prompt, model=model)
    #     print("answer:", answer)
    #     prompt_check_dataset[i].append(answer) if answer.find("CHECKING") != -1 else None
    #     if len(prompt_check_dataset[i])>=5:
    #         prompt = prompts.get_task_prompt(context, query, version=prompts.PromptEnums.TASK_QA)
    #         answer = call_llm(query=prompt, model=model)
    #         print("direct answer:", answer)
    #         prompt_check_dataset[i].append(answer)
    #
    #     print('-' * 10, f" {i + 1}/{len(prompt_check_dataset)} ", '-' * 10)
    #
    # print("*" * 20, " finished processing dataset | data size ", len(prompt_check_dataset), " ", "*" * 20)
    # print('*' * 100)
    #
    # save_xls_dataset(dataset, os.path.join('datasets', f'prompt_sensitive_dataset.xlsx'),
    #                  ['task', 'query', 'context', 'answer', "GPT_Prompt_Sensitive_answer", "GPT_Direct_answer"])

    results_prompt_check = []
    results_qa = []
    results_gpt = []
    results_cong_psqa = []
    dataset, xls_file = load_xls_dataset(os.path.join('datasets', 'prompt_sensitive_dataset.xlsx'))
    datas = dataset['dataset'][st:end]

    for d in datas:
        if len(d) < 6:
            continue
        [task, query, context, answer, Prompt_Sensitive_answer, gpt_answer] = d[:6]
        results_prompt_check.append(
            {"instruction": prompts.get_task_prompt(context, query, version=prompt_flag, specific_instruction=task),
             "input": "", "output": Prompt_Sensitive_answer})
        # answer_st = 0
        # anser_end = len(Prompt_Sensitive_answer)
        # if 'ANSWER:\n' in Prompt_Sensitive_answer:
        #     answer_st = Prompt_Sensitive_answer.rfind('ANSWER:\n') + 7
        # if 'CHECKING' in Prompt_Sensitive_answer:
        #     anser_end = Prompt_Sensitive_answer.rfind('CHECKING')
        # answer_pure = Prompt_Sensitive_answer[answer_st:anser_end]
        # if 'CHECKING' in answer_pure:
        #     anser_end = answer_pure.find('CHECKING')
        #     answer_pure = answer_pure[:anser_end]
        # answer_pure = answer_pure.strip()
        # results_prompt_check.append(
        #     {"instruction": prompts.get_task_prompt(context, query, version=prompts.PromptEnums.TASK_QA,
        #                                             specific_instruction=task),
        #      "input": "", "output": answer_pure})
        results_qa.append(
            {"instruction": prompts.get_task_prompt(context, query, version=prompts.PromptEnums.TASK_QA),
             "input": "", "output": answer})
        results_gpt.append(
            {"instruction": prompts.get_task_prompt(context, query, version=prompts.PromptEnums.TASK_QA),
             "input": "", "output": gpt_answer})

    with open(os.path.join('datasets', f'TrustworthyLLM_PromptSensitive_Finetuning_Dataset.json'), "w") as f:
        json.dump(results_prompt_check, f, indent=2)
    with open(os.path.join('datasets', f'TrustworthyLLM_QA_Finetuning_Dataset.json'), "w") as f:
        json.dump(results_qa, f, indent=2)
    with open(os.path.join('datasets', f'TrustworthyLLM_GPT_Finetuning_Dataset.json'), "w") as f:
        json.dump(results_gpt, f, indent=2)

    benchmark_dataset, xls_file = load_xls_dataset(os.path.join('datasets', 'TrustworthyLLM Benchmark.xlsx'))
    benchmark_cleaned: List = benchmark_dataset['benchmark']
    psqa_i = 0
    for sample in benchmark_cleaned:
        [id, context, pos_query, pos_answer, neg_query] = sample
        res = {
            "instruction": prompts.get_task_prompt(context, pos_query, prompts.PromptEnums.TASK_COGNITION),
            "input": "",
            "output": "Sufficient"
        }
        results_cong_psqa.append(res)
        res = {
            "instruction": prompts.get_task_prompt(context, neg_query, prompts.PromptEnums.TASK_COGNITION),
            "input": "",
            "output": "Insufficient"
        }
        results_cong_psqa.append(res)
        while len(datas[psqa_i]) < 6:
            psqa_i += 1
            psqa_i = psqa_i % end
        [task, query, context, answer, Prompt_Sensitive_answer, gpt_answer] = datas[psqa_i][:6]
        res = {"instruction": prompts.get_task_prompt(context, query, version=prompts.PromptEnums.TASK_QA),
               "input": "", "output": answer}
        results_qa.append(res)
        res = {
            "instruction": prompts.get_task_prompt(context, query, version=prompt_flag, specific_instruction=task),
            "input": "",
            "output": Prompt_Sensitive_answer
        }
        results_cong_psqa.append(res)
        psqa_i = (psqa_i + 1) % end
    with open(os.path.join('datasets', "TrustworthyLLM_Cognition_PSQA_Finetuning_Dataset.json"), 'w') as file:
        json.dump(results_cong_psqa, file, indent=2)


# generate_cognition_finetuning_dataset(with_qa=False)
generate_prompt_sensitive_finetuning_dataset()
