# @Time : 2024/1/24 11:10
# @Author : Li Jiaqi
# @Description :
from load_data import load_xls_dataset, load_hagrid_dataset, clean_context, save_xls_dataset
from chatgpt import ModelEnums, call_llm
import prompts
import os
import json

if __name__ == '__main__':
    model = ModelEnums.TEMP
    prompt_flag = prompts.PromptEnums.TASK_COGNITION
    neg_words = ['not mentioned', 'not provided', 'not given', "'t mentioned", "'t provided", "'t given", 'not provide',
                 "'t provide", "'t mention", "not mention"]

    dataset, xls_file = load_xls_dataset(os.path.join('datasets', 'TrustworthyLLM Benchmark.xlsx'))
    reference_sheets = ["benchmark"]
    dataset = {key: value for key, value in dataset.items() if key in reference_sheets}

    for k in dataset.keys():
        print("*" * 20, " processing dataset ", k, " | data size ", len(dataset[k]), " ", "*" * 20)
        pos_correct_num = 0
        neg_correct_num = 0
        pos_sample_count = 0
        neg_sample_count = 0
        # for i in range(len(dataset[k])):
        for i in range(0, len(dataset[k]), 10):
            pos_sample_count += 1
            neg_sample_count += 1
            pos_correct, neg_correct = False, False
            [id, context, pos_query, pos_answer, neg_query] = dataset[k][i]
            pos_answer = pos_answer[1:-1].split(',')
            pos_answer = [answer.strip()[1:-1].lower() for answer in pos_answer]
            print("context:", context)
            print("pos_query:", pos_query)
            print("pos_answer_gt:", pos_answer)
            prompt1 = prompts.get_task_prompt(context, pos_query, version=prompt_flag)
            answer1 = call_llm(query=prompt1, model=model, modelName="default")
            print("pos_answer:", answer1)
            answer1_lower = answer1.lower()
            suf_num = answer1_lower.count('sufficient') - answer1_lower.count('insufficient')
            insuf_num = answer1_lower.count('insufficient')
            if suf_num == insuf_num == 0:
                continue
            elif suf_num>=insuf_num:
                pos_correct = True
                pos_correct_num += 1

            print("neg_query", neg_query)
            prompt2 = prompts.get_task_prompt(context, neg_query, version=prompt_flag)
            answer2 = call_llm(query=prompt2, model=model, modelName="default")
            print("neg_answer:", answer2)
            answer2_lower = answer2.lower()
            suf_num = answer2_lower.count('sufficient') - answer2_lower.count('insufficient')
            insuf_num = answer2_lower.count('insufficient')
            if suf_num<=insuf_num:
                neg_correct = True
                neg_correct_num += 1
            print(f'correct: {pos_correct} {neg_correct}')
            print(
                f'correct rate: {100 * pos_correct_num / pos_sample_count:.2f} {100 * neg_correct_num / neg_sample_count:.2f}') if pos_sample_count > 0 and neg_sample_count > 0 else None
            print('-' * 10, f" {i + 1}/{len(dataset[k])} ", '-' * 10)
            dataset[k][i].append(answer1)
            dataset[k][i].append(pos_correct)
            dataset[k][i].append(answer2)
            dataset[k][i].append(neg_correct)

        print("*" * 20, " finished processing dataset ", k, " | data size ", len(dataset[k]), " ", "*" * 20)
        print("*" * 20, f" positive correct samples: {pos_correct_num} negative correct samples: {neg_correct_num} ",
              "*" * 20)
        print('*' * 80)
