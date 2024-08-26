# @Time : 2024/2/1 18:06
# @Author : Li Jiaqi
# @Description :
import json
import os
from itertools import permutations

from load_data import save_xls_dataset, load_xls_dataset

def generate_trustworthy_benchmark():
    file_path = os.path.join("datasets", "original", "ASQA.json")
    res = []
    empty_res = []
    clean_dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.readlines()[0]
        data = json.loads(content)['train']
        for line in data.values():
            res.append(line)
            has_context = False
            context = {}
            for pair in line['qa_pairs']:
                if pair['context'] != "No context provided":
                    context[pair['context']] = pair
                    has_context = True
            if has_context and len(context) > 1:
                empty_res.append(list(context.values()))
    # sample {question,context,[short_answers]}
    # [id,context,pos_query,pos_answer,neg_query]
    benchmark = []
    id = 0
    for sample_collection in empty_res:
        for [sample1, sample2] in permutations(sample_collection, 2):
            context = f"This is a passage about {sample1['wikipage']}:\n{sample1['context']}"
            benchmark.append(
                [str(id), context, sample1['question'], sample1['short_answers'], sample2['question']])
            id += 1
            pos_context = context.lower()
            pos_answers = [x.lower() for x in sample1['short_answers']]
            neg_context = f"This is a passage about {sample2['wikipage']}:\n{sample2['context']}".lower()
            neg_answers = [x.lower() for x in sample2['short_answers']]
            pos_answer_in_context = False
            neg_answer_not_in_context = True
            for answer in pos_answers:
                if answer in pos_context:
                    pos_answer_in_context = True
                    break
            for answer in pos_answers:
                if answer in neg_context:
                    neg_answer_not_in_context = False
                    break
            if pos_answer_in_context and neg_answer_not_in_context:
                clean_dataset.append(benchmark[-1])

    save_xls_dataset({"benchmark": clean_dataset, 'benchmark_uncleaned': benchmark}, os.path.join('datasets',"TrustworthyLLM Benchmark.xlsx"),
                     ['id', 'context', 'pos_query', 'pos_answer', 'neg_query'])

def generate_trustworthy_benchmark_coqa():
    file_path = os.path.join("datasets", "original", "coqa_abg_train.json")
    res = []
    dataset=[]
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for sample in data['data']:
            res.append(sample)
    for sample in res:
        if sample['ambiguity']!='non_ambiguous':
            context = sample['story']
            for k,clarification_dicts in sample.items():
                if k.startswith('clarification_turn'):
                    pos_query = ""
                    pos_answer = ""
                    neg_query = ""
                    for clarification_dict in clarification_dicts['answers']:
                        if clarification_dict['org_ans']=='Unknown':
                            neg_query = clarification_dicts['question']+clarification_dict['clr_ans']
                        else:
                            pos_query =  clarification_dicts['question']+clarification_dict['clr_ans']
                            pos_answer = clarification_dict['org_ans']
                    if pos_query!="" and pos_answer!="" and neg_query!="":
                        dataset.append([len(dataset),context,pos_query,pos_answer,neg_query])



    print("total samples: ", len(res))
    #save_xls_dataset({"benchmark": res}, os.path.join('datasets',"TrustworthyLLM Benchmark2.xlsx"),
    #                  ['id', 'context', 'pos_query', 'pos_answer', 'neg_query'])


# generate_trustworthy_benchmark()
generate_trustworthy_benchmark_coqa()
