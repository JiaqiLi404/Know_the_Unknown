from load_data import load_xls_dataset, load_hagrid_dataset, clean_context, save_xls_dataset
from chatgpt import ModelEnums, call_llm
import prompts
import os
import json

if __name__ == '__main__':
    model = ModelEnums.PROMPT_CENTERED_QA_COGNITION
    neg_words = prompts.neg_words


    dataset, xls_file = load_xls_dataset(os.path.join('benchmark_records','final',f'TrustworthyLLM Benchmark - {model}.xlsx'))
    dataset = dataset["benchmark"]

    correct=wrong=unknown=0


    for record in dataset:
        [id, context, pos_query, pos_answer, neg_query, pos_answer_pred, pos_correct, neg_answer_pred, neg_correct] = record
        # pos_correct=neg_correct
        # pos_answer_pred=neg_answer_pred

        if pos_correct=='True':
            correct+=1
            continue
        pos_answer_pred = pos_answer_pred.lower()

        pos_reject = False
        for word in neg_words:
            if pos_answer_pred.find(word) != -1:
                pos_reject = True
                break

        if pos_reject:
            unknown+=1
        else:
            wrong+=1

    print(f"correct: {correct}, wrong: {wrong}, unknown: {unknown}")
