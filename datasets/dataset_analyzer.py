# @Time : 2024/1/31 9:43
# @Author : Li Jiaqi
# @Description :
import json
import os
import prompts


def count_vulnerable_samples(file_name):
    file_path = os.path.join("datasets", "original", file_name)
    print("dataset:", file_name)
    if file_name == "simplified-nq-train.jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            res = []
            empty_res = []
            content = f.readlines()
            for line in content:
                data = json.loads(line)
                res.append(data)
                if len(data['annotations'][0]['long_answer']) == 0 and len(
                        data['annotations'][0]['short_answers']) == 0:
                    empty_res.append(data)
            print(
                f"total samples: {len(res)} | vulnerable samples: {len(empty_res)} | proportion:{len(empty_res) / len(res) * 100:.2f}% \n")
    elif file_name == "self-rag-train.jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            res = []
            empty_res = []
            content = f.readlines()
            for line in content:
                data = json.loads(line)
                res.append(data)
                if "Irrelevant" in data['output']:
                    empty_res.append(data)
            print(
                f"total samples: {len(res)} | vulnerable samples: {len(empty_res)} | proportion:{len(empty_res) / len(res) * 100:.2f}% \n")
    elif file_name == "ASQA.json":
        with open(file_path, "r", encoding="utf-8") as f:
            res = []
            empty_res = []
            content = f.readlines()[0]
            data = json.loads(content)['train']
            for line in data.values():
                res.append(line)
                for pair in line['qa_pairs']:
                    if pair['context'] != "No context provided":
                        answer = pair['short_answers']
                        answer = " ".join(answer).lower()
                        for word in prompts.neg_words:
                            if answer.find(word) != -1:
                                empty_res.append(line)
                                break
            print(
                f"total samples: {len(res)} | vulnerable samples: {len(empty_res)} | proportion:{len(empty_res) / len(res) * 100:.2f}% \n")
    elif file_name == "QAMPARI_train_data.jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            res = []
            empty_res = []
            content = f.readlines()
            for line in content:
                data = json.loads(line)
                res.append(data)
                if len(data['answer_list']) == 0:
                    empty_res.append(data)
            print(
                f"total samples: {len(res)} | vulnerable samples: {len(empty_res)} | proportion:{len(empty_res) / len(res) * 100:.2f}% \n")
    elif file_name == "eli5":
        import nlp
        eli5 = nlp.load_dataset('eli5')['train_eli5']
        res = []
        empty_res = []
        for sample in eli5:
            res.append(sample)
        print(
            f"total samples: {len(res)} | vulnerable samples: {len(empty_res)} | proportion:{len(empty_res) / len(res) * 100:.2f}% \n")
    elif file_name == "hotpot_train_v1.1.json":
        res = json.loads(open(file_path).read())
        empty_res = []
        for d in res:
            answer = d['answer'].lower()
            for word in prompts.neg_words:
                if answer.find(word) != -1:
                    empty_res.append(d)
                    break
        print(
            f"total samples: {len(res)} | vulnerable samples: {len(empty_res)} | proportion:{len(empty_res) / len(res) * 100:.2f}% \n")



# count_vulnerable_samples("simplified-nq-train.jsonl")
# count_vulnerable_samples("self-rag-train.jsonl")
count_vulnerable_samples("ASQA.json")
# count_vulnerable_samples("QAMPARI_train_data.jsonl")
# count_vulnerable_samples("eli5")
# count_vulnerable_samples("hotpot_train_v1.1.json")
# reformat_hotpot()
