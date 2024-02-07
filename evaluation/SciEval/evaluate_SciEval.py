import json
import argparse
import tqdm
import os
import glob
import re
import json
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_file", type=str,
                        choices=['./datasets/bai-scieval-valid.json', './datasets/bai-scieval-test.json'],
                        default='./datasets/bai-scieval-valid.json')  # 评测数据集
    parser.add_argument('--data_file', type=str, default= str(filename.split('/')[-2]) + '_' + str(filename.split('/')[-1]) + '_scieval_valid.json')  # 模型输出数据位置
    parser.add_argument('--outcomes_file', type=str, default=str(filename.split('/')[-2]) + '_' + str(filename.split('/')[-1])+'_scieval_valid_outcomes.json')  # 模型评测结果位置
    args = parser.parse_args()
    return args


def get_results(args, tokenizer, model):
    out_path = os.path.join('./results/', args.data_file)
    input_path = os.path.join('./', args.target_file)
    with open(input_path, 'r') as reader:
        label_data = json.load(reader)

    label_data = dict([(label["id"], label) for label in label_data])
    outputs = []
    bars = tqdm(label_data.items())
    for id, label in bars:
        bars.set_description(f'id<{id}>')
        prompt = label['prompt']
        question = label['question']
        prompt = prompt + '\ninput:' + question
        inputs = tokenizer([prompt], padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
        output_ = model.generate(**inputs, do_sample=False, max_new_tokens=512)
        output = output_.tolist()[0][len(inputs["input_ids"][0]):]
        output = tokenizer.decode(output)
        if output:
            output = output.strip()
        else:
            output = ''
        outputs.append({'id': id, 'pred': output})

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)


def eval(args):
    input_path = os.path.join('./results/', args.data_file)
    with open(input_path, 'r', encoding='utf-8') as reader:
        data = json.load(reader)

    """
    Predict label format:
    [{
            "id": "1",
            "pred": "A"
    }]
    """

    source_path = os.path.join('./', args.target_file)
    with open(source_path, 'r') as reader:
        label_data = json.load(reader)

    label_data = dict([(label["id"], label) for label in label_data])

    category_judge = {
        "biology": [0, 0, 0, 0],
        "chemistry": [0, 0, 0, 0],
        "physics": [0, 0, 0, 0]
    }
    category_num = {
        "biology": [0, 0, 0, 0],
        "chemistry": [0, 0, 0, 0],
        "physics": [0, 0, 0, 0]
    }
    ability_index = {
        "Base Knowledge": 0,
        "Knowledge Application": 1,
        "Scientific Calculation": 2,
        "Research Ability": 3,
    }
    index_ability = dict([(value, key) for key, value in ability_index.items()])

    all_cnt = 0

    for d in data:
        data_id = d["id"]
        pred = d["pred"]

        answer = label_data[data_id]["answer"][0]
        question_type = label_data[data_id]["type"]
        question_category = label_data[data_id]["category"]
        ability = label_data[data_id]["ability"]
        category_num[question_category][ability_index[ability]] += 1
        if question_type == "multiple-choice":
            if len(pred) > 0:
                if answer.lower() == pred[0].lower():
                    category_judge[question_category][ability_index[ability]] += 1
                    all_cnt += 1
        elif question_type == "judge":
            if answer.lower() in pred.lower():
                category_judge[question_category][ability_index[ability]] += 1
                all_cnt += 1
        elif question_type == "filling":
            if answer.lower() in pred.lower():
                category_judge[question_category][ability_index[ability]] += 1
                all_cnt += 1
        else:
            raise ValueError

    results = {}
    for category in category_judge.keys():
        # print(f"==== {category} ====")
        results[category] = {}
        category_j = category_judge[category]
        category_n = category_num[category]
        for i in range(len(category_j)):
            if category_n[i] == 0:
                continue
            results[category][index_ability[i]] = category_j[i] / category_n[i]
            print(index_ability[i], category_j[i] / category_n[i])
        results[category]["all"] = sum(category_j) / sum(category_n)

    results["all"] = all_cnt / len(data)

    return results


if __name__ == '__main__':
    filenames = []
    directory_paths = [
                    '/path/to/SciGLM_checkpoint/',
                    ]
    for directory_path in directory_paths:
        for root, dirs, files in os.walk(directory_path):
            for dir in dirs:
                if dir.startswith('checkpoint-') and 'checkpoint-' in dir:
                    ckpt_dir = os.path.abspath(os.path.join(root, dir))
                    print(ckpt_dir)
                    filenames.append(ckpt_dir)
    print(filenames)          
    for filename in filenames:
        tokenizer = AutoTokenizer.from_pretrained(filename, trust_remote_code=True)
        model = AutoModel.from_pretrained(filename, trust_remote_code=True).bfloat16().cuda()
        args = parse_args()
        get_results(args, tokenizer, model)
        if 'valid' in args.target_file:
            results = eval(args)
            with open(os.path.join('./results/', args.outcomes_file), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)