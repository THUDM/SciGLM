import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from models import glm, gpt
from prompt import cot_prompt


def read_json(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


process = 37434
source = 'enhanced_data/enhanced_sft_qa_all.json'
datas = read_json(source)
out_file = 'api_gen_q_enhanced/api_gen_q_gpt4-turbo.json'
with open(out_file, 'a', encoding='utf-8') as f:
    for idx, item in enumerate(tqdm(datas)):
        if idx < process:
            continue
        print(idx)
        q = item['content']
        a = item['summary']
        query = cot_prompt + q + '\nA:'
        for i in range(1):
            tol = 3
            while tol > 0:
                try:
                    # output = glm(query)
                    output = gpt(query, n=1, stop=None)[0].split('\n')
                    solution = ''
                    for sen in output:
                        solution = solution + sen
                    print('output:', solution, '\n')
                    break
                except Exception as e:
                    print(f'error:{e}\n')
                    tol -= 1
            if tol == 0:
                continue
            new_line = {'content': q, 'summary': solution, 'real_answer': a}
            json.dump(new_line, f, ensure_ascii=False)
            f.write('\n')
