import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from models import glm, gpt


def read_json(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


def is_choice(s):
    for c in s:
        if c not in ['A', 'B', 'C', 'D', 'E', 'F', ' ']:
            return False
    return True


process = 10194
source = 'api_gen_q_enhanced/filtered/gpt4_revised_left_real.json'
datas = read_json(source)
prompt = '下面将输入三段文字，第一段是一个理科的题目，第二段是该题目对应的一个解答。但是，这个解答是错误的，请反思其错误，然后参考第三段文字给出的正确答案分步地给出详细的正确的解题过程。注意，输出格式为“反思：...正确解答：...”。文段1：'
out_file = 'api_gen_q_enhanced/revised/api_gen_q_gpt4-turbo_revised_not_choice.json'
with open(out_file, 'a', encoding='utf-8') as f:
    for idx, item in enumerate(tqdm(datas)):
        if idx < process:
            continue
        print(idx)
        q = item['content']
        a = item['summary']
        ra = item['real_answer']
        if is_choice(ra):
            continue
        query = prompt + q + '\n文段2：' + a + '\n文段3：' + ra + '\n输出：'
        # inputs = tokenizer(query, return_tensors="pt").to('cuda')
        # print(inputs)
        for i in range(1):
            tol = 3
            while tol > 0:
                try:
                    # output = glm(query)
                    output = gpt(query, n=1, stop=None)[0].split('\n')
                    solution = ''
                    for sen in output:
                        solution = solution + sen
                    print('ori_output:', solution, '\n')
                    if '正确解答：' in solution:
                        solution = solution.split('正确解答：')[-1]
                        print('trunc_output:', solution, '\n')
                        break
                    else:
                        tol -= 1
                except Exception as e:
                    print(f'error:{e}\n')
                    tol -= 1
            if tol == 0:
                continue
            new_line = {'content': q, 'summary': solution, 'real_answer': ra, 'idx': idx}
            json.dump(new_line, f, ensure_ascii=False)
            f.write('\n')
