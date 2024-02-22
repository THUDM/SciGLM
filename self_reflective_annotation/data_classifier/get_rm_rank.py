import json
import pandas as pd

preds = pd.read_csv('ckpt_gen_preds/ckpt_gen_preds_rm4_q.csv').values[:, 1]
source = 'ckpt_gen_q/ckpt_gen_q.json'
all_data = []
with open(source, "rb") as fp:
    for idx, line in enumerate(fp.readlines()):
        obj = json.loads(line)
        question, answer, real_answer = obj['content'], obj['summary'], obj['real_answer']
        rank_data = {'content': question, 'summary': answer, 'real_answer': real_answer, 'score': float(preds[idx])}
        all_data.append(rank_data)

assert len(preds) == len(all_data)
sorted_data = sorted(all_data, key=lambda x: x['score'])

invalid_json = 'filtered/gen_q_neg_rm4.json'
valid_json = 'filtered/gen_q_pos_rm4.json'
with open(valid_json, 'w', encoding='utf-8') as f1, open(invalid_json, 'w', encoding='utf-8') as f2:
    for i in range(len(preds)):
        if sorted_data[i]['score']<=0:
            new_line = {'content': sorted_data[i]['content'], 'summary': sorted_data[i]['summary'], 'real_answer':sorted_data[i]['real_answer'], 'score': sorted_data[i]['score']}
            json.dump(new_line, f2, ensure_ascii=False)
            f2.write('\n')
        else:
            new_line = {'content': sorted_data[i]['content'], 'summary': sorted_data[i]['summary'], 'real_answer':sorted_data[i]['real_answer'], 'score': sorted_data[i]['score']}
            json.dump(new_line, f1, ensure_ascii=False)
            f1.write('\n')
