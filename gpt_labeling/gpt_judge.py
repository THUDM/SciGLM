import openai
import json
from tqdm import tqdm

openai.api_key = ""
openai.api_base = ""
judge_model = 'gpt-4-1106-preview'

source = 'revised/api_gen_q_gpt4-turbo_revised_not_choice.json'
out_file1 = 'filtered/gpt4_revised_filtered_not_choice.json'
out_file2 = 'filtered/gpt4_revised_left_not_choice.json'


# real_answer_source = 'filtered/gpt4_left.json'


def get_label(ans, real_ans):
    prompt = '下面将输入两段文字，第一段文字为某道理科题目的解答，第二段是这道题目的标准答案。请判断第一段解答得到的答案与标准答案是否一致，并根据判断直接输出‘0’或’1‘，不需要输出任何别的信息。如果答案一致，请输出‘1’；否则，只要答案不匹配，或者第一个文段中没有明确指出答案也没有输出latex表达式，请输出‘0’；如果第一段解答与标准答案之间关系模糊，请输出‘0’。\n'
    qry = prompt + '文段1：' + ans + '\n' + '文段2：' + real_ans + '\n输出：'
    lbl = ''
    cnt = 10
    while lbl == '' and cnt:
        out = ''
        try:
            chat_comp = openai.ChatCompletion.create(model=judge_model, messages=[{"role": "user", "content": qry}])
            out = chat_comp.choices[0].message.content[0]
        except Exception as e:
            print(f'发生错误:{e}\n')
        # print the chat completion
        # print('answer: ', chat_comp.choices[0].message.content)
        if out == '0' or out == '1':
            lbl = out
        else:
            cnt -= 1
    if not cnt:
        return 0
    return int(lbl)


json_list = []
process = 0
read_limit = 50000
with open(source, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx < read_limit:
            json_list.append(json.loads(line))
        else:
            break

# answer_list = []
# with open(real_answer_source, 'r', encoding='utf-8') as f:
#     for idx, line in enumerate(f):
#         if idx < read_limit:
#             answer_list.append(json.loads(line)['real_answer'])
#         else:
#             break
# assert len(answer_list) == len(json_list)

with open(out_file1, 'w', encoding='utf-8') as f1, open(out_file2, 'w', encoding='utf-8') as f2:
    for idx, item in enumerate(tqdm(json_list)):
        if idx < process:
            continue
        label = get_label(item['summary'], item['real_answer'])
        if label == 1:
            json.dump(item, f1, ensure_ascii=False)
            f1.write('\n')
        else:
            json.dump(item, f2, ensure_ascii=False)
            f2.write('\n')
