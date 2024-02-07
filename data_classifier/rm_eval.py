import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from modelscope import AutoTokenizer, AutoModel, snapshot_download

model_dir = snapshot_download("ZhipuAI/chatglm3-6b-base", revision="v1.0.0")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
base_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()

max_length = 1024

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data_js, tokenizer):
        self.data_js = data_js
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        prompt = self.data_js[idx]['content']
        answer = self.data_js[idx]['summary']
        prompt_answer = prompt + '【答案】' + answer

        encoded_pair = self.tokenizer.encode_plus(
            prompt_answer,
            padding='max_length',
            max_length=max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].squeeze(),
            'attention_mask': encoded_pair['attention_mask'].squeeze(),
        }


key_token1 = 'True'
key_token1 = tokenizer.encode(key_token1)[-1]
print(key_token1, '\n')
key_token2 = 'False'
key_token2 = tokenizer.encode(key_token2)[-1]
print(key_token2, '\n')


class ChatGLM_Filter(nn.Module):
    def __init__(self, base, hidden_size, num_classes=1):
        super(ChatGLM_Filter, self).__init__()
        self.base_model = base
        # self.LN = nn.Linear(65204, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]
        # print(outputs)
        # trans_output = self.LN(outputs)
        value = outputs
        return value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, '\n')
hidden_size = base_model.config.hidden_size
filter_model = ChatGLM_Filter(base_model, hidden_size, 1)
filter_model.load_state_dict(torch.load("ckpt/rm_best_checkpoint_4.pt"))
filter_model.to(device)


def read_json(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


target_file = 'api_gen_q_enhanced/gpt4_revised_left_real.json'
data_js=read_json(target_file)
batch_size = 3
dataset = MyDataset(data_js, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size)

preds = []
for batch in tqdm(dataloader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    values = filter_model(input_ids, attention_mask)
    outputs = torch.softmax(values, dim=1)
    outputs_1 = outputs[:, key_token1]
    outputs_2 = outputs[:, key_token2]
    pred = outputs_1 - outputs_2
    preds.extend(pred.tolist())

df = pd.DataFrame(preds, columns=['preds'])
df.to_csv('api_gen_preds/gpt4_revised_q.csv')
