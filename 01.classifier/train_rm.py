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
        prompt_answer = self.data_js[idx]['prompt_answer']
        label = self.data_js[idx]['label']
        label = tokenizer.encode('True')[-1] if label == 1 else tokenizer.encode('False')[-1]

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
            'label': torch.tensor(label)
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


# 加载训练集、验证集和测试集数据
train_js = 'data/rm_train.json'
test_js = 'data/rm_test.json'
val_js = 'data/rm_valid.json'


def read_json(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


train_json = read_json(train_js)  # 以CSV文件为例，具体根据实际情况选择数据加载方式
val_json = read_json(val_js)
test_json = read_json(test_js)

# 创建自定义数据集
train_dataset = MyDataset(train_json, tokenizer)
val_dataset = MyDataset(val_json, tokenizer)
test_dataset = MyDataset(test_json, tokenizer)

# # 创建数据加载器
batch_size = 3  # 设置批大小
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# # 设置设备和模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, '\n')
hidden_size = base_model.config.hidden_size
filter_model = ChatGLM_Filter(base_model, hidden_size, 1)
filter_model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(filter_model.parameters(), lr=3e-6, eps=1e-4)
num_epochs = 2
# 训练和验证循环
best_val_loss = 1000000
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    print(f"{epoch}/{num_epochs} training")
    # 训练
    filter_model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # while True:
        #     try:
        #         labels = [float(label) for label in batch['label']]
        #         break
        #     except Exception as e:
        #         print(e, '\n')
        labels = batch['label'].to(device)
        # labels = torch.tensor(labels).to(device).bfloat16()
        # print(labels.shape)

        optimizer.zero_grad()
        outputs = filter_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.float(), labels).half()
        print(f'损失:{loss}\n')
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # 验证
    filter_model.eval()
    val_loss = 0.0
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # while True:
            #     try:
            #         labels = [float(label) for label in batch['label']]
            #         break
            #     except Exception as e:
            #         print(e, '\n')
            labels = batch['label'].to(device)
            # labels = torch.tensor(labels).to(device).bfloat16()
            outputs = filter_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.float(), labels).half()
            val_loss += loss.item()
            val_labels.extend(labels.tolist())

    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ")

    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(filter_model.state_dict(), "ckpt/rm_best_checkpoint_4.pt")

print("训练完成！")
#
# import matplotlib.pyplot as plt
#
# epochs = range(1, num_epochs + 1)
# plt.plot(epochs, train_losses, 'r', label='Training Loss')
# plt.plot(epochs, val_losses, 'b', label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('loss.png')
# plt.show()

# 加载最佳模型进行推断
best_model = ChatGLM_Filter(base_model, hidden_size, 1)
best_model.load_state_dict(torch.load("ckpt/rm_best_checkpoint_4.pt"))
best_model.to(device)
best_model.eval()

# 进行推断
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # while True:
        #     try:
        #         labels = [float(label) for label in batch['label']]
        #         break
        #     except Exception as e:
        #         print(e, '\n')
        labels = batch['label'].to(device)
        outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.softmax(outputs, dim=1)
        outputs_1 = outputs[:, key_token1]
        outputs_2 = outputs[:, key_token2]
        preds = [key_token1 if outputs_1[i] > outputs_2[i] else key_token2 for i in range(len(outputs_1))]
        test_preds.extend(preds)
        test_labels.extend(labels.tolist())
    print("推断结果：")
    for i in range(len(test_preds)):
        print(f"样本{i + 1}: 预测{test_preds[i]}，实际{test_labels[i]}")

cnt = 0
for i in range(len(test_preds)):
    if test_preds[i] == test_labels[i]:
        cnt += 1
test_acc = cnt / len(test_preds)
print(f"test acc: {test_acc:.4f}")
