#!/usr/bin/env python
# coding: utf-8
#k-fold cross validation 训练时，训练k个模型，预测时候取平均

# ## 1 导入工具包

# 导入transformers
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup


# 导入torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F


# 常用包
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import argparse
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
LR=2e-5
MAX_LEN = 126
device_ids=[0,3]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
BATCH_SIZE = 32
class_names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]


# ## 2 加载数据

train=pd.read_csv('data/train.csv',encoding = "utf-8")

print("train.shape",train.shape)
print("train null nums")
print(train.shape[0]-train.count())


# ## 3 数据预处理
# 填充缺失值
train['text'] = train['text'].fillna('')

# ### 4 将文本映射为id表示
PRE_TRAINED_MODEL_NAME = 'guwenbert-base'#/home/sl/guwenbert-base
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# ## 5 构建数据集
# ### 5.1 自定义数据集

class TitleDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,item):
        """
        item 为数据索引，迭代取第item条数据
        """
        text=str(self.texts[item])
        label=self.labels[item]
        
        encoding=self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'texts':text,
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'labels':torch.tensor(label,dtype=torch.long)
        }
        


# ### 5.2 划分数据集并创建生成器
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=1)
split = kfold.split(train) #长度为k,用for循环，每一个元素里面是train,test


def create_data_loader(df,tokenizer,max_len,batch_size):
    ds=TitleDataset(
        texts=df['text'].values,
        labels=df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4 # windows多线程
    )


class TitleClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TitleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        output = self.drop(pooled_output) # dropout
        return self.out(output)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval() # 不启用 BatchNormalization 和 Dropout
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# def get_predictions(model, data_loader):
#     model = model.eval()

#     texts = []
#     predictions = []
#     prediction_probs = []
#     real_values = []

#     with torch.no_grad():
#         for d in data_loader:
#             texts = d["texts"]
#             input_ids = d["input_ids"].to(device)
#             attention_mask = d["attention_mask"].to(device)
#             targets = d["labels"].to(device)

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
#             _, preds = torch.max(outputs, dim=1)

#             probs = F.softmax(outputs, dim=1)

#             texts.extend(texts)
#             predictions.extend(preds)
#             prediction_probs.extend(probs)
#             real_values.extend(targets)

#     predictions = torch.stack(predictions).cpu().numpy()
#     predictions = predictions.tolist()
#     prediction_probs = torch.stack(prediction_probs).cpu()
#     prediction_probs = prediction_probs.tolist()
#     real_values = torch.stack(real_values).cpu()
#     real_values = real_values.tolist()
#     right = list(map(lambda x,y : x==y ,predictions,real_values))
    
#     return texts, predictions, prediction_probs, real_values




for ks, (train_ids, test_ids) in enumerate(split):  #每一折交叉验证
    ks = ks+1 #让fold索引从1开始
    train_data_loader = create_data_loader(train.loc[train_ids], tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(train.loc[test_ids], tokenizer, MAX_LEN, BATCH_SIZE)

    model = TitleClassifier(len(class_names))
    model = torch.nn.DataParallel(model,device_ids=device_ids)
    model = model.cuda(device = device_ids[0])

    # ## 7 模型训练

    EPOCHS = 5 # 训练轮数

    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=total_steps//10,
      num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)


    history = defaultdict(list) # 记录loss和acc
    best_accuracy = 0

    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_ids)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        test_acc, test_loss = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(test_ids)
        )

        
        print("the %s fold test accuracy:"%str(ks),test_acc)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        

        if test_acc > best_accuracy:
            torch.save(model.state_dict(), 'the_%s_fold_best_state.bin'%str(ks))
                torch.save(model, 'the_%s_fold_best_model.bin'%str(ks))
            best_accuracy = test_acc
            print("best model saved!!!!")
