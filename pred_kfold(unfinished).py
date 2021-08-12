#!/usr/bin/env python
# coding: utf-8

# 导入transformers
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup


# 导入torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# 常用包
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

import argparse
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging

#
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
k = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

test = pd.read_csv('data/test.csv',encoding = "utf-8")
test['text']=test['text'].fillna('')

PRE_TRAINED_MODEL_NAME = 'guwenbert-base'#/home/sl/guwenbert-base
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


MAX_LEN = 126
class_names=[0,1]
BATCH_SIZE = 32

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


def create_pred_data_loader(df,tokenizer,max_len,batch_size):
    ds=TitleDataset(
        texts=df['text'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4 # windows多线程
    )


test_data_loader = create_data_loader(test, tokenizer, MAX_LEN, BATCH_SIZE)
pred_data_loader = create_pred_data_loader(pred, tokenizer, MAX_LEN, BATCH_SIZE)




class TitleClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TitleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        output = self.drop(pooled_output) # dropout
        return self.out(output)

# load the fine-tuned model


model_names = locals()
for i in range(k):
    model_names['model_%s' % i] = TitleClassifier(len(class_names))
    model_names['model_%s' % i] = torch.nn.DataParallel(model_names['model_%s' % i])   
    model_names['model_%s' % i] = model_names['model_%s' %i].cuda(device = device_ids[0])
    model_names['model_%s' % i].load_state_dict(torch.load('the_%s_fold_best_state.bin'%i))
#加载了model0,model1,...modelk-1

def get_probs(model, pred_data_loader):#输出值用于返回混淆矩阵
    model = model.eval()

    texts = []
    prediction_probs = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["texts"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)

            texts.extend(texts)
            prediction_probs.append(probs)

    prediction_probs = torch.stack(prediction_probs).cpu()
    prediction_probs = prediction_probs.tolist()
    
    return texts, prediction_probs #what shape??嵌套列表？


texts,prediction_probs = get_probs(model_0, pred_data_loader)

added_probs = [[0 for i in range(len(class_names))] for _ in range(len(texts))]
prediction_probs = []

for i in range(k):
    model_names['model_%s' % i]
    texts,prediction_probs = get_probs(model, pred_data_loader)
    for i in range(len(prediction_probs)):
        each_list = prediction_probs[i]
        for idx in range(k):
            added_probs[i][k] += each_list[k]
final_preds = []

for small_list in added_probs:
    pred_class_id = small_list.index(max(small_list))
    final_preds.append(pred_class_id)



def data_pred(pred_data,model):
    texts, id_ids, y_pred_ids = [], [], []
    for index, row in pred_data.iterrows():
        id = row['id']
        text = row['text']
        encoded_text = tokenizer.encode_plus(
              text,
              max_length=MAX_LEN,
              add_special_tokens=True,
              return_token_type_ids=False,
              pad_to_max_length=True,
              return_attention_mask=True,
              return_tensors='pt',
            )
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        id_ids.append(id)
        y_pred_ids.append(class_names[prediction])
        # print(f'Sample text: {text}')
#         print(f' label  : {class_names[prediction]}')
        texts.append(text)
    return texts, id_ids, y_pred_ids

# res_data = test
# texts, id_ids, y_pred_ids = data_pred(res_data,model)
# df_save = pd.DataFrame()
# df_save['id'] = id_ids
# df_save['label'] = y_pred_ids

# with open("res.txt","w",encoding="utf-8")as rf:
#     i = 0
#     while i < len(texts):
#         rf.write(texts[i] + "," +str(id_ids[i])+","+ str(y_pred_ids[i]) + "\n")
#         i += 1
# df_save.to_csv('result_bertlargewwm_epoch5.csv',index=False)
