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
device_ids=[0,1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

test = pd.read_csv('data/test_0721.csv',encoding = "utf-8")
test['text']=test['text'].fillna('')

PRE_TRAINED_MODEL_NAME = 'sikubert'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


MAX_LEN = 200
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
            padding='max_length', 
            return_attention_mask=True,
            truncation=True,
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
#pred_data_loader = create_pred_data_loader(pred, tokenizer, MAX_LEN, BATCH_SIZE)


#BERT finetune模块==========================================================================
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
#BERT+LSTM模块===================================================================================
class bert_lstm(nn.Module):
    def __init__(self, hidden_dim,output_size,n_layers,bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()
 
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        #Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert=BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
          
        #self.sig = nn.Sigmoid()
 
    def forward(self, x, hidden):
        batch_size = x.size(0)
        #生成bert字向量
        x=self.bert(x)[0]     #bert 字向量
        
        # lstm_out
        #x = x.float()
        lstm_out, (hidden_last,cn_last) = self.lstm(x, hidden)
        #print(lstm_out.shape)   #[32,100,768]
        #print(hidden_last.shape)   #[4, 32, 384]
        #print(cn_last.shape)    #[4, 32, 384]
        
        #修改 双向的需要单独处理
        if self.bidirectional:
            #正向最后一层，最后一个时刻
            hidden_last_L=hidden_last[-2]
            #print(hidden_last_L.shape)  #[32, 384]
            #反向最后一层，最后一个时刻
            hidden_last_R=hidden_last[-1]
            #print(hidden_last_R.shape)   #[32, 384]
            #进行拼接
            hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
            #print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out=hidden_last[-1]   #[32, 384]
            
            
        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        #print(out.shape)    #[32,768]
        out = self.fc(out) #全连接就是线性
        
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        number = 1
        if self.bidirectional:
            number = 2
        
        if (USE_CUDA):
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda()
                     )
        else:
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float()
                     )
        
        return hidden

PRE_TRAINED_MODEL_NAME = 'sikubert'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
sents = list(test['text'])
y_test = test['labels'].to_list()

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
output_size = 2
hidden_dim = 384   #768/2
n_layers = 2
bidirectional = True  #这里为True，为双向LSTM
lr=learning_rate=2e-5
class_names=[0,1]


bertlstm_model = bert_lstm(hidden_dim, output_size,n_layers, bidirectional)                        
bertlstm_model = bertlstm_model.cuda() 
bertlstm_model.load_state_dict(torch.load('best_state_bertlstm0812.bin'))


small_batches =[]
sent_size = len(sents)
step = 64
for i in range(0,sent_size,step):
    small_batches.append(sents[i:i+step])
    

def get_lstm_probs(model, sents):
    model = model.eval()
    prediction_probs = []
    sent_id=tokenizer(sents,padding=True,truncation=True,max_length=MAX_LEN,return_tensors="pt")
  
    pred_h = model.init_hidden(len(sents))
    inputs = sent_id["input_ids"].cuda() 
    pred_h = tuple([each.data for each in pred_h])
    prob_list = []
    with torch.no_grad(): 
        output = model(inputs,pred_h) 
        probs = F.softmax(output, dim=1)
    prediction_probs.extend([line for line in probs.cpu().numpy()])
    return prediction_probs

def get_bert_probs(model, data_loader):#输出值用于返回混淆矩阵
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
            prediction_probs.extend([line for line in probs.cpu().numpy()])
    return texts, prediction_probs 


loaded_models = []
model_names = locals()
num_bert_models = 5
for i in range(num_bert_models):
    model_names['model_%s' % i] = TitleClassifier(len(class_names))
    model_names['model_%s' % i] = torch.nn.DataParallel(model_names['model_%s' % i])   
    model_names['model_%s' % i] = model_names['model_%s' %i].cuda(device = device_ids[0])
    model_names['model_%s' % i].load_state_dict(torch.load('5fold/the_%s_fold_best_state.bin'%i))
    loaded_models.append(model_names['model_%s' % i])
#加载了model0,model1,...modelk-1
loaded_models.append(bertlstm_model)

all_test_size = len(test)
avg_prob = [np.zeros(2, dtype = float) for i in range(all_test_size)]
k = len(loaded_models)
for model in loaded_models:
    if "lstm" not in model:
        texts,prediction_probs = get_probs(model, test_data_loader)
        for i in range(all_test_size):
            avg_prob[i] += prediction_probs[i]/k
    else:
        prediction_probs = []
        for sents in small_batches:
            prediction_probs.extend(pred_lstm_sent_list(bertlstm_model, sents))
        for i in range(all_test_size):
            avg_prob[i] += prediction_probs[i]/k


      
y_pred = [int(round(i[1])) for i in avg_prob]
print(y_pred)
print(classification_report(y_test, y_pred, target_names = [str(label) for label in class_names]))                                                                                                                

y_test = test["labels"]
id_ids = test["ids"]
df_save = pd.DataFrame()
df_save['id'] = id_ids
df_save['label'] = y_pred
df_save.to_csv('result_ensemble.csv',index=False)
# with open("res.txt","w",encoding="utf-8")as rf:
#     i = 0
#     while i < len(texts):
#         rf.write(texts[i] + "," +str(id_ids[i])+","+ str(y_pred_ids[i]) + "\n")
#         i += 1
#
