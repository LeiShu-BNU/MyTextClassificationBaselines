#!/usr/bin/env python
# coding: utf-8
# ## 1 导入工具包
#前序模型结构
from train_BERTBiLSTM import bert_lstm
# 导入transformers
import transformers
from transformers import BertModel, BertTokenizer, AdamW,  get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# 导入torch
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.nn.utils.rnn import pad_sequence

# 常用包
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
#from textwrap import wrap
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

MAX_LEN = 200
RANDOM_SEED = 42
batch_size = 64
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(RANDOM_SEED)

test = pd.read_csv('data/test_0721.csv',encoding = "utf-8")
test['text']=test['text'].fillna('')


# ### 4 将文本映射为id表示
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
bertlstm_model = model.cuda() 
bertlstm_model.load_state_dict(torch.load('best_state_bertlstm0812.bin'))


small_batches =[]
sent_size = len(sents)
step = 64
for i in range(0,sent_size,step):
    small_batches.append(sents[i:i+step])
    

def pred_lstm_sent_list(model, sents):

    model = model.eval()
    sent_id=tokenizer(sents,padding=True,truncation=True,max_length=MAX_LEN,return_tensors="pt")
  
    pred_h = model.init_hidden(len(sents))
    inputs = sent_id["input_ids"].cuda() 
    pred_h = tuple([each.data for each in pred_h])
    pred_list = []
    with torch.no_grad(): 
        output = model(inputs,pred_h) 
        output = torch.nn.Softmax(dim=1)(output)
        preds = torch.max(output,1)[1]  
        pred_list.extend(preds) 
    pred_list = torch.stack(pred_list).cpu()
    return pred_list

y_pred = []
for sents in small_batches:
    y_pred.extend(pred_lstm_sent_list(bertlstm_model, sents))
print(classification_report(y_test, y_pred, target_names = [str(label) for label in class_names]))
        










