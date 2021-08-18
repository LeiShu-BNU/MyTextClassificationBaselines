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
        










