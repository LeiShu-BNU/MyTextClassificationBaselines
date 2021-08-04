#!/usr/bin/env python
# coding: utf-8
# ## 1 导入工具包

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
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(RANDOM_SEED)

# ## 2 加载数据

data=pd.read_csv('data/train0721.csv',encoding = "utf-8")

#data = data[:1000] #先toy model 测试一下能否跑通
print("data.shape",data.shape)

# ## 3 数据预处理
# 填充缺失值
print("data null nums")
print(data.shape[0]-data.count())


data['text'] = data['text'].fillna('')

#剔除标点符号,\xa0 空格
# def pretreatment(texts):
#     result_texts=[]
#     punctuation='。，？！：%&~、；&|,.?!:%&~()”“;""'#中文括号和书名号保留，因为是专名
#     for text in texts:
#         text= ''.join([c for c in text if c not in punctuation])
#         text= ''.join(text.split())   #\xa0
#         result_texts.append(text)
    
#     return result_texts

# ### 4 将文本映射为id表示
PRE_TRAINED_MODEL_NAME = 'sikubert'#/home/sl/guwenbert-base
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
texts = list(data['text'])
print(len(texts))
text_id=tokenizer(texts,
	padding=True,
	truncation=True,
	max_length=MAX_LEN,
	return_tensors='pt')
text_id['input_ids']

#text_id 长度为所有训练集长度，每一行是token id，长度为句子长度

X=text_id['input_ids']
y=torch.from_numpy(data['label'].values).float()

X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,shuffle=True,stratify=y,random_state=2020)
X_valid,X_test,y_valid,y_test=train_test_split(X_test,y_test,test_size=0.5,shuffle=True,stratify=y_test,random_state=2020)

# ## 5 构建数据集
# ### 5.1 自定义数据集
# create Tensor datasets
train_data = TensorDataset(X_train, y_train)
valid_data = TensorDataset(X_valid, y_valid)
test_data = TensorDataset(X_test,y_test)

# dataloaders
batch_size = 32

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,drop_last=True)
#len(train_loader)为最多能构建的BATCH数量
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,drop_last=True)
num_of_train_eps = len(train_loader)

# ##模型构建

# converting

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging
logging.set_verbosity_info()

PRE_TRAINED_MODEL_NAME = 'guwenbert-base'
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

#BERT+BILSTM:  https://blog.csdn.net/zhangtingduo/article/details/108474401
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


output_size = 2
hidden_dim = 384   #768/2
n_layers = 2
bidirectional = True  #这里为True，为双向LSTM
lr=learning_rate=2e-5
class_names=[0,1]
print_every = 100
EPOCHS = 5
clip=5 # gradient clipping

model = bert_lstm(hidden_dim, output_size,n_layers, bidirectional)
device_ids = [0,3]                                                                            
model = torch.nn.DataParallel(model,device_ids=[0,3])                                         
model = model.cuda(device = device_ids[0]) 

#print(model)

criterion = nn.CrossEntropyLoss()


#为不同的层设置不同的学习率和权重衰减，见https://github.com/hemingkx/CLUENER2020的项目
bert_optimizer = list(model.bert.named_parameters())
lstm_optimizer = list(model.lstm.named_parameters())
classifier_optimizer = list(model.fc.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.0001},
    {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
     'lr': learning_rate * 5, 'weight_decay': 0.0001},
    {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
     'lr': learning_rate * 5, 'weight_decay': 0.0},
    {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
     'lr': learning_rate * 5, 'weight_decay': 0.0001},
    {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
     'lr': learning_rate * 5, 'weight_decay': 0.0}
]#bilstm的学习率设置为BERT的5倍，若不设置，则用全局lr
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * EPOCHS
train_steps_per_epoch = len(train_loader) // batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=(EPOCHS // 10) * train_steps_per_epoch,
                                            num_training_steps=EPOCHS * train_steps_per_epoch)
# ## 7 模型训练
best_val_loss = float("inf")
# train for some number of epochs
for e in range(EPOCHS):
    model.train()
    # initialize hidden state because bilstm know nothing about the data at the beginning of each epoch
    h = model.init_hidden(batch_size)#2个torch.Size([4, 32, 384])构成的tuple
    counter = 0
 
    # batch loop
    for inputs, labels in train_loader:
        counter += 1
        
        if(USE_CUDA):
            inputs, labels = inputs.cuda(), labels.cuda().long()
        h = tuple([each.data for each in h])
        model.zero_grad()
        output= model(inputs, h)
        loss = criterion(output.squeeze(), labels.long())
        loss.backward()
        optimizer.step()
        # loss stats
        if counter % print_every == 0: #每训练print_every句话就eval一次
            model.eval()
            with torch.no_grad():
                val_h = model.init_hidden(batch_size)
                val_losses = []
                for inputs, labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])

                    if(USE_CUDA):
                        inputs, labels = inputs.cuda(), labels.cuda().long()

                    output = model(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.long())
                    val_losses.append(val_loss.item())

            
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            model.train()
    #at epoch end
    model.eval()
    with torch.no_grad():
        val_h = model.init_hidden(batch_size)
        val_losses = []
        for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])

            if(USE_CUDA):
                inputs, labels = inputs.cuda(), labels.cuda().long()

            output = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.long())
            val_losses.append(val_loss.item())
    if np.mean(val_losses) < best_val_loss:
        best_val_loss = np.mean(val_losses)
        torch.save(model.state_dict(), 'best_state.bin')
        print("best model saved!!!!")
    model.train()
torch.save(model.state_dict(), 'last_state.bin')



##测试
#
test_losses = [] # track loss
num_correct = 0

# init hidden state
h = model.init_hidden(batch_size)
 
model.eval()
# iterate over test data
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    if(USE_CUDA):
        inputs, labels = inputs.cuda(), labels.cuda()
    output = net(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
     
    output=torch.nn.Softmax(dim=1)(output)
    pred=torch.max(output, 1)[1]
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not USE_CUDA else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
 
# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))







