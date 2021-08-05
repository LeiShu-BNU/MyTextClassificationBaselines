#!/usr/bin/env python
# coding: utf-8

# ## 1 导入工具包

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
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
#=============================================================
#variables
MAX_LEN = 200
BATCH_SIZE = 32
EPOCHS = 5 # 训练轮数
LR = 2e-5
class_names=[0,1]
#=============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# ## 2 加载数据===============================================

train=pd.read_csv('data/train_0712.csv',encoding = "utf-8")


print("train.shape",train.shape)
print("train null nums")
print(train.shape[0]-train.count())

# ## 3 数据预处理
# 填充缺失值
train['text'] = train['text'].fillna('')

# ### 4 将文本映射为id表示
# 使用`BertTokenizer`进行分词
PRE_TRAINED_MODEL_NAME = 'sikubert'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


#手工特征在自定义数据集的return里面返回

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
        sent_length = len(text)
        commas = len(re.findall("[。，？！.,]",text))
        
        encoding=self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_tensors='pt',
        )
        
        return {
            'texts':text,
            'sent_len': sent_length,
            'num_of_commas':commas,
            'input_ids':encoding['input_ids'].flatten(),
            'labels':torch.tensor(label,dtype=torch.long)
        }
        
# ### 5.2 划分数据集并创建生成器


df_train, df_test = train_test_split(train, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
#df_train.shape, df_val.shape, df_test.shape
#
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


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)

# ##模型构建

# converting
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging
logging.set_verbosity_info()

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


class TitleClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TitleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size+2, n_classes) #加上后面拼接的手工特征的长度
    def forward(self, input_ids, sent_len, num_of_commas):

        _,pooled_output = self.bert(
            input_ids,
            return_dict = False
        )#pooled_output.size:(bs,768)
        batch_len = len(input_ids) #不一定都等于BS，最后一个batch长度不满BS
        sent_len = torch.reshape(sent_len, (batch_len,1))
        num_of_commas = torch.reshape(num_of_commas, (batch_len,1))
        
        drop = self.drop(pooled_output) # dropout
        concat_features = torch.cat([drop,sent_len,num_of_commas],dim=1)
        return self.linear(concat_features)

#实例化模型

model = TitleClassifier(len(class_names))
model = model.to(device)

# ## 7 模型训练

optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=total_steps//10,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)



def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        targets = d["labels"].to(device)
        sent_len = d['sent_len'].to(device)
		num_of_commas = d['num_of_commas'].to(device)

        outputs = model(
            input_ids,
            sent_len, 
            num_of_commas
        )
        _, preds = torch.max(outputs, dim=1) #OUTPUTS size:(bs,2)
        loss = loss_fn(outputs, targets)#标签是 one-hot 编码的形式，即只有一个位置是 1，其他位置都是 0
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        #先回传再截断（防止梯度爆炸和消失）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval() # 验证预测模式

    losses = []
    correct_predictions = 0
    pred_all = []
    labels_all = []
    with torch.no_grad():
        for d in data_loader:
	        input_ids = d["input_ids"].to(device)
	        targets = d["labels"].to(device)
	        labels_all.extend(targets)
	        sent_len = d['sent_len'].to(device)
			num_of_commas = d['num_of_commas'].to(device)

	        outputs = model(
	            input_ids,
	            sent_len, 
	            num_of_commas
	        )
            _, preds = torch.max(outputs, dim=1)
            pred_all.extend(preds)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    pred_all = torch.stack(pred_all).cpu().numpy()
    labels_all = torch.stack(labels_all).cpu().numpy()
    f1 = f1_score(labels_all,pred_all)
    AUC = roc_auc_score(labels_all,pred_all)
    return correct_predictions.double() / n_examples, np.mean(losses), f1,AUC



history = defaultdict(list) # 记录loss和acc
best_auc = 0

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
        len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss , f1,AUC= eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc} f1 {f1} AUC {AUC}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(f1)
    history['val_AUC'].append(AUC)
    if AUC > best_auc:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_auc = AUC
        print("best model saved!!!!")



test_acc, test_loss , f1, AUC= eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

print("test_acc:",test_acc, "test_loss:",test_loss ,"f1:", f1,"test AUC:", AUC)



def get_predictions(model, data_loader):#输出值用于返回混淆矩阵
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["texts"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu().numpy()
    predictions = predictions.tolist()
    prediction_probs = torch.stack(prediction_probs).cpu()
    prediction_probs = prediction_probs.tolist()
    real_values = torch.stack(real_values).cpu()
    real_values = real_values.tolist()
    right = list(map(lambda x,y : x==y ,predictions,real_values))
    
    return texts, predictions, prediction_probs, real_values


y_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)

print(classification_report(y_test, y_pred, target_names=[str(label) for label in class_names]))
