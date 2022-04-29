# coding: utf-8
# ## 1 导入工具包
# torch: 1.7.1
# tensorflow: 4
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch import nn, optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.utils import logging

# ##模型构建

class DoubanClassifier(nn.Module):
    def __init__(self, n_classes):
        super(DoubanClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids,attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False)
        output = self.drop(pooled_output) # dropout
        return self.out(output)
class DoubanDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len
    def __len__(self): #用来返回dataset的长度，也就是batch数量
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
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'texts':text,
            'input_ids':encoding['input_ids'].flatten(),
            'labels':torch.tensor(label,dtype=torch.long),
            "attention_mask":encoding['attention_mask']
        }

def create_data_loader(df,tokenizer,max_len,batch_size) -> DataLoader:
    ds=DoubanDataset(
        texts=df['text'].values,
        labels=df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    ) #dict-like object
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4 # windows多线程
    )


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
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(input_ids=input_ids,
                attention_mask=attention_mask)
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
    model = model.eval() # 验证预测模式

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


def get_predictions(model, data_loader):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            contents = d["texts"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            if "labels" in d:
                targets = d["labels"].to(device)
                real_values.extend(targets)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)

            texts.extend(contents)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            

    predictions = torch.stack(predictions).cpu().numpy().tolist()
    prediction_probs = torch.stack(prediction_probs).cpu().tolist()
    real_values = torch.stack(real_values).cpu().tolist()
    return texts, predictions, prediction_probs, real_values

if __name__=="__main__":
    MAX_LEN = 200
    RANDOM_SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 16
    LR = 2e-5 
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda",torch.cuda.is_available())
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        torch.cuda.manual_seed(RANDOM_SEED)


    # 加载数据
    train_data = pd.read_csv('douban/data/train_all.txt', sep="\t")
    dev_data = pd.read_csv('douban/data/dev_all.txt', sep="\t")
    test_data = pd.read_csv('douban/data/test_all.txt', sep="\t")
    # train_data = train_data[:1000]  # 先toy model 测试一下能否跑通
    print("data.shape", train_data.shape)


    train_data.columns = ["text", "label"]
    dev_data.columns = ["text", "label"]
    test_data.columns = ["text", "label"]

    # # 数据预处理
    # # 填充缺失值
    # print("data null nums")
    # print(train_data.shape[0]-train_data.count())
    # train_data['text'] = train_data['text'].fillna('')


    # 将文本映射为id表示
    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # 构建数据集
    # 5.1 自定义数据集

    # dataloaders
    train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(dev_data, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)
    

    print("len(train_data_loader)",len(train_data_loader))

    
    logging.set_verbosity_info()
    class_names=[0, 1]

    model = DoubanClassifier(len(class_names))
    #device_ids = [2, 3]
    #model = torch.nn.DataParallel(model,device_ids=device_ids)                                         
    model = model.cuda() #device = device_ids[0]


    # ##模型训练


    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    print("num of batches:", len(train_data_loader))
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
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
            len(train_data)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(dev_data)
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_bert_state.bin') #model.module.state_dict()
            torch.save(model, 'best_model_bert.bin')
            best_accuracy = val_acc
            print ("best model saved!")


    # test using trained best model

    bert_model = DoubanClassifier(len(class_names))                        
    bert_model = bert_model.cuda()
    bert_model.load_state_dict(torch.load('best_model_bert_state.bin'))

    test_acc, _ = eval_model(
    bert_model,
    test_data_loader,
    loss_fn,
    device,
    len(test_data)
    )

    print(test_acc.item())

    y_texts, y_pred, y_pred_probs, y_real = get_predictions(
    model,
    test_data_loader
    )

    # with open("predictions.txt","a",encoding="utf-8")as rf:
    #     i = 0
    #     while i < len(y_texts):
    #         if str(y_pred[i])!= str(y_real[i]):
    #             rf.write(y_texts[i] + "," +str(y_pred[i])+","+ str(y_pred_probs[i])+","+ str(y_real[i]) + "\n")
    #         i += 1
    #     print("proceed",i,"test data")




