# %%
from sklearn.metrics import precision_recall_fscore_support

# %%
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW
from collections.abc import Iterable
from torchtext import data
from transformers import BertModel
import torch.nn as nn
from transformers.optimization import *
import torch.optim as optim
import math
from torch.nn.utils.rnn import pad_sequence

# %%

from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# %%
import logging
import os
import sys
import argparse
import random

import numpy as np
import tqdm
import pickle
import pandas as pd
from matplotlib import pyplot as plt

# %%
df_train = pd.read_csv("snli_1.0/snli_1.0_train.txt", sep="\t" )
df_dev = pd.read_csv("snli_1.0/snli_1.0_dev.txt", sep="\t")
df_test = pd.read_csv("snli_1.0/snli_1.0_test.txt", sep="\t")

# %%
df_train[:10]

# %%
df_train.drop(df_train.columns.difference(['gold_label','sentence1', 'sentence2']), axis=1, inplace=True) ## to drop the columns
df_test.drop(df_test.columns.difference(['gold_label','sentence1', 'sentence2']), axis=1, inplace=True) ## to drop the columns
df_dev.drop(df_dev.columns.difference(['gold_label','sentence1', 'sentence2']), axis=1, inplace=True) ## to drop the columns

# %%
df_train[:10]

# %%
df_train = df_train[df_train['gold_label'] != '-']
df_dev = df_dev[df_dev['gold_label'] != '-']
df_test = df_test[df_test['gold_label'] != '-']

# %%
df_train


# %%
# Analyzing the data
df_train.groupby('gold_label').count()

# %%
df_test.groupby('gold_label').count()

# %%
df_dev.groupby('gold_label').count()

# %%
#  dropping the rows from the data with NaN values from df_train

df_train = df_train.dropna()


# %%
print(len(df_train), len(df_test), len(df_dev))

# %%
for i, row in df_train.iterrows():
    df_train.at[i, 'input'] = " <cls> " + str(row[1]) + " <sep> " + str(row[2]) + " <cls> "
    if row[0] == 'neutral':
        df_train.at[i, 'label'] = int(0)
    elif row[0] == 'entailment':
        df_train.at[i, 'label'] = int(1)
    elif row[0] == 'contradiction':
        df_train.at[i, 'label'] = int(2)
    else:
        print(str(row[0]))
        break

    if(i % 100000) == 0 and i:
        print("Completed: %s" %(i))

# %%
for i, row in df_dev.iterrows():
    df_dev.at[i, 'input'] = " <cls> " + str(row[1]) + " <sep> " + str(row[2]) + " <cls> "

    if row[0] == 'neutral':
        df_dev.at[i, 'label'] = int(0)
    elif row[0] == 'entailment':
        df_dev.at[i, 'label'] = int(1)
    elif row[0] == 'contradiction':
        df_dev.at[i, 'label'] = int(2)
    else:
        print(row[0])
        break

    if(i % 100000) == 0 and i:
        print("Completed: %s" %(i))

# %%
for i, row in df_test.iterrows():
    df_test.at[i, 'input'] = " <cls> " + str(row[1]) + " <sep> " + str(row[2]) + " <cls> "

    if row[0] == 'neutral':
        df_test.at[i, 'label'] = int(0)
    elif row[0] == 'entailment':
        df_test.at[i, 'label'] = int(1)
    elif row[0] == 'contradiction':
        df_test.at[i, 'label'] = int(2)
    else:
        print(row[0])
        break

    if(i % 100000) == 0 and i:
        print("Completed: %s" %(i))

# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# %%

df_train.head()

# %%
##Convert the dataset into a list

##train
inp_train = df_train['input'].tolist()
labels_train = df_train['label'].astype(int).tolist()

##test
inp_test = df_test['input'].tolist()
labels_test = df_test['label'].astype(int).tolist()

##dev
inp_dev = df_dev['input'].tolist()
labels_dev = df_dev['label'].astype(int).tolist()

# %%
inp_train[:10]

# %%
labels_train[:10]

# %%
lr = 2e-5
num_epochs = 10
MAX_LEN = 128
batch_size = 128
dataset = 'snli'


# In[4]:


ending_path = ('%s_%s_%s_%s' %(dataset, MAX_LEN, batch_size, str(lr).replace("-", "")))


# %%
ending_path

# %%
def create_attention_masks(input_ids):
    """Creates attention masks from input IDs."""
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

# %% [markdown]
# # TRAIN

# %%
tokenized_texts_train = [tokenizer.tokenize(inp) for inp in inp_train]
input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_train]


# %%

input_ids_train = pad_sequence([torch.tensor(x) for x in input_ids_train], batch_first=True, padding_value=0)
attention_masks_train = create_attention_masks(input_ids_train)

# %%
print(len(input_ids_train), len(attention_masks_train))

# %%
train_inputs = torch.tensor(input_ids_train)
train_labels = torch.tensor(labels_train)
train_masks = torch.tensor(attention_masks_train)

# %%
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# %% [markdown]
# # DEV

# %%
tokenized_texts_dev = [tokenizer.tokenize(inp) for inp in inp_dev]
input_ids_dev = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_dev]

# %%

input_ids_dev = pad_sequence([torch.tensor(x) for x in input_ids_dev], batch_first=True, padding_value=0)
attention_masks_dev = create_attention_masks(input_ids_dev)

# %%
print(len(input_ids_dev), len(attention_masks_dev))

# %%
dev_inputs = torch.tensor(input_ids_dev)
dev_labels = torch.tensor(labels_dev)
dev_masks = torch.tensor(attention_masks_dev)

# %%
dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)
dev_sampler = RandomSampler(dev_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)


# %% [markdown]
# # TEST

# %%
tokenized_texts_test = [tokenizer.tokenize(inp) for inp in inp_test]
input_ids_test = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_test]

# %%

input_ids_test = pad_sequence([torch.tensor(x) for x in input_ids_test], batch_first=True, padding_value=0)
attention_masks_test = create_attention_masks(input_ids_test)

# %%
print(len(input_ids_test), len(attention_masks_test))

# %%
test_inputs = torch.tensor(input_ids_test)
test_labels = torch.tensor(labels_test)
test_masks = torch.tensor(attention_masks_test)

# %%
test_data = TensorDataset(test_inputs, test_masks, test_labels)
#test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# %% [markdown]
# ## Model

# %%
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model.to(device)

# %%
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# %%
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

# %%
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    labels_flat = labels_flat.cpu().detach().numpy()
    return np.sum(pred_flat == labels_flat), pred_flat

# %%
def train(i):
    model.train()
    total_loss = 0.0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    train_len = 0
    f_acc = 0

    ## adaptive lr
    optimizer.param_groups[0]['lr'] *= (0.1)**(1/40.)


    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        if b_labels.size(0) == 1:
            continue
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        pred = outputs[1].detach().cpu().numpy()
        batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
        f_acc += batch_f_acc
        loss = outputs[0]
        loss.sum().backward()
        optimizer.step()


        labels_flat = b_labels.flatten().cpu().detach().numpy()
        total_actual_label = np.concatenate((total_actual_label, labels_flat))
        total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

        total_loss += outputs[0].sum()
        train_len += b_input_ids.size(0)

        if step%100 == 0 and step:
            precision, recall, f1_measure, _ =             precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
            print("Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                         (train_len*100.0/train_inputs.size(0), i, step,                            total_loss/train_len, f_acc*100.0/train_len,                            precision*100., recall*100., f1_measure*100.))

    #if torch.cuda.device_count() > 1:
        #p = 100
       # path = save_model_path + '/e_' + str(i) + "_" + str(p) + ".ckpt"
        #torch.save(model.module.state_dict(), path)
    #else:
        #torch.save(model.state_dict(), path)
    precision, recall, f1_measure, _ =     precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
    print("Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                 (train_len*100.0/train_inputs.size(0), i, step,                    total_loss/train_len, f_acc*100.0/train_len,                    precision*100., recall*100., f1_measure*100.))
    return total_loss/train_len

# %%


# %%
def dev(i):
    model.eval()
    val_len = 0
    total_loss = 0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    f_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(dev_dataloader):
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            if b_labels.size(0) == 1:
                continue

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            pred = outputs[1].detach().cpu().numpy()
            batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
            f_acc += batch_f_acc

            labels_flat = b_labels.flatten().cpu().detach().numpy()
            total_actual_label = np.concatenate((total_actual_label, labels_flat))
            total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

            val_len += b_input_ids.size(0)
            total_loss += outputs[0].sum()

            if step%100 == 0 and step:
                precision, recall, f1_measure, _ =                 precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
                print("Eval: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                             (val_len*100.0/dev_inputs.size(0), i, step,                              total_loss/val_len, f_acc*100.0/val_len,                                precision*100., recall*100., f1_measure*100.))

        precision, recall, f1_measure, _ =         precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
        print("Validation: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                     (val_len*100.0/dev_inputs.size(0), i, step,                      total_loss/val_len, f_acc*100.0/val_len,                        precision*100., recall*100., f1_measure*100.))
    return total_loss/val_len


# %%
def test(i):
    model.eval()
    val_len = 0
    total_loss = 0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    f_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            if b_labels.size(0) == 1:
                continue

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            pred = outputs[1].detach().cpu().numpy()
            batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
            f_acc += batch_f_acc

            labels_flat = b_labels.flatten().cpu().detach().numpy()
            total_actual_label = np.concatenate((total_actual_label, labels_flat))
            total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

            val_len += b_input_ids.size(0)
            total_loss += outputs[0].sum()

            if step%100 == 0 and step:
                precision, recall, f1_measure, _ =                 precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
                print("Eval: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                             (val_len*100.0/test_inputs.size(0), i, step,                              total_loss/val_len, f_acc*100.0/val_len,                                precision*100., recall*100., f1_measure*100.))

        precision, recall, f1_measure, _ =         precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
        print("Test: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %                     (val_len*100.0/test_inputs.size(0), i, step,                      total_loss/val_len, f_acc*100.0/val_len,                        precision*100., recall*100., f1_measure*100.))
        return total_actual_label, total_predicted_label

# %%
train_loss = []
val_loss = []
import pickle as pkl
for i in range(num_epochs):
    train_loss.append(train(i))
    val_loss.append(dev(i))
    actual, predicted = test(0)

    pkl.dump(actual, open('actual', 'wb'))

    pkl.dump(predicted, open('predicted', 'wb'))

pkl.dump(train_loss, open('train_loss.pkl', 'wb'))

pkl.dump(val_loss, open('val_loss.pkl', 'wb'))






# %%

# save the model
torch.save(model.state_dict(), 'model_state_dict_10Epochs.pth')
torch.save(model, 'model_10Epochs.pth')

