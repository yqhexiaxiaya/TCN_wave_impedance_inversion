# coding:UTF-8
# User  :yqhe
# Date  :2022/3/29 19:16
import math

import tcn
import torch
import load_data
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt


Epoch=200
Batch_size=5
Train_rate=0.001


#load data
train_set, val_data, val_labels=load_data.dataset()
val_data=val_data.cuda()
val_labels=val_labels.cuda()


train_loader=Data.DataLoader(dataset=train_set,batch_size=Batch_size,shuffle=True)

#net
cnn=tcn.TemporalConvNet(1,[8,16,32,64,128,128,64,32,16])
cnn.cuda()

#train
optimizer=torch.optim.Adam(cnn.parameters(),lr=Train_rate)
loss_fn=nn.MSELoss()
scheduler = ExponentialLR(optimizer, gamma=0.9)
flag=0

for epoch in range(Epoch):
    for step,train_set in enumerate(train_loader):  #从Data.DataLoader中迭代取出数据集和索引
        train_data,train_labels=train_set
        train_labels = torch.squeeze(train_labels,1)
        train_data=train_data.cuda()
        train_labels=train_labels.cuda()
        train_output=cnn(train_data)
        loss=loss_fn(train_output,train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            val_output=cnn(val_data)
            mseloss = loss_fn(val_output, val_labels)
            # print('Epoch:', epoch + 1, '|mesloss:%.4f' % mseloss)
            val_accuracy=1-mseloss/torch.var(val_output)
            print('Epoch:', epoch + 1, '|val_accuracy:%.4f' % val_accuracy)
            # if mseloss<0.010:
            if val_accuracy<1.0000 and val_accuracy>0.9980:
                flag=1
                break
    if epoch>50:
        scheduler.step()
    if flag==1:
        break

'''保存参数'''
torch.save(cnn.state_dict(),'./model/parameters.pkl')  #state_dict就是一个简单的Python dictionary，其功能是将每层与层的参数张量之间一一映射。

'''
#test
test_output=cnn(test_data)
pred_y=torch.max(test_output,1)[1].cpu().numpy()
acc=float((pred_y==test_labels).sum ())/float(np.size(test_labels,0))
print('accuracy:%.4f'%acc)
'''











