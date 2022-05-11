# coding:UTF-8
# User  :yqhe
# Date  :2022/3/26 10:21
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


def dataset():

    #load train data
    train_data=np.fromfile("./train/train_data.bin", dtype=np.float32)
    train_data = np.reshape(train_data, [2801, 1047], 'F')
    train_data = train_data.transpose(1, 0)
    train_data = torch.from_numpy(train_data)
    train_data = train_data.unsqueeze(1)

    train_labels=np.fromfile("./train/train_label.bin",dtype=np.float32)
    train_labels = np.reshape(train_labels, [2801, 1047], 'F')
    train_labels = train_labels.transpose(1, 0)
    train_labels = torch.from_numpy(train_labels)
    train_labels = train_labels.unsqueeze(1)

    # train_data=torch.from_numpy(train_data)
    # train_labels=torch.from_numpy(train_labels)

    '''load validation data'''
    val_data=np.fromfile("./train/val_data.bin",dtype=np.float32)
    val_data = np.reshape(val_data, [2801, 50], 'F')
    val_data = val_data.transpose(1, 0)
    val_data = torch.from_numpy(val_data)
    val_data = val_data.unsqueeze(1)


    val_labels=np.fromfile("./train/val_label.bin",dtype=np.float32)
    val_labels = np.reshape(val_labels, [2801, 50], 'F')
    val_labels = val_labels.transpose(1, 0)
    val_labels = torch.from_numpy(val_labels)
    #val_labels = val_labels.unsqueeze(1)
    #val_labels = np.reshape(val_labels, [4, 1, 2801])


    # seismic_data=np.fromfile("./train/seismic_data.bin",dtype=np.float32)
    # seismic_data=np.reshape(seismic_data,[2801,13601],'F')
    # seismic_data=seismic_data.transpose(1,0)
    # seismic_data=torch.from_numpy(seismic_data)
    # seismic_data=seismic_data.unsqueeze(1)
    #
    # labels=np.fromfile("./train/label.bin",dtype=np.float32)
    # labels=np.reshape(labels,[2801,13601],'F')



    '''generate train set'''
    train_set = Data.TensorDataset(train_data, train_labels)

    return train_set, val_data, val_labels #seismic_data,labels









