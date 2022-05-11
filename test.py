# coding:UTF-8
# User  :yqhe
# Date  :2022/5/2 16:01
import tcn
import load_data
import torch
import numpy as np
import matplotlib.pyplot as plt

train_set, val_data, val_labels,seismic_data,labels=load_data.dataset()


cnn=tcn.TemporalConvNet(1,[8,16,32,64,128,128,64,32,16])
cnn.load_state_dict(torch.load('./model/parameters.pkl'))
cnn.cuda()

pred_inversion=[]
cnn.eval()
with torch.no_grad():
    for i in range(seismic_data.size(0)):
        print(i)
        input_x = seismic_data[i, :, :]
        input_x = input_x.unsqueeze(0)
        input_x = input_x.cuda()
        pred = cnn(input_x)
        pred = pred.transpose(1, 0)
        pred = pred.cpu().numpy()
        pred.astype(np.single())
        pred_inversion.append(pred)
        del pred

pred = np.array(pred_inversion)
pred = np.squeeze(pred)
pred = pred.transpose(1, 0)


# fig1=plt.figure(1)
# ax1 = fig1.add_subplot(111)
# plt.title('Pediction')
# plt.imshow(pred)
# ax1.set_aspect(0.6/ax1.get_data_ratio(), adjustable='box')
# #plt.colorbar(orientation='horizontal',fraction=0.05, pad=0.17)
# plt.show()


# fig3=plt.figure(3)
# ax3 = fig3.add_subplot(111)
# plt.title('error')
# plt.imshow(pred-labels)
# ax3.set_aspect(0.6/ax3.get_data_ratio(), adjustable='box')
# plt.colorbar(orientation='horizontal',fraction=0.05, pad=0.17)
# plt.show()



# fig2=plt.figure()
# ax2 = fig2.add_subplot(111)
# plt.title('labels')
# plt.imshow(labels)
# ax2.set_aspect(0.6/ax2.get_data_ratio(), adjustable='box')
# plt.colorbar(orientation='horizontal',fraction=0.05, pad=0.17)
# plt.show()



plt.figure()
plt.subplot(1, 3, 1)
plt.plot(pred[:, 12000], label='prediction')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(labels[:,12000], label='label')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(pred[:, 12000] - labels[:, 12000], label='error')
plt.legend()
plt.show()





