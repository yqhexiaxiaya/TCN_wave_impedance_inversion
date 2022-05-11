# TCN_wave_impedance_inversion
TCN网络反演波阻抗(一维卷积)_pytorch_地震勘探_Marmousi2
## 1. 使用必备
### 本代码使用pytorch用户
### 请自行安装python库
## 2.说明
1.本代码核心TCN来源于[github开源资源](https://github.com/locuslab/TCN)\
2.由于上传文件大小限制，train文件夹里目前只有训练和验证数据集，所以代码目前只能训练，最终的测试成图还需要测试集的输入，测试集可以自行下载Marmousi2数据然后制作，也可以私信我2622895104@qq.com。
## 3.数据
-数据来源于Marmousi2([AGL Elastic Marmousi - SEG Wiki](https://wiki.seg.org/wiki/AGL_Elastic_Marmousi))\
-标签是由Marmousi2中的纵波速度和密度相乘得到\
-数据的制作流程：波阻抗(标签)->反射系数->和雷克子波褶积得到地震记录\
-源数据中有13601道，每道采样点2801个\
-训练集是从中抽出的1047道数据，采样间隔为13\
-验证集是从训练集中随机抽取50道\
-测试集就是所有13601道数据
## 4.参数
目前，我设置了epoch为200，已经够了\
也可自行更改
