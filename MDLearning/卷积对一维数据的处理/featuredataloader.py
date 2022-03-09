import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import glob
import os
import random
from torchvision import  datasets
from torchvision import transforms     #变换器
import pandas as pd


# 创建子类
class featureDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, root, srcfilename, filename,mode):
        super(featureDataset, self).__init__()
        self.root=root
        self.srcfilename=srcfilename
        self.filename=filename
        self.datas,self.labels = self.load_csv(self.root,self.srcfilename,self.filename)

        # 裁剪,按照train,test对数据集裁剪,6:2:2
        if mode == 'train':
            self.datas = self.datas[0:int(0.6 * len(self.datas))]
            self.labels = self.labels[0:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.datas = self.datas[int(0.6 * len(self.datas)):int(0.8 * len(self.datas))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.datas = self.datas[int(0.8 * len(self.datas)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    # 返回数据集大小
    def __len__(self):
        return len(self.datas)

    # 得到数据内容和标签
    def __getitem__(self, index):
        # 这里实现data一维到二维tensor的转换
        li = self.datas[index]
        li = list(li)
        li = list(map(float, li))
        data = torch.Tensor(li)
        data = data.view(7, 7)
        data = data.unsqueeze(0)  # 加一个channels维度
        data = data.unsqueeze(0)
        data = data.view(1,7, 7)
        data = torch.Tensor(data)
        # label = int(self.labels[index])
        label = torch.tensor(self.labels[index])

        return data, label



    #自定义获取Data，Lable的函数
    #root是和此文件同一个目录的文件夹，存放数据集
    def load_csv(self,root,srcfilename,filename):
            if not os.path.exists(os.path.join(root,filename)):
                #存放数据集的数据

                datas, labels = [], []
                #在这里获取每条原始记录
                df = pd.read_csv(srcfilename,header=None)
                print(df)
                #将Df中数据按照每行存入列表featuredatas[]
                for idx in range (len(df)):
                    hx = df.loc[df.index[idx]]  # 获取每一行
                    print('hx')
                    print(hx)
                    hx = list(hx)
                    print(hx)
                    hx = list(map(float, hx))
                    print(hx)
                    print(hx[0:-1])
                    listdata = hx[0:-1]

                    print('listdata:', listdata)
                    datas.append(listdata)
                    labels.append(int(hx[-1]))
                #print(len(featuredatas),featuredatas)
                print('len(datas:',len(datas), datas)
                print('len(label:',len(labels), labels)
                print(datas[1])

                with open(os.path.join(root,filename),mode='w',newline='') as f:
                    writer = csv.writer(f)
                    for i in range (len(datas)):               #遍历
                        #获取每条记录的label
                        data = datas[i]
                        label =labels[i]
                        data.append(int(label))
                        writer.writerows([data])
                        print('i:',i)
                        #writer.writerow([data,label])
                    print('write into csv:',filename)

            if os.path.exists(os.path.join(root,filename)):
                #read from csv file
                datas,labels = [], []
                with open(os.path.join(root, filename)) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        data,label = row[0:-1],row[-1]
                        label = int(label) #注意思label眼转换为int
                        labels.append(label)
                        datas.append(data)
            print('len(datas:', len(datas), datas)
            print('len(label:', len(labels), labels)
            print('data[1]:',datas[1])
            assert len(datas)==len(labels)
            return datas,labels



def main():
    db_trainefe = featureDataset('UNSW-datas', 'netdata2.csv', 'netout.csv', 'train')

    # batchsz=32
    # db = featureDataset('UNSW-datas', 'netdata2.csv','netout.csv','train')
    # train_loader = DataLoader(db, batch_size=batchsz, shuffle=True,
    #                           num_workers=8)  # shufftle随机取得[data,labekl]片,但是不会重复;num_workers一次8个进程
    #
    # print(db)
    # print('dataset大小为：', db.__len__())


if __name__ == '__main__':
    main()