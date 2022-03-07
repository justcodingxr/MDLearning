# -*-coding:utf-8-*-
#自定义数据加载标签分类等
import csv
import glob
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import  datasets
from torchvision import transforms     #变换器
from PIL import Image                  #图片读取工具



class PokeemonDataset(Dataset):
    def __init__(self,root,resize,mode):
        super(PokeemonDataset, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}
        #遍历目录
        for name in sorted(os.listdir(os.path.join(root))):
            #过滤文件
            if not os.path.isdir(os.path.join(root,name)):
                continue
            #保存文件夹名称
            self.name2label[name]  = len(self.name2label.keys())
        print(self.name2label)
        self.images,self.labels = self.load_csv('PokemonImages.csv')
        #裁剪,按照train,test对数据集裁剪,6:2:2
        if mode == 'train':
            self.images = self.images[0:int(0.6*len(self.images))]
            self.labels = self.labels[0:int(0.6*len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    #拿到 图片,标签的数据对 image,label,并将它保存到一个csv文件里面
    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images = []#存储图片路径,filename存[路径,label]
            #按照标签顺序将图片(路径)存到images,最终filename村了所有(图片,图片标签),图片标签可以根据路径中的值推断
            for name in self.name2label.keys():
                #images +=glob.glob(os.path.join(self.root,name,'*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                #images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                #images += glob.glob(os.path.join(self.root, name, '*.gif'))
                #images += glob.glob(os.path.join(self.root, name, '*.JPG'))
            print(len(images),images)
            #1166['Pokemon-ResNet/bulbasaur/00000226.png',
            #保存
            random.shuffle(images)
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                writer = csv.writer((f))
                for img in images:#Pokemon-ResNet/bulbasaur/00000226.png
                    #sep表示同用路径分割符号,img.split(os.sep)[-2]获取分割后的倒数第二个元素
                    name = img.split(os.sep)[-2]
                    #再由名字得到对应label
                    label = self.name2label[name]
                    #['Pokemon-ResNet/bulbasaur/00000226.png',label]
                    writer.writerow([img,label])
                print('write into csv:',filename)

        #if os.path.exists(os.path.join(self.root,filename))
        # read from csv file
        images,labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader((f))
            for row in reader:
                # ['Pokemon-ResNet/bulbasaur/00000226.png',label]
                img,label = row
                label = int(label) #注意思label眼转换为int
                images.append(img)
                labels.append(label)

        assert len(images)==len(labels)

        return images,labels

    def __len__(self):
        return len(self.images)     #获取裁剪后的各个部分的len

    #normalize会导致可视化出问题
    def denormalize(self,x_hot):
        mean=[0.485,0.456,0.406]
        std=[0.229,0.224,0.225]
        #x-hot=(x-mean)/std,所以x=     x = x_hot*std+mean
        #mean[3]=>mean[3,1,1]
        mean=torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std=torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hot*std+mean
        return x

    def __getitem__(self,idx):
        #['Pokemon-ResNet/bulbasaur/00000226.png'],[label]
        img,label = self.images[idx],self.labels[idx]

        #变换器,将图片路径转换为图片像素
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),#中心裁剪
            transforms.Resize((self.resize,self.resize)), #   (self.resize,self.resize)括号不可少
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])    #在rgb三个通道上的均值方差,分布由0-1=>-1-1之间
        ])
         #将图片路径转换为图片像素
        img = tf(img)
        label = torch.tensor(label)

        return img,label



def main():
    import  visdom
    import time
    viz = visdom.Visdom()
    import torchvision

    #方式一如果图片按照耳二级目录规则存放,可以简化 PokeemonDataset读取图片
    # trf = transforms.Compose([
    #     transforms.Resize((64,64)),
    #     transforms.ToTensor(),
    # ])
    # db=torchvision.datasets.ImageFolder(root='Pokemon',transform=trf)
    # print(db.class_to_idx)#打印编码方式 ,label信息

    #方式二 ,更加适用广泛
    #指定与imageloader同一级的Pokemon为root
    db = PokeemonDataset('Pokemon',224,'train')

    #获取单张iter(db) 获取images,labels的迭代器,next遍历
    #x,y = next(iter(db))


    loader = DataLoader(db,batch_size=32,shuffle=True,num_workers=8) #shufftle随机取得到图片,但是不会重复;num_workers一次取8个图片
    for x,y in loader:
        #label是tensor,转换为numpy在转换为str
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch-label'))
        #一行显示8帐
        #方式一
        #viz.images(x,nrow=8,win='batch',opts=dict(title='batch'))
        #方式二
        viz.images(db.denormalize(x),nrow=8,win='batch',opts=dict(title='batch'))
        time.sleep(0.1)

if __name__ == '__main__':
    main()