# -*-coding:utf-8-*-
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import  datasets
from  torchvision import transforms
from torch import nn,optim
import visdom
import time
from PIL import Image
import numpy as np
from featuredataloader import featureDataset
from ResNet import  ResNet18


#batchsz = 2
lr = 1e-3
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
torch.manual_seed(1234)

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

db_train = featureDataset('UNSW-datas', 'netdata2.csv', 'netout.csv','train')
train_loader = DataLoader(db_train,  shuffle=True,
                          num_workers=2)  # shufftle随机取得[data,labekl]片,但是不会重复;num_workers一次8个进程

db_val = featureDataset('UNSW-datas', 'netdata2.csv', 'netout.csv','val')
val_loader = DataLoader(db_val,  shuffle=True,
                          num_workers=2)  # shufftle随机取得[data,labekl]片,但是不会重复;num_workers一次8个进程
db_test = featureDataset('UNSW-datas', 'netdata2.csv', 'netout.csv','test')
test_loader = DataLoader(db_test,  shuffle=True,
                          num_workers=2)  # shufftle随机取得[data,labekl]片,但是不会重复;num_workers一次8个进程
#验证和测试用同一个函数
def evalute(model,loader):
    correct_count = 0
    total = len(loader.dataset)#dataset是DataLoader的属性
    for x,y in loader:
        with torch.no_grad():#测试不用更新梯度
            logits = model(x)
            predmax_index = logits.argmax(dim=1)#predmax-index是预测的logits中概率最大值对应的索引
            correct_count += torch.eq(predmax_index,y).sum().float().item()
    return  correct_count/total


def main():

    model = ResNet18(10)
    optimzer = optim.Adam(model.parameters(),lr=lr)
    criteon = nn.CrossEntropyLoss()

    #保存最好状态
    best_acc,best_epoch=0,0

    for epoch in range(epochs):
        print('epoch:',epoch)
        #enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        #for batchidx,(x,y) in enumerate(train_loader):


        for x, y in train_loader:
            x, y = iter(train_loader).next()
            print('x.shape:', x.shape, 'lable.shape:', y.shape)
            #x:[b,3,224,224],y:[b]
            #x,y=x.to(device),y.to(device)
            logits = model(x)#模型中输入x,获得一个[a,5]格式的logits,表示5类
            print('logits:',logits)
            loss = criteon(logits,y)#logits与标签输入CrossEntropyLoss()得到交叉商损失
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            print('now_loss:', loss)

          #做测试验证
        if epoch%2 == 0:
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                 best_acc = val_acc
                 best_epoch = epoch
                 torch.save(model.state_dict(),'best_model.mdl')
                 print('now_best_acc:',best_acc)
    print('best acc:',best_acc,'best_epoch:',best_epoch)
    model.load_state_dict(torch.load('best_model.mdl'))#将valdedao的最好的模型用来test
    print('loaded from ckpt!')

    #最好的模型来测试,model已经被覆盖了
    test_acc = evalute(model,test_loader)
    print('test acc:',test_acc)


if __name__ == '__main__':
    main()