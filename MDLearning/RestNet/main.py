# -*-coding:utf-8-*-
import torch
from torch.utils.data import DataLoader
from torchvision import  datasets
from  torchvision import transforms
#from letnet5 import LetNet5
#letnet5和resnet输入[b,3,32,32]一样,输出都是10类可以替换模型
from ResNet import ResNet18
from torch import nn,optim


batchse = 32#应该为32最好
def load_train_data():
    # datasets会自动加载数据集到cifar文件夹
    cifar_train = datasets.CIFAR10('cifar10', True, transform=
    transforms.Compose([transforms.Resize((32, 32)),
                        transforms.ToTensor(),

                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                        ]), download=True)
    return cifar_train


# True表示训练数据集,False表示测试数据集
def load_test_data():
    cifar_test = datasets.CIFAR10('cifar10', False, transform=
    transforms.Compose([transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485,0.456,0.406],
                                             std=[0.229,0.224,0.225])]), download=True)
    return cifar_test


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #DataLoader分批次加载,DataLoader是一个迭代器
    #cifar10数据集已经分好了训练集和数据集,DataLoader负责安排每次训练的数量
    #x,lable=iter(cifar_train).next()得到DataLoader的一个batch

    cifar_train = DataLoader(load_train_data(),batch_size=batchse,shuffle=True)
    cifar_test  = DataLoader(load_test_data(), batch_size=batchse, shuffle=True)#load_test_data()

    # shuffle表示加载时随机
    # https://www.cnblogs.com/ranjiewen/p/10128046.html

    x, lable = iter(cifar_train).next()
    print('x.shape:',x.shape,'lable.shape:',lable.shape)



    #model = LetNet5()
    model =ResNet18()
    print(model)
    criteria = nn.CrossEntropyLoss()
    #优化器
    optimizer = optim.Adam(model.parameters(),lr=1e-3)#0.01
    for epoch in range(1000):
        model.train()
        for batchidx,(x,label) in enumerate(cifar_train):
            #[b,32,32,2]输入,[b]标签
            logits = model(x)
            loss = criteria(logits,label)

            #backprop
            optimizer.zero_grad()#清零
            loss.backward()
            optimizer.step()
            print('batchidx:',batchidx, loss.item())
        print('epoch:',epoch,loss.item())    #item将标量转为numpy输出

        model.eval()
        with torch.no_grad():
            #test
            total_correct = 0
            toltal_num = 0 ;
            for x,label in cifar_test:
                # [b,10]
                logits = model(x)
                # [b]
                pre = logits.argmax(dim=1)
                total_correct +=torch.eq(pre,label).sum().item()
                toltal_num+=x.size(0)
            acc = total_correct/toltal_num
            print(epoch,acc)

if __name__ == '__main__':
    main()

