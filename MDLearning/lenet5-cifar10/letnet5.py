import torch
from torch import nn
from torch.nn import functional as F
#lenet5是最简单的卷及神经网络版本

class LetNet5(nn.Module):
    #类的初始化方法,为此法人0而建
    def __init__(self):
        super(LetNet5,self).__init__()
         #定义网络结构,一个卷及层,一个pool曾,在一个卷基层...

        #卷及层:在定统一好输入输出channels后,fc_unit输入[b,3,32,32],卷及层中间每层输出的结点是可以确定的,所以定义卷基层不需要中间节点个数
        self.conv_unit = nn.Sequential(
            #取名conv_unit
            #[b,3,32,32]b张照片,输入channels为3,32*32像素点
            #输出6channels
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding= 0),
            #3个输入通道,6个输出通道,卷积核尺度5,stride步长,padding边界扩充,目的是不改变输出map的大小
            #池化层,MaxPool和AvgPool2d的计算方式不同
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            #第二个卷及层
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # 第二个池化层,MaxPool和AvgPool2d的计算方式不同
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

            #全链接层,pytorch没有专门打平的类,Sequential中所有的类都会i要求已经有的
            #所以flatten不能写在这里
    )

        #flatten
        #fc_unit全链接层
        self.fc_unit=nn.Sequential(
            nn.Linear(16*5*5,120),#见out,将本来一张图得到的所有特征图信息打平
            nn.ReLU(),
            nn.Linear(120,84),#输入120,输出84
            nn.ReLU(),
            nn.Linear(84,10)#最后iu输出10类
        )

        #试图得到fc_unit全链接层的第一个Linear参数
        tmp = torch.randn(2,3,32,32)#假定一个数据[b,3,32,32],b=2张
        out = self.conv_unit(tmp) #计算出卷及层输出维度,然后就可以定义fc_unit全链接层的第一个Linear参数
        print('conv_out',out.shape)
        #conv_out torch.Size([2, 16, 5, 5])




    def forward(self,x):#forward是涵盖在网络模块中的,backward不需要自己实现
        batchse=x.size(0)   #相当与得到[b, 3, 32, 32]的0维度数值

        #[b, 3, 32, 32]=>[b, 16, 5, 5],(测试过)
        x=self.conv_unit(x) #输入x获得卷及后的输出

        # flatten,卷及后的输出打平
        x=x.view(batchse,16*5*5)#传入b,和-1也行,将后面的维度打平成一个维度[b,16*5*5]

        #[b,16*5*5]=>logits[b,10]=>
        #logits,进行softmax前全链接网络fc_unit之后的数据常叫logtis
        logits = self.fc_unit(x)
        return logits
        #CrossEntropyLoss()取代pred = F.softmax(logits,dim=1)  #在logits[b,10]的1维softmax

        #输入x[b,3,32,32]=>卷基网络conv_unit(x)=>flatten打平=x.view(batchse,16*5*5)=>全链接曾logits = self.fc_unit(x) =>pre= F.softmax(logits)或CrossEntropyLoss()
#测试模块

    #可以在网络模块定义一个loss评价计算方法,分类问题交叉熵比MSE更好

def test_conv_out():
    net=LetNet5()

def test_net_out():
    net=LetNet5()
    tmp = torch.randn(7, 3, 32, 32)
    out = net(tmp)
    print('net.shape:',out.shape)
if __name__ == '__main__':
    test_conv_out()
    test_net_out()

