# -*-coding:utf-8-*-
import torch
from torch import  nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        #加一个旁路,让x与out数量一样才可以相加;stride=stride和下面必须是一样的
        self.extra = nn.Sequential()#如果in=out,extra(x)相当于不变
        if ch_in!=ch_out:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                #nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # out = F.relu(self.conv1(x))
        # out = F.relu(self.conv2(out))
        out = self.extra(x)+out
        return out

    #NesBlock类init时给了输入输出,因为是不定的,对于ResNet18输入输出可以通过网络结构和实际输入数据得到
class ResNet_3(nn.Module):
    def __init__(self,n):
       super(ResNet_3, self).__init__()

       #预处理卷基层
       self.conv1=nn.Sequential(
           nn.Conv2d(1,16,kernel_size=3,stride=2,padding=0),
           nn.BatchNorm2d(16)
       )

       #follow 4 ResBlock,惨差层
       self.b1 = ResBlock(16,32,stride=2)
       self.b11 = ResBlock(32, 32, stride=2)
       self.b2 = ResBlock(32,64,stride=2)
       self.b22 = ResBlock(64, 64, stride=2)
       self.b3 = ResBlock(64, 128,stride=2)
       self.b33 = ResBlock(128, 128, stride=2)
       self.b4 = ResBlock(128, 256,stride=2)#chanmel一般不超过512

       #全链接层
       self.outLinear=nn.Linear(256,n)#这里的256由测试决定,受stride影响

    def forward(self,x):
        x = F.relu(self.conv1(x))

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        #print('x.shape:',x.shape)
        x=F.adaptive_max_pool2d(x,[1,1])#转为[b,256,1,1]，不用管size了，看b4通道数为256，所以Linear(256，n）
        #print('x.shape:', x.shape)
        x=x.view(x.size(0),-1)#打平
        #print('x.shape:', x.shape)
        x = self.outLinear(x)
        #print('x.shape:', x.shape)
        return x
#一般channels会慢慢增大,为了避免参数也增大,调节stride
if __name__ == '__main__':
    temp=torch.randn(6,1,14,14)
    res = ResNet_3(10)
    out=res(temp)
    print(out.shape)
    #打印所用参数
    print(sum(map(lambda  p:p.numel(),res.parameters())))


