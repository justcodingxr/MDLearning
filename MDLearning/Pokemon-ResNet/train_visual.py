# -*-coding:utf-8-*-
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
import visdom
import time
from PIL import Image

from imageloader import PokeemonDataset
from ResNet import ResNet18

batchsz = 32
lr =1e-3
epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
torch.manual_seed(1234)

db_train = PokeemonDataset('Pokemon', 224, 'train')
train_loader = DataLoader(db_train, batch_size=batchsz, shuffle=True,
                          num_workers=8)  # shufftle随机取得到图片,但是不会重复;num_workers一次取8个图片
db_val = PokeemonDataset('Pokemon', 224, 'val')
val_loader = DataLoader(db_val, batch_size=batchsz, shuffle=True,
                        num_workers=8)  # shufftle随机取得到图片,但是不会重复;num_workers一次取8个图片
db_test = PokeemonDataset('Pokemon', 224, 'test')
test_loader = DataLoader(db_test, batch_size=batchsz, shuffle=True,
                         num_workers=8)  # shufftle随机取得到图片,但是不会重复;num_workers一次取8个图片

# batch_size=batchsz,从而得到一个batchidx,
# 在下面训练的for循环中将batchidx与[image,label]组合,
# for batchidx,(x,label) in enumerate(train_loader)


viz = visdom.Visdom()
# for x, y in train_loader:
#     # label是tensor,转换为numpy在转换为str
#     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-label'))
#     # 一行显示8帐
#     # 方式二
#     # normalize会导致可视化出问题,作用在数据集上,  不是DataLoader,DataLoader只是个迭代器
#     viz.images(db_train.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
#     time.sleep(1)


# 验证和测试用同一个函数
def evalute(model, loader):
    correct_count = 0
    total = len(loader.dataset)  # dataset是DataLoader的属性
    for x, y in loader:
        with torch.no_grad():  # 测试不用更新梯度
            logits = model(x)
            predmax_index = logits.argmax(dim=1)  # predmax-index是预测的logits中概率最大值对应的索引
            correct_count += torch.eq(predmax_index, y).sum().float().item()
    return correct_count / total


def main():
    model = ResNet18(5)
    optimzer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    # 保存最好状态
    best_acc, best_epoch = 0, 0

    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    t=0

    for epoch in range(epochs):
        # enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        for batchidx, (x, y) in enumerate(train_loader):
            # x:[b,3,224,224],y:[b]
            # x,y=x.to(device),y.to(device)
            logits = model(x)  # 模型中输入x,获得一个[a,5]格式的logits,表示5类
            loss = criteon(logits, y)  # logits与标签输入CrossEntropyLoss()得到交叉商损失
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            print('now_loss:', loss)
            viz.line([loss.item()], [t], win='loss', update='append')
            t+=1

        # 做测试验证
        if epoch % 2 == 0:
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_model.mdl')
                print('now_best_acc:', best_acc)
                #可视化更好的acc
                viz.line([val_acc], [t], win='val_acc', update='append')

    print('best acc:', best_acc, 'best_epoch:', best_epoch)
    model.load_state_dict(torch.load('best_model.mdl'))  # 将valdedao的最好的模型用来test
    print('loaded from ckpt!')

    # 最好的模型来测试,model已经被覆盖了
    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()
# -*-coding:utf-8-*-
