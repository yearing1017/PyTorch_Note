import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# ================================================================== #
#                     1. 自动求导例子 1                                #
# ================================================================== #

# 使用单个数字创建tensor
# requires_grad=True:将会追踪该tensor的所有操作，用于之后计算梯度
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# 一个公式.
y = w * x + b    # y = 2 * x + 3

# 由上述requires_grad=True,可有backward()计算梯度.
y.backward()

# 张量的所有梯度将会自动累加到.grad属性.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 

# ================================================================== #
#                    2. 自动求导例子 2                                 #
# ================================================================== #

# 创建 tensor shape 为(10, 3)和(10, 2)，随机初始化
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 一个全连接层.
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# 损失函数和优化器.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward 计算.
pred = linear(x)

# 计算损失.
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward 计算.
loss.backward()

# 打印梯度.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 计算完一个batch的数据后进行权重更新.
optimizer.step()

# 在权重更新之后，再次查看损失.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# 建一个numpy array
x = np.array([[1,2],[3,4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()

# ================================================================== #
#                         4. 使用官方的datasets方法                    #
# ================================================================== #

# Download CIFAR-10 dataset.
# transform 包含了如何对数据进行处理
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# read data from disk 只读一组数据
image, label = train_dataset[0]
print (image.size())
print (label)

# 数据加载器：设置batch大小，线程，是否打乱等
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# 遍历加载器 得到每个batch文件
data_iter = iter(train_loader)

# next方法得到一个batch
images, labels = data_iter.next()

# 每次batch大小进行喂入数据
for images, labels in train_loader:
    # Training code should be written here.
    pass

# ================================================================== #
#                5. 使用自己的数据                                     #
# ================================================================== #

# 标准创建自己的数据类的形式.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. 初始化工作，transform 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. 从路径去读文件
        # 2. torchvision.Transform 处理数据
        # 3. 返回一对 image & label
        pass
    def __len__(self):
        # 0 代替你的数据的数量 
        return 0 

# 像下面一样使用自己数据 构造loader加载器. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# 一个上述的例子（139-190）
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import cv2
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('bag_data'))

    # 实现__getitem__方法可将对象变为迭代器
    def __getitem__(self, idx):
        img_name = os.listdir('bag_data')[idx]
        imgA = cv2.imread('bag_data/'+img_name)
        imgA = cv2.resize(imgA, (160, 160))
        # 读入mask图片，0表示以灰度模式读
        imgB = cv2.imread('bag_data_msk/'+img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)       # onehot编码为2类
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        # 对输入放入图片进行transform转换
        if self.transform:
            imgA = self.transform(imgA)    

        return imgA, imgB

# 实例化一个对象
bag = BagDataset(transform)
# 此处调用len方法会调用类的内置方法__len__
train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
# random_split:按照给定的长度将数据集划分成没有重叠的新数据集组合
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

# 数据加载时会调用 __getitem__内置方法
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

# ================================================================== #
#                        6. 预训练模型的使用                            #
# ================================================================== #

# 下载并载入预训练好的模型ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# 如果只想微调 top layer，如下设置
for param in resnet.parameters():
    param.requires_grad = False  # 不需跟踪记录参数

# 替换 top layer.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# forward计算
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)

# ================================================================== #
#                      7. 保存和载入模型的使用方法                       #
# ================================================================== #

# 保存和载入.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# 只保存训练好模型的参数
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
