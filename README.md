# PyTorch_Note
⏰PyTorch学习笔记
## 💡 1. Pytorch_tutorial
- [Pytorch_60min.md](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_60min.md)：官方60分钟入门Pytorch
- [Pytorch_Basic.py](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_Basic.py)：自动求导、数据集的使用、模型保存及载入
- [Pytorch_linearRegression.py](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_linearRegression.py)：线性回归例子实现完整训练
- [Pytorch_logisticRegression.py](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_logisticRegression.py)：MINIST+逻辑回归实现训练测试
- [Pytorch_NNdemo.py](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_NNdemo.py)：MINIST+简易神经网络实现训练测试
- [Pytorch_CNN](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_CNN.py)：MINST+卷积神经网络训练测试
- [pytorch_cuda.ipynb](https://github.com/yearing1017/PyTorch_Note/blob/master/pytorch_cuda.ipynb)：pytorch有关cuda的基本操作与概念
- [LeNet.ipynb](https://github.com/yearing1017/PyTorch_Note/blob/master/LeNet.ipynb)：pytorch搭建LeNet网络
- [ResNet.ipynb](https://github.com/yearing1017/PyTorch_Note/blob/master/ResNet.ipynb)：pytorch搭建ResNet
- [Pytorch_图像增强](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_图像增强.md)：总结了pytorch中主要用到的7中图像增强的方法
- [DenseNet_PyTorch实现](https://github.com/yearing1017/PyTorch_Note/blob/master/DenseNet_PyTorch.md)；回顾DenseNet的核心架构以及使用PyTorch进行实现
- [PyTorch保存模型两种方式的比较](https://zhuanlan.zhihu.com/p/94971100)：保存模型和保存模型参数及load使用的方式
- [PyTorch对于数据集的处理方式-torch.utils.data](https://www.cnblogs.com/Bella2017/p/11791216.html)：subset根据索引来获取子集
- [PyTorch常用代码段转载](https://zhuanlan.zhihu.com/p/104019160)：非常高人气的文章，适合查阅

## 💡 2. Pytorch_已解决问题_1

- 在跑unet的模型时，遇到该错误:

`RuntimeError: Given groups=1, weight of size 64 3 3 3, expected input[4, 64, 158, 158] to have 3 channels, but got 64 channels instead`

- 问题是输入本来该是 3 channels，但却是64通道。

- 解决思路：打印了一下输入的size:[4,3,160,160],本来以为没错误，就一直在找。

- 实际问题：因为我在以下代码部分有两个卷积操作，我的第二个卷积的输入应该是第一个卷积的输出，我却设定了两者相同。如下：
```python
class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super.__init__()
		# 构建一个“容器网络”
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=3),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels,out_channels,kernel_size=3),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	def forward(self,x):
		return self.double_conv(x)
```

- 在第25行的卷积中，我的in_channels和第一个卷积的一样，但却应该是第一个的输出，所以改为out_channels,如下：
```python
class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super.__init__()
		# 构建一个“容器网络”
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,kernel_size=3),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels,out_channels,kernel_size=3),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	def forward(self,x):
		return self.double_conv(x)
```

## 💡 3. Pytorch_cv2_Tensor相关

- 一个灰度的图片，只有一个通道，只是cv2读取image，打印shape，仅仅显示[H, W]

- 若使用了torchvision.ToTensor()方法，再打印shape，会打印出[C, H, W],且进行了归一化，取值范围为[0,1.0]的torch.FloatTensor
```python
import cv2
import torch
import torchvision.transforms

image_name = 'Dataset/2d_images/ID_0000_Z_0142.tif'
image = cv2.imread(image_name, 0)
print(image.shape) # (512,512)
image_tensor = torchvision.transforms.ToTensor()(image)
print(image_tensor.shape) # torch.Size([1, 512, 512])
```

- 还有一种暴力方法得到想要的shape，先resize，再reshape，最后再转为tensor，这期间没有进行归一化
```python
import torch
import torchvision.transforms
import cv2

image_name = 'Dataset/2d_images/ID_0000_Z_0142.tif'
image = cv2.imread(image_name, 0)
print(image.shape)
image = cv2.resize(image, (160, 160))
image_new = image.reshape((1, 160, 160))
image_tensor = torch.FloatTensor(image_new)
print(image_new.shape)
print(image_tensor.shape)

# 输出：
(512, 512)
(1, 160, 160)
torch.Size([1, 160, 160])
```
- `cv2.resize(img, (width, height))`参数是:先宽后高

- **class torchvision.transforms.ToTensor**:
  - 把一个取值范围是`[0,255]`的`PIL.Image`或者`shape`为`(H,W,C)`的`numpy.ndarray`，转换成形状为`[C,H,W]`，取值范围是`[0,1.0]`的`torch.FloatTensor`
  
- **class torchvision.transforms.Normalize(mean, std)**:
  - 给定均值：`(R,G,B)` 方差：`（R，G，B）`，将会把`Tensor`正则化。即：`Normalized_image=(image-mean)/std`
  
- cv2.imread(img, 1)：返回结果为`type: numpy.ndarray `，多维数组

## 💡 4. Pytorch模型构造

###  4.1 继承Module类来构造模型

- 这里定义的MLP类重载了Module类的__init__函数和forward函数。它们分别用于创建模型参数和定义前向计算。前向计算也即正向传播。
```python
import torch
from torch import nn

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层


    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
```

###  4.2 使用Sequential类定义模型

- 它可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数来逐一添加Module的实例
- 模型的前向计算就是将这些实例按添加的顺序逐一计算
```python
import torch
from torch import nn
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),# in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, img):
        feature = self.conv(img)
        # view相当于reshape，这里的img.shape[0]是batch_size，-1代表自动计算出来的H*W*Channels
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```

### 4.3 ModuleList类

- ModuleList接收一个子模块的列表作为输入，然后类似List那样进行append和extend操作:
```python
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError
# 输出
Linear(in_features=256, out_features=10, bias=True)
ModuleList(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
```
- 既然Sequential和ModuleList都可以进行列表化构造网络，那二者区别是什么呢。
- ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），而且没有实现forward功能需要自己实现，所以上面执行net(torch.zeros(1, 784))会报NotImplementedError；
- 而Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现。

### 4.4 ModuleDict类
- ModuleDict接收一个子模块的字典作为输入, 然后也可以类似字典那样进行添加访问操作:
```python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError
# 输出
Linear(in_features=784, out_features=256, bias=True)
Linear(in_features=256, out_features=10, bias=True)
ModuleDict(
  (act): ReLU()
  (linear): Linear(in_features=784, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
```
- 和ModuleList一样，ModuleDict实例仅仅是存放了一些模块的字典，并没有定义forward函数需要自己定义。
- 同样，ModuleDict也与Python的Dict有所不同，ModuleDict里的所有模块的参数会被自动添加到整个网络中。

## 💡 5. Pytorch的CrossEntropyLoss

- 错误描述：语义分割实验中，在对label进行onehot编码之后，将其变为(4,4,640,640)，定义loss如下：
```python
criterion = nn.CrossEntropyLoss().to(device)
loss = criterion(output, label)
```
- 报错：**大致为，期望的target是3维，却得到了一个4维。**
- 查看官方文档如下：
![](https://github.com/yearing1017/PyTorch_Note/blob/master/image/5-4.png)
- 该损失函数包含了**softmax函数**，**该损失函数期望的target是在像素值为（0，C-1）的一个标注图。与标注好的label相对应，每个像素值标注了类别0-3。共4类**
![](https://github.com/yearing1017/PyTorch_Note/blob/master/image/5-3.png)
- 上图详细解释了loss函数的要求的shape。对于语义分割的4维向量来说：**要求input即网络的预测为(N,C,H,W)，target为(N, H, W)，且target[i]在0-C-1之间。**
- **改动：去掉onehot，直接读入标注的label，因为符合上述要求。**

## 💡 6. tensorboardX的简单使用
- 安装：
`pip install tensorboardX`
- 简单数据的记录: `writer.add_scalar(名称，数值，x轴坐标)`
```python
from tensorboardX import SummaryWriter
writer.add_scalar('train_loss', train_loss/len(train_dataloader), epo)
```

## 💡 7. 预训练模型参数的使用
- 在训练时，会考虑是否采用在例如Image数据集上预训练得到的参数，但是有的时候预训练得到的网络结构是当前训练网络的一部分
- 例如：deeplabv3-resnet基于残差网络，我们若使用resnet的预训练参数，则需要判断那些层可以使用（其中deeplabv3中新添加了aspp和移除了FC）
- 实例代码如下：
```python
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 基于ResNet的deeplabv3
class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes, num_groups=None, weight_std=False, beta=False, pretrained=False):
        self.inplanes = 64 # 控制残差块的输入通道数 planes:输出通道数
        # nn.BatchNorm2d和nn.GroupNorm两种不同的归一化方法
        self.norm = nn.BatchNorm2d
        self.conv = Conv2d if weight_std else nn.Conv2d
        super(ResNet, self).__init__()

        if not beta:
            # 整个ResNet的第一个conv
            self.conv1 = self.conv(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            # 第一个残差模块的conv
            self.conv1 = nn.Sequential(
                self.conv(3, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 建立残差块部分
        self.layer1 = self._make_layer(block, 64,  block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        # block4开始为dilation空洞卷积
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, dilation=2)
        # aspp,512 * block.expansion是经过残差模块的输出通道数
        self.aspp = ASPP(512 * block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)
        # 遍历模型进行初始化
        for m in self.modules():
            if isinstance(m, self.conv):        #isinstance：m类型判断    若当前组件为 conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  #正太分布初始化
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm): #若为batchnorm
                m.weight.data.fill_(1)          #weight为1
                m.bias.data.zero_()             #bias为0

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        # stride!=1 代表后续残差块中有stride=2，尺寸大小改变，所以第一个残差块中的stride也该用来修改尺寸
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )
        # laysers 存放产生的残差块，最后根据此列表进行生成网络
        layers = []
        # 在多个残差块中，只有第一个残差块的输入输出通道不一致，所以先单独添加带downsample的block
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)


    def forward(self, x):
        # x.shape:[batch_size, channels, H, w]
        size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)
        #x = x.reshape(-1, x.shape[1])
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x
    # 根据具体的网络层来载入模型参数    
    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=4, pretrained=pretrained, **kwargs)
    return model
```
