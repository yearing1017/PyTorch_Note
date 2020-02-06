### 1. 前言

- DenseNet是CVPR2017的best paper，之前写过论文笔记，详见[博文](https://yearing1017.site/2019/10/29/DenseNet-CVPR2017/)。
- 本文先回顾DenseNet的网络架构与思想，再使用Pytorch框架实现该网络。

### 2. DenseNet回顾

#### 2.1 核心思想

- DenseNet最显著的特性就是密集（Dense），这种密集体现在它各层特征图之间的连接方式上，下面这张图展示了**DenseNet的核心组件Dense Block（密集连接块）**：

 

- 其中，$x_0$表示网络的输入，$H_l$代表第$l$层的映射函数，$x_l$代表第$l$层的输出特征图。
- 所以，可以得到第$l$层的输出特征图$x_l$为：第$l$个映射函数$H_l(·)$对前一层（第$l-1$层）的映射输出：

$$
x_l = H_l(x_{l-1})
$$

- DenseNet很容易被误认为是ReNet的加强版，**实际上二者在不同层特征图的融合方式上存在很大的区别**。ResNet将不同层的特征进行“add”，也就是直接进行相加（或者采用1x1的旁路卷积），比如	其**第2个特征图$x_2$为本层映射输出$H(x_1)$和前一层特征图$x_1$之和**，表达式如下：

$$
x_l = H_l(x_{l-1}) + x_{l-1}
$$

- DenseNet为了进一步促进不同层之间的信息流，**将前面所有层输出都拼接到本层映射**，拼接指将不同层的特征图“concat”，也就是在通道上进行拼接。对于第$l$个映射函数$Hl(·)$来说，它的输入为前面所有特征图的拼接，计算方式如下：

$$
x_l = H_l([x_0,x_1...x_{l-1}])
$$

- 上面的公式就是**DenseNet最核心的“Dense”思想**，简洁有效。

#### 2.2 网络结构细节

- **Composite function组合函数**：
  - 非线性转换 + 归一化 + 卷积操作，实现上选用**ReLU + BN + Conv**，用$H_l(·)$表示。

- **Pooling layer 池化层**：

  - **池化层即transition layer**，负责下采样操作，对应下图中Dense Block之间的部分。transition层包含：归一化 + 1x1卷积 + 池化，文中使用**BN + 1x1Conv + 2x2Avg Pooling**。

  ![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/pytorch/2-2.jpg)

- **Growth Rate 增长率**：

  - **“第$l$个映射函数$H_l(·)$的输入，需要将前面所有层特征图按照通道进行拼接”**，假设Dense Block的输入通道数是$k_0$，并假设映射函数$H_l(·)$的输出通道数为k，那么第$l$层的输入通道数为前面所有层通道数之和：

  $$
  k_0 + k \times (l-1)
  $$

  - 某一个Dense Block的层数$l$与输入通道数$k_0$确定后，$k$就成为了唯一的超参数，作者将$k$定义为增长率 Growth Rate。

- **Bottleneck layers 瓶颈层**
  - **3x3卷积所需的计算量远远高于1x1卷积，所以在输入特征图的通道数较多的情况下，可以先使用1x1的卷积减少输入特征图的通道数，再使用3x3卷积**。论文中依旧是先进行归一化和非线性映射，再使用卷积，具体为：**BN-ReLU-Conv(1× 1)-BN-ReLU-Conv(3×3)**
- 具体的参数表如下：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/pytorch/2-3.jpg)

### 3. PyTorch实现DenseNet

#### 3.1 瓶颈层 Bottleneck Layers

- **瓶颈层主要作用是使用1x1的卷积降低特征图通道数，这样使得后面3x3卷积的计算量的到减轻**。根据论文中所述，1x1卷积的输出通道数为4k，也就是4倍的增长率。计算的顺序为BN + ReLU + Conv1x1 + BN + ReLU + Conv3x3，对应的代码实现如下：

```python
class Bottleneck(nn.Module):
    def __init__(self, channels_in, growth_rate):
        super().__init__()
        self.growth_rate = growth_rate
        self.channels_in = channels_in
        self.out_channels_1x1 = 4*self.growth_rate
        self.layers = nn.Sequential(nn.BatchNorm2d(num_features=self.channels_in),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.channels_in, out_channels=self.out_channels_1x1, kernel_size=1, padding=0,bias=False),
                                    nn.BatchNorm2d(num_features=self.out_channels_1x1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.out_channels_1x1, out_channels=self.growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        out = self.layers(x)
        # 重点：这里是x前面所有层的输出特征图
        out = torch.cat((x, out), dim=1)
        return out 
```

- 上述代码种最重要的一句是`out = torch.cat((x, out), dim=1)`，这行代码**是在前向卷积计算以后调用的**，也就是说当前第$l$层（Bottleneck层）的forward的返回值是映射函数$H_l(·)$输出的k个特征图与前面所有的特征图（共$l*k$个）拼接，这是个递归的过程！对应原理图中本层与后面层（后面Bottleneck层）相连的线，保证每一层计算都不会影响到上一层的特征图，**使得前层特征图可以不断的以累积拼接的形式向后层传递**。另外，**每个Bottleneck层的输出通道数都是相同的，均为增长率k**。

#### 3.2 Transition Layers 转换层

- 转换层为各个Dense Block之间的部分，也就是论文中对应的Pooling layer。
- 计算过程为BN+ ReLU + Conv1x1 + Avg Pooling2x2，代码如下：

```python
class TransitionLayer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        # 1：定义转换层的输入输出通道数
        self.channels_in = channels_in
        self.channels_out = channels_out
        # 2：BN+ReLU+Conv1x1+AvgPool2x2
        self.layers = nn.Sequential(nn.BatchNorm2d(num_features=channels_in),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.AvgPool2d(kernel_size=2))

    def forward(self, x):
        out = self.layers(x)
        return out
```

#### 3.3 制作DenseBlock

- DenseBlock内部全部使用Bottleneck层，目的是尽可能用1x1卷积降低通道数以保证3x3卷积计算量不会太大。代码如下：

```python
def make_dense_block(num_bottleneck, growth_rate, channels_in):
    """
    根据Bottleneck制作Dense Block
    :param num_bottleneck: 目标Dense Block层数
    :param growth_rate: 增长率，即通道数
    :param channels_in: 输入通道数
    :return: 返回nn.Sequential类型的Dense Block
    """
    # 1：创建容器
    layers = []
    # 2：每一个bottleneck层的输入通道数是前面所有bottleneck层输出通道数之和
    #    每一个bottleneck层输出通道数都是增长率k，即论文中growth rate
    current_channels = channels_in
    for i in range(num_bottleneck):
        # 3：给Dense Block添加Bottleneck层
        layers.append(BottleneckLayer(channels_in=current_channels, growth_rate=growth_rate))
        # 4：每次添加current_channels都增大growth rate
        current_channels += growth_rate
    return nn.Sequential(*layers)
```

- 创建Dense Block仅仅是一个循环过程，每添加一个Bottleneck层都会使得下一层的输入通道数增加k个，因为每一层输入都是前面所有层的输出。**

#### 3.4 FirstConv首个卷积层

- 这不是DenseNet提出的概念，而是为了代码清晰所以单独作为一个类来实现。
- **FirstConv负责将输入图片从3个通道变为和自己想要的m个通道，从而输入到后面的DenseBlock层，代码如下：**

```python
class FirstConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        """
        DenseNet第一个卷积层，将输入图片从3通道变为其它自定义通道数
        :param channels_in: 输入图片通道数
        :param channels_out: 自己设定的输出通道数
        """
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(num_features=channels_out),
                                    nn.ReLU(),
                                    nn.AvgPool2d(kernel_size=2))
    def forward(self, x):
        return self.layers(x)
```

### 4. 整体的DenseNet实现

- 使用DenseNet-121(k=32)如结构图。**对应4个Dense Block包含的Bottleneck层数分别是6、12、24和16。**
- 首先需要使用一个卷积+池化降低特征图长宽并将通道数设定到某个值，作为后面Dense Block的输入；
- 随后是**4个Dense Block，每个Dense Block由多个Bottleneck层组成。前3个Dense Block后面都紧跟一个Transition层，且Transition层输出通道数为其输入通道数的0.5倍（即compression=0.5）。**
- **最后一个Dense Block后面没有Transition层，而是7x7全局均值池化层，池化层后面是用于分类的全连接层。**

```python
class DenseNet(nn.Module):
    def __init__(self, growth_rate, channels_in, num_dense_block, num_bottleneck, num_channels_before_dense, compression, num_classes):
        """
        DenseNet核心代码
        :param growth_rate: 增长率
        :param channels_in: 输入数据通道数
        :param num_dense_block: 需要几个Dense Block，暂时不用此参数
        :param num_bottleneck: 用list表示每个DenseBlock包含的bottleneck个数，如list(6, 12, 24, 16)表示DenseNet121
        :param num_channels_before_dense: 第一个卷积层的输出通道数
        :param compression: 压缩率，Transition层的输出通道数为Compression乘输入通道数
        :param num_classes:类别数
        """
        super().__init__()
        self.growth_rate = growth_rate
        self.channel_in = channels_in
        self.num_dense_block = num_dense_block
        self.num_bottleneck = num_bottleneck

        # 1：定义第1个卷积层
        self.first_conv = FirstConv(channels_in=channels_in, channels_out=num_channels_before_dense)

        # 2：定义第1个Dense Block，其输出通道数为输入通道数加上层数*增长率
        self.dense_1 = make_dense_block(num_bottleneck=num_bottleneck[0], channels_in=num_channels_before_dense,
                                        growth_rate=growth_rate)
        dense_1_out_channels = int(num_channels_before_dense + num_bottleneck[0]*growth_rate)
        self.transition_1 = TransitionLayer(channels_in=dense_1_out_channels,
                                            channels_out=int(compression*dense_1_out_channels))

        # 3：定义第2个Dense Block，其输出通道数为输入通道数加上层数*增长率
        self.dense_2 = make_dense_block(num_bottleneck=num_bottleneck[1], channels_in=int(compression*dense_1_out_channels),
                                        growth_rate=growth_rate)
        dense_2_out_channels = int(compression*dense_1_out_channels + num_bottleneck[1]*growth_rate)
        self.transition_2 = TransitionLayer(channels_in=dense_2_out_channels,
                                            channels_out=int(compression*dense_2_out_channels))

        # 4：定义第3个Dense Block，其输出通道数为输入通道数加上层数*增长率
        self.dense_3 = make_dense_block(num_bottleneck=num_bottleneck[2], channels_in=int(compression * dense_2_out_channels),
                                        growth_rate=growth_rate)
        dense_3_out_channels = int(compression * dense_2_out_channels + num_bottleneck[2] * growth_rate)
        self.transition_3 = TransitionLayer(channels_in=dense_3_out_channels,
                                            channels_out=int(compression * dense_3_out_channels))

        # 5：定义第4个Dense Block，其输出通道数为输入通道数加上层数 * 增长率
        self.dense_4 = make_dense_block(num_bottleneck=num_bottleneck[3],
                                        channels_in=int(compression * dense_3_out_channels),
                                        growth_rate=growth_rate)
        dense_4_out_channels = int(compression * dense_3_out_channels + num_bottleneck[3] * growth_rate)

        # 6：定义最后的7x7池化层，和分类全连接层
        self.BN_before_classify = nn.BatchNorm2d(num_features=dense_4_out_channels)
        self.pool_before_classify = nn.AvgPool2d(kernel_size=7)
        self.classify = nn.Linear(in_features=dense_4_out_channels, out_features=num_classes)


    def forward(self, x):
        out_1 = self.first_conv(x)
        out_2 = self.transition_1(self.dense_1(out_1))
        out_3 = self.transition_2(self.dense_2(out_2))
        out_4 = self.transition_3(self.dense_3(out_3))
        out_5 = self.dense_4(out_4)
        out_6 = self.BN_before_classify(out_5)
        out_7 = self.pool_before_classify(out_6)
        out_8 = self.classify(out_7.view(x.size(0), -1))
        return out_8
```

- 输出测试size是否正确：

```python
x = torch.randn(size=(4, 3, 224, 224))
densenet = DenseNet(channels_in=3, compression=0.5, growth_rate=12, num_classes=10,num_bottleneck=[6, 12, 24, 16],
                        num_channels_before_dense=32,
                        num_dense_block=4)
out = densenet(x)
```

