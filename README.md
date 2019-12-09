# PyTorch_Note
⏰PyTorch学习笔记
## Pytorch_tutorial
- [Pytorch_Basic.py](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_Basic.py)：自动求导、数据集的使用、模型保存及载入
- [Pytorch_linearRegression.py](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_linearRegression.py)：线性回归例子实现完整训练
- [Pytorch_logisticRegression.py](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_logisticRegression.py)：MINIST+逻辑回归实现训练测试
- [Pytorch_NNdemo.py](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_NNdemo.py)：MINIST+简易神经网络实现训练测试
- [Pytorch_CNN](https://github.com/yearing1017/PyTorch_Note/blob/master/Pytorch_CNN.py)：MINST+卷积神经网络训练测试

## Pytorch_已解决问题_1
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

## 60分钟熟悉Pytorch
- 本部分为官方的中文文档内容，放在首页为了每次方便查阅
### 张量

- `Tensor`（张量）类似于`NumPy`的`ndarray`，但还可以在GPU上使用来加速计算

```python
from __future__ import print_function
import torch
```

- 创建一个没有初始化的5*3矩阵：

```python
x = torch.empty(5, 3)
print(x)
# 输出结果
'''
tensor([[2.2391e-19, 4.5869e-41, 1.4191e-17],
        [4.5869e-41, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])
'''
```

- 创建一个随机初始化矩阵：

```python
x = torch.rand(5, 3)
print(x)

# 输出结果
tensor([[0.5307, 0.9752, 0.5376],
        [0.2789, 0.7219, 0.1254],
        [0.6700, 0.6100, 0.3484],
        [0.0922, 0.0779, 0.2446],
        [0.2967, 0.9481, 0.1311]])
```

- 构造一个填满`0`且数据类型为`long`的矩阵:

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 输出结果
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

- 直接从数据构造张量：

```python
x = torch.tensor([5.5, 3])
print(x)

# 输出结果
tensor([5.5000, 3.0000])
```

- 根据已有的tensor建立新的tensor。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如dtype等：

```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 重载 dtype
print(x)                                      # 结果size一致

# 输出结果
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 1.6040, -0.6769,  0.0555],
        [ 0.6273,  0.7683, -0.2838],
        [-0.7159, -0.5566, -0.2020],
        [ 0.6266,  0.3566,  1.4497],
        [-0.8092, -0.6741,  0.0406]])
```

- 获取张量的形状：

```python
print(x.size())
# torch.Size([5, 3])
```

### 运算

- 以加法为例：

```python
y = torch.rand(5, 3)
print(x + y)

# 输出
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])

# 加法形式二
print(torch.add(x, y))

# 输出
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

- 加法：给定一个输出张量作为参数

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 输出
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

- 加法：原位/原地操作（in-place）

```python
# adds x to y
y.add_(x)
print(y)

# 输出
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

> 注意：
>
> 任何一个in-place改变张量的操作后面都固定一个`_`。例如`x.copy_(y)`、`x.t_()`将更改x

- 也可以使用像标准的NumPy一样的各种索引操作：

```python
print(x[:, 1])

# 输出
tensor([-0.6769,  0.7683, -0.5566,  0.3566, -0.6741])
```

- 改变形状：如果想改变形状，可以使用`torch.view`

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 表示根据其他的数值来计算该位置的数值
print(x.size(), y.size(), z.size())

# 输出
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

- 如果是仅包含一个元素的tensor，可以使用`.item()`来得到对应的python数值

```python
x = torch.randn(1)
print(x)
print(x.item())

# 输出
tensor([0.0445])
0.0445479191839695
```

> 后续阅读：
>
> 超过100种tensor的运算操作，包括转置，索引，切片，数学运算， 线性代数，随机数等，具体访问[这里](https://pytorch.org/docs/stable/torch.html)

### 桥接Numpy

- 一个Torch张量与一个NumPy数组的转换很简单

- Torch张量和NumPy数组将共享它们的底层内存位置，因此当一个改变时,另外也会改变。

```python
a = torch.ones(5)
print(a)

# 输出
tensor([1., 1., 1., 1., 1.])

b = a.numpy()
print(b)

# 输出
[1. 1. 1. 1. 1.]

a.add_(1)
print(a)
print(b)

# 输出
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

- numpy数组转换为张量

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# 输出
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

- CPU上的所有张量(CharTensor除外)都支持与Numpy的相互转换。
- 张量可以使用`.to`方法移动到任何设备（device）上：

```python
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype
    
# 输出
tensor([1.0445], device='cuda:0')
tensor([1.0445], dtype=torch.float64)
```

### 自动求导

#### 张量

- PyTorch中，所有神经网络的核心是 `autograd` 包。
- `autograd` 包为张量上的所有操作提供了自动求导机制。它是一个在运行时定义（define-by-run）的框架，这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的。
- 看例子
- `torch.Tensor` 是这个包的核心类。
  - 如果设置它的属性 `.requires_grad` 为 `True`，那么它将会追踪对于该张量的所有操作。
  - 当完成计算后可以通过调用 `.backward()`，来自动计算所有的梯度。
  - 这个张量的所有梯度将会自动累加到`.grad`属性.
- 要阻止一个张量被跟踪历史，可以调用 `.detach()` 方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。为了防止跟踪历史记录（和使用内存），可以将代码块包装在 `with torch.no_grad():` 中。在评估模型时特别有用，因为模型可能具有 `requires_grad = True` 的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。
- 还有一个类对于autograd的实现非常重要：`Function`。`Tensor` 和 `Function` 互相连接生成了一个无圈图(acyclic graph)，它编码了完整的计算历史。每个张量都有一个 `.grad_fn` 属性，该属性引用了创建 `Tensor` 自身的`Function`（除非这个张量是用户手动创建的，即这个张量的 `grad_fn` 是 `None` ）。
- 如果需要计算导数，可以在 `Tensor` 上调用 `.backward()`。如果 `Tensor` 是一个标量（即它包含一个元素的数据），则不需要为 `backward()` 指定任何参数，但是如果它有更多的元素，则需要指定一个 `gradient` 参数，该参数是形状匹配的张量。

```python
import torch

# 创建一个张量并设置requires_grad=True用来追踪其计算历史
x = torch.ones(2,2,requires_grad=True)
print(x)
'''
输出
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
'''

# 对张量做一次运算
y = x + 2
print(y)
'''
输出
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
y是计算的结果，所以它有grad_fn属性。
'''

print(y.grad_fn)
# 输出 <AddBackward0 object at 0x7f1b248453c8>

# 对y进行更多操作
z = y * y * 3
out = z.mean() # 对所有元素求均值
print(z, out)

# 输出
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)


# .requires_grad_(...) 原地改变了现有张量的 requires_grad 标志。如果没有指定的话，默认输入的这个标志是 False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# 输出
False
True
<SumBackward0 object at 0x7f1b24845f98>
```

#### 梯度

- 反向传播，因为 `out` 是一个标量，因此 `out.backward()` 和 `out.backward(torch.tensor(1.))` 等价。

```python
out.backward() # 自动计算所有的梯度

# 输出导数 d(out)/dx
print(x.grad)

# 输出
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```



### 神经网络

- 使用`torch.nn`包来构建神经网络。
- 我们已经学习了`autograd`，`nn`包则依赖于`autograd`包来定义模型并对它们求导。一个`nn.Module`包含各个层和一个`forward(input)`方法，该方法返回`output`。
- 一个神经网络的典型训练过程如下：
  - 定义包含一些可学习参数（或者叫权重）的神经网络
  - 在输入数据集上迭代
  - 通过网络处理输入
  - 计算损失（输出和正确答案的距离）
  - 将梯度反向传播给网络的参数
  - 更新网络的权重，一般使用一个简单的规则：`weight = weight - learning_rate * gradient`

#### 定义网络

- 定义这样一个网络：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    # 输入图像channel：1；输出channel：6；5x5卷积核
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
	
  def forward(self, x):
    # 2x2 Max pooling
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    # 如果是方阵,则可以只使用一个数字进行定义
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x)) # 见下面该函数
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
	def num_flat_features(self, x):
    size = x.size()[1:]  # 除去batch_size的其他所有维度,pytorch中为[batch_size,channle,h,w]
    num_features = 1
    for s in size:
    	num_features *= s
    return num_features
net = Net()
print(net)

# 输出
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)

```

- 我们只需要定义 `forward` 函数，`backward`函数会在使用`autograd`时自动定义，`backward`函数用来计算导数。可以在 `forward` 函数中使用任何针对张量的操作和计算。
- 一个模型的可学习参数可以通过`net.parameters()`返回

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# 输出
10
torch.Size([6, 1, 5, 5])

```

- 尝试一个随机的32x32的输入：

```python
input = torch.randn(1, 1, 32, 32) # 1分别代表batch_size,channle
out = net(input)
print(out)
# 输出：上网络结果最后输出10个分类
tensor([[ 0.0399, -0.0856,  0.0668,  0.0915,  0.0453, -0.0680, -0.1024,  0.0493,
         -0.1043, -0.1267]], grad_fn=<AddmmBackward>)

```

- 清零所有参数的梯度缓存，然后进行随机梯度的反向传播：

```python
net.zero_grad()
out.backward(torch.randn(1, 10))

```

> 注意：
>
> `torch.nn`只支持小批量处理（mini-batches）。整个`torch.nn`包只支持小批量样本的输入，不支持单个样本。
>
> 比如，`nn.Conv2d` 接受一个4维的张量，即`nSamples x nChannels x Height x Width`
>
> 如果是一个单独的样本，只需要使用`input.unsqueeze(0)`来添加一个“假的”批大小维度。

#### 损失函数

- 一个损失函数接受一对(output, target)作为输入，计算一个值来估计网络的输出和目标值相差多少。
- nn包中有很多不同的[损失函数](https://pytorch.org/docs/stable/nn.html)。`nn.MSELoss`是比较简单的一种，它计算输出和目标的均方误差（mean-squared error）。

```python
output = net(input)
target = torch.randn(10)     # 本例子中使用模拟数据
target = target.view(1, -1)  # 使目标值与数据值形状一致
criterion = nn.MSELoss()     # 均方误差函数

loss = criterion(output, target)
print(loss)

# 输出
tensor(1.0263, grad_fn=<MseLossBackward>)

```

- 现在，如果使用`loss`的`.grad_fn`属性跟踪反向传播过程，会看到计算图如下：

```python
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss

```

- 所以，当我们调用`loss.backward()`，整张图开始关于loss微分，图中所有设置了`requires_grad=True`的张量的`.grad`属性累积着梯度张量。

#### 反向传播

- 我们只需要调用`loss.backward()`来反向传播权重。我们需要清零现有的梯度，否则梯度将会与已有的梯度累加。
- 现在，我们将调用`loss.backward()`，并查看conv1层的偏置（bias）在反向传播前后的梯度。

```python
net.zero_grad()     # 清零所有参数（parameter）的梯度缓存

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 输出
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([ 0.0084,  0.0019, -0.0179, -0.0212,  0.0067, -0.0096])

```

#### 更新权重

- 最简单的更新规则是随机梯度下降法（SGD）:

- ```python
  weight = weight - learning_rate * gradient
  
  ```

- 在使用神经网络时，可能希望使用各种不同的更新规则，如SGD、Nesterov-SGD、Adam、RMSProp等。为此，我们构建了一个较小的包`torch.optim`，它实现了所有的这些方法。使用它很简单：

```python
import torch.optim as optim

# 创建优化器（optimizer）
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练的迭代中：
optimizer.zero_grad()   # 清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新参数

```

