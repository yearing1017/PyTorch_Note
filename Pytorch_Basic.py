import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

# ================================================================== #
#                     1. Basic autograd example 1                    #
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
#                    2. Basic autograd example 2                     #
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

