import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 超参数
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())
                                          
# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# 逻辑回归模型
model = nn.Linear(input_size, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epoches):
    for i, (images,labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1,28*28)
        
        # forward 计算
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印信息
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, running_loss))

# 测试模型
# 在测试阶段，为了效率，不需进行梯度计算
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1,28*28)
        outputs = model(images)
        # 分类问题，挑选出最大概率的一个
        # outputs是输出的10个类别的概率，输出最大的索引，1代表输出列方向索引
        # _ 代表忽略返回的最大概率值
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# 保存模型参数
torch.save(model.state_dict(), 'model.ckpt')



