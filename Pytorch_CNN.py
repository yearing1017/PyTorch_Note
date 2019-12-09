import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 超参数
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# 网络搭建
class ConvNet(nn.Moudle):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            # input_channel:1   output_channel:16
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
     
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # size(0)提取tensor的第一维：batch_size
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
      
model = ConvNet(num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward 计算
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
# 测试
# 有关eval模式和train模式的区别见下
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


"""

在训练模型时会在前面加上：model.train()
在测试模型时会在前面使用：model.eval()
虽然不适用这两个语句程序也能运行，但运行的细节不一样。
比如Batch Normalization 和 Dropout。
Batch NormalizationBN的作用主要是对网络中间的每层进行归一化处理，并且使用变换重构（Batch Normalization Transform）保证每层提取的特征分布不会被破坏。
训练时是针对每个mini-batch的，但是测试是针对单张图片的，即不存在batch的概念。
由于网络训练完成后参数是固定的，因此每个batch的均值和方差是不变的，因此直接结算所以batch的均值和方差。
DropoutDropout能够克服Overfitting，在每个训练批次中，通过忽略一半的特征检测器，可以明显的减少过拟合现象。
详细见文章：《Dropout: A Simple Way to Prevent Neural Networks from Overtting》
eval()就是保证BN和dropout不发生变化，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果！！！
"""















