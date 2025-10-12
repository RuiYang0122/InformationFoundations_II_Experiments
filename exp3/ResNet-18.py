import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l    
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np
import matplotlib.pyplot as plt

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, X):
        #输入X先经过第一个卷积层，再经过批量归一化层，最后经过ReLU激活函数
        Y = F.relu(self.bn1(self.conv1(X)))
        #Y再经过第二个卷积层，再经过批量归一化层，最后经过ReLU激活函数
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        #残差连接
        Y += X
        return F.relu(Y)
    
b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))

X = torch.rand(1, 3, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 调整学习率和批次大小以适应CIFAR-10
lr, num_epochs, batch_size = 0.01, 20, 128

# 在创建优化器时添加权重衰减
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 添加L2正则化

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.Resize(224),  # 调整大小到224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转增强
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 类别名称
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 加载数据集 - 保存到d:/Information_HW/Data/CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(
    root='d:/Information_HW/Data/CIFAR-10', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='d:/Information_HW/Data/CIFAR-10', train=False, download=True, transform=transform_test)

# 创建数据加载器
train_iter = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_iter = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 添加保存预测结果的函数
def save_predictions(model, test_loader, classes, output_dir, num_images=50):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 将模型设置为评估模式
    model.eval()
    device = next(model.parameters()).device  # 获取模型所在设备
    
    # 收集足够的测试图像
    images_list = []
    labels_list = []
    
    # 随机选择一个起始批次
    start_batch = random.randint(0, len(test_loader) - 2)
    
    # 从起始批次开始收集图像
    data_iter = iter(test_loader)
    for _ in range(start_batch):
        next(data_iter)  # 跳过前面的批次
    
    # 收集足够的图像
    while len(images_list) < num_images:
        try:
            images, labels = next(data_iter)
            images, labels = images.to(device), labels.to(device)
            images_list.append(images)
            labels_list.append(labels)
        except StopIteration:
            break  # 如果数据用完了就停止
    
    # 合并收集到的图像和标签
    all_images = torch.cat(images_list)[:num_images]
    all_labels = torch.cat(labels_list)[:num_images]
    
    # 获取模型预测
    with torch.no_grad():
        outputs = model(all_images)
        _, predicted = outputs.max(1)
    
    # 保存图像及预测结果
    for i in range(num_images):
        img = all_images[i].cpu().numpy().transpose((1, 2, 0))
        # 反归一化，使用与训练时相同的参数
        img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img, 0, 1)  # 确保像素值在合理范围

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f'True: {classes[all_labels[i]]}\nPred: {classes[predicted[i]]}', fontsize=10)
        plt.axis('off')
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f'prediction_{i}.png'), dpi=150, 
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    print(f'已成功保存 {num_images} 张预测结果图片到 {output_dir}')

# 训练模型
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 保存预测结果
output_directory = 'D:\Information_HW\Experiment_code\exp3\img'
save_predictions(net, test_iter, classes, output_directory, num_images=50)

d2l.plt.show()