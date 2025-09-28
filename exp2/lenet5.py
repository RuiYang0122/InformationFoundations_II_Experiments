import os
os.environ['TORCH_HOME'] = 'D:\Information_HW\Data\torch_cache'
import torch
from torch import nn
from d2l import torch as d2l
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    
#Lenet网络核心结构 选用Sigmoid激活函数
# net = torch.nn.Sequential(
#     Reshape(),
#     torch.nn.Conv2d(1, 6, kernel_size=5, padding=2),
#     torch.nn.Sigmoid(),
#     torch.nn.AvgPool2d(kernel_size=2, stride=2),
#     torch.nn.Conv2d(6, 16, kernel_size=5),
#     torch.nn.Sigmoid(),
#     torch.nn.AvgPool2d(kernel_size=2, stride=2),
#     torch.nn.Flatten(),
#     torch.nn.Linear(16 * 5 * 5, 120),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(120, 84),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(84, 10)
# )
    
#修改后的 LeNet 网络结构，使用 ReLU 激活函数
net = torch.nn.Sequential(
    Reshape(),
    torch.nn.Conv2d(1, 6, kernel_size=5, padding=2),
    torch.nn.ReLU(),  
    torch.nn.AvgPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(6, 16, kernel_size=5),
    torch.nn.ReLU(),  
    torch.nn.AvgPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(16 * 5 * 5, 120),
    torch.nn.ReLU(),  
    torch.nn.Linear(120, 84),
    torch.nn.ReLU(),  
    torch.nn.Linear(84, 10)
)


#测试网络结构
X = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

batch_size = 256
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#使用GPU计算模型在数据集上的精度
def evaluate_accuracy_gpu(net, data_iter, device='cuda'):
    
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())  # 累加当前批次的正确预测数和总样本数
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, lr, device):
    #初始化权重
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr) #优化器：随机梯度下降（SGD）
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3) #训练损失、正确预测数和总样本数
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            #更新模型参数
            optimizer.step()
            # 计算当前batch的损失和准确率
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            #绘图
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    # 训练损失值、训练准确率、测试准确率
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    # 每秒处理的样本数量
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

#学习率为0.9，训练10个epoch 激活函数：Sigmoid
#lr, num_epochs = 0.9, 10

#学习率为0.01，训练10个epoch 激活函数：ReLU
lr, num_epochs = 0.01, 10

train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()

save_path = r'D:\Information_HW\Experiment_code\exp2\lenet5_weights.pth'
torch.save(net.state_dict(), save_path)

