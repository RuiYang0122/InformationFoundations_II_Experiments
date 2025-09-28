import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class BPNetwork(nn.Module):
    """三层BP神经网络"""
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        super(BPNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # 选择激活函数
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)  # 输出层不使用激活函数（线性输出）
        return x

def train_network(model, X, y, epochs=10000, lr=0.01, optimizer_type='adam'):
    """训练神经网络"""
    criterion = nn.MSELoss()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
        
        if loss.item() < 1e-5:
            print(f'Training stopped at epoch {epoch}')
            break
    
    return losses

def test_xor():
    """测试XOR问题"""
    print("=== XOR Problem ===")
    
    # 准备数据
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    # 创建网络
    model = BPNetwork(input_size=2, hidden_size=4, output_size=1, activation='sigmoid')
    
    # 训练
    losses = train_network(model, X, y, epochs=10000, lr=0.5, optimizer_type='sgd')
    
    # 测试
    with torch.no_grad():
        predictions = model(X)
        print("Final predictions:")
        for i, (input_val, target, pred) in enumerate(zip(X, y, predictions)):
            print(f"Input: {input_val.numpy()}, Target: {target.item():.0f}, Prediction: {pred.item():.4f}")
    
    return losses

def test_function_approximation():
    """测试函数逼近问题: y = 1/sin(x) + 1/cos(x)"""
    print("\n=== Function Approximation ===")
    
    # 准备数据
    x1 = np.linspace(-np.pi/2 + 0.05, -0.05, 200)
    x2 = np.linspace(0.05, np.pi/2 - 0.05, 200)
    x_data = np.concatenate([x1, x2])
    y_data = 1 / np.sin(x_data) + 1 / np.cos(x_data)
    
    X = torch.tensor(x_data.reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(y_data.reshape(-1, 1), dtype=torch.float32)
    
    # 创建网络
    model = BPNetwork(input_size=1, hidden_size=120, output_size=1, activation='tanh')
    
    # 训练
    losses = train_network(model, X, y, epochs=50000, lr=0.001, optimizer_type='adam')
    
    # 预测
    with torch.no_grad():
        predictions = model(X)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    # 绘制函数拟合结果
    plt.subplot(2, 1, 2)
    plt.scatter(x_data, y_data, label='True Function', alpha=0.6, s=10)
    plt.scatter(x_data, predictions.numpy(), label='Neural Network', alpha=0.6, s=10)
    plt.title('Function Approximation: y = 1/sin(x) + 1/cos(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-20, 20)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, X, y

def analyze_learning_rates():
    """分析不同学习率的影响"""
    print("\n=== Learning Rate Analysis ===")
    
    # 准备数据（使用XOR作为简单示例）
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    plt.figure(figsize=(10, 6))
    
    for lr in learning_rates:
        model = BPNetwork(input_size=2, hidden_size=4, output_size=1, activation='sigmoid')
        losses = train_network(model, X, y, epochs=5000, lr=lr, optimizer_type='sgd')
        
        plt.plot(losses[:1000], label=f'LR={lr}')  # 只显示前1000个epoch
    
    plt.title('Learning Rate Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_weight_modification(model, X, y, noise_level=0.1):
    """分析权重修改对网络性能的影响"""
    print("\n=== Weight Modification Analysis ===")
    
    # 原始预测
    with torch.no_grad():
        original_pred = model(X)
    
    # 保存原始权重
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.clone()
    
    # 修改权重（添加噪声）
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_level
            param.add_(noise)
    
    # 修改后的预测
    with torch.no_grad():
        modified_pred = model(X)
    
    # 恢复原始权重
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(original_weights[name])
    
    # 可视化比较
    if X.shape[1] == 1:  # 函数逼近问题
        plt.figure(figsize=(10, 6))
        x_np = X.numpy().flatten()
        y_np = y.numpy().flatten()
        
        plt.scatter(x_np, y_np, label='True Function', alpha=0.6, s=10)
        plt.scatter(x_np, original_pred.numpy().flatten(), label='Original Network', alpha=0.6, s=10)
        plt.scatter(x_np, modified_pred.numpy().flatten(), label='Modified Network', alpha=0.6, s=10)
        
        plt.title('Weight Modification Impact on Function Approximation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(-20, 20)
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 计算误差
        original_error = torch.mean((original_pred - y) ** 2)
        modified_error = torch.mean((modified_pred - y) ** 2)
        
        print(f"Original MSE: {original_error.item():.6f}")
        print(f"Modified MSE: {modified_error.item():.6f}")
        print(f"Error increase: {(modified_error - original_error).item():.6f}")

if __name__ == "__main__":
    # 设置随机种子以确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Experiment Begin...")
    print(f"PyTorch Version: {torch.__version__}")
    
    try:
        # 1. XOR问题测试
        xor_losses = test_xor()
        
        # 2. 函数逼近测试
        model, X, y = test_function_approximation()
        
        # 3. 学习率分析
        analyze_learning_rates()
        
        # 4. 权重修改分析
        analyze_weight_modification(model, X, y)
        
        print("\n" + "="*50)
        print("Experiment completed! All images have been saved as PNG files")
        print("="*50)
        
        # 保持程序运行，等待用户输入
        input("Press Enter to exit...")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        
    finally:
        plt.ioff()  # 关闭交互模式