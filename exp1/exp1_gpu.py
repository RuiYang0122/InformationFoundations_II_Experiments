import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleNN(nn.Module):
    """简单的三层神经网络"""
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # 选择激活函数
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

def train_model(model, X, y, epochs=10000, lr=0.01, optimizer_type='adam'):
    """训练模型"""
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
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
        
        if loss.item() < 1e-5:
            print(f'Early stopping: loss reached threshold after {epoch} epochs')
            break
    
    return losses

def xor_experiment():
    """XOR问题实验"""
    print("=" * 50)
    print("XOR Problem Experiment")
    print("=" * 50)
    
    # 准备XOR数据
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)
    
    # 创建和训练模型
    model = SimpleNN(2, 8, 1, 'sigmoid').to(device)
    losses = train_model(model, X, y, epochs=10000, lr=0.2)
    
    # 测试结果
    with torch.no_grad():
        predictions = model(X)
        print("\nXOR Prediction Results:")
        for i in range(len(X)):
            print(f"Input: {X[i].cpu().numpy()} -> Output: {predictions[i].item():.4f}, Target: {y[i].item()}")
    
    return model, losses

def function_approximation_experiment():
    """函数逼近实验: y = 1/sin(x) + 1/cos(x)"""
    print("=" * 50)
    print("Function Approximation Experiment: y = 1/sin(x) + 1/cos(x)")
    print("=" * 50)
    
    # 准备数据
    x1 = np.linspace(-np.pi/2 + 0.05, -0.05, 200)
    x2 = np.linspace(0.05, np.pi/2 - 0.05, 200)
    X_np = np.concatenate([x1, x2]).reshape(-1, 1)
    y_np = (1 / np.sin(X_np) + 1 / np.cos(X_np)).reshape(-1, 1)
    
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    y = torch.tensor(y_np, dtype=torch.float32).to(device)
    
    # 创建和训练模型
    model = SimpleNN(1, 120, 1, 'tanh').to(device)
    losses = train_model(model, X, y, epochs=50000, lr=0.001)
    
    # 预测和可视化
    with torch.no_grad():
        predictions = model(X)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(X_np, y_np, alpha=0.6, s=10, label='True Function', color='blue')
    plt.scatter(X_np, predictions.cpu().numpy(), alpha=0.6, s=10, label='Neural Network Prediction', color='red')
    plt.ylim(-20, 20)
    plt.title('Function Approximation Results')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    
    return model, losses, X, y, predictions

def learning_rate_experiment():
    """学习率影响实验"""
    print("=" * 50)
    print("Learning Rate Impact Experiment")
    print("=" * 50)
    
    # 准备数据（使用XOR作为简单示例）
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    colors = ['blue', 'green', 'red', 'purple']
    
    plt.subplot(2, 2, 3)
    
    for lr, color in zip(learning_rates, colors):
        model = SimpleNN(2, 4, 1, 'sigmoid').to(device)
        losses = train_model(model, X, y, epochs=2000, lr=lr)
        plt.plot(losses[:1000], label=f'LR={lr}', color=color)  # 只显示前1000轮
    
    plt.title('Convergence with Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

def weight_modification_experiment(model, X, y):
    """权重修改实验"""
    print("=" * 50)
    print("Weight Modification Experiment")
    print("=" * 50)
    
    # 原始预测
    with torch.no_grad():
        original_pred = model(X)
    
    # 修改部分权重
    with torch.no_grad():
        for param in model.parameters():
            # 随机修改10%的权重
            noise = torch.randn_like(param) * 0.1
            mask = torch.rand_like(param) < 0.1  # 只修改10%的权重
            param.data += noise * mask
    
    # 修改后的预测
    with torch.no_grad():
        modified_pred = model(X)
    
    # 可视化对比
    plt.subplot(2, 2, 4)
    X_np = X.cpu().numpy()
    plt.scatter(X_np, y.cpu().numpy(), alpha=0.6, s=10, label='True Values', color='blue')
    plt.scatter(X_np, original_pred.cpu().numpy(), alpha=0.6, s=10, label='Prediction Before Modification', color='green')
    plt.scatter(X_np, modified_pred.cpu().numpy(), alpha=0.6, s=10, label='Prediction After Modification', color='red')
    plt.ylim(-20, 20)
    plt.title('Comparison Before and After Weight Modification')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    # 计算预测差异
    mse_original = torch.mean((original_pred - y) ** 2).item()
    mse_modified = torch.mean((modified_pred - y) ** 2).item()
    
    print(f"MSE before modification: {mse_original:.6f}")
    print(f"MSE after modification: {mse_modified:.6f}")
    print(f"Performance degradation: {((mse_modified - mse_original) / mse_original * 100):.2f}%")

def main():
    """主实验函数"""
    print("Starting neural network function approximation experiment...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 实验1: XOR问题
    xor_model, xor_losses = xor_experiment()
    
    # 实验2: 函数逼近
    func_model, func_losses, X, y, predictions = function_approximation_experiment()
    
    # 实验3: 学习率影响
    learning_rate_experiment()
    
    # 实验4: 权重修改实验
    weight_modification_experiment(func_model, X, y)
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 50)
    print("All experiments completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()