# LeNet-5 MNIST 实验

## 推荐使用Typora打开并阅读本readme文件

## 文件结构
- `lenet5.py`  
  实验的核心代码，定义并训练 LeNet-5 网络（支持 Sigmoid 与 ReLU 激活函数版本）。
- `lenet5_weights.pth`  
  训练完成后的模型权重文件，可以用于模型加载与推理。
- `report.docx`  
  实验报告文档，包含知识梳理、代码构建过程、实验结果及总结。
- `test.py`  
  用于测试模型的脚本，可以加载训练好的权重文件对手写数字进行预测。
- `test_img/`  
  存放测试图片（手写数字图像），可用于 `test.py` 进行推理验证。

## 环境依赖
- Python 3.x
- PyTorch
- torchvision
- d2l (Dive into Deep Learning)
- matplotlib

安装依赖：
```bash
pip install torch torchvision matplotlib d2l
```

## 运行方法

### 1. 训练模型
在命令行中运行：
```bash
python lenet5.py
```
训练完成后会生成权重文件 `lenet5_weights.pth`。

### 2. 测试模型
将需要测试的图片放入 `test_img/` 文件夹，然后运行：
```bash
python test.py
```
程序会加载 `lenet5_weights.pth` 并输出预测结果。

## 实验说明
- 本实验基于 **LeNet-5 网络结构**，在 **MNIST 数据集** 上进行训练与测试。
- 对比了 **Sigmoid** 和 **ReLU** 两种激活函数在训练效果上的差异。
- 训练结果表明，合适的学习率对 ReLU 激活函数的训练效果提升显著。
