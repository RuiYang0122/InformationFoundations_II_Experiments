# 神经网络函数逼近实验

## 实验描述
本实验使用PyTorch实现BP神经网络，完成了XOR问题和函数 $y = \frac{1}{\sin(x)} + \frac{1}{\cos(x)}$ 的逼近，并探究了学习率与权重修改对网络性能的影响。

## 环境依赖
### 硬件要求
- 推荐使用带有NVIDIA GPU的电脑（实验在CUDA 12.6环境下完成）
- CPU也可运行，但训练速度较慢

### 软件环境
- Python 3.9+
- 依赖库详见 `requirements.txt`

## 安装与运行
1.  **创建并激活Conda环境（推荐）**
    ```bash
    conda create -n nn_exp python=3.9
    conda activate nn_exp
    ```

2.  **安装依赖**
    ```bash
    # 如果拥有CUDA环境（Linux/Windows）:
    pip install -r requirements.txt

    # 如果只有CPU（MacOS或无需GPU）:
    # 请先卸载原有的torch（如果有），然后安装CPU版本
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
    pip install numpy matplotlib
    ```

3.  **运行实验**
    在命令行中直接运行主程序，所有实验将依次执行并显示结果图表。
    ```bash
    python exp1_gpu.py
    ```

## 文件说明
- `exp1_gpu.py`: 主程序文件，包含GPU版本实验代码。
- `exp1_cpu.py`: 主程序文件，包含CPU版本实验代码。
- `requirements.txt`: 项目依赖库列表。
- `report.pdf`: 实验报告，包含实验分析、结果与结论。
- `README.md`: 本说明文件。

