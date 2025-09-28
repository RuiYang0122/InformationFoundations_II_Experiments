import os
import torch
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
# 设置标准输出为UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体显示 - 使用Windows系统更常见的字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

# 定义网络结构（与训练时保持一致）
def create_lenet_model():
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
    return net

# 预处理图片函数
def preprocess_image(image_path):
    # 打开图片并转换为灰度
    img = Image.open(image_path).convert('L')
    # 调整大小为28x28
    img = img.resize((28, 28), Image.LANCZOS)
    # 转换为numpy数组
    img_array = np.array(img)
    # 反转颜色（使数字为黑色，背景为白色）
    img_array = 255 - img_array
    # 归一化
    img_array = img_array / 255.0
    # 转换为张量并添加维度
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor, img_array

# 加载模型
def load_model(model_path, device):
    net = create_lenet_model()
    # 加载模型权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()  # 设置为评估模式
    return net

# 识别手写数字
def recognize_digits(image_dir, model_path):
    # 确定设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    net = load_model(model_path, device)
    
    # 获取test_img目录下的所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) 
                  if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f'在{image_dir}目录下未找到图片文件')
        return
    
    # 预测并显示结果
    plt.figure(figsize=(12, 8))
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        try:
            # 预处理图片
            img_tensor, original_array = preprocess_image(img_path)
            img_tensor = img_tensor.to(device)
            
            # 进行预测
            with torch.no_grad():
                output = net(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # 打印完整的概率分布（保留4位小数）
                prob_dist = [f'{p:.4f}' for p in probabilities[0].tolist()]
                print(f'图片 {img_file}: 完整概率分布: {prob_dist}')
            
            # 显示图片和预测结果（置信度保留4位小数）
            plt.subplot(len(image_files) // 3 + 1, 3, i + 1)
            plt.imshow(original_array, cmap='gray')
            plt.title(f'预测: {predicted_class}, 置信度: {confidence:.4f}')
            plt.axis('off')
            print(f'图片 {img_file}: 预测数字为 {predicted_class}，置信度为 {confidence:.4f}')
        except Exception as e:
            print(f'处理图片 {img_file} 时出错: {str(e)}')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 设置图片目录和模型路径
    IMAGE_DIR = r'D:\Information_HW\Experiment_code\exp2\test_img'
    # 请根据实际保存的模型文件修改路径
    MODEL_PATH = r'D:\Information_HW\Experiment_code\exp2\lenet5_weights.pth'
    
    # 检查目录是否存在
    if not os.path.exists(IMAGE_DIR):
        print(f'图片目录 {IMAGE_DIR} 不存在')
    elif not os.path.exists(MODEL_PATH):
        print(f'模型文件 {MODEL_PATH} 不存在')
    else:
        recognize_digits(IMAGE_DIR, MODEL_PATH)