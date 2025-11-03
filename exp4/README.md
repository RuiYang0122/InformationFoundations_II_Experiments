# Selective Search 实验

## 推荐使用Typora打开本文件阅读相关须知

**课程**: 信息基础2 - 机器学习与深度学习  
**实验**: 实验四 - Selective Search（选择性搜索）

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 测试你的图像
将测试图像放入 `img` 文件夹，然后运行：
```bash
python batch_test.py
```

### 3. 查看结果
结果保存在 `outputs` 文件夹中。

---

## 📁 项目结构

```
exp4/
├── img/                          # 📂 放置测试图像
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
│
├── outputs/                      # 📂 输出结果（自动生成）
│   ├── 1_result.png
│   ├── 2_result.png
│   ├── 3_result.png
│   └── all_results_comparison.png
│
├── selective_search.py           # 核心算法实现
├── batch_test.py      # 测试脚本
└── requirements.txt              # 依赖包
```

---

## 💻 使用方法

### 方法一：批量测试（推荐）

测试 `img` 文件夹中的所有图像：
```bash
python batch_test.py
```

自定义输入/输出目录：
```bash
python batch_test.py --input 你的图像文件夹 --output 输出文件夹
```

### 方法二：基础演示

运行内置测试：
```bash
python selective_search.py
```

---

## 📊 运行示例

```bash
# 批量测试
python batch_test.py

# 输出：
# ============================================================
# Selective Search 批量测试
# ============================================================
# 
# 📁 输入目录: img
# 📁 输出目录: outputs
# 🖼️  找到 3 张图像
# 
# [1/3] 处理: 1.png
#    📏 调整尺寸: 1920x1080 → 600x338
#    🔍 执行 Selective Search...
#    ✓ 完成！生成 156 个候选区域
#    ⏱️  耗时: 8.32 秒
#    💾 保存结果: 1_result.png
# ...
```

---

## 🎯 主要功能

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| **batch_test.py** | 批量测试多张图像 | 测试你的图片 ⭐ |
| selective_search.py | 算法实现 + 演示 | 理解算法原理 |
| test_selective_search.py | 单张测试 + 参数对比 | 深入分析 |

---


## 🔧 参数调整

在 `selective_search.py` 或 `batch_test.py` 中修改参数：

```python
ss = SelectiveSearch(
    scale=1.0,      # 分割尺度（越大区域越少）
    sigma=0.8,      # 高斯平滑参数
    min_size=50     # 最小区域大小
)
```

---

## 📋 依赖包

- `numpy` - 数值计算
- `opencv-python` - 图像处理
- `scikit-image` - 图像分割
- `matplotlib` - 可视化
- `python-docx` - Word 文档生成

---

## 📚 算法原理

Selective Search 通过以下步骤生成候选区域：

1. **初始分割** - 使用 Felzenszwalb 算法分割图像
2. **特征提取** - 计算颜色、纹理特征
3. **相似度计算** - 综合颜色、纹理、尺寸、空间位置
4. **层次合并** - 迭代合并相似区域
5. **候选生成** - 记录所有合并过程产生的区域

---