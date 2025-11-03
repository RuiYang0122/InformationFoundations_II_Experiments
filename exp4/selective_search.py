"""
Selective Search算法实现
实验四：选择性搜索（Selective Search）
"""

import numpy as np
import cv2
import os
from skimage import segmentation, color
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class SelectiveSearch:
    """
    Selective Search实现类
    用于从图像中生成候选区域（Region Proposals）
    """
    
    def __init__(self, scale=1.0, sigma=0.8, min_size=50):
        """
        初始化Selective Search参数
        
        Args:
            scale: 图像分割的尺度参数
            sigma: 高斯模糊的标准差
            min_size: 最小区域大小
        """
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
    
    def _generate_segments(self, image):
        """
        使用图像分割算法生成初始区域
        使用Felzenszwalb's efficient graph based segmentation
        
        Args:
            image: 输入图像
            
        Returns:
            segments: 分割后的区域标签图
        """
        # 使用Felzenszwalb算法进行初始分割
        segments = segmentation.felzenszwalb(
            img_as_float(image), 
            scale=self.scale * 100, 
            sigma=self.sigma, 
            min_size=self.min_size
        )
        return segments
    
    def _calc_color_hist(self, image, mask):
        """
        计算区域的颜色直方图
        
        Args:
            image: 输入图像
            mask: 区域掩码
            
        Returns:
            hist: 颜色直方图（归一化）
        """
        # 为每个通道计算25 bins的直方图
        bins = 25
        hist = np.array([])
        
        for channel in range(image.shape[2]):
            channel_hist = np.histogram(
                image[:, :, channel][mask], 
                bins=bins, 
                range=(0.0, 1.0)
            )[0]
            hist = np.concatenate([hist, channel_hist])
        
        # 归一化
        hist = hist / np.sum(hist)
        return hist
    
    def _calc_texture_gradient(self, image, mask):
        """
        计算区域的纹理特征（使用梯度）
        
        Args:
            image: 输入图像
            mask: 区域掩码
            
        Returns:
            texture_hist: 纹理直方图
        """
        # 转换为灰度图
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # 计算8个方向的梯度
        texture_hist = np.array([])
        for angle in range(8):
            # 使用Sobel算子
            dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 计算梯度方向
            gradient = np.sqrt(dx**2 + dy**2)
            
            # 只考虑当前区域的梯度
            masked_gradient = gradient[mask]
            
            # 计算直方图
            hist = np.histogram(masked_gradient, bins=10)[0]
            texture_hist = np.concatenate([texture_hist, hist])
        
        # 归一化
        texture_hist = texture_hist / np.sum(texture_hist)
        return texture_hist
    
    def _calc_similarity(self, region1, region2, image):
        """
        计算两个区域的相似度
        综合考虑颜色、纹理、尺寸和重叠情况
        
        Args:
            region1, region2: 区域字典
            image: 输入图像
            
        Returns:
            similarity: 相似度分数
        """
        # 颜色相似度（使用直方图交集）
        color_sim = np.minimum(region1['color_hist'], region2['color_hist']).sum()
        
        # 纹理相似度
        texture_sim = np.minimum(region1['texture_hist'], region2['texture_hist']).sum()
        
        # 尺寸相似度（鼓励合并小区域）
        size1 = region1['size']
        size2 = region2['size']
        size_sim = 1.0 - (size1 + size2) / image.size
        
        # 重叠相似度（鼓励合并相邻区域）
        bbox1 = region1['bbox']
        bbox2 = region2['bbox']
        
        # 计算外接矩形
        min_x = min(bbox1[0], bbox2[0])
        min_y = min(bbox1[1], bbox2[1])
        max_x = max(bbox1[2], bbox2[2])
        max_y = max(bbox1[3], bbox2[3])
        
        bbox_size = (max_x - min_x) * (max_y - min_y)
        fill_sim = 1.0 - (bbox_size - size1 - size2) / image.size
        
        # 综合相似度
        similarity = color_sim + texture_sim + size_sim + fill_sim
        
        return similarity
    
    def _extract_regions(self, image, segments):
        """
        从分割结果中提取区域信息
        
        Args:
            image: 输入图像
            segments: 分割标签图
            
        Returns:
            regions: 区域字典列表
        """
        regions = {}
        
        # 为每个分割区域计算特征
        for segment_id in np.unique(segments):
            # 创建掩码
            mask = segments == segment_id
            
            # 跳过太小的区域
            if np.sum(mask) < self.min_size:
                continue
            
            # 计算边界框
            y_coords, x_coords = np.where(mask)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            # 存储区域信息
            regions[segment_id] = {
                'mask': mask,
                'bbox': (min_x, min_y, max_x, max_y),
                'size': np.sum(mask),
                'color_hist': self._calc_color_hist(image, mask),
                'texture_hist': self._calc_texture_gradient(image, mask),
                'labels': [segment_id]
            }
        
        return regions
    
    def _merge_regions(self, region1, region2):
        """
        合并两个区域
        
        Args:
            region1, region2: 要合并的区域
            
        Returns:
            merged_region: 合并后的区域
        """
        merged_region = {
            'mask': region1['mask'] | region2['mask'],
            'size': region1['size'] + region2['size'],
            'labels': region1['labels'] + region2['labels']
        }
        
        # 更新边界框
        merged_region['bbox'] = (
            min(region1['bbox'][0], region2['bbox'][0]),
            min(region1['bbox'][1], region2['bbox'][1]),
            max(region1['bbox'][2], region2['bbox'][2]),
            max(region1['bbox'][3], region2['bbox'][3])
        )
        
        # 更新颜色直方图（加权平均）
        total_size = merged_region['size']
        merged_region['color_hist'] = (
            region1['color_hist'] * region1['size'] + 
            region2['color_hist'] * region2['size']
        ) / total_size
        
        # 更新纹理直方图
        merged_region['texture_hist'] = (
            region1['texture_hist'] * region1['size'] + 
            region2['texture_hist'] * region2['size']
        ) / total_size
        
        return merged_region
    
    def _get_neighbors(self, regions):
        """
        获取相邻区域对
        
        Args:
            regions: 区域字典
            
        Returns:
            neighbors: 相邻区域对的集合
        """
        neighbors = []
        
        # 简化实现：检查边界框是否相邻
        region_ids = list(regions.keys())
        for i, id1 in enumerate(region_ids):
            for id2 in region_ids[i+1:]:
                bbox1 = regions[id1]['bbox']
                bbox2 = regions[id2]['bbox']
                
                # 检查是否相邻（边界框有重叠或相邻）
                if not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
                       bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
                    neighbors.append((id1, id2))
        
        return neighbors
    
    def hierarchical_search(self, image):
        """
        执行层次化分组算法（Hierarchical Grouping）
        
        Args:
            image: 输入图像（RGB）
            
        Returns:
            candidate_boxes: 候选区域列表
        """
        # 将图像转换为float
        image_float = img_as_float(image)
        
        # 1. 生成初始分割
        print("生成初始分割区域...")
        segments = self._generate_segments(image_float)
        print(f"初始区域数量: {len(np.unique(segments))}")
        
        # 2. 提取区域特征
        print("提取区域特征...")
        regions = self._extract_regions(image_float, segments)
        
        # 3. 初始化候选区域列表
        candidate_boxes = []
        for region in regions.values():
            candidate_boxes.append(region['bbox'])
        
        # 4. 迭代合并相似区域
        print("开始迭代合并...")
        iteration = 0
        while len(regions) > 1:
            iteration += 1
            if iteration % 10 == 0:
                print(f"迭代 {iteration}, 当前区域数: {len(regions)}")
            
            # 获取相邻区域
            neighbors = self._get_neighbors(regions)
            
            if not neighbors:
                break
            
            # 计算所有相邻区域对的相似度
            similarities = {}
            for id1, id2 in neighbors:
                sim = self._calc_similarity(
                    regions[id1], 
                    regions[id2], 
                    image_float
                )
                similarities[(id1, id2)] = sim
            
            # 找到最相似的区域对
            max_sim_pair = max(similarities.items(), key=lambda x: x[1])
            id1, id2 = max_sim_pair[0]
            
            # 合并区域
            merged_region = self._merge_regions(regions[id1], regions[id2])
            
            # 更新区域列表
            new_id = max(regions.keys()) + 1
            regions[new_id] = merged_region
            
            # 添加到候选框列表
            candidate_boxes.append(merged_region['bbox'])
            
            # 删除已合并的区域
            del regions[id1]
            del regions[id2]
            
            # 限制迭代次数避免过长时间运行
            if iteration > 100:
                print("达到最大迭代次数")
                break
        
        print(f"完成！总共生成 {len(candidate_boxes)} 个候选区域")
        return candidate_boxes
    
    def visualize_results(self, image, boxes, max_boxes=100):
        """
        可视化Selective Search结果
        
        Args:
            image: 原始图像
            boxes: 候选框列表
            max_boxes: 最多显示的框数量
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # 随机选择一部分框进行显示
        if len(boxes) > max_boxes:
            import random
            boxes_to_show = random.sample(boxes, max_boxes)
        else:
            boxes_to_show = boxes
        
        # 绘制候选框
        for bbox in boxes_to_show:
            x, y, x_max, y_max = bbox
            rect = mpatches.Rectangle(
                (x, y), x_max - x, y_max - y,
                fill=False, 
                edgecolor='red', 
                linewidth=1
            )
            ax.add_patch(rect)
        
        ax.set_title(f'Selective Search Results (showing {len(boxes_to_show)} of {len(boxes)} boxes)')
        ax.axis('off')
        
        return fig


def main():
    """
    主函数：演示Selective Search的使用
    """
    print("=" * 60)
    print("Selective Search 实验演示")
    print("=" * 60)
    
    # 读取测试图像
    # 这里使用OpenCV生成一个测试图像
    print("\n1. 加载测试图像...")
    
    # 创建一个简单的测试图像
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # 添加一些彩色矩形作为"物体"
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.rectangle(image, (200, 100), (350, 250), (0, 255, 0), -1)
    cv2.rectangle(image, (400, 200), (550, 350), (0, 0, 255), -1)
    cv2.circle(image, (300, 300), 50, (255, 255, 0), -1)
    
    # 转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"图像尺寸: {image.shape}")
    
    # 2. 创建Selective Search实例
    print("\n2. 初始化Selective Search...")
    ss = SelectiveSearch(scale=1.0, sigma=0.8, min_size=20)
    
    # 3. 执行Selective Search
    print("\n3. 执行Selective Search算法...")
    candidate_boxes = ss.hierarchical_search(image)
    
    # 4. 显示结果
    print("\n4. 可视化结果...")
    fig = ss.visualize_results(image, candidate_boxes, max_boxes=50)
    
    # 保存结果
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'selective_search_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {output_path}")
    
    # 5. 统计信息
    print("\n" + "=" * 60)
    print("实验结果统计")
    print("=" * 60)
    print(f"生成的候选区域总数: {len(candidate_boxes)}")
    print(f"候选框尺寸范围:")
    
    areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in candidate_boxes]
    print(f"  - 最小面积: {min(areas):.0f} 像素")
    print(f"  - 最大面积: {max(areas):.0f} 像素")
    print(f"  - 平均面积: {np.mean(areas):.0f} 像素")
    
    plt.close()


if __name__ == "__main__":
    main()