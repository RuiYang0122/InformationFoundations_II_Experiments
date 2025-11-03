"""
批量测试脚本 - 输出详细统计数据
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from selective_search import SelectiveSearch
import time


class DetailedSelectiveSearch(SelectiveSearch):
    """扩展的 Selective Search，记录详细统计信息"""
    
    def __init__(self, scale=1.0, sigma=0.8, min_size=50):
        super().__init__(scale, sigma, min_size)
        self.stats = {}
    
    def hierarchical_search_with_stats(self, image):
        """带统计信息的层次化搜索"""
        
        # 1. 生成初始分割
        print("      - 初始分割...")
        segments = self._generate_segments(image)
        initial_regions = len(np.unique(segments))
        self.stats['initial_segments'] = initial_regions
        print(f"        初始分割块数: {initial_regions}")
        
        # 2. 提取区域特征
        print("      - 提取特征...")
        regions = self._extract_regions(image, segments)
        valid_regions = len(regions)
        print(f"        有效区域数: {valid_regions}")
        
        # 3. 初始化候选区域列表
        candidate_boxes = []
        for region in regions.values():
            candidate_boxes.append(region['bbox'])
        
        # 4. 迭代合并
        print("      - 迭代合并...")
        iteration = 0
        merge_count = 0
        
        while len(regions) > 1:
            iteration += 1
            
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
                    image
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
            
            merge_count += 1
            
            # 限制迭代次数
            if iteration > 100:
                break
        
        self.stats['merge_count'] = merge_count
        self.stats['final_regions'] = len(regions)
        self.stats['total_candidates'] = len(candidate_boxes)
        
        print(f"        合并次数: {merge_count}")
        print(f"        最终区域数: {len(regions)}")
        print(f"        候选框总数: {len(candidate_boxes)}")
        
        return candidate_boxes


def calculate_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def filter_boxes_by_size(boxes, min_area=500, max_area=50000):
    """根据面积筛选候选框"""
    filtered = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        area = (x_max - x_min) * (y_max - y_min)
        if min_area <= area <= max_area:
            filtered.append(box)
    return filtered


def nms(boxes, iou_threshold=0.5):
    """非极大值抑制 (NMS)"""
    if len(boxes) == 0:
        return []
    
    # 计算每个框的面积
    areas = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        area = (x_max - x_min) * (y_max - y_min)
        areas.append(area)
    
    # 按面积排序（大的优先）
    indices = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    
    keep = []
    while len(indices) > 0:
        # 选择面积最大的框
        current = indices[0]
        keep.append(current)
        
        # 计算与其他框的 IoU
        remaining = []
        for i in indices[1:]:
            iou = calculate_iou(boxes[current], boxes[i])
            if iou < iou_threshold:
                remaining.append(i)
        
        indices = remaining
    
    return [boxes[i] for i in keep]


def batch_test_with_stats(input_dir='img', output_dir='outputs'):
    """
    带详细统计的批量测试
    """
    
    print("=" * 70)
    print("Selective Search 详细统计测试")
    print("=" * 70)
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"\n❌ 错误：找不到目录 '{input_dir}'")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(input_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"\n❌ 错误：在 '{input_dir}' 中没有找到图像文件")
        return
    
    print(f"\n📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🖼️  找到 {len(image_files)} 张图像")
    print()
    
    # 创建增强版 Selective Search 实例
    ss = DetailedSelectiveSearch(scale=1.0, sigma=0.8, min_size=50)
    
    # 存储所有结果
    all_results = []
    all_stats = []
    
    # 逐个处理图像
    for idx, filename in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 处理: {filename}")
        
        # 读取图像
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"   ❌ 无法读取图像，跳过")
            continue
        
        # 转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小
        max_size = 600
        h, w = image.shape[:2]
        original_size = (w, h)
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            print(f"   📏 调整尺寸: {w}x{h} → {new_w}x{new_h}")
        
        # 执行 Selective Search（带统计）
        print(f"   🔍 执行 Selective Search（详细模式）...")
        start_time = time.time()
        
        try:
            candidate_boxes = ss.hierarchical_search_with_stats(image)
            elapsed_time = time.time() - start_time
            
            # 获取统计信息
            stats = ss.stats.copy()
            stats['elapsed_time'] = elapsed_time
            
            print(f"   ✓ 完成！")
            print()
            print("   📊 详细统计:")
            print(f"      初始分割块数: {stats['initial_segments']}")
            print(f"      合并次数: {stats['merge_count']}")
            print(f"      最终区域数: {stats['final_regions']}")
            print(f"      候选框总数: {stats['total_candidates']}")
            print(f"      处理时间: {elapsed_time:.2f} 秒")
            
            # 筛选候选框
            print()
            print("   🔧 候选框筛选:")
            
            # 1. 按面积筛选
            filtered_boxes = filter_boxes_by_size(
                candidate_boxes,
                min_area=500,
                max_area=image.shape[0] * image.shape[1] * 0.8
            )
            print(f"      面积筛选后: {len(filtered_boxes)} 个")
            stats['filtered_by_area'] = len(filtered_boxes)
            
            # 2. NMS
            iou_threshold = 0.5
            final_boxes = nms(filtered_boxes, iou_threshold=iou_threshold)
            print(f"      NMS后 (IoU={iou_threshold}): {len(final_boxes)} 个")
            stats['final_boxes_after_nms'] = len(final_boxes)
            stats['iou_threshold'] = iou_threshold
            
            # 可视化结果（使用筛选后的框）
            display_boxes = final_boxes[:30]  # 最多显示30个
            stats['displayed_boxes'] = len(display_boxes)
            
            fig = ss.visualize_results(image, display_boxes, max_boxes=len(display_boxes))
            
            # 保存结果
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_result.png"
            output_path = os.path.join(output_dir, output_filename)
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 保存结果: {output_filename}")
            
            # 记录结果
            all_results.append({
                'filename': filename,
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'output': output_filename
            })
            
            all_stats.append({
                'filename': filename,
                **stats
            })
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # 生成详细统计报告
    print("=" * 70)
    print("处理完成！详细统计报告")
    print("=" * 70)
    print()
    
    if all_stats:
        # 创建统计表格
        print(f"{'图像':<15} {'初始块':<8} {'合并次数':<8} {'候选框':<8} {'NMS后':<8} {'耗时(秒)':<10}")
        print("-" * 70)
        
        for stat in all_stats:
            print(f"{stat['filename']:<15} "
                  f"{stat['initial_segments']:<8} "
                  f"{stat['merge_count']:<8} "
                  f"{stat['total_candidates']:<8} "
                  f"{stat['final_boxes_after_nms']:<8} "
                  f"{stat['elapsed_time']:<10.2f}")
        
        print()
        
        # 计算平均值
        avg_initial = np.mean([s['initial_segments'] for s in all_stats])
        avg_merge = np.mean([s['merge_count'] for s in all_stats])
        avg_candidates = np.mean([s['total_candidates'] for s in all_stats])
        avg_final = np.mean([s['final_boxes_after_nms'] for s in all_stats])
        avg_time = np.mean([s['elapsed_time'] for s in all_stats])
        
        print("📊 平均统计:")
        print(f"   - 平均初始分割块数: {avg_initial:.1f}")
        print(f"   - 平均合并次数: {avg_merge:.1f}")
        print(f"   - 平均候选框数: {avg_candidates:.1f}")
        print(f"   - 平均NMS后候选框: {avg_final:.1f}")
        print(f"   - 平均处理时间: {avg_time:.2f} 秒")
        print(f"   - IoU 阈值: {all_stats[0]['iou_threshold']}")
        
        # 保存统计数据到文件
        stats_file = os.path.join(output_dir, 'statistics.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Selective Search 详细统计报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("实验参数:\n")
            f.write(f"  - Scale: 1.0\n")
            f.write(f"  - Sigma: 0.8\n")
            f.write(f"  - Min Size: 50\n")
            f.write(f"  - IoU Threshold: {all_stats[0]['iou_threshold']}\n\n")
            
            f.write("每张图像的详细统计:\n\n")
            for stat in all_stats:
                f.write(f"图像: {stat['filename']}\n")
                f.write(f"  初始分割块数: {stat['initial_segments']}\n")
                f.write(f"  合并目标区域数: {stat['merge_count']}\n")
                f.write(f"  输出候选框数: {stat['displayed_boxes']}\n")
                f.write(f"  IoU 阈值: {stat['iou_threshold']}\n")
                f.write(f"  处理时间: {stat['elapsed_time']:.2f} 秒\n\n")
            
            f.write("\n平均统计:\n")
            f.write(f"  平均初始分割块数: {avg_initial:.1f}\n")
            f.write(f"  平均合并次数: {avg_merge:.1f}\n")
            f.write(f"  平均候选框数: {avg_candidates:.1f}\n")
            f.write(f"  平均NMS后候选框: {avg_final:.1f}\n")
            f.write(f"  平均处理时间: {avg_time:.2f} 秒\n")
        
        print()
        print(f"✓ 统计数据已保存到: {stats_file}")
        
    print()
    print("=" * 70)


def main():
    """主函数"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='带详细统计的批量测试')
    parser.add_argument('--input', default='img', help='输入图像文件夹')
    parser.add_argument('--output', default='outputs', help='输出文件夹')
    
    args = parser.parse_args()
    
    batch_test_with_stats(args.input, args.output)


if __name__ == "__main__":
    main()