#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从浮选泡沫图像中提取形态学特征

此脚本提供一个简单的接口，用于输入一张浮选图片并提取其形态学特征。
特征包括灰度均值、大小均值、色调、大小标准差、红色均值、绿色均值、蓝色均值、长宽比、
承载率、红色相对分量、大小峰度、大小偏斜度、粗度、非均匀度、大小分布、
低频能量和高频能量等共17维特征。
"""

import numpy as np
import os
import sys
import cv2
import argparse
from skimage import measure, morphology
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter, convolve
from sklearn.cluster import KMeans
import json
from scipy import stats

# 尝试导入FCM相关库
try:
    import skfuzzy as fuzz
    HAS_FCM = True
except ImportError:
    HAS_FCM = False
    print("警告: 未安装scikit-fuzzy，将使用KMeans作为FCM替代方案")


# ======================== 形态学特征提取器 ======================== #
class MorphologicalFeatureExtractor:
    """
    形态学特征提取器，从浮选泡沫图像中提取用户指定的17维特征
    """
    
    def __init__(self):
        self.feature_dim = 17  # 提取17维形态学特征
    
    def extract_features_from_image(self, image_path):
        """
        从单张图像中提取形态学特征
        
        参数:
            image_path (str): 图像文件路径
        
        返回:
            np.ndarray: 形态学特征向量 (17维)
        """
        try:
            # 使用完整分割流程
            orig, illum_corrected, enhanced, markers, segmented = full_flotation_foam_segmentation(image_path)
            
            # 提取区域属性
            props = measure.regionprops(segmented)
            
            # 初始化特征向量
            features = np.zeros(self.feature_dim)
            
            if len(props) > 0:
                # 过滤掉背景区域和面积小于20的区域
                valid_props = [prop for prop in props if prop.label != 0 and prop.area >= 20]
                
                if len(valid_props) > 0:
                    # 1. 灰度均值
                    gray_img = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
                    features[0] = np.mean(gray_img)
                    
                    # 2. 大小均值 (气泡面积均值)
                    areas = [prop.area for prop in valid_props]
                    features[1] = np.mean(areas)
                    
                    # 3. 色调 (HSV空间中H通道的均值)
                    hsv_img = cv2.cvtColor(orig, cv2.COLOR_RGB2HSV)
                    features[2] = np.mean(hsv_img[:, :, 0]) / 180.0  # 归一化到[0, 1]
                    
                    # 4. 大小标准差 (气泡面积标准差)
                    features[3] = np.std(areas)
                    
                    # 5-7. 红色均值、绿色均值、蓝色均值
                    features[4] = np.mean(orig[:, :, 0])  # R通道均值
                    features[5] = np.mean(orig[:, :, 1])  # G通道均值
                    features[6] = np.mean(orig[:, :, 2])  # B通道均值
                    
                    # 8. 长宽比 (气泡的平均宽高比)
                    aspect_ratios = [prop.axis_major_length / prop.axis_minor_length 
                                     for prop in valid_props if prop.axis_minor_length > 0]
                    features[7] = np.mean(aspect_ratios) if aspect_ratios else 1.0
                    
                    # 9. 承载率 (气泡覆盖面积比例)
                    total_bubble_area = sum(areas)
                    image_area = orig.shape[0] * orig.shape[1]
                    features[8] = total_bubble_area / image_area
                    
                    # 10. 红色相对分量 (R/(R+G+B))
                    total_rgb = features[4] + features[5] + features[6]
                    features[9] = features[4] / total_rgb if total_rgb > 0 else 0.0
                    
                    # 11. 大小峰度 (气泡面积分布的峰度)
                    features[10] = stats.kurtosis(areas)
                    
                    # 12. 大小偏斜度 (气泡面积分布的偏斜度)
                    features[11] = stats.skew(areas)
                    
                    # 13. 粗度 (基于灰度共生矩阵的纹理特征)
                    features[12] = self._calculate_coarseness(gray_img)
                    
                    # 14. 非均匀度 (气泡大小分布的变异系数)
                    features[13] = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0.0
                    
                    # 15. 大小分布 (气泡大小的熵)
                    features[14] = self._calculate_size_entropy(areas)
                    
                    # 16-17. 低频能量和高频能量 (使用傅里叶变换)
                    low_freq_energy, high_freq_energy = self._calculate_frequency_energy(enhanced)
                    features[15] = low_freq_energy
                    features[16] = high_freq_energy
            
            return features
            
        except Exception as e:
            print(f"提取形态学特征时出错: {e}")
            # 返回零向量作为默认值
            return np.zeros(self.feature_dim)
    
    def _calculate_coarseness(self, gray_img):
        """计算图像粗度特征"""
        # 使用高斯差分计算粗度
        g1 = gaussian_filter(gray_img, sigma=1)
        g2 = gaussian_filter(gray_img, sigma=2)
        diff = np.abs(g2 - g1)
        return np.mean(diff)
    
    def _calculate_size_entropy(self, areas):
        """计算气泡大小分布的熵"""
        # 计算面积直方图
        hist, _ = np.histogram(areas, bins=10, density=True)
        # 移除零值并计算熵
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    
    def _calculate_frequency_energy(self, enhanced_img):
        """计算图像的低频能量和高频能量"""
        # 傅里叶变换
        f_transform = np.fft.fft2(enhanced_img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 计算低频和高频区域
        rows, cols = enhanced_img.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建低通和高通掩码
        mask_low = np.zeros((rows, cols), np.uint8)
        mask_high = np.zeros((rows, cols), np.uint8)
        
        # 低频区域 (中心区域)
        radius_low = min(crow, ccol) // 4
        cv2.circle(mask_low, (ccol, crow), radius_low, 1, -1)
        
        # 高频区域 (外围区域)
        radius_high = min(crow, ccol) // 2
        cv2.circle(mask_high, (ccol, crow), radius_high, 1, -1)
        mask_high = cv2.bitwise_not(mask_high) | (1 - mask_low)
        
        # 计算能量
        low_freq_energy = np.sum(magnitude_spectrum * mask_low) / (rows * cols)
        high_freq_energy = np.sum(magnitude_spectrum * mask_high) / (rows * cols)
        
        return low_freq_energy, high_freq_energy


# ======================== 图像预处理与分割 ======================== #
def apply_msrcr(img):
    """
    实现多尺度Retinex with Color Restoration (MSRCR)光照校正
    """
    img = img.astype(np.float32) + 1.0  # 避免log(0)
    
    # 计算多个尺度的SSR并求平均
    sigma_list = [15, 80, 250]  # 不同尺度的高斯核
    msr = np.zeros_like(img)
    for sigma in sigma_list:
        # 高斯滤波
        blurred = gaussian_filter(img, sigma=sigma, mode='reflect')
        # SSR: log(I) - log(F*I)
        ssr = np.log(img) - np.log(blurred)
        msr += ssr
    msr /= len(sigma_list)
    
    # 色彩恢复
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restore = 46 * (np.log(125 * img) - np.log(img_sum))
    msrcr = 1.0 * (msr * color_restore) + 0.0
    
    return msrcr


def stretch_to_0_255(img):
    """
    将图像灰度值线性拉伸到[0, 255]范围
    """
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min == 0:
        return np.zeros_like(img, dtype=np.uint8)
    stretched = 255.0 * (img - img_min) / (img_max - img_min)
    return stretched.astype(np.uint8)


def apply_kmeans_as_fcm_proxy(gray_img):
    """
    用KMeans聚类作为FCM的代理实现
    """
    # 将图像展平
    h, w = gray_img.shape
    data = gray_img.reshape(-1, 1).astype(np.float32)
    
    # 执行KMeans聚类
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data)
    labels = kmeans.labels_.reshape(h, w)
    
    # 选择平均灰度最高的聚类作为前景
    cluster_means = [np.mean(gray_img[labels == i]) for i in range(2)]
    brightest_cluster = np.argmax(cluster_means)
    
    # 创建二值图
    proxy_fcm = np.zeros_like(gray_img, dtype=np.uint8)
    proxy_fcm[labels == brightest_cluster] = 255
    
    return proxy_fcm


def extract_markers_improved_h_max(gray_img):
    """
    基于面积重构的改进H顶帽变换来提取标识点
    """
    # 确保输入是uint8
    if gray_img.dtype != np.uint8:
        gray_img = stretch_to_0_255(gray_img)
    
    # 1. H-maxima 变换: 找到所有高度大于30的局部极大值区域
    h_max = morphology.h_maxima(gray_img, 30)
    
    # 2. 面积开运算: 移除面积小于100的连通区域
    markers = morphology.area_opening(h_max, area_threshold=100)
    
    # 为了可视化，返回二值图
    marker_binary = (markers > 0).astype(np.uint8) * 255
    
    return marker_binary


def full_flotation_foam_segmentation(image_path, use_fcm=True):
    """
    完整的浮选泡沫图像预处理与分割流程
    
    返回:
        tuple: (原始图像, 光照校正图像, 增强图像, 标记点, 分割结果)
    """
    # 步骤 0: 读取原始图像
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转为RGB格式
    
    # 步骤 1: 光照不均校正 (MSRCR)
    img_msrcr = apply_msrcr(original_img)
    illum_corrected = stretch_to_0_255(img_msrcr)
    
    # 步骤 2: 图像灰度化
    gray_img = 0.3 * illum_corrected[:, :, 0] + \
               0.59 * illum_corrected[:, :, 1] + \
               0.11 * illum_corrected[:, :, 2]
    gray_img = gray_img.astype(np.uint8)
    
    # 步骤 3: 灰度增强
    enhanced_img = stretch_to_0_255(gray_img)
    
    # 步骤 4: FCM聚类（使用KMeans作为代理）
    if use_fcm:
        clustered_img = apply_kmeans_as_fcm_proxy(enhanced_img)
    else:
        clustered_img = enhanced_img.copy()
    
    # 步骤 5: 标识点提取
    marker_binary = extract_markers_improved_h_max(clustered_img)
    
    # 步骤 6: 应用标记控制分水岭分割
    marker_labels = morphology.label(marker_binary, connectivity=2)
    inverted_img = 255 - enhanced_img.astype(np.float32)
    segmented_labels = watershed(inverted_img, markers=marker_labels, watershed_line=True)
    
    return original_img, illum_corrected, enhanced_img, marker_labels, segmented_labels


# ======================== 浮选泡沫特征提取器 ======================== #
class FoamMorphologicalFeatureExtractor:
    """浮选泡沫图像形态学特征提取器"""
    
    def __init__(self):
        """初始化特征提取器"""
        self.extractor = MorphologicalFeatureExtractor()
        print("形态学特征提取器已初始化")
        print(f"将提取 {self.extractor.feature_dim} 维形态学特征")
    
    def extract_from_image(self, image_path, save_segmented=False, output_dir=None):
        """
        从单张图像中提取形态学特征
        
        参数:
            image_path (str): 图像文件路径
            save_segmented (bool): 是否保存分割后的图像
            output_dir (str): 分割图像保存目录
        
        返回:
            np.ndarray: 17维形态学特征向量
        """
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 检查图像格式是否支持
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in valid_extensions:
            raise ValueError(f"不支持的图像格式: {ext}，支持的格式有: {', '.join(valid_extensions)}")
        
        try:
            print(f"正在从图像中提取形态学特征: {image_path}")
            
            # 如果需要保存分割后的图像，获取并保存中间结果
            if save_segmented:
                if not output_dir:
                    output_dir = os.path.dirname(image_path)
                
                # 确保输出目录存在
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # 获取分割结果
                orig, illum_corrected, enhanced, markers, segmented = full_flotation_foam_segmentation(image_path)
                
                # 保存各种处理后的图像
                self.save_segmented_images(image_path, orig, illum_corrected, enhanced, markers, segmented, output_dir)
                
            # 提取特征
            features = self.extractor.extract_features_from_image(image_path)
            
            print("特征提取完成")
            return features
        except Exception as e:
            print(f"特征提取失败: {e}")
            # 返回零向量作为默认值
            return np.zeros(self.extractor.feature_dim)
    
    def save_segmented_images(self, image_path, orig, illum_corrected, enhanced, markers, segmented, output_dir):
        """
        保存分割过程中的各种图像
        """
        try:
            # 获取原始文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 保存原始图像
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
            
            # 保存光照校正后的图像
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_illum_corrected.png"), cv2.cvtColor(illum_corrected, cv2.COLOR_RGB2BGR))
            
            # 保存灰度增强后的图像
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_enhanced.png"), enhanced)
            
            # 保存标记点图像
            markers_vis = (markers > 0).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_markers.png"), markers_vis)
            
            # 保存分割结果（彩色标签）- 过滤面积小于50的区域
            from skimage.color import label2rgb
            
            # 创建过滤后的分割标签
            filtered_segmented = segmented.copy()
            props = measure.regionprops(segmented)
            for prop in props:
                if prop.label != 0 and prop.area < 50:
                    filtered_segmented[filtered_segmented == prop.label] = 0
            
            # 使用过滤后的标签创建彩色分割图
            segmented_rgb = label2rgb(filtered_segmented, bg_label=0, kind='overlay')
            segmented_rgb = (segmented_rgb * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_segmented.png"), cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))
            
            # 在原始图像上叠加过滤后的分割边界
            overlay = orig.copy()
            boundaries = filtered_segmented == 0  # 分水岭线
            overlay[boundaries] = [255, 0, 0]  # 红色边界
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_segmented_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            print(f"分割图像已保存到: {output_dir}")
        except Exception as e:
            print(f"保存分割图像失败: {e}")
    
    def print_feature_details(self, features):
        """
        打印特征的详细信息
        """
        if len(features) != self.extractor.feature_dim:
            print("警告: 特征维度不匹配，无法打印详细信息")
            return
        
        feature_names = [
            "灰度均值", "大小均值", "色调", "大小标准差",
            "红色均值", "绿色均值", "蓝色均值", "长宽比",
            "承载率", "红色相对分量", "大小峰度", "大小偏斜度",
            "粗度", "非均匀度", "大小分布", "低频能量", "高频能量"
        ]
        
        print("\n=== 形态学特征详细信息 ===")
        for i, (name, value) in enumerate(zip(feature_names, features)):
            print(f"{i+1}. {name}: {value:.4f}")
    
    def save_features(self, features, output_path):
        """
        保存特征到文件（npy和json格式）
        """
        try:
            # 创建输出目录（如果不存在）
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 保存特征为npy格式
            np.save(output_path, features)
            print(f"特征已保存到: {output_path}")
            
            # 保存为JSON格式
            json_path = os.path.splitext(output_path)[0] + '.json'
            feature_names = [
                "灰度均值", "大小均值", "色调", "大小标准差",
                "红色均值", "绿色均值", "蓝色均值", "长宽比",
                "承载率", "红色相对分量", "大小峰度", "大小偏斜度",
                "粗度", "非均匀度", "大小分布", "低频能量", "高频能量"
            ]
            
            features_dict = {
                "features": [{"name": name, "value": float(value)}
                             for name, value in zip(feature_names, features)]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(features_dict, f, ensure_ascii=False, indent=2)
            print(f"特征JSON文件已保存到: {json_path}")
        except Exception as e:
            print(f"保存特征失败: {e}")


# ======================== 命令行接口 ======================== #
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从浮选泡沫图像中提取形态学特征')
    parser.add_argument('--image', default=r'D:\zzb_data\zzb_project\I3D\pic\image.png', 
                        help='输入图像文件路径')
    parser.add_argument('--output', default=r'D:\zzb_data\zzb_project\I3D\pic\features.npy', 
                        help='输出特征文件路径（默认不保存）')
    parser.add_argument('--print-details', default=False, action='store_true', 
                        help='打印特征详细信息')
    parser.add_argument('--save-segmented', default=True, action='store_true', 
                        help='保存分割后的图像')
    parser.add_argument('--segmented-dir', help='分割图像保存目录（默认与输出特征在同一目录）')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 创建特征提取器
        feature_extractor = FoamMorphologicalFeatureExtractor()
        
        # 提取特征
        features = feature_extractor.extract_from_image(
            args.image,
            save_segmented=args.save_segmented,
            output_dir=args.segmented_dir or os.path.dirname(args.output)
        )
        
        # 打印详细信息（如果需要）
        if args.print_details:
            feature_extractor.print_feature_details(features)
        
        # 保存特征（如果需要）
        if args.output:
            feature_extractor.save_features(features, args.output)
        
        print("\n处理完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()