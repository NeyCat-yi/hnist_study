import torch
import torch.nn as nn
import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import json

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入TimeSformer模型定义
from multi_model_ensemble import TimeSformerModel, create_data_transforms

class VideoInferencer:
    """TimeSformer模型推理类"""
    
    def __init__(self, model_path, num_classes=5, num_frames=8, device=None):
        """
        初始化推理器
        
        参数:
            model_path: 模型权重文件路径
            num_classes: 分类类别数
            num_frames: 每个视频片段的帧数
            device: 运行设备，默认为cuda（若可用）或cpu
        """
        self.num_classes = num_classes
        self.num_frames = num_frames
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 创建并加载模型
        self.model = self._create_and_load_model(model_path)
        
        # 创建数据变换
        _, self.transform = create_data_transforms()
        
        # 加载标签映射（如果有）
        self.labels = self._load_labels()
    
    def _create_and_load_model(self, model_path):
        """创建模型并加载权重"""
        # 创建TimeSformer模型实例
        model = TimeSformerModel(
            num_classes=self.num_classes,
            pretrained=False,  # 不加载预训练权重，因为我们要加载自己的权重
            num_frames=self.num_frames
        )
        
        # 将模型移至指定设备
        model = model.to(self.device)
        
        # 加载模型权重
        try:
            print(f"加载模型权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查checkpoint结构，如果是完整的训练检查点则提取model_state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 直接加载权重（如果checkpoint本身就是model_state_dict）
                model.load_state_dict(checkpoint)
            
            print("模型权重加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        # 设置为评估模式
        model.eval()
        
        return model
    
    def _load_labels(self):
        """加载标签映射"""
        labels_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'dataset', 'labels.json')
        if os.path.exists(labels_file):
            with open(labels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _sample_frames(self, video_path, num_frames):
        """从视频中采样指定数量的帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError(f"无法读取视频文件: {video_path}")
        
        # 均匀采样帧
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 如果读取失败，使用上一帧
                if frames:
                    frames.append(frames[-1])
                else:
                    # 如果第一帧就读取失败，使用空白帧
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        
        # 如果帧数不足，重复最后一帧
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        return np.array(frames)
    
    def preprocess_frames(self, frames):
        """预处理视频帧"""
        processed_frames = []
        
        # 对每一帧应用变换
        for frame in frames:
            # 转换为PIL图像格式（部分变换需要）
            frame_pil = Image.fromarray(frame)
            # 应用变换
            processed_frame = self.transform(frame_pil)
            processed_frames.append(processed_frame)
        
        # 堆叠成批次
        # [num_frames, 3, 224, 224] -> [1, 3, num_frames, 224, 224]
        video_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0)
        
        return video_tensor
    
    def infer_single_video(self, video_path):
        """推理单个视频"""
        # 检查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        # 采样帧
        frames = self._sample_frames(video_path, self.num_frames)
        # 预处理帧
        # 注意：这里需要PIL库，如果没有安装可以使用以下方式
        # 简单预处理替代方案：
        processed_frames = []
        for frame in frames:
            # 调整大小
            frame = cv2.resize(frame, (224, 224))
            # 转换为张量并归一化
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            # 应用ImageNet标准化
            frame = (frame - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            processed_frames.append(frame)
        # 堆叠成批次
        video_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        # 推理
        with torch.no_grad():
            output = self.model(video_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # 转换为CPU并转为numpy
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        probabilities = probabilities.cpu().numpy()[0]
        
        # 获取类别名称（如果有）
        class_name = f"类别{predicted_class}"
        if self.labels and str(predicted_class) in self.labels:
            class_name = self.labels[str(predicted_class)]
        
        return {
            'video_path': video_path,
            'predicted_class': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }
    
    def infer_batch_videos(self, video_dir):
        """批量推理目录中的视频"""
        # 检查目录是否存在
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"视频目录不存在: {video_dir}")
        
        # 获取目录中的所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []
        
        for root, _, files in os.walk(video_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            print(f"在目录 {video_dir} 中未找到视频文件")
            return []
        
        print(f"找到 {len(video_files)} 个视频文件，开始批量推理...")
        
        results = []
        
        for video_path in tqdm(video_files, desc="批量推理"):
            try:
                result = self.infer_single_video(video_path)
                results.append(result)
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                continue
        
        return results

# 添加缺少的导入
from PIL import Image

if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='TimeSformer模型推理脚本')
    parser.add_argument('--model_path', type=str, 
                        default= r"D:\code\QT_Fish\人工智能大赛\timesformer_models\stage1_best.pth",
                        help='模型权重文件路径')
    parser.add_argument('--video_path', type=str,default=r"D:\zzb_data\zzb_project\视频理解\dataset\test\1\20140608115252Sb.avi", help='单个视频文件路径')
    parser.add_argument('--video_dir', type=str, help='批量视频文件目录路径')
    parser.add_argument('--num_frames', type=int, default=8, help='每个视频片段的帧数')
    parser.add_argument('--num_classes', type=int, default=5, help='分类类别数')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.video_path and not args.video_dir:
        parser.error("必须指定 --video_path 或 --video_dir")
    
    # 创建推理器
    inferencer = VideoInferencer(
        model_path=args.model_path,
        num_classes=args.num_classes,
        num_frames=args.num_frames
    )
    
    # 执行推理
    if args.video_path:
        # 单个视频推理
        result = inferencer.infer_single_video(args.video_path)
        print("\n推理结果:")
        print(f"视频路径: {result['video_path']}")
        print(f"预测类别: {result['predicted_class']} ({result['class_name']})")
        print(f"置信度: {result['confidence']:.4f}")
        print("各类别概率:")
        for i, prob in enumerate(result['probabilities']):
            class_name = f"类别{i}"
            if inferencer.labels and str(i) in inferencer.labels:
                class_name = inferencer.labels[str(i)]
            print(f"  {class_name}: {prob:.4f}")
    
    if args.video_dir:
        # 批量视频推理
        results = inferencer.infer_batch_videos(args.video_dir)
        
        if results:
            print(f"\n批量推理完成，共处理 {len(results)} 个视频文件")
            
            # 保存结果到JSON文件
            output_file = os.path.join(os.path.dirname(args.model_path), 'inference_results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"推理结果已保存至: {output_file}")
            
            # 统计各类别数量
            class_counts = {}
            for result in results:
                class_name = result['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("\n各类别统计:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}")