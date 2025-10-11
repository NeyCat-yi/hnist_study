import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models.video import r2plus1d_18
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
import os
from typing import Dict, List, Tuple, Optional
import cv2
from transformers import TimesformerForVideoClassification, TimesformerConfig
import warnings
warnings.filterwarnings('ignore')

class FoamVideoDataset(torch.utils.data.Dataset):
    """泡沫视频数据集"""
    
    def __init__(self, data_dir: str, labels_file: str, num_frames: int = 64, 
                 transform=None, is_training: bool = True):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.transform = transform
        self.is_training = is_training
        
        # 加载标签
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.labels_map = json.load(f)
        
        # 获取视频文件列表
        self.video_files = []
        self.labels = []
        
        for class_name, class_idx in self.labels_map.items():
            class_dir = os.path.join(data_dir, str(class_idx))
            if os.path.exists(class_dir):
                for video_file in os.listdir(class_dir):
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        self.video_files.append(os.path.join(class_dir, video_file))
                        self.labels.append(class_idx - 1)  # 转换为0-based索引
    
    def __len__(self):
        return len(self.video_files)
    
    def load_video_frames(self, video_path: str) -> torch.Tensor:
        """加载视频帧 - 每隔5帧抽取一帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            # 如果视频为空，创建黑色帧
            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]
        
        # 每隔5帧抽取一帧的采样策略
        if len(frames) > 0:
            # 从第0帧开始，每隔5帧抽取一帧 (即取第0, 5, 10, 15, ...帧)
            sampled_frames = []
            for i in range(0, len(frames), 5):  # 步长为5
                sampled_frames.append(frames[i])
            
            # 如果采样帧数不足目标帧数，进行处理
            if len(sampled_frames) >= self.num_frames:
                # 如果采样帧数超过目标帧数，取前num_frames帧
                frames = sampled_frames[:self.num_frames]
            else:
                # 如果采样帧数不足，重复最后一帧或循环重复
                frames = sampled_frames
                while len(frames) < self.num_frames:
                    if len(sampled_frames) > 0:
                        # 循环重复采样帧
                        frames.extend(sampled_frames[:min(len(sampled_frames), self.num_frames - len(frames))])
                    else:
                        # 如果没有采样帧，添加黑色帧
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # 转换为tensor
        frames = np.stack(frames)  # (T, H, W, C)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        # 加载视频帧
        frames = self.load_video_frames(video_path)
        
        # 应用变换
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label

class TimeSformerModel(nn.Module):
    """TimeSformer模型包装器"""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True, num_frames: int = 64):  # 5个类别(1-5对应标签0-4)
        super().__init__()
        
        # 先创建适合指定帧数的配置
        config = TimesformerConfig(
            num_labels=num_classes,
            num_frames=num_frames,
            image_size=224
        )
        
        if pretrained:
            try:
                # 尝试使用预训练的TimeSformer
                self.model = TimesformerForVideoClassification.from_pretrained(
                    "facebook/timesformer-base-finetuned-k400"
                )
                # 修改分类头
                self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
                # 重要：更新模型配置以使用指定的帧数
                self.model.config.num_frames = num_frames
                print("✅ 成功加载预训练TimeSformer模型并更新配置")
            except Exception as e:
                print(f"⚠️ 无法加载预训练模型，使用随机初始化: {e}")
                # 使用自定义配置从头开始训练
                self.model = TimesformerForVideoClassification(config)
        else:
            # 从头开始训练
            self.model = TimesformerForVideoClassification(config)
        
        self.freeze_backbone = True
    
    def freeze_backbone_layers(self):
        """冻结骨干网络"""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # 打印输入数据维度信息
        print(f"Input shape before permute: {x.shape}")
        
        # x shape: (B, C, T, H, W)
        # TimeSformer expects (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # 打印permute后的数据维度
        print(f"Input shape after permute: {x.shape}")
        
        # 检查模型配置的帧数是否与输入匹配
        print(f"Model config num_frames: {self.model.config.num_frames}")
        print(f"Input num_frames: {x.shape[1]}")
        
        try:
            outputs = self.model(pixel_values=x)
            return outputs.logits
        except RuntimeError as e:
            # 发生错误时打印详细信息以方便调试
            print(f"Error in TimeSformer forward pass: {e}")
            # 打印模型配置的关键参数
            print(f"Model config hidden_size: {self.model.config.hidden_size}")
            print(f"Model config image_size: {self.model.config.image_size}")
            print(f"Model config patch_size: {self.model.config.patch_size}")
            raise

class R2Plus1DModel(nn.Module):
    """R(2+1)D模型包装器"""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5):  # 5个类别(1-5对应标签0-4)
        super().__init__()
        
        self.model = r2plus1d_18(pretrained=pretrained)
        
        # 修改分类头
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    
    def __init__(self, alpha: float = 0.1, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.class_weights = class_weights
    
    def forward(self, pred, target):
        num_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * (1 - self.alpha) + self.alpha / num_classes
        loss = -torch.sum(smooth_target * torch.log_softmax(pred, dim=1), dim=1)
        
        # 应用类别权重
        if self.class_weights is not None:
            weights = self.class_weights[target]
            loss = loss * weights
            
        return loss.mean()

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device, model_name: str, class_weights=None, resume_training=True):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.best_acc = 0.0
        self.patience_counter = 0
        self.patience = 5
        self.start_epoch = 0
        self.resume_training = resume_training
        
        # 类别权重
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights).to(device)
        else:
            self.class_weights = None
        
        # 检查是否有已保存的模型
        self.checkpoint_path = f'models/best_{self.model_name.lower()}_model.pth'
        if self.resume_training and os.path.exists(self.checkpoint_path):
            print(f"发现已保存的{self.model_name}模型: {self.checkpoint_path}")
            self.load_checkpoint()
        else:
            print(f"未找到已保存的{self.model_name}模型，将从头开始训练")
    
    def load_checkpoint(self):
        """加载检查点"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_acc = checkpoint.get('best_acc', 0.0)
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"成功加载{self.model_name}模型检查点:")
            print(f"  - 最佳验证准确率: {self.best_acc:.4f}")
            print(f"  - 将从第{self.start_epoch + 1}轮开始训练")
        except Exception as e:
            print(f"加载{self.model_name}模型检查点失败: {e}")
            print("将从头开始训练")
            self.best_acc = 0.0
            self.start_epoch = 0
    
    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # 打印预测结果和真实标签以检查模型学习情况
        print(f"验证集预测结果: {all_preds}")
        print(f"验证集真实标签: {all_targets}")
        print(f"预测类别分布: {np.bincount(all_preds, minlength=5)}")
        print(f"真实类别分布: {np.bincount(all_targets, minlength=5)}")
        
        return avg_loss, accuracy * 100, f1 * 100, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs: int, lr: float = 1e-4, weight_decay: float = 1e-4):
        """完整训练流程"""
        criterion = LabelSmoothingCrossEntropy(alpha=0.1, class_weights=self.class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 如果是从检查点恢复，尝试加载优化器和调度器状态
        if self.resume_training and os.path.exists(self.checkpoint_path) and self.start_epoch > 0:
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("成功加载优化器状态")
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("成功加载学习率调度器状态")
            except Exception as e:
                print(f"加载优化器/调度器状态失败: {e}，使用默认状态")
        
        if self.start_epoch == 0:
            print(f"开始训练 {self.model_name} 模型...")
        else:
            print(f"继续训练 {self.model_name} 模型 (从第{self.start_epoch + 1}轮开始)...")
        
        for epoch in range(self.start_epoch, epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            
            # 验证
            val_loss, val_acc, val_f1, _, _ = self.validate(val_loader, criterion)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%')
            
            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_acc': self.best_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_f1': val_f1
                }, f'models/best_{self.model_name.lower()}_model.pth')
                print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.patience:
                print(f'验证准确率连续{self.patience}个epoch未提升，早停训练')
                break
        
        print(f'{self.model_name} 训练完成，最佳验证准确率: {self.best_acc:.2f}%')

class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, timesformer_model, r2plus1d_model, device):
        self.timesformer_model = timesformer_model.to(device)
        self.r2plus1d_model = r2plus1d_model.to(device)
        self.device = device
        
        self.timesformer_model.eval()
        self.r2plus1d_model.eval()
    
    def predict_single_clip(self, clip: torch.Tensor, timesformer_weight: float = 0.6) -> Tuple[np.ndarray, int]:
        """预测单个视频片段"""
        clip = clip.unsqueeze(0).to(self.device)  # 添加batch维度
        
        with torch.no_grad():
            # TimeSformer预测
            timesformer_logits = self.timesformer_model(clip)
            timesformer_probs = torch.softmax(timesformer_logits, dim=1).cpu().numpy()[0]
            
            # R(2+1)D预测 - 需要调整帧数
            if clip.size(2) > 32:  # 如果帧数超过32，进行下采样
                indices = torch.linspace(0, clip.size(2) - 1, 32).long()
                r2plus1d_clip = clip[:, :, indices]
            else:
                r2plus1d_clip = clip
            
            r2plus1d_logits = self.r2plus1d_model(r2plus1d_clip)
            r2plus1d_probs = torch.softmax(r2plus1d_logits, dim=1).cpu().numpy()[0]
        
        # 加权平均
        ensemble_probs = timesformer_weight * timesformer_probs + (1 - timesformer_weight) * r2plus1d_probs
        predicted_class = np.argmax(ensemble_probs)
        
        return ensemble_probs, predicted_class
    
    def predict_video(self, video_path: str, num_clips: int = 5, timesformer_weight: float = 0.6) -> Dict:
        """预测整个视频"""
        # 加载视频
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return {'error': '无法读取视频文件'}
        
        # 分割为多个片段
        clip_length = 64
        clips = []
        
        if len(frames) >= clip_length * num_clips:
            # 均匀分割
            for i in range(num_clips):
                start_idx = i * len(frames) // num_clips
                end_idx = start_idx + clip_length
                if end_idx > len(frames):
                    end_idx = len(frames)
                    start_idx = max(0, end_idx - clip_length)
                
                clip_frames = frames[start_idx:end_idx]
                if len(clip_frames) < clip_length:
                    # 重复帧
                    while len(clip_frames) < clip_length:
                        clip_frames.extend(clip_frames[:min(len(clip_frames), clip_length - len(clip_frames))])
                
                clip_frames = np.stack(clip_frames[:clip_length])
                clip_tensor = torch.from_numpy(clip_frames).float() / 255.0
                clip_tensor = clip_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
                clips.append(clip_tensor)
        else:
            # 视频太短，创建单个片段
            while len(frames) < clip_length:
                frames.extend(frames[:min(len(frames), clip_length - len(frames))])
            
            clip_frames = np.stack(frames[:clip_length])
            clip_tensor = torch.from_numpy(clip_frames).float() / 255.0
            clip_tensor = clip_tensor.permute(3, 0, 1, 2)
            clips = [clip_tensor]
        
        # 预测每个片段
        all_probs = []
        all_predictions = []
        
        for clip in clips:
            probs, pred = self.predict_single_clip(clip, timesformer_weight)
            all_probs.append(probs)
            all_predictions.append(pred)
        
        # 集成结果
        avg_probs = np.mean(all_probs, axis=0)
        final_prediction = np.argmax(avg_probs)
        
        # 多数投票
        vote_prediction = max(set(all_predictions), key=all_predictions.count)
        
        return {
            'final_prediction': int(final_prediction),
            'vote_prediction': int(vote_prediction),
            'confidence': float(avg_probs[final_prediction]),
            'all_probabilities': avg_probs.tolist(),
            'clip_predictions': all_predictions
        }

class VideoTransform:
    """视频变换类，对每一帧应用相同的变换"""
    def __init__(self, frame_transform):
        self.frame_transform = frame_transform
    
    def __call__(self, video_tensor):
        # video_tensor shape: (C, T, H, W)
        C, T, H, W = video_tensor.shape
        
        # 重新排列为 (T, C, H, W) 以便对每帧应用变换
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        # 对每一帧应用变换
        transformed_frames = []
        for t in range(T):
            frame = video_tensor[t]  # (C, H, W)
            transformed_frame = self.frame_transform(frame)
            transformed_frames.append(transformed_frame)
        
        # 重新堆叠并转换回 (C, T, H, W)
        transformed_video = torch.stack(transformed_frames, dim=1)  # (C, T, H, W)
        
        return transformed_video

def create_data_transforms():
    """创建数据变换"""
    # 训练时的帧变换
    train_frame_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证时的帧变换
    val_frame_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 包装为视频变换
    train_transform = VideoTransform(train_frame_transform)
    val_transform = VideoTransform(val_frame_transform)
    
    return train_transform, val_transform

def main(resume_training: bool = True):
    """主函数
    
    Args:
        resume_training: 是否启用断点续训，默认为True
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'断点续训: {"启用" if resume_training else "禁用"}')
    
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    
    # 数据路径
    train_data_dir =r'D:\zzb_data\zzb_project\I3D\dataset\train_augmented'
    test_data_dir = 'd:/zzb_data/zzb_project/I3D/dataset/test_augmented'
    labels_file = 'd:/zzb_data/zzb_project/I3D/dataset/labels.json'
    
    # 创建数据变换
    train_transform, val_transform = create_data_transforms()
    
    # 创建数据集
    train_dataset = FoamVideoDataset(train_data_dir, labels_file, num_frames=64, 
                                   transform=train_transform, is_training=True)
    test_dataset = FoamVideoDataset(test_data_dir, labels_file, num_frames=64, 
                                  transform=val_transform, is_training=False)
    
    # 快速计算类别权重（基于文件数量而非加载数据）
    print("\n计算类别权重...")
    class_counts = {}
    for class_folder in os.listdir(train_data_dir):
        class_path = os.path.join(train_data_dir, class_folder)
        if os.path.isdir(class_path):
            video_files = [f for f in os.listdir(class_path) if f.endswith(('.avi', '.mp4', '.mov'))]  
            class_counts[int(class_folder) - 1] = len(video_files)  # 类别1-5对应标签0-4
    
    total_samples = sum(class_counts.values())
    class_weights = []
    for i in range(len(class_counts)):
        class_weights.append(total_samples / (len(class_counts) * class_counts[i]))
    
    print(f"类别分布: {class_counts}")
    print(f"类别权重: {dict(enumerate(class_weights))}")
    
    # 由于数据已经完全平衡，使用简单的随机采样
    print("数据集已平衡，使用随机采样")
    sampler = None  # 使用默认的随机采样
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        sampler=sampler,  # 使用加权采样器而不是shuffle
        num_workers=2
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    
    # 训练TimeSformer模型
    print('\n=== 训练TimeSformer模型 ===')
    timesformer_model = TimeSformerModel(num_classes=5, pretrained=True)
    timesformer_model.freeze_backbone_layers()  # 冻结骨干网络
    
    timesformer_trainer = ModelTrainer(timesformer_model, device, 'TimeSformer', class_weights, resume_training)
    timesformer_trainer.train(train_loader, test_loader, epochs=50, lr=1e-4, weight_decay=1e-4)
    
    # 训练R(2+1)D模型
    print('\n=== 训练R(2+1)D模型 ===')
    r2plus1d_model = R2Plus1DModel(num_classes=5, pretrained=True, dropout=0.5)
    
    r2plus1d_trainer = ModelTrainer(r2plus1d_model, device, 'R2Plus1D', class_weights, resume_training)
    r2plus1d_trainer.train(train_loader, test_loader, epochs=50, lr=1e-3, weight_decay=1e-4)
    
    # 加载最佳模型
    print('\n=== 加载最佳模型进行集成测试 ===')
    
    # 加载TimeSformer最佳模型
    timesformer_checkpoint = torch.load('models/best_timesformer_model.pth')
    timesformer_model.load_state_dict(timesformer_checkpoint['model_state_dict'])
    
    # 加载R(2+1)D最佳模型
    r2plus1d_checkpoint = torch.load('models/best_r2plus1d_model.pth')
    r2plus1d_model.load_state_dict(r2plus1d_checkpoint['model_state_dict'])
    
    # 创建集成预测器
    ensemble_predictor = EnsemblePredictor(timesformer_model, r2plus1d_model, device)
    
    # 在测试集上评估集成模型
    print('\n=== 集成模型评估 ===')
    all_predictions = []
    all_targets = []
    
    for data, target in test_loader:
        for i in range(data.size(0)):
            clip = data[i]
            true_label = target[i].item()
            
            # 预测
            _, pred = ensemble_predictor.predict_single_clip(clip, timesformer_weight=0.6)
            
            all_predictions.append(pred)
            all_targets.append(true_label)
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    cm = confusion_matrix(all_targets, all_predictions)
    
    print(f'集成模型准确率: {accuracy * 100:.2f}%')
    print(f'集成模型F1分数: {f1 * 100:.2f}%')
    print('混淆矩阵:')
    print(cm)
    
    # 测试不同权重组合
    print('\n=== 测试不同权重组合 ===')
    weights = [0.5, 0.6, 0.7, 0.8]
    best_weight = 0.6
    best_f1 = f1
    
    for weight in weights:
        predictions = []
        for data, target in test_loader:
            for i in range(data.size(0)):
                clip = data[i]
                _, pred = ensemble_predictor.predict_single_clip(clip, timesformer_weight=weight)
                predictions.append(pred)
        
        current_f1 = f1_score(all_targets, predictions, average='weighted')
        print(f'权重 {weight:.1f}: F1分数 {current_f1 * 100:.2f}%')
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_weight = weight
    
    print(f'\n最佳权重组合: TimeSformer {best_weight:.1f}, R(2+1)D {1-best_weight:.1f}')
    print(f'最佳F1分数: {best_f1 * 100:.2f}%')
    
    print('\n训练和评估完成！')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='多模型集成训练')
    parser.add_argument('--no-resume', action='store_true', 
                       help='禁用断点续训，从头开始训练')
    
    args = parser.parse_args()
    resume_training = not args.no_resume
    
    main(resume_training=resume_training)