# Vedio MAE

对于Vedio Transformer，它们通常源自基于图像的 Transformer，并且严重依赖于大规模图像数据的预训练模型

用 自监督的训练方法（遮住视频的部分，然后通过不对称的 编码器解码器架构来还原），由于视频的高冗余和时间相关性，采用更高的掩码率来创建更有挑战性的任务，使学习到的表征捕获更有用的时空结构。



与图片和nlp那边的MAE(Bert)不同的是：1、视频可以被视为静态外观的时间演变，并且帧之间存在对应关系。除非考虑特定的屏蔽策略，否则这种时间相关性可能会导致重建过程中的信息泄漏（即屏蔽的时空内容重新出现）。从这个意义上说，对于每个被遮罩的立方体，很容易在相邻帧中找到相应的且未被遮罩的副本。这一属性将使学习模型识别出一些难以推广到新场景的“快捷”特征。

问题：时间冗余和时间相关性

解决办法：

1、使用极高的掩码比例来下采样，这种简单的策略不仅有效地提高了预训练性能，而且由于非对称编码器-解码器架构而大大降低了计算成本。



2、使用 tube masking ，即其中所有帧的掩蔽图都是一样的

![屏幕截图 2025-10-29 112343](C:\Users\34356\Desktop\md\resources\屏幕截图 2025-10-29 112343.png)

## 模型流程：

input image(3 * H * W) 被分为大小 16 * 16 * tubelet_size 的 patches，每个 patch 被表示成 tokens，然后用高掩码(75%) 随机遮蔽掉 tokens 的部分，然后把剩余的 tokens 送入 transformer 的编码器，再由解码器还原



## 代码详解：

### Embedding：

输入视频：(batch_size, num_frames, num_channels, height, width)	加上 位置编码(正余弦编码)后

输出 embedding：(batch_size, seq_len, hidden_size)

```python
	# permute to (batch_size, num_channels, num_frames, height, width)
    pixel_values = pixel_values.permute(0, 2, 1, 3, 4) # 第2维和第1维交换
    
    # 3D卷积 使得 输入(batch_size, num_channels, num_frames, height, width)
    # 变为(batch_size, hidden_size, num_frames//tubelet_size, height//patch_size, width//patch_size)
    self.projection = nn.Conv3d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )
    
	# 从维度 2 开始一直扁平化到最后一维，再交换 1 和 2 维
    # 最终形状: (batch_size, num_patches, hidden_size)
    embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
```

输入:

(B, 16, 3, 224, 224)
    ↓ pixel_values.permute(0, 2, 1, 3, 4) # 第2维和第1维交换

(B, 3, 16, 224, 224)
    ↓ Conv3d(in=3, out=768, k=(2,16,16), s=(2,16,16))
(B, 768, 8, 14, 14)
    ↓ flatten(2)  # 把 (8,14,14) 拉平成一个维度
(B, 768, 8x14x14=1568)
    ↓ transpose(1, 2)  # 交换通道维和 patch 维
(B, 1568, 768)



### Attention:

#### 1、线性层得到 Q / K / V

```python
# 三个线性层 输入 hidden_size，输出 all_head_size
# hindden_size 和 all_head_size 值相同
self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

# weight:(all_head_size, hidden_size)
 # nn.functional.linear: input @ weight.T + bias ; nn.functional.linear 不同于 nn.Linear 的是：直接做运算而不是定义层
k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
keys = nn.functional.linear(input=hidden_states, weight=self.key.weight, bias=k_bias)
values = nn.functional.linear(input=hidden_states, weight=self.value.weight, bias=self.v_bias)
queries = nn.functional.linear(input=hidden_states, weight=self.query.weight, bias=self.q_bias)
```

输入 hidden_states: (B, 1568, 768)
    ↓ 使用 nn.functional.linear 执行线性变换
queries / keys / values: (B, 1568, 768)  # 通过权重矩阵（self.query.weight 等）做线性变换

#### 2、拆成多头格式 (view + transpose)

```python
# 经过 reshape 和 transpose，变成 (batch_size, num_heads, seq_length, head_size)
key_layer = keys.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
value_layer = values.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
query_layer = queries.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

```

(B, 1568, 768)
    ↓ view(B, 1568, num_heads=12, head_dim=64)

(B, 1568, 12, 64)
    ↓ transpose(1, 2)   # 把 seq_len 和 head 维交换

(B, 12, 1568, 64)       # 记为: query_layer / key_layer / value_layer

#### 3、计算注意力分数 QKᵀ

```python
attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
```

query: (B, 12, 1568, 64)
key:   (B, 12, 1568, 64)
    ↓ key.transpose(2, 3)

key^T: (B, 12, 64, 1568)
    ↓ torch.matmul(query, key^T)

attn_scores:
(B, 12, 1568, 1568)   # 每个 head 上，query 序列对 key 序列的相似度
    ↓ * scaling (1/√64)
(形状不变)

#### 4、softmax + dropout

```python
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
```

(B, 12, 1568, 1568)
    ↓ softmax(dim=-1)

attention_probs:
(B, 12, 1568, 1568)
    ↓ dropout (训练时)

attention_probs_drop:
(B, 12, 1568, 1568)

#### 5、加权 V 得到注意力输出

```python
attn_output = torch.matmul(attn_weights, value)
attn_output = attn_output.transpose(1, 2).contiguous()
```

attention_probs_drop: (B, 12, 1568, 1568)
value:                (B, 12, 1568, 64)
    ↓ torch.matmul(attention_probs_drop, value)

attn_output:
(B, 12, 1568, 64)
    ↓ transpose(1, 2)  # 把 head 维和 seq_len 维交换

(B, 1568, 12, 64)

#### 6、从多头拼回 hidden_size

```python
# 取回原来的形状 (batch_size, seq_length, all_head_size)
new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
context_layer = context_layer.reshape(new_context_layer_shape)
```

(B, 1568, 12, 64)
    ↓ reshape  →  把 (12,64) 合并

(B, 1568, 12×64=768)

#### 7、SelfOutput

用一个可学习的线性变换，把“多头产出的特征空间”
 重新投影回模型统一使用的 hidden 表示空间

```python
hidden_states = self.dense(hidden_states)
hidden_states = self.dropout(hidden_states)
```

输入: context_layer
(B, 1568, 768)
    ↓ self.dense: Linear(768 → 768)

(B, 1568, 768)
    ↓ dropout

(B, 1568, 768)



# 训练

## 导包

先下载所需要的包

```bash
pip install torch torchvision transformers decord pytorchvideo evaluate
```



```python
import os
import torch
import numpy as np
import evaluate
from torch.utils.data import Dataset
from transformers import (
    VideoMaeConfig, 
    VideoMaeForVideoClassification, 
    VideoMaeImageProcessor, 
    TrainingArguments, 
    Trainer
)
# decord 是目前最快的视频读取库，VideoMAE 官方推荐
from decord import VideoReader, cpu
```

