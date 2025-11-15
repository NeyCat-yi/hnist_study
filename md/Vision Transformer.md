#  Vision Transformer

ViT是把transformer应用到CV方向的里程碑式工作

transformer架构不好直接用于图像的原因是 序列太长：224*224的图像拉直超过是5万长度的序列(还要乘channels)

**ViT的解决方案是把图片切分成一个个的 patch 再通过一个线性层映射减少序列长度再加上位置信息和分类头通过一个 Transformer 的 Encoder 提取表征后用MLP做分类**





![VIT架构图](C:\Users\34356\Desktop\屏幕截图 2025-10-16 214533.png)

### 1、把图片切分成 patches

原序列：[batch, channels, height, weight] 要转换成：[batch, num_patches, patch_dim]

```python
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
num_patches = (image_size_h // patch_size) * (image_size_w // patch_size) # patch 个数
patch_dim = channels * patch_size ** 2 # patch 维度

# 将图像切分为patches并映射到dim维度的线性层
# 为了提高训练稳定性，添加了LayerNorm层
# 将每个patch转换成dim维度的向量
to_patch_embedding = nn.Sequential( # Sequential模块将多个层组合在一起
    # 输入图像 [batch, channels, height, width] 转变为 [batch, num_patches, patch_dim]
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
    nn.LayerNorm(patch_dim),
    nn.Linear(patch_dim, dim), # 缩成更小的维度了 序列长度就降低了
    nn.LayerNorm(dim)
)
x = to_patch_embedding(img)
```

### 2、在序列中加上 位置信息 和分类头

```python
pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
cls_token = nn.Parameter(torch.randn(1, 1, dim))

cls_tokens = repeat(cls_token, '() n d -> b n d', b = b) # 此处的 b 是 x 的 batch
x = torch.cat((cls_tokens, x), dim=1) # dim = 1表示在num_patch维度上拼接 (在第二个维度上拼接)(拼接时，其他维度要保持一致)
x += self.pos_embedding[:, :(n + 1)] # 控制在一共只有 n+1 个位置信息
```

### 3、传入一个 Transformer 的 Encoder 然后用一个 MLP 分类

```python
mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes) 
        )
x = transformer(x)
mlp_head(x)
```

### 4、补充

vision transformer 在经过足够大的数据训练下能获得很好的结果去做下游的迁移任务

1、用标准的transformer架构直接用于图像，尽可能的少做修改

#### 1、输入：

原图 x = 224 * 224 	切成 16 * 16 大小的 patch 则 变成 (224 / 16) * (224 / 16)  = 14 * 14 = 196 个 patch

每个 patch 为 16 * 16 * 3(RGB通道) = 768 总的输入是 input = 196 * 768 (196个向量，每个向量维度是768)

#### 2、线性投射层（全连接层）

维度：768(输入的维度) * 768     做了一个矩阵乘法 [196 * 768] * [768 * 768]

3、给序列加上位置信息和一个分类头

序列变为 197 * 768    再加上一个可学习的位置编码

![全流程公式](C:\Users\34356\Desktop\md\resources\9d726ed7-b4ad-4968-a86b-e601fb49a762.png)



  