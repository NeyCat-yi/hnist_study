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

input image(3 * H * W) 被分为大小 16 * 16 的 patches，每个 patch 被表示成 tokens，然后用高掩码(75%) 随机遮蔽掉 tokens 的部分，然后把剩余的 tokens 送入 transformer 的编码器，再有解码器还原

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

