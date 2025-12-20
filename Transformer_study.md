# Transformer 知识梳理

## 梦开始的地方：Transformer

<img src="C:\Users\34356\Desktop\md\resources\b6042a23-c498-4000-9aae-2aec20a32aaf.png" style="zoom:80%;" />

编码器解码器架构，使用多头自注意力机制进行全局建模。

## 把Transformer应用到图像领域经典之作：ViT

![](C:\Users\34356\Desktop\md\resources\屏幕截图 2025-10-16 214533.png)

把图片切分成一个个的 patches，再加入 cls embedding 和 position embedding 一起喂入 Transformer Encoder，再用一个 MLP 做分类头

图片(224 * 224) 切成 16 * 16 的 patch 则 有 224/16 * 224/16 = 196 个patch 每个 patch = 16 * 16 * 3 = 768 

把 图片(b, c, h, w) 转换成(b, num_patch, patch_dim) # 把每个 patch 展平成一维的 token

加入分类头和位置信息后，再把 patch_dim 映射到 dim 维，即 Transformer 接收的维度 原文是把 768 映射到 128 ，减少了六倍长度的 token

## CV界的Bert：MAE

![](C:\Users\34356\Desktop\md\resources\11a772d7-6586-461e-9478-0ec733422ead.png)

输入遵循 ViT，使用高掩码率遮蔽掉图片的部分(默认75%)，然后喂入编码器提取特征，然后用解码器还原。

## 把ViT应用到视频理解领域：TimeSformer

![](C:\Users\34356\Desktop\md\resources\60881339-8b78-40df-ab8a-12d781ac55ab.png)

设计了五种注意力方法

S：只做空间注意力

ST：同时做空间和时间的注意力，效果最好内存消耗最大

T+S：先做时间注意力，再做空间注意力

L+G：先做局部注意力，再做全局注意力

T+W+H：轴注意力，按照时间轴，长，宽，做注意力



## 站在巨人的肩膀上：VedioMAE

<img src="C:\Users\34356\Desktop\md\resources\a8f14e96-7bf4-4f49-bf3f-9ef7c4027e15.png"  />

  

下采样 视频帧，用一个高掩码(90%以上)遮蔽视频帧的部分，使用 tube masking 即所有视频帧都在同一个区域被遮蔽。把处理好的立方体传入 Encoder(ViT)，再用 Decoder 还原(非对称结构)



## 
