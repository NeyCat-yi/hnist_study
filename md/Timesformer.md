# Timesformer

下面把这个 `Block` 在两种注意力类型下的**计算流程、张量形状、直觉理解、以及优缺点**都讲清楚。先约定符号：

- 输入 `x` 形状：`[B, N, C]`，其中
   `N = 1 + T * H * W`（含 1 个 CLS token + T 帧 × H×W 个 patch token）
- `B`：batch size，`T`：时间维帧数，`H×W`：每帧的patch网格大小，`C`：通道维/嵌入维
- `num_heads`：注意力头数

`Block` 里共有两类注意力：

- `self.attn`：空间注意力（space attn），也可在 joint 模式下充当「时空联合注意力」
- （仅 divided 模式有）`self.temporal_attn`：时间注意力（temporal attn）
- 另有 `MLP`、`LayerNorm`、`DropPath`、以及一个 `temporal_fc`（线性层，给 temporal 残差做投影/融合）

------

# joint_space_time（时空联合注意力）

## 代码路径

`if self.attention_type in ['space_only', 'joint_space_time']:` 这一分支走同样的两行，joint 模式等价为**对整段序列（包含时间+空间的所有 token）一次性做标准 MHSA**：

```python
x = x + self.drop_path(self.attn(self.norm1(x)))
x = x + self.drop_path(self.mlp(self.norm2(x)))
```

## 具体做了什么

1. **LayerNorm**：对整段序列的每个 token 做归一化：`[B, N, C] -> [B, N, C]`
2. **一次 MHSA（多头注意力）**：
   - 在 `Attention` 里把 `x` 线性映射成 `q, k, v`，再按 `num_heads` 切分：
      `q, k, v` 形状均为 `[B, num_heads, N, C_head]`（`C_head = C / num_heads`）
   - 计算注意力分数：`attn = softmax(q @ k^T / sqrt(C_head))`，得到 `[B, num_heads, N, N]`
   - 加权求和得到输出：`[B, num_heads, N, C_head] -> [B, N, C]`
3. **残差 + DropPath**：加回 `x`（ResNet风格），稳定训练
4. **MLP + 残差**：`[B, N, C] -> [B, N, C]`，完成一层 Transformer block

## 直觉

- 把**所有时空位置**视作一个长序列，一次注意力就能让任意帧的任意patch与其它任何帧/patch交互，**时空信息完全混合**。
- 最简单，和标准 ViT block 完全一致（只是 token 数更长：含时间维）。

## 复杂度

- 注意力的时空 token 总数是 `N ≈ T * H * W`（忽略CLS的常数1），复杂度 ~ `O(N^2)` = `O((T H W)^2)`，当 `T`、`H`、`W` 稍大，**显著昂贵**。

## 适用场景与优缺点

- **优点**：实现简单、时空交互能力强；对长程依赖建模直接、明确。
- **缺点**：计算/显存开销大；训练时更容易受 `N` 增长限制。

------

# divided_space_time（分离式时空注意力）

> 这是一种「先时间后空间」的解耦策略：**时间注意力**只让同一空间位置跨帧交互，**空间注意力**只让同一帧内空间位置彼此交互，二者分步进行，再与 MLP 组合。它源于 TimeSformer 里的 divided attention 思路。

整体流程分三段：**Temporal → Spatial → MLP**。下面逐步拆解：

## A. Temporal（时间注意力）

### 代码关键段

```python
# 1) 去掉 CLS，只对 patch token 做 temporal
xt = x[:, 1:, :]  # [B, T*H*W, C]

# 2) 重新排列：每个 (h,w) 位置上一条长度为 T 的时间序列
xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)  # [(B*H*W), T, C]

# 3) 对长度为 T 的序列做 MHSA（temporal_attn）
res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))  # [(B*H*W), T, C]

# 4) 改回原形状并线性投影
res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)  # [B, T*H*W, C]
res_temporal = self.temporal_fc(res_temporal)  # [B, T*H*W, C]

# 5) temporal 残差与原 patch token 相加
xt = x[:, 1:, :] + res_temporal  # [B, T*H*W, C]
```

### 形状与直觉

- `[(B*H*W), T, C]` 意味着：**对每个空间位置 (h,w)**，把它在 **T 帧上的 token** 取出来形成一条时间序列，只在时间维上做注意力。
- 这样每个空间位置的 token 可以随时间相互交流，建模**同一点随时间的运动/变化**。
- 复杂度：对每条长度为 `T` 的序列做注意力，共有 `B*H*W` 条，复杂度 ~ `O(B H W * T^2)`。

## B. Spatial（空间注意力）

### 代码关键段

```python
# 1) 处理 CLS：将最初的 CLS token 复制 T 次，每帧一个
init_cls_token = x[:, 0, :].unsqueeze(1)              # [B, 1, C]
cls_token = init_cls_token.repeat(1, T, 1)            # [B, T, C]
cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)  # [(B*T), 1, C]

# 2) 按帧拆分 patch：每个样本每帧一个序列（长度 H*W），与对应的帧CLS拼接
xs = xt  # [B, T*H*W, C]
xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)      # [(B*T), H*W, C]
xs = torch.cat((cls_token, xs), 1)                                          # [(B*T), 1+H*W, C]

# 3) 对每一帧的空间 token 做 MHSA（共享参数 self.attn）
res_spatial = self.drop_path(self.attn(self.norm1(xs))) # [(B*T), 1+H*W, C]

# 4) 处理帧级 CLS：取每帧的 CLS 输出，再对 T 帧做平均，得到一个全视频 CLS
cls_token = res_spatial[:, 0, :]                               # [(B*T), C]
cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T) # [B, T, C]
cls_token = torch.mean(cls_token, 1, True)                     # [B, 1, C]

# 5) 空间残差（去掉帧级CLS，留下空间 token）
res_spatial = res_spatial[:, 1:, :]  # [(B*T), H*W, C]
res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)  # [B, T*H*W, C]
res = res_spatial
x = xt  # [B, T*H*W, C]
```

### 形状与直觉

- 把 **每一帧** 的 `H×W` 个 patch 和一个**帧内 CLS** 组成序列，做一次**空间注意力**。
- 这样建模的是同一帧内的**空间关系**（物体不同部位/不同位置之间的依赖）。
- 空间注意力完成后：
  - 帧内的 CLS 输出被**跨帧平均**，凝聚成**全视频 CLS**（把 T 个帧级 CLS 压成 1 个视频级 CLS）。
  - 空间 token（不含帧内 CLS）按原顺序重排回 `[B, T*H*W, C]`。
- 复杂度：对每帧长度 `H*W` 的序列做注意力，共有 `B*T` 条，复杂度 ~ `O(B T * (H W)^2)`。

## C. 残差融合 + MLP

### 代码关键段

```python
# 把初始的全局 CLS（init_cls_token）与空间/时间后的 token 残差拼回
x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)  # [B, 1+T*H*W, C]
x = x + self.drop_path(self.mlp(self.norm2(x)))
```

### 解释

- **CLS 处理要点**：
  - 进入 spatial 前，用 **初始 CLS** 复制成 T 个帧级 CLS，分别参与每帧的空间注意力
  - 空间注意力后，将 T 个帧级 CLS **平均**为一个全视频 CLS
  - 最后与原始 `init_cls_token` 对齐并相加（残差思路），保证 CLS 表征稳定
- **Patch token**：temporal + spatial 的两个残差**相加融合**回原 token，再接 MLP

## 直觉与收益

- **分而治之**：
  - Temporal 只在同一空间位置跨帧交互 → 擅长建模**时间运动/动态**
  - Spatial 只在同一帧内交互 → 擅长建模**空间结构/纹理**
- 计算上把一个巨大的 `N×N` 交互拆成两块更可控的子问题（T×T 与 (HW)×(HW)），通常**更省算/显存**，也更稳定。

## 复杂度（对比 joint）

- **joint**：`O((T H W)^2)`
- **divided**：`O(B H W * T^2) + O(B T * (H W)^2)`（去掉 B 后的相对量级）
  - 当 `T` 与 `H×W` 都不小，divided 仍然不便宜，但**显著优于 joint** 的全局 `N^2`。

## 适用场景与优缺点

- **优点**：
  - 对视频更友好的结构归纳偏置（时间/空间分离）
  - 计算/显存相对可控，往往**更容易 scale 到更长 T 或更大分辨率**
  - CLS 的帧级融合（平均）让全视频语义更稳定
- **缺点**：
  - 时空交互被拆分为两步，**跨帧跨位置的直接交互**需要通过两步间接实现（不如 joint 那样一次到位）
  - 需要更小心的张量重排与 CLS 处理（实现复杂度更高）

------

# 小结对照

| 特性       | joint_space_time          | divided_space_time                       |
| ---------- | ------------------------- | ---------------------------------------- |
| 注意力范围 | 一次性对所有时空 token    | 先同位点跨帧（T），再同帧内跨位置（H×W） |
| 交互路径   | 任意时空位置直接交互      | 跨帧跨位置需经两步（Temporal + Spatial） |
| 复杂度     | `O((T H W)^2)`            | `O(T^2·H·W) + O((H·W)^2·T)`              |
| 实现难度   | 低（标准 ViT block）      | 中（需拆分与重排、CLS 聚合）             |
| 归纳偏置   | 弱（更通用）              | 强（更贴合视频结构）                     |
| 适合       | 小/中等序列，注重全局混合 | 更长 T、更大分辨率；注重时空解耦         |

------

# 额外说明：`Attention(with_qkv)` 的作用

- 你提供的 `Attention` 支持 `with_qkv=False` 时**不再线性生成 Q/K/V**，而是把输入当成已经按多头切好的 `q=k=v`。
- 在这个 `Block` 中两次实例化 `Attention` 都使用默认 `with_qkv=True`，因此**都会自己从 `x` 里线性得到 Q/K/V**；没有走「外部已备好 qkv」的分支。

如果你之后想做**更细的可视化或调试**（比如打印每一步的形状、查看某个 head 的注意力热图），我可以把这段 `Block` 封装成带 `verbose` 开关的版本，并在关键 `rearrange` 前后插入断言与 `print`，或者给出一个**最小可运行示例**（随机视频张量）帮助你验证 `H、W、T、B` 的维度推导是否正确。