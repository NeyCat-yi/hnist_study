# Word Embedding 与序列建模：代码逐行详解与实战笔记

> 适用场景：理解如何从“词表索引序列”得到“稠密向量表示（embedding）”，并为后续的 RNN/Transformer 等序列模型做输入准备。

---

## 1. 背景与目标

我们有**源序列**（src）与**目标序列**（tgt）。每个序列由**词表索引**组成（整数 ID），长度不一。为了并行训练，需要：

1. 把不同长度的序列做 **padding** 到统一长度（常用 padding 值是 `0`）；  
2. 用 `nn.Embedding` 把索引序列变为**稠密向量序列**（形状一般是 `[batch, seq_len, model_dim]`）。

本笔记详细解释以下代码（略有润色排版）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 关于word embedding，以序列建模为例
# 构建序列，序列的字符以其在词表中的索引的形式表示
batch_size = 2

# 单词表大小
max_num_src_words = 8
max_num_tgt_words = 8

# 序列的最大长度
max_src_seq_len = 5
max_tgt_seq_len = 5

# 维度
model_dim = 8

# 把 源序列 和 目标序列 的长度定下来
src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# 单词索引构成的句子, 构建 batch，并且做了 padding, 默认值为 0
src_seq = torch.cat([
    torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max_src_seq_len - L)), 0)
    for L in src_len
])
tgt_seq = torch.cat([
    torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max_tgt_seq_len - L)), 0)
    for L in tgt_len
])

# 构造 embedding（注意：可选 padding_idx=0）
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words + 1, model_dim)
src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)
```

---

## 2. 关键超参数

- `batch_size = 2`：一个 batch 有两条样本。
- 词表大小：
  - `max_num_src_words = 8`
  - `max_num_tgt_words = 8`
  - **注意**：后面 `nn.Embedding(max_num_*_words + 1, ...)` 多加了 `1`，通常是为了**预留 index=0 作为 padding**（也方便未来词表扩充）。
- 序列最大长度：`max_src_seq_len = max_tgt_seq_len = 5`。
- 向量维度：`model_dim = 8`，也就是每个 token 会被映射到长度为 8 的向量。

---

## 3. 序列真实长度（未 padding 前）

```python
src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([4, 3]).to(torch.int32)
```

- `src_len` 表示本 batch 的两条 **源序列** 真正长度分别为 2、4。  
- `tgt_len` 表示两条 **目标序列** 真正长度分别为 4、3。  
- 这些长度仅用来指导后续**随机生成索引序列**与**padding**，本身不会被直接喂给 Embedding。

> 小提示：长度张量用 `int32` 或 `int64` 都行；真正用于索引的张量必须是 **长整型（`torch.long` / `int64`）**。

---

## 4. 生成批序列并做 Padding

以 `src_seq` 为例（`tgt_seq` 同理）：

```python
src_seq = torch.cat([
    torch.unsqueeze(
        F.pad(
            torch.randint(1, max_num_src_words, (L,)),  # 生成长度为 L 的随机索引序列
            (0, max_src_seq_len - L)                    # 在右侧 pad 到统一长度
        ),
        0  # 在第 0 维（batch 维）增加一维，形状从 [L] -> [1, max_src_seq_len]
    )
    for L in src_len
])
```

逐步理解：

1. **随机索引序列**：`torch.randint(1, max_num_src_words, (L,))`  
   - 取值范围是 `[1, max_num_src_words-1]`（上界开区间），即 `[1, 7]`。  
   - 这样可以**避免 0**（留给 padding）。
2. **Padding**：`F.pad(x, (0, max_len - L))`  
   - 对 1D 序列，`(left, right)` 表示在最后一维左/右侧分别补多少个值，默认补 `0`。  
   - 这里在**右侧**补 `max_len - L` 个 `0`，让序列变成统一长度 `max_len=5`。
3. **Unsqueeze + Cat**：  
   - `unsqueeze(..., 0)` 把形状 `[max_len]` 变成 `[1, max_len]`，方便后续按 batch 维拼接。  
   - `torch.cat([...])` 把两条样本在第 0 维拼起来，最终 `src_seq` 形状是 **`[batch_size, max_src_seq_len] = [2, 5]`**。

`tgt_seq` 同理，最终形状也是 `[2, 5]`。序列中的 0 都是 padding 位置。

---

## 5. 构造 Embedding 表并查表

```python
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words + 1, model_dim)

src_embedding = src_embedding_table(src_seq)  # [2, 5, 8]
tgt_embedding = tgt_embedding_table(tgt_seq)  # [2, 5, 8]
```

- `nn.Embedding(num_embeddings, embedding_dim)` 会创建一个形状为 `[num_embeddings, embedding_dim]` 的**查找表**。  
- 输入是**索引张量**（`long` 类型，形如 `[2, 5]`），输出是对应的**向量张量**（`float` 类型，形如 `[2, 5, 8]`）。  
- 这里把 `num_embeddings` 设为 `max_num_*_words + 1`，为 index=0 的 padding 预留空间。

> **强烈建议**：在 `Embedding` 里显式指定 `padding_idx=0`：
>
> ```python
> nn.Embedding(max_num_src_words + 1, model_dim, padding_idx=0)
> ```
>
> 这样，padding 的行向量在训练中不会被更新（梯度为 0），可避免 “模型把注意力学到 padding 上”。

---

## 6. 形状与数据类型总览

| 名称 | 作用 | 形状 | dtype | 备注 |
|---|---|---|---|---|
| `src_len` | 源序列真实长度 | `[2]` | `int32` | 值为 `[2, 4]` |
| `tgt_len` | 目标序列真实长度 | `[2]` | `int32` | 值为 `[4, 3]` |
| `src_seq` | 源索引序列（已 padding） | `[2, 5]` | `long` | 取值范围含 0（padding）与 1~7 |
| `tgt_seq` | 目标索引序列（已 padding） | `[2, 5]` | `long` | 同上 |
| `src_embedding` | 源序列的向量表示 | `[2, 5, 8]` | `float32` | 每个 token → 8 维向量 |
| `tgt_embedding` | 目标序列的向量表示 | `[2, 5, 8]` | `float32` | 同上 |

---

## 7. 随机性的说明与可复现实验

由于使用了 `torch.randint(...)`，每次运行会得到不同的索引序列。若希望**复现实验**，可在生成前设置随机种子：

```python
torch.manual_seed(42)
```

---

## 8. 常见坑位与调试建议

1. **索引越界**：如果 `nn.Embedding(num_embeddings=9, ...)`，允许的索引是 `0~8`。若你用到了 `9` 会报错。  
   - 本例中 `torch.randint(1, 8, ...)` 的上界是开区间，不会生成 `8`，因此安全。
2. **dtype 不匹配**：传给 `Embedding` 的输入必须是 `long`。如果你从别处拿的张量是 `int32/float`，需 `.long()`。  
3. **忘记 `padding_idx`**：虽然不是硬性错误，但最好设置 `padding_idx=0`，避免模型把 padding 学出非零向量。  
4. **mask 的使用**：下游（如 Transformer）需要利用长度或 0 值位置去构造 **padding mask**，屏蔽注意力。

---

## 9. 进阶：由长度或 0 值构造 Padding Mask

如果你保留了 `src_len`，可以直接基于长度构造 mask（`True` 表示是 padding 位置，需要被“遮住”）：

```python
# 基于长度构造 key padding mask: [batch, seq_len]
batch_indices = torch.arange(max_src_seq_len).unsqueeze(0)  # [1, seq_len]
src_key_padding_mask = batch_indices >= src_len.unsqueeze(1)  # [2, 5], bool

# 或者：基于索引是否为 0 来构造
src_key_padding_mask_alt = (src_seq == 0)  # [2, 5], bool
```

在 PyTorch 的 `nn.Transformer` 中，`src_key_padding_mask` 可直接作为参数传入，用于注意力计算时屏蔽 padding。

---

## 10. 完整可复现示例（改进版，含 padding_idx 与 mask）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)  # 复现

batch_size = 2
max_num_src_words = 8
max_num_tgt_words = 8
max_src_seq_len = 5
max_tgt_seq_len = 5
model_dim = 8

src_len = torch.tensor([2, 4], dtype=torch.int32)
tgt_len = torch.tensor([4, 3], dtype=torch.int32)

# 生成并 padding
def make_batch(lengths, max_len, vocab_high_exclusive):
    seqs = []
    for L in lengths.tolist():
        # 1~(vocab_high_exclusive-1)，避免 0（留给 padding）
        x = torch.randint(1, vocab_high_exclusive, (L,), dtype=torch.long)
        x = F.pad(x, (0, max_len - L), value=0)  # 右侧补 0
        x = x.unsqueeze(0)  # [1, max_len]
        seqs.append(x)
    return torch.cat(seqs, dim=0)  # [batch, max_len]

src_seq = make_batch(src_len, max_src_seq_len, max_num_src_words)
tgt_seq = make_batch(tgt_len, max_tgt_seq_len, max_num_tgt_words)

# Embedding（显式指定 padding_idx=0）
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim, padding_idx=0)
tgt_embedding_table = nn.Embedding(max_num_tgt_words + 1, model_dim, padding_idx=0)

src_embedding = src_embedding_table(src_seq)  # [2, 5, 8]
tgt_embedding = tgt_embedding_table(tgt_seq)  # [2, 5, 8]

# 由长度构造 padding mask（True 表示 padding）
idx = torch.arange(max_src_seq_len).unsqueeze(0)
src_key_padding_mask = idx >= src_len.unsqueeze(1)  # [2, 5], bool

print("src_seq:\n", src_seq)
print("src_embedding shape:", src_embedding.shape)
print("src_key_padding_mask:\n", src_key_padding_mask)
```

---

## 11. 小结

- 用 `randint + pad + unsqueeze + cat` 可以从**不同长度的索引序列**构造出**定长批序列**；  
- 用 `nn.Embedding` 把**索引**转成**稠密向量**，形状为 `[batch, seq_len, model_dim]`；  
- 记得为 padding 预留索引 `0`，并设置 `padding_idx=0`，下游用 **padding mask** 屏蔽无效位置。

祝你在后续的 RNN/Transformer 中玩得开心！🚀
