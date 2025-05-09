---
layout: post
title: attention_mask
subtitle: 🫥 encoder- and decoder-only standard attention_mask
tags: [huggingface]
gh-repo: Peg-Wu
gh-badge: [star, fork, follow]
cover-img: /assets/img/blog_imgs/2025-05-09-attention_mask/cover.png
comments: true
mathjax: true
author: Pengpeng Wu
---

{: .box-success}
🚅 看了一下huggingface官方是怎么写标准的encoder-only和decoder-only的attention_mask的，写的还是非常有意思的~

- transformers：4.51.3



- 我们都知道，transformers的attention_mask在传入模型前是二维的张量，形状为batch_size x seq_len，对于每个样本，**<u>有效token的位置是1，填充token的位置是0</u>**
- 对attention_mask的进一步处理在模型的forward阶段，**<u>一般写在模型的backbone中</u>**，下面我们来看一下代码：
- 代码位置：**transformers.modeling_utils.ModuleUtilsMixin.get_extended_attention_mask**，PreTrainedModel继承了ModuleUtilsMixin类，所以可以直接在模型的forward中使用get_extended_attention_mask函数

```python
import torch

"""encoder-only mask"""
# (1, 5) -> (batch_size, seq_len)
attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

# (1, 1, 1, 5) -> (batch_size, 1, 1, seq_len)
attention_mask = attention_mask[:, None, None, :]

# 1 -> 0; 0 -> -inf
attention_mask = (1 - attention_mask) * torch.finfo(type=torch.float32).min
attention_mask  # (batch_size, 1, 1, seq_len)

"""
Output:
tensor([[[[-0.0000e+00, -0.0000e+00, -0.0000e+00, -3.4028e+38, -3.4028e+38]]]])
"""


"""decoder-only mask"""
# (1, 5) -> (batch_size, seq_len)
attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

batch_size, seq_len = attention_mask.shape

seq_ids = torch.arange(seq_len)

# (1, 5, 5) -> (batch_size, seq_len, seq_len)
causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_len, 1) <= seq_ids[None, :, None]

# (1, 1, 5, 5) -> (batch_size, 1, seq_len, seq_len)
attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]

# 1 -> 0; 0 -> -inf
attention_mask = (1 - attention_mask) * torch.finfo(type=torch.float32).min
attention_mask  # (batch_size, 1, seq_len, seq_len)

"""
Output:
tensor([[[[-0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
          [-0.0000e+00, -0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
          [-0.0000e+00, -0.0000e+00, -0.0000e+00, -3.4028e+38, -3.4028e+38],
          [-0.0000e+00, -0.0000e+00, -0.0000e+00, -3.4028e+38, -3.4028e+38],
          [-0.0000e+00, -0.0000e+00, -0.0000e+00, -3.4028e+38, -3.4028e+38]]]])
"""
```

- 不论是encoder-only还是decoder-only的模型，最终生成的attention_mask都是4维的张量，其中batch_size后面的1维其实是给nheads预留的，后面会通过广播机制将attention_mask与attention_weight**<u>相加</u>**（query和key点积并缩放后，softmax前的attention_weight，形状是batch_size x nheads x seq_len x seq_len）
- attention_mask中0的部分表示不被掩码，负无穷的部分表示被掩码
- 👾 总体上写的还是非常有意思的，包含很多增加维度和广播运算的操作，值得反复观看~