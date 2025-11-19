---
layout: post
title: compute_metrics
subtitle: 📚 compute_metrics in Trainer
tags: [huggingface]
gh-repo: Peg-Wu
gh-badge: [star, fork, follow]
cover-img: /assets/img/blog_imgs/2025-11-19-compute_metrics/cover.png
comments: true
mathjax: true
author: Pengpeng Wu
---

{: .box-success}
⚙️ 介绍传入Trainer的compute_metrics函数中EvalPrediction.predictions和EvalPrediction.label_ids分别是什么?

- 先来看一下用法：

```python
import evaluate

clf_metrics = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return clf_metrics.compute(predictions=preds, references=labels)

# compute_metrics: Callable[[EvalPrediction], Dict]
```
![compute_metrics](/assets/img/blog_imgs/2025-11-19-compute_metrics/compute_metrics.png)
