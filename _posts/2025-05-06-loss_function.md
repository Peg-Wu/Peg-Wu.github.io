---
layout: post
title: self.loss_function
subtitle: 😢 The holiday is over
tags: [huggingface]
gh-repo: Peg-Wu
gh-badge: [star, fork, follow]
cover-img: /assets/img/blog_imgs/2025-05-06-loss_function/cover.jpg
comments: true
mathjax: true
author: Pengpeng Wu
---

{: .box-success}
🙂 最近看Qwen3模型的时候，发现transformers在模型的forward中直接调用了self.loss_function，但在init初始化时并没有显式添加这一属性，猜测huggingface团队可能对loss进行了统一和封装，于是就详细探索了一下~

- transformers：4.51.3
- 如下图，可以看到：foward中直接使用了self.loss_function计算了损失，Qwen3MoeForCausalLM，Qwen3MoeForSequenceClassification，Qwen3MoeForTokenClassification和Qwen3MoeForQuestionAnswering都是这样的做法

![1]( /assets/img/blog_imgs/2025-05-06-loss_function/1.png)

- 当我定位self.loss_funciton的位置时，发现其定义在父类PreTrainedModel中：
  - 首先，他会检测我们的对象中是否包含_loss_function属性，如果存在该属性，则直接使用self.\_loss_function定义的损失函数；这个属性可以通过在init方法中写self.loss_function = ... 来添加；这样就和之前老版本的写法是一致的~
  - 接着，如果我们没有手动添加self.loss_function = ... ，他会去检测对象中是否包含loss_type属性，然后通过LOSS_MAPPING字典映射找到loss_type对应的损失函数；如果loss_type不在LOSS_MAPPING中或者loss_type为None，则默认会使用LOSS_MAPPING中的“ForCausalLM”对应的损失函数

![2]( /assets/img/blog_imgs/2025-05-06-loss_function/2.png)

- 然后，我们需要看一下loss_type这个属性是怎么被添加的：
  - 从下图可以看到：loss_type这个属性是在调用父类PreTrainedModel的init方法时被自动添加的
  - 他会直接取我们最终定义的模型的类名，然后检测类名中是否包含LOSS_MAPPING中匹配的字段，将匹配的字段作为self.loss_type，如果没有匹配到任何字段，则self.loss_type为None

![3]( /assets/img/blog_imgs/2025-05-06-loss_function/3.png)

- 我们来看看LOSS_MAPPING长什么样子：
  - 其实看到这个字典，就已经能猜到huggingface官方做了什么了~😂
  - 总而言之，如果我们定义的模型做的是一些常见的任务，如：掩码任务（ForMaskedLM），自回归（ForCausalLM），序列分类（ForSequenceClassification）等，huggingface官方已经帮我们把这些常见任务的损失函数给写好了。我们只需要在写模型类名的时候包含LOSS_MAPPING中的相应字段，就可以直接调用self.loss_function使用这些损失函数了，不需要自己再重复造轮子~
  - huggingface工程上做的是真的好，值得学习~💕

![4]( /assets/img/blog_imgs/2025-05-06-loss_function/4.png)