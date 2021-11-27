---
layout: post
title: Weekly Report
date: 2021-10-15 11:29
comments: true
external-url:
categories: Others
---

> my daily lift.

## 10.9～10.15


一开始尝试使用pytorch复现ptr-net，但是发现框架基础不足，该计划不太现实。随后，阅读了pytorch官方Tutorials及Docs，并做了一些小的练习。接下来的计划是：

1. 完成上周RNN的笔记
2. 再次尝试复现ptr-net

## 10.16～10.25


花费了几天的时间看[github ptr-net pytorch](https://github.com/shirgur/PointerNet)，难以推进下去，主要原因是attention的细节了解不足。在22号时找到[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)之后终于豁然开朗，到25号下午为止刚完成模型代码的debug，暂未进行实验。

在复现过程中，遇到的值得注意的一点就是：pytorch在反向传播的过程中，不允许需要梯度计算的参数进行就地操作inplace operation，否则会报错：

```text
one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [4, 6]], which is output 0 of SoftmaxBackward, is at version 1; expected version 0 instead. 
Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
```

常见的会犯错误的就地操作有

- `x += 1 `
- `alpha[visted] = 0`

由于是25号下午刚发现的bug，对于`就地操作`还有待进一步总结。

接下来的计划是，首先继续进行ptrnet的实验，查看模型复现效果，然后根据模型复现及实验中遇到的细节问题完善笔记。

## 10.26～10.31


这周进度有点慢了，总结如下：

10.26~10.27，由于理发师的一些过失，效率很低，以后应该及时调整心态；

10.28～10.29，尝试跑了代码，平均loss较高，在1左右，之后询问了师兄师姐如何调整参数，看了"ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!"的一点开头，写了一部分blog，没有写完；

10.30~10.31，参与了CCPC；

今晚之前把blog赶完吧。

## 11.1～11.7

- 阅读lil's[Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)，`Born for Translation`一节及对原论文的一些标注帮助我更清楚的认识了attention mechanism。

- 在阅读lil's blog 的时候发现她使用了Jekyll

  <img src="https://i.loli.net/2021/11/07/eMVDOwIbQ8FrlHA.png" alt=" " style="zoom:30%;" />

  查了一下之后认为比较适合，花了点时间了解Jekyll，熟悉git操作；在本地做了一些搭建尝试后做了线上部署。

- 做实验，调参。一开始使用随机产生的数据集，求得tsp最优解还要花费$O(2^n * n^2)$，慢。换为使用论文中[提供的数据集](http://goo.gl/NDcOIG)，做了TSP5，10，20的实验，在TSP5～10的情况下ptrnet可以得到很好的结果，提供的TSP20数据集比较小，同时结果也较差。此外lr设定过大容易训练到后面的时候颠簸；
- 读ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!，看到很多不认识的名词，去看了强化学习。

接下来：了解强化学习之后再去看看ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!

## 11.8～11.14

强化学习基础

## 11.15～11.21

本周因为大作业的事情，进度又比较少；

顺着上周继续看了一点点强化学习（因为论文里提到了基线），看了transformer（因为论文里提到了transformer），把ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!的网络模型看懂了；

下周想：看一下A3C、2OPT、禁忌搜索，看一下ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!的代码。





