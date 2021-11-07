---
layout: post
title: An Introduction to Reinforcement Learning
date: 2021-08-03 17:51
comments: true
external-url:
categories: MachineLearning
---

> 未完成

## Markov Decision Processes

> 内容参考自Stanford公开课CS221[Lecture 7: Markov Decision Processes - Value Iteration](https://www.youtube.com/watch?v=9g32v7bK3Co)

**MDP问题**

在学习一些算法如状压DP时，有这样的思想：目标解存在于状态空间中，我们从目前的**“状态”**，做出一定的**“动作”**，转移到下一个状态，直到搜到我们希望的最终解。

<img src="{{ '/assets/imgs/1.png' | relative_url }}" style="zoom:50%;">

考虑电车问题：

>$n$ 个节点编号 1 ～ $n$，某时刻选择走路将从 $s$ 到 $s+1$ 并花费1 mins，选择坐电车将从 $s$ 到 $2s$ 并花费 2 mins；
>
>如何用最短的时间从 $1$ 到 $n$ ？ 

在这类搜索问题中，我们做出“动作”，得到的“状态”、“奖励”等是**确定**的：我们在$s$选择做电车，一定会花费2 mins，并得到状态$2s$。对这类“确定”的问题，我们可以在搜索空间进行搜索，最终找到一系列的“动作”（如：走路 - 电车 - 电车 - 电车 - 走路）作为我们的目标解。

Markov Decision Processes (MDP)在上面问题的基础上，引入了**不确定性**：做出一个动作，但这个动作不是转移到下一个状态，而是以多种概率转移到多个状态。

> $n$ 个节点编号 1 ～ $n$，某时刻选择走路将从 $s$ 到 $s+1$ 并花费1 mins，选择坐电车将花费2mins，有$0.5$的可能从 $s$ 到 $2s$，有0.5的可能留在原地；
>
> 如何用最短的时间从 $1$ 到 $n$ ？ 

<img src="{{ '/assets/imgs/2.png' | relative_url }}" style="zoom:50%;">

对于该类问题，形式化的表述如下：


$$
\begin{align}
& States: \text{可能的states集合}
\\
& s_{state} \in States: \text{初态}
\\
& Actions(s): \text{状态s可能的动作a集合}
\\
& T(s, a, s'): \text{在状态s做出a之后到达}s'\text{的概率}
\\
& Reward(s, a, s'): \text{在状态s做出a之后到达}s'\text{的收益}
\\
&
IsEnd(s): 状态s是否为终态
\\
&
0 < \gamma < 1: \text{折扣系数discount factor, 默认为1}
\end{align}
$$


> $关于\gamma$的具体概念将在后面展开。

为了展示状态之间的转移关系，引入`choice node`的概念：在状态$s$，选择动作$a \in Action(s)$将转移到一个$\text{choice node}(s, a)$，该node有一至多条出边，每条边有两个值：$T(s, a, s')$及$Reward(s, a, s')$代表走该条边的概率以及获得的reward。考虑以下MDP问题

> 初态in，末态end，在每一步你有两种操作可选：stay继续游戏，有0.6的概率继续游戏并得到4¥，有0.4的概率结束游戏并得到5¥，quit结束游戏，有1的概率结束游戏并得到10¥；
>
> 如何得到尽可能多的钱币？

对该问题的状态转移关系可视化即为：

<img src="{{ '/assets/imgs/3.png' | relative_url }}" style="zoom:50%;">

上图中，蓝色节点代表状态节点$s$，引出的蓝色边代表做出的动作$a$，橘色代表`choice node`，引出的橘色边代表$s,a$可能的转移；在`in`状态，选择动作`stay`将转移到概率节点`(in, stay)`，该节点展示有 $T(in, stay, in)$ = 0.6的概率到达状态节点`in`，并获$Reward(in, stay, in)$ = 4，有 $T(in, stay, end)$ = 0.4的概率转移到`end`终态，并获得 $Reward(in, stay, in)$ = 5。

**解的形式**

对于确定性问题，最终解为一个最终的答案序列 (如走路 - 电车 - 电车 - 电车 - 走路)；当面对MDP问题时，情况可能更复杂一些，我们需要的解不再是一个序列，而是一组策略 $\pi $ : $s \Rightarrow a$的映射，其中 $s \in States$，$a \in Action(s)$ 。即对于每一个状态$s$，我们可以使用$\pi(s)$指该步的动作。

<img src="{{ '/assets/imgs/4.png' | relative_url }}" style="zoom:50%;">

**评估解**

对于一个状态-动作path：$s_0, a_1r_1s_1, a_2r_2s_2, \dots $，其中$s$代表状态，$r$代表收益，在状态$s_0$时认为path总收益是由以下迭代式得到的：


$$
\begin{align}
u_1 & = r_1 + \gamma u_2
\\
u_2 & = r_2 + \gamma u_3
\\
& \dots
\end{align}
$$
即$u_1 = r_1 + \gamma r_2 + \gamma ^2 r_3 + \gamma ^3 r_4 +\dots$ 

可以看到，我们不是做简单的累加 $\sum$ ，而是增加了一个影响因子 $\gamma$ ，其代表我们对“未来”赋予的影响度。当$\gamma = 1$时，代表我们认为未来与现在同样重要，当$\gamma = 0$时，代表我们只关注于现在。后文中可以看到，$\gamma$ 的增加对保持算法收敛同样是有益的。

当位于状态$s$时，对于策略$\pi(s)$，由于不确定性$\pi(s)$将引出许多path

<img src="{{ '/assets/imgs/5.png' | relative_url }}" style="zoom:50%;">

定义$V_\pi(s)$为从$s$状态开始执行策略$\pi(s)$的期望收益，$Q_\pi(s, a)$为概率节点$(s, a)$的期望收益

<img src="{{ '/assets/imgs/6.png' | relative_url }}" style="zoom:50%;">

可以得出以下的迭代式


$$
V_\pi(s) = 

\begin{cases}
0  & \text{if  } IsEnd(s)= True\\
Q_\pi(s, \pi(s))  &  otherwise.
\end{cases}

\\

Q_\pi(s, a) = \sum_{s'} T(s, a, s')[R(s, a, s') + \gamma V_\pi(s')]
$$


 以上面提到的赌钱游戏为例，设$\gamma = 1$，$\pi(in) = stay$


$$
\begin{align}
V_{\pi}(end) & = 0
\\
V_{\pi}(in) & = Q_\pi(in, stay) = 0.4 * (5 + V_{\pi}(end)) + 0.6 * (4 + V_{\pi}(end))
\\
& = 6.5
\end{align}
$$
定义$V_{opt}(s)$为从$s$状态开始执行策略$\pi(s)$的最大期望收益，$Q_{opt}(s, a)$为概率节点$(s, a)$的最大期望收益，有


$$
V_{opt}(s) = 

\begin{cases}
0  & \text{if  } IsEnd(s)= True\\
max_{a\in Actions(s)}Q_\pi(s, a)  &  otherwise.
\end{cases}

\\
Q_{opt}(s, a) = \sum_{s'} T(s, a, s')[R(s, a, s') + \gamma V_{opt}(s')]
$$




至此，根据对策略$\pi(s)$收益的定义，可以得知最优解为


$$
\pi_{opt}(s) = \text{argmax}_{a\in Actions(s)}Q_{opt}(s, a)
$$




