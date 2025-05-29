“**soft-F1**” 是 F1-score 的一种 **可微分（soft/differentiable）版本**，通常用在机器学习模型训练中，特别是深度学习里需要对损失函数进行梯度下降优化时。

---

## 为什么需要 soft-F1？

* 传统的 **F1-score** 是基于硬分类结果（预测类别是正或负）计算的，是离散的、不连续的。
* 这导致 F1-score 不能直接用作模型训练的目标函数（loss function），因为它不可微分，没法通过梯度下降更新参数。
* 为了优化模型的 F1-score，人们设计了 **soft-F1**，用模型预测的概率（连续值）来近似计算 F1，从而使其变得可微分。

---

## soft-F1 的基本思想

* 把传统的 TP、FP、FN 都换成 **软计数**，用预测概率代替硬分类标签。

假设：

* 真实标签：$y_i \in \{0,1\}$
* 预测概率：$\hat{y}_i \in [0,1]$

则：

$$
\text{soft TP} = \sum_i \hat{y}_i \cdot y_i
$$

$$
\text{soft FP} = \sum_i \hat{y}_i \cdot (1 - y_i)
$$

$$
\text{soft FN} = \sum_i (1 - \hat{y}_i) \cdot y_i
$$

然后用这些软计数计算 **soft Precision** 和 **soft Recall**：

$$
\text{soft Precision} = \frac{\text{soft TP}}{\text{soft TP} + \text{soft FP} + \epsilon}
$$

$$
\text{soft Recall} = \frac{\text{soft TP}}{\text{soft TP} + \text{soft FN} + \epsilon}
$$

其中 $\epsilon$ 是防止除零的小常数。

最后计算 soft-F1：

$$
\text{soft-F1} = 2 \times \frac{\text{soft Precision} \times \text{soft Recall}}{\text{soft Precision} + \text{soft Recall} + \epsilon}
$$

---

## 应用场景

* 训练模型时，直接用 soft-F1 作为损失函数的负值，最大化 F1。
* 适合处理不平衡数据，因为 F1 同时考虑了精确率和召回率。
* 在 NLP、医学影像等领域常用。

---
