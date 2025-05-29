F-score（也叫 F-measure）是一种用于衡量分类模型精度的指标，它是 **精确率（Precision）**和**召回率（Recall）** 的调和平均。F-score 有不同的变种，其中最常用的是 **F1-score** 和 **F0.5-score**，它们在精确率和召回率之间赋予不同的权重。

---

## 1. **基础定义**

* **Precision（精确率）** = TP / (TP + FP)
  指所有被预测为正类的样本中，有多少是真正的正类。

* **Recall（召回率）** = TP / (TP + FN)
  指所有实际为正类的样本中，有多少被正确预测为正类。

* **Fβ-score 的通用公式**：

  $$
  F_\beta = (1 + \beta^2) \cdot \frac{{\text{Precision} \cdot \text{Recall}}}{{\beta^2 \cdot \text{Precision} + \text{Recall}}}
  $$

其中，β 是一个可调参数：

* 当 β = 1：**F1-score**，表示**精确率和召回率同等重要**。
* 当 β < 1：例如 β = 0.5（**F0.5-score**），表示**精确率比召回率更重要**。
* 当 β > 1：表示**召回率比精确率更重要**。

---

## 2. **F1-score**

$$
F_1 = 2 \cdot \frac{{\text{Precision} \cdot \text{Recall}}}{{\text{Precision} + \text{Recall}}}
$$

* F1 是精确率和召回率的调和平均。
* 用于平衡精确率和召回率，适合两者同等重要的场景。

---

## 3. **F0.5-score**

$$
F_{0.5} = 1.25 \cdot \frac{{\text{Precision} \cdot \text{Recall}}}{{0.25 \cdot \text{Precision} + \text{Recall}}}
$$

* 在这个公式中，**精确率的权重更高**。
* 适合那些**假阳性代价很高**的场景，比如垃圾邮件检测（不希望把正常邮件误判为垃圾邮件）。

---

## 4. **什么时候用哪个？**

| 指标             | 使用场景                    |
| -------------- | ----------------------- |
| **F1-score**   | 当精确率和召回率同样重要时。          |
| **F0.5-score** | 当**更在意精确率**（少出错、宁缺毋滥）时。 |
| **F2-score**   | 当**更在意召回率**（宁可错杀、不放过）时。 |

---
