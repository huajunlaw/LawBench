
---

# 📘 模型选择题评估模块文档

## 简介

该模块用于评估模型对 JEC\_KD（Judgment and Examination Corpus - Knowledge Distillation）数据集中的 **多项选择题**的回答准确性。

数据集中的每道题有 4 个选项（A、B、C、D），评估规则非常严格：

* **预测必须只包含正确选项字母（如 A）**
* **预测中不得包含任何其他选项字母**
* **如果预测中没有包含任何选项字母，则视为“弃答”**

---

## 📂 模块组成

包含两个核心函数：

1. `compute_jec_kd(data_dict)`
2. `multi_choice_judge(prediction, option_list, answer_token)`

---

## 🧩 函数详细说明

---

### 🔹 `compute_jec_kd`

```python
def compute_jec_kd(data_dict):
```

#### 📌 功能

对一个由题目、预测和参考答案组成的数据集进行批量评估，计算：

* **准确率（Accuracy）**
* **弃答率（Abstention Rate）**

#### 🧾 参数

| 名称          | 类型           | 说明                                                  |
| ----------- | ------------ | --------------------------------------------------- |
| `data_dict` | `List[Dict]` | 数据列表，每个元素包含 `origin_prompt`、`prediction`、`refr` 三个键 |

每个元素的示例格式如下：

```python
{
    "origin_prompt": "问题文本",
    "prediction": "模型输出的文本",
    "refr": "正确答案：A"  # 注意格式
}
```

#### 🔁 实现逻辑

* 提取参考答案中的选项字母。
* 调用 `multi_choice_judge` 判断预测是否准确。
* 统计准确题目数量和弃答数量。
* 最终计算准确率与弃答率。

#### ✅ 返回值

| 字段                | 类型      | 说明                 |
| ----------------- | ------- | ------------------ |
| `score`           | `float` | 所有题目的准确率（正确数 / 总数） |
| `abstention_rate` | `float` | 弃答的比例（弃答数 / 总数）    |

#### 📎 示例返回：

```python
{
    "score": 0.72,
    "abstention_rate": 0.1
}
```

---

### 🔹 `multi_choice_judge`

```python
def multi_choice_judge(prediction, option_list, answer_token):
```

#### 📌 功能

对一条预测结果进行判定：

* 是否 **准确回答**
* 是否 **弃答**

#### 🧾 参数

| 名称             | 类型          | 说明                                |
| -------------- | ----------- | --------------------------------- |
| `prediction`   | `str`       | 模型输出文本                            |
| `option_list`  | `List[str]` | 所有可能的选项，例如 `["A", "B", "C", "D"]` |
| `answer_token` | `str`       | 正确选项字母（如 `"B"`）                   |

#### 🔁 判断逻辑

* 统计 `prediction` 中包含哪些选项。
* 如果一个都没包含 → 弃答
* 如果只包含正确选项 → 正确
* 如果包含多个选项或错误选项 → 错误

#### ✅ 返回值

| 字段           | 类型    | 说明                   |
| ------------ | ----- | -------------------- |
| `score`      | `int` | 是否正确（1 = 正确；0 = 错误）  |
| `abstention` | `int` | 是否弃答（1 = 弃答；0 = 有作答） |

#### 📎 示例

```python
multi_choice_judge("我认为是A", ["A", "B", "C", "D"], "A")
# 返回: {"score": 1, "abstention": 0}

multi_choice_judge("我认为是A或B", ["A", "B", "C", "D"], "A")
# 返回: {"score": 0, "abstention": 0}

multi_choice_judge("我不确定", ["A", "B", "C", "D"], "C")
# 返回: {"score": 0, "abstention": 1}
```

---

## 🧪 示例数据

```python
data_dict = [
    {
        "origin_prompt": "以下哪项是正确的？",
        "prediction": "我认为答案是A",
        "refr": "正确答案：A"
    },
    {
        "origin_prompt": "以下哪项最不可能？",
        "prediction": "可能是B或C",
        "refr": "正确答案：B"
    },
    {
        "origin_prompt": "哪个说法是对的？",
        "prediction": "我不确定",
        "refr": "正确答案：C"
    }
]
```

---

## 🛠️ 使用建议

* 确保参考答案格式为 `"正确答案：X"`，其中 `X` 为选项字母。
* 可以将本模块与模型输出日志整合，形成一体化的评估流程。
* 可扩展为支持更多选项或多选任务。

---

## 🔚 总结

本模块适用于对语言模型在中文多项选择任务中的回答进行严格评估。它可用于：

* 模型调优后的验证
* 不同模型之间的比较
* 准确率与稳定性评估

如需更进一步的功能（如 Top-k 预测支持、多选题、详细错误报告），该模块可扩展实现。

---
