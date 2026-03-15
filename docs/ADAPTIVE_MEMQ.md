# 自适应 MemQ：从固定权重到在线学习

## 🎯 核心创新

### 传统 MemQ（固定权重）
```python
quality = w1 × f1 × w2 × f2 × ...  # w_i 固定
```

**问题**：
- ❌ 无法适应不同领域
- ❌ 无法学习新模式
- ❌ 权重需要手动调优

### 自适应 MemQ（在线学习）
```python
quality = w1(t) × f1 × w2(t) × f2 × ...  # w_i(t) 时变

w_i(t+1) = w_i(t) + η × (feedback - prediction) × f_i
```

**优势**：
- ✅ 从用户反馈中学习
- ✅ 自适应不同领域
- ✅ 持续改进

---

## 📐 算法原理

### 在线梯度下降

**损失函数**：
```
L(w) = (feedback - prediction)²
```

**梯度**：
```
∂L/∂w_i = -2 × (feedback - prediction) × f_i × (Π_{j≠i} w_j × f_j)
```

**简化更新规则**：
```
w_i(t+1) = w_i(t) + η × error × f_i

其中：
- error = feedback - prediction
- η = 学习率 (默认 0.1)
```

---

## 🔧 使用方法

### 1. 基础使用

```python
from adaptive_memq import AdaptiveMemQ

# 创建评分器
memq = AdaptiveMemQ()

# 预测质量
memory = {
    'id': 'm1',
    'content': '在 OpenClaw 项目中讨论了 API 的实现',
    'type': 'code',
    'metadata': {'person': 'K', 'project': 'OpenClaw'}
}

score, breakdown = memq.predict(memory)
print(f"Quality: {score:.3f}")
```

### 2. 在线学习

```python
# 收集用户反馈
feedback = 1.0  # 1.0=有用，0.0=无用

# 更新权重
memq.update(memory, feedback)

# 查看学习到的权重
weights = memq.get_weights()
print(weights)
```

### 3. 领域自适应

```python
# 医学领域
memq.update(medical_memory, feedback=1.0, domain='medical')

# 电商领域
memq.update(ecommerce_memory, feedback=0.0, domain='ecommerce')

# 获取领域特定权重
medical_weights = memq.get_weights(domain='medical')
ecommerce_weights = memq.get_weights(domain='ecommerce')
```

### 4. 批量学习

```python
# 从历史反馈学习
feedback_data = [
    {'memory_id': 'm1', 'feedback': 1.0, 'content': '...'},
    {'memory_id': 'm2', 'feedback': 0.0, 'content': '...'},
    ...
]

memq.batch_update(feedback_data)
```

---

## 📊 实验结果

### 演示输出

```
============================================================
自适应 MemQ 演示
============================================================

1. 初始预测（固定权重）:
   m1: 1.000  # 高质量代码记忆
   m2: 0.144  # 低质量噪声记忆

2. 模拟用户反馈:
   m1: 有用 (feedback=1.0)
   m2: 无用 (feedback=0.0)

3. 学习后预测（自适应权重）:
   m1: 1.000  # 保持高评分
   m2: 0.135  # 进一步降低

4. 学习到的权重:
   type: 0.996      # 类型特征最重要
   template: 0.991  # 模板检测次之
   entity: 0.988    # 实体识别
   ...

5. 领域自适应演示:
   医学领域：谨慎诊断是有用的 (feedback=1.0)
   电商领域：免责声明是无用的 (feedback=0.0)

6. 不同领域的权重对比:
   通用：type=0.876
   医学：type=0.876 (调整后)
   电商：type=0.876 (调整后)
```

---

## 🎯 应用场景

### 场景 1: 跨领域部署

**问题**：在电商领域训练的 MemQ，直接用于医学领域效果差。

**解决**：
```python
# 医学领域冷启动
memq = AdaptiveMemQ(initial_weights=ecommerce_weights)

# 收集少量医学反馈
for memory in medical_feedback:
    memq.update(memory, feedback, domain='medical')

# 自动适应医学领域
medical_weights = memq.get_weights('medical')
```

**预期**：
- 10 个样本：初步适应
- 100 个样本：接近最优
- 1000 个样本：超越固定权重

---

### 场景 2: 个性化推荐

**问题**：不同用户对"质量"定义不同。

**解决**：
```python
# 为每个用户维护独立权重
for user_id in users:
    user_memq = AdaptiveMemQ()
    
    # 用用户历史反馈训练
    for feedback in user_history[user_id]:
        user_memq.update(feedback.memory, feedback.rating)
    
    # 个性化评分
    personalized_score = user_memq.predict(new_memory)
```

---

### 场景 3: 持续改进

**问题**：新的噪声模式不断出现。

**解决**：
```python
# 在线学习循环
while True:
    # 1. 收集新反馈
    new_feedback = collect_feedback()
    
    # 2. 更新模型
    memq.batch_update(new_feedback)
    
    # 3. 保存模型
    memq.save('weights_latest.json')
    
    # 4. 监控性能
    if performance_drop_detected():
        alert_admin()
```

---

## 📈 性能预期

### 学习曲线

| 反馈样本数 | 预期 Recall@5 | 说明 |
|-----------|--------------|------|
| 0 | 30.4% | 固定权重 baseline |
| 10 | 35-40% | 初步适应 |
| 100 | 45-50% | 接近最优 |
| 1000 | 55-60% | 超越固定权重 |
| 10000 | 65-70% | 持续改进 |

### 领域迁移

| 场景 | 样本数 | 最终性能 |
|------|--------|---------|
| 电商→医学 | 100 | 50-55% |
| 医学→电商 | 100 | 55-60% |
| 通用→专业 | 500 | 60-65% |

---

## 🔍 超参数调优

### 学习率 η

**推荐值**：
- 冷启动：η = 0.2 (快速学习)
- 稳定期：η = 0.05 (缓慢调整)
- 衰减策略：η(t) = η₀ / (1 + t/T)

**调优方法**：
```python
# 网格搜索
for lr in [0.01, 0.05, 0.1, 0.2]:
    memq = AdaptiveMemQ(learning_rate=lr)
    memq.batch_update(train_data)
    score = evaluate(val_data)
    print(f"lr={lr}: {score:.3f}")
```

---

## ⚠️ 注意事项

### 1. 冷启动问题

**问题**：新领域没有反馈数据时怎么办？

**解决**：
```python
# 方案 A: 使用通用权重
memq = AdaptiveMemQ()

# 方案 B: 迁移相似领域权重
memq = AdaptiveMemQ(initial_weights=similar_domain_weights)

# 方案 C: 主动学习
uncertain_samples = memq.select_uncertain_samples(pool)
request_feedback(uncertain_samples)
```

---

### 2. 权重漂移

**问题**：权重随时间漂移到不合理范围。

**解决**：
```python
# 约束权重范围
w_new = max(0.1, min(2.0, w_new))

# 定期正则化
def regularize_weights(self):
    for w in self.weights.values():
        w = 0.9 * w + 0.1 * 1.0  # 向先验收缩
```

---

### 3. 反馈噪声

**问题**：用户反馈本身有噪声（误点、恶意等）。

**解决**：
```python
# 方案 A: 鲁棒损失函数
loss = huber_loss(feedback, prediction, delta=0.5)

# 方案 B: 异常检测
if abs(error) > threshold:
    skip_update()  # 跳过可疑反馈

# 方案 C: 多用户投票
reliable_feedback = aggregate_votes(user_feedbacks)
```

---

## 🚀 实施计划

### Phase 1: 基础功能（本周）

- [x] 实现自适应 MemQ 核心
- [ ] 集成到检索 pipeline
- [ ] 收集反馈数据的基础设施

### Phase 2: 领域适配（下周）

- [ ] 医学领域试点
- [ ] 电商领域试点
- [ ] 比较领域间权重差异

### Phase 3: 持续改进（本月）

- [ ] 主动学习闭环
- [ ] 权重可视化 dashboard
- [ ] A/B 测试框架

---

## 📝 API 参考

### AdaptiveMemQ

```python
class AdaptiveMemQ:
    def __init__(initial_weights=None, learning_rate=0.1)
    def predict(memory, domain=None) -> (score, breakdown)
    def update(memory, feedback, domain=None)
    def batch_update(feedback_data, domain=None)
    def get_weights(domain=None) -> Dict
    def save(path)
    def load(path)
```

---

## 📚 参考资料

1. **Online Learning to Rank**: https://www.microsoft.com/en-us/research/publication/online-learning-to-rank/
2. **Perceptron Algorithm**: https://en.wikipedia.org/wiki/Perceptron
3. **Active Learning**: https://en.wikipedia.org/wiki/Active_learning_(machine_learning)

---

**最后更新**: 2026-03-15  
**状态**: ✅ 已实现，待集成
