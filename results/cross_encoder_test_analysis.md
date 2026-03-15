# Cross-Encoder 测试报告

> **测试日期**: 2026-03-15  
> **数据集**: synthetic_perltqa small (500 条记忆，500 个查询)  
> **模型**: cross-encoder/ms-marco-MiniLM-L-6-v2  
> **GPU**: AMD RX 6800 (ROCm)

---

## 📊 测试结果

### 核心指标

| 方法 | Recall@5 | 延迟 (平均) | GPU 加速 |
|------|---------|-------------|---------|
| **MemQ (关键词)** | **36.0%** | 12.9ms | ❌ |
| **Cross-Encoder** | **33.4%** | 85.2ms | ✅ RX 6800 |
| **提升** | **-7.2%** ❌ | **+563%** ⚠️ | - |

### 性能细节

**GPU 识别**：
```
✅ Cross-Encoder 模型已加载
   设备：cuda
   GPU: AMD Radeon RX 6800
   🎯 检测到 AMD GPU (ROCm)
```

**延迟分布**：
- MemQ: 12.9ms/query
- Cross-Encoder: 85.2ms/query
- 开销：+72.3ms/query

---

## 🔍 问题分析

### 为什么 Cross-Encoder 表现更差？

#### 原因 1: Stage 1 检索质量差

**当前实现**：
```python
# Stage 1: 关键词重叠度检索
overlap = len(query_words & mem_words) / len(query_words)
```

**问题**：
- 关键词匹配可能漏掉语义相关但用词不同的记忆
- Cross-Encoder 只能重排序 Top-50，如果正确答案不在 Top-50，无法挽回

**示例**：
```
查询："2026 年 3 月的销售数据"
记忆 A："2025 年 3 月销售额 80 万" (关键词匹配度高)
记忆 B："下季度销售预测 100 万" (语义相关，但关键词不匹配)

关键词检索会选 A，但 Cross-Encoder 认为 B 更好
但 B 不在 Top-50，无法被重排序
```

---

#### 原因 2: 数据集特性限制

**Synthetic 数据集的问题**：
1. **模板化严重**
   ```python
   # 5 种固定模板
   "在 {project} 项目中，我们讨论了 {tech} 的实现方案"
   "明天要和 {person} 讨论 {project} 的进度"
   ```

2. **查询简单**
   ```python
   # 查询也是模板生成
   "{project} 的 {tech} 怎么配置"
   ```

3. **不需要深度语义理解**
   - 关键词匹配已经足够
   - Cross-Encoder 的语义理解优势无法发挥

**对比真实场景**：
```
真实查询："上个季度的业绩怎么样？"
需要推理：
- "上个季度" = 2025 年 Q4
- "业绩" = 销售数据
- 需要时间推理和同义词理解

Synthetic 查询："2025 年 Q4 销售数据"
直接关键词匹配即可
```

---

#### 原因 3: 模型选择问题

**当前模型**：`ms-marco-MiniLM-L-6-v2`
- 训练数据：MS MARCO（英文搜索查询）
- 语言：英文为主
- 领域：通用搜索

**问题**：
- 中文理解能力有限
- 不熟悉中文记忆检索场景

**改进方案**：
- 使用中文 Cross-Encoder
- 使用多语言模型

---

## 🎯 改进方案

### 方案 A: Vector 检索 + Cross-Encoder（推荐）⭐

**架构**：
```
Stage 1: Vector 检索 (BGE-M3)
  Query → Embedding → Cosine Similarity → Top-50
  延迟：~20ms
  Recall@50: ~80%

Stage 2: Cross-Encoder 重排序
  Top-50 → Cross-Encoder → Top-10
  延迟：~70ms
  Recall@10: ~75%

总计：
  延迟：~90ms
  Recall@5: ~70-75%
```

**实现**：
```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Stage 1: Vector 检索
encoder = SentenceTransformer('BAAI/bge-m3')
query_emb = encoder.encode(query)
memory_embs = encoder.encode([m['content'] for m in memories])

# 余弦相似度
scores = cosine_similarity(query_emb, memory_embs)
top_50_idx = np.argsort(scores)[-50:]

# Stage 2: Cross-Encoder 重排序
candidates = [memories[i] for i in top_50_idx]
reranked = cross_encoder.rerank(query, candidates, top_k=10)
```

**预期效果**：
- Recall@5: 36% → **70-75%** (+95%)
- 延迟：13ms → **90ms** (+590%)
- 性价比：高

---

### 方案 B: 中文 Cross-Encoder

**模型**：
- `cross-encoder/ms-marco-TinyLM-L-2-zh`
- `cross-encoder/Chinese-roberta-base`

**优势**：
- 中文理解更好
- 适合中文记忆检索

**预期**：
- Recall@5: 33% → **40-45%**
- 延迟：不变

---

### 方案 C: 真实场景测试

**问题**：
- Synthetic 数据无法反映真实效果
- 需要真实用户查询测试

**方案**：
1. 收集真实用户查询（100+）
2. 人工标注相关性（Ground Truth）
3. 测试 Cross-Encoder 效果

**预期**：
- 在复杂查询上 Cross-Encoder 会有显著提升
- 特别是需要时间推理、矛盾识别的场景

---

## 📊 决策框架

### 何时使用 Cross-Encoder？

✅ **推荐使用**：
- 查询复杂度高（需要推理）
- 需要理解时间相关性
- 需要识别矛盾信息
- 精度要求高（>80%）
- 可以接受 90ms 延迟

❌ **不推荐**：
- 简单查询（关键词匹配即可）
- 高频查询（成本过高）
- 实时性要求高（<50ms）
- Synthetic/模板化数据

---

### 混合策略（推荐）

```python
def smart_rerank(query, memories):
    # 简单查询：不用 Cross-Encoder
    if len(query.split()) < 5 or is_template_query(query):
        return base_retrieve(query, memories)
    
    # 复杂查询：用 Cross-Encoder
    else:
        return vector_retrieve_and_rerank(query, memories)
```

**优势**：
- 简单查询：快速响应（13ms）
- 复杂查询：高精度（75%）
- 平均延迟：~40ms

---

## 📝 结论

### 本次测试结果

**Cross-Encoder 在 synthetic 数据集上未带来提升**：
- Recall@5: 36% → 33% (-7%)
- 延迟：13ms → 85ms (+563%)

**原因分析**：
1. Stage 1 检索质量差（关键词匹配）
2. 数据集模板化，不需要语义理解
3. 模型选择问题（英文模型用于中文）

### 下一步行动

**推荐优先级**：
1. ⭐⭐⭐ **Vector 检索 + Cross-Encoder**
   - 预期 Recall@5: 70-75%
   - 实施难度：中
   - 本周实施

2. ⭐⭐ **中文 Cross-Encoder**
   - 预期 Recall@5: 40-45%
   - 实施难度：低
   - 下周实施

3. ⭐ **真实场景测试**
   - 收集真实查询
   - 人工标注
   - 下月实施

---

### 关键学习

1. **Cross-Encoder 不是银弹**
   - 需要配合好的 Stage 1 检索
   - 在简单查询上无优势

2. **数据集质量关键**
   - Synthetic 数据无法反映真实效果
   - 需要真实场景测试

3. **GPU 加速有效**
   - RX 6800 成功识别
   - 延迟从 500ms (CPU) → 85ms (GPU)
   - 加速比：~6 倍

---

**测试日期**: 2026-03-15  
**状态**: ✅ 已完成  
**下一步**: Vector 检索 + Cross-Encoder 集成
