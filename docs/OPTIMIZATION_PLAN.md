# MemQ 性能优化方案

## 问题分析

### 当前性能

| 规模 | Recall@5 | 预期 | 差距 |
|------|----------|------|------|
| baseline (200) | 72.5% | 80% | -7.5% |
| small (500) | 63.4% | 75% | -11.6% |
| medium (2000) | **30.4%** | 70% | **-39.6%** ❌ |

### 根因分析

#### 1. Embedding 模型容量不足

**问题**：
- Qwen3-4B 只有 4B 参数
- 2000 条记忆的语义空间过于拥挤
- 余弦相似度区分度下降

**证据**：
```
MRR: 0.185 → 平均排名 5.41
说明相关记忆被埋在后面
```

**解决方案**：
- 升级到 Qwen3-8B
- 或使用 BGE-M3（专为检索优化）

#### 2. LanceDB 索引参数问题

**问题**：
- 默认 IVF_PQ 参数不适合 2000+ 条数据
- num_partitions 太小，搜索范围有限

**默认配置**：
```python
num_partitions = 1  # 太小！
num_sub_vectors = 16
```

**建议配置**：
```python
num_partitions = 64  # sqrt(2000) ≈ 45
num_sub_vectors = 32  # 增加精度
```

#### 3. 混合检索权重问题

**问题**：
- 默认 RRF 权重 (k=60) 不适合
- BM25 和 Vector 的贡献不平衡

**当前**：
```python
score = 1/(60+rank_vector) + 1/(60+rank_bm25)
```

**建议**：
```python
# 增加 Vector 权重
score = 0.7/(40+rank_vector) + 0.3/(40+rank_bm25)
```

#### 4. 数据质量问题

**问题**：
- 模板化严重（5 种模板）
- 语义多样性不足
- 查询 - 记忆匹配过于简单

**改进**：
- 增加模板数量（20+）
- 引入 paraphrase 增强
- 添加负样本训练

---

## 优化方案

### 方案 1: 优化索引参数 ⭐⭐⭐

**实施难度**：低  
**预期提升**：+15-20%  
**耗时**：1-2 小时

```python
# 创建优化的索引
table.create_index(
    metric="cosine",
    index_type="IVF_PQ",
    num_partitions=64,
    num_sub_vectors=32,
    index_cache_size=256
)
```

**验证**：
```bash
python scripts/test_index_params.py
```

### 方案 2: 升级 Embedding 模型 ⭐⭐⭐⭐

**实施难度**：中  
**预期提升**：+20-30%  
**耗时**：2-4 小时

**选项 A: Qwen3-8B**
```python
model = "Qwen3-Embedding-8B"
# 优点：与当前兼容
# 缺点：速度慢 2 倍
```

**选项 B: BGE-M3**
```python
model = "BAAI/bge-m3"
# 优点：专为检索优化，支持多语言
# 缺点：需要额外安装
```

**验证**：
```bash
python scripts/test_embedding_models.py
```

### 方案 3: 调整混合权重 ⭐⭐

**实施难度**：低  
**预期提升**：+5-10%  
**耗时**：30 分钟

```python
# 测试不同权重
alphas = [0.5, 0.6, 0.7, 0.8]
for alpha in alphas:
    score = alpha * vector_score + (1-alpha) * bm25_score
```

**验证**：
```bash
python scripts/test_hybrid_weights.py
```

### 方案 4: 增加重排序 ⭐⭐⭐⭐⭐

**实施难度**：中  
**预期提升**：+10-15%  
**耗时**：2-3 小时

```python
# 使用 Cross-Encoder rerank top-50
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 检索 top-50
results = retrieve(query, k=50)

# 重排序
pairs = [[query, r['text']] for r in results]
scores = reranker.predict(pairs)

# 取 top-10
final_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:10]
```

**验证**：
```bash
python scripts/test_reranker.py
```

### 方案 5: 改进数据生成 ⭐⭐⭐

**实施难度**：高  
**预期提升**：+10-20%  
**耗时**：1-2 天

```python
# 1. 增加模板多样性
MEMORY_TEMPLATES = {
    "code": [
        "在 {project} 项目中，我们讨论了 {tech} 的实现方案",
        "{person} 负责 {project} 的 {tech} 模块开发",
        "关于 {project} 的 {tech} 问题，{person} 提出了新方案",
        # ... 20+ 模板
    ],
    # ...
}

# 2. 使用 LLM 生成 paraphrase
def paraphrase(text):
    prompt = f"Rewrite this sentence while keeping the meaning: {text}"
    return llm.generate(prompt)

# 3. 添加负样本
def generate_hard_negative(memory):
    # 保持关键词，改变语义
    ...
```

---

## 实施计划

### Phase 1: 快速优化（今天）

1. **优化索引参数** (1 小时)
   ```bash
   python scripts/apply_index_optimization.py
   ```

2. **调整混合权重** (30 分钟)
   ```bash
   python scripts/test_hybrid_weights.py --alpha 0.7
   ```

3. **重新测试 medium** (1 小时)
   ```bash
   python scripts/final_ab_test.py --optimized
   ```

**预期**：Recall@5 从 30.4% → **50-55%**

### Phase 2: 模型升级（明天）

1. **测试 Qwen3-8B** (2 小时)
   ```bash
   python scripts/test_embedding_models.py --model Qwen3-8B
   ```

2. **测试 BGE-M3** (2 小时)
   ```bash
   python scripts/test_embedding_models.py --model BGE-M3
   ```

3. **选择最佳模型** (30 分钟)

**预期**：Recall@5 从 55% → **70-75%**

### Phase 3: 重排序（后天）

1. **集成 Cross-Encoder** (2 小时)
2. **测试 rerank 效果** (1 小时)
3. **优化 rerank 参数** (1 小时)

**预期**：Recall@5 从 75% → **80-85%**

### Phase 4: 数据改进（本周）

1. **增加模板多样性** (4 小时)
2. **LLM paraphrase 增强** (4 小时)
3. **生成 hard negative** (2 小时)

**预期**：Recall@5 从 85% → **90%+**

---

## 验证脚本

### 快速验证

```bash
# 1. 测试索引参数
python scripts/test_index_params.py \
  --partitions 64 \
  --sub-vectors 32

# 2. 测试混合权重
python scripts/test_hybrid_weights.py \
  --alpha 0.7

# 3. 测试 Embedding 模型
python scripts/test_embedding_models.py \
  --model Qwen3-8B

# 4. 测试重排序
python scripts/test_reranker.py \
  --model cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 完整验证

```bash
# 运行完整 A/B 测试
python scripts/final_ab_test.py \
  --optimized \
  --model Qwen3-8B \
  --alpha 0.7 \
  --rerank
```

---

## 目标性能

| 阶段 | Recall@5 | 耗时 |
|------|----------|------|
| 当前 | 30.4% | - |
| Phase 1 | 50-55% | 2 小时 |
| Phase 2 | 70-75% | 4 小时 |
| Phase 3 | 80-85% | 4 小时 |
| Phase 4 | 90%+ | 10 小时 |

**最终目标**：Recall@5 ≥ 90%，MRR ≥ 0.85

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Qwen3-8B 速度太慢 | 中 | 中 | 使用 batch 编码 |
| 索引优化效果不佳 | 低 | 高 | 准备回滚方案 |
| 重排序引入延迟 | 中 | 中 | 只 rerank top-50 |
| 数据生成质量差 | 低 | 中 | 人工审核样本 |

---

**最后更新**: 2026-03-15  
**状态**: 🔄 准备实施 Phase 1
