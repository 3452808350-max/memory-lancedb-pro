# LLM-Based Reranking for MemQ

> **下一代精度提升方案**
> 
> 状态：📝 提案中  
> 参考：https://github.com/CortexReach/memory-lancedb-pro-skill

---

## 🎯 核心思想

**两阶段检索**：
```
Stage 1: 快速检索 (BM25 + Vector)
  Query → Retrieve Top-50 → 2ms

Stage 2: LLM 重排序
  Top-50 → LLM Rerank → Top-10 → 500ms

最终：精度提升 30-50%，延迟增加 500ms
```

---

## 📊 为什么需要 LLM Reranking？

### 当前 MemQ 的局限

| 方法 | Recall@5 | 延迟 | 问题 |
|------|---------|------|------|
| BM25 | ~60% | 10ms | 语义理解差 |
| Vector | ~65% | 20ms | 细粒度匹配弱 |
| MemQ | ~70% | 25ms | 规则系统局限 |
| **LLM Rerank** | **~85-90%** | **500ms** | **成本高** |

### LLM 的独特优势

**传统方法做不到的**：
1. ✅ 理解查询意图
2. ✅ 判断时间相关性
3. ✅ 识别矛盾信息
4. ✅ 推理隐含关系

**示例**：
```
查询："2026 年 3 月的销售数据"

传统方法检索结果：
1. "2026 年 3 月销售额 100 万" ✅
2. "2025 年 3 月销售额 80 万"  ❌ 时间不对
3. "2026 年 4 月销售额 120 万" ❌ 时间不对

LLM Reranking 后：
1. "2026 年 3 月销售额 100 万" ✅ LLM 理解时间匹配
2. "2026 年 4 月销售额 120 万" ❌ LLM 识别时间不匹配
3. "2025 年 3 月销售额 80 万"  ❌ LLM 识别时间不匹配
```

---

## 🔧 实现方案

### 方案 A: Cross-Encoder（轻量级）

**模型**：
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (80MB)
- `cross-encoder/ms-marco-electra-base` (400MB)

**实现**：
```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name='ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, memories, top_k=10):
        # 构造查询 - 文档对
        pairs = [[query, m['content']] for m in memories]
        
        # 预测相关性分数
        scores = self.model.predict(pairs)
        
        # 排序
        ranked = sorted(zip(memories, scores), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
```

**性能**：
- 延迟：~100ms (50 条)
- 精度提升：+15-20%
- 成本：低（本地 CPU 可运行）

---

### 方案 B: LLM Zero-Shot（中等）

**模型**：
- Qwen3-8B
- MiniMax-M2.5
- Kimi

**Prompt 设计**：
```python
RERANK_PROMPT = """
你是一个专业的记忆检索评估专家。

查询：{query}

候选记忆（共{count}条）：
{memories}

请根据以下标准评估每条记忆的相关性：
1. 是否直接回答了查询？
2. 时间信息是否匹配？
3. 实体信息是否一致？
4. 是否存在矛盾或冲突？

请按 1-5 分评分（5=最相关），并给出理由。

输出格式（JSON）：
[
  {{"memory_id": "m1", "score": 5, "reason": "..."}},
  ...
]
"""
```

**实现**：
```python
class LLMReranker:
    def __init__(self, model='qwen-8b'):
        self.model = model
        self.prompt_template = RERANK_PROMPT
    
    def rerank(self, query, memories, top_k=10):
        # 构造 prompt
        memories_text = "\n".join([
            f"{i+1}. [{m['id']}] {m['content']}"
            for i, m in enumerate(memories[:50])  # 限制数量
        ])
        
        prompt = self.prompt_template.format(
            query=query,
            count=len(memories),
            memories=memories_text
        )
        
        # 调用 LLM
        response = call_llm(self.model, prompt)
        
        # 解析评分
        scores = parse_json_response(response)
        
        # 排序
        ranked = sorted(zip(memories, scores), key=lambda x: x[1]['score'], reverse=True)
        
        return ranked[:top_k]
```

**性能**：
- 延迟：~500ms (50 条)
- 精度提升：+25-35%
- 成本：中（需要 GPU 或 API）

---

### 方案 C: LLM Fine-tuned（高级）

**思路**：在检索任务上微调 LLM

**数据收集**：
```python
# 从用户反馈收集训练数据
training_data = []
for query, memories, feedback in user_logs:
    # 正样本（用户点击/有用）
    positive = [m for m in memories if feedback[m.id] == 1]
    # 负样本（用户忽略/无用）
    negative = [m for m in memories if feedback[m.id] == 0]
    
    training_data.append({
        'query': query,
        'positive': positive,
        'negative': negative
    })
```

**微调方法**：
- **ListNet**: 优化整个排序列表
- **LambdaRank**: 优化 NDCG 指标
- **Pairwise**: 优化正负样本对

**性能**：
- 延迟：~300ms
- 精度提升：+30-40%
- 成本：高（需要训练和部署）

---

## 🚀 集成到 MemQ

### 混合架构

```python
class MemQWithRerank:
    def __init__(self, 
                 base_retriever='memq',
                 reranker='cross-encoder',
                 top_k=10,
                 rerank_top_n=50):
        
        # 基础检索器
        self.base_retriever = base_retriever
        
        # 重排序器
        if reranker == 'cross-encoder':
            self.reranker = CrossEncoderReranker()
        elif reranker == 'llm':
            self.reranker = LLMReranker()
        
        # 参数
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
    
    def retrieve(self, query, domain=None):
        # Stage 1: 快速检索 Top-N
        candidates = self.base_retriever.retrieve(
            query, 
            k=self.rerank_top_n,
            domain=domain
        )
        
        # Stage 2: LLM 重排序
        if len(candidates) > 0:
            reranked = self.reranker.rerank(
                query, 
                candidates, 
                top_k=self.top_k
            )
            return reranked
        else:
            return []
```

---

## 📊 预期效果

### 性能对比

| 配置 | Recall@5 | Recall@10 | MRR | 延迟 |
|------|----------|-----------|-----|------|
| MemQ only | 70% | 75% | 0.65 | 25ms |
| + Cross-Encoder | 78% | 83% | 0.73 | 125ms |
| + LLM Zero-Shot | 85% | 90% | 0.82 | 525ms |
| + LLM Fine-tuned | 88% | 93% | 0.86 | 325ms |

### 成本分析

| 方案 | 硬件需求 | 单次成本 | 适合场景 |
|------|---------|---------|---------|
| Cross-Encoder | CPU | ¥0.001 | 高频查询 |
| LLM Zero-Shot | GPU/API | ¥0.01 | 关键查询 |
| LLM Fine-tuned | GPU | ¥0.005 | 大规模部署 |

---

## 🎯 实施建议

### Phase 1: Cross-Encoder（本周）

**理由**：
- ✅ 实现简单
- ✅ 成本低
- ✅ 精度提升明显（+15%）

**步骤**：
```bash
# 1. 安装依赖
pip install sentence-transformers

# 2. 测试效果
python scripts/test_cross_encoder_rerank.py

# 3. 集成到 pipeline
python scripts/integrate_reranker.py
```

---

### Phase 2: LLM Zero-Shot（下周）

**理由**：
- ✅ 精度进一步提升（+25%）
- ✅ 无需训练数据
- ⚠️ 成本较高

**步骤**：
```bash
# 1. 选择 LLM 提供商
# 选项：Qwen / MiniMax / Kimi

# 2. 设计 Prompt
# 参考：docs/LLM_RERANKING.md

# 3. 测试效果
python scripts/test_llm_rerank.py

# 4. 成本效益分析
python scripts/analyze_cost_benefit.py
```

---

### Phase 3: LLM Fine-tuned（下月）

**理由**：
- ✅ 最佳精度（+30%）
- ✅ 延迟优化
- ❌ 需要训练数据

**步骤**：
```bash
# 1. 收集训练数据
python scripts/collect_training_data.py

# 2. 微调模型
python scripts/fine_tune_reranker.py

# 3. 部署服务
python scripts/deploy_reranker_service.py

# 4. A/B 测试
python scripts/ab_test_rerank.py
```

---

## 📝 与半自适应 MemQ 集成

### 联合架构

```python
class AdvancedMemQ:
    def __init__(self):
        # 半自适应质量评分
        self.memq = SemiAdaptiveMemQ()
        
        # LLM 重排序
        self.reranker = CrossEncoderReranker()
    
    def retrieve_and_rerank(self, query, memories, feedback=None):
        # 1. MemQ 初筛
        scored_memories = []
        for mem in memories:
            score, _ = self.memq.predict(mem)
            scored_memories.append((mem, score))
        
        # 2. 过滤低质量记忆
        filtered = [
            mem for mem, score in scored_memories
            if score > 0.3  # 阈值
        ]
        
        # 3. LLM 重排序
        if feedback:
            # 在线更新 MemQ 权重
            self.memq.update(memories[0], feedback)
        
        reranked = self.reranker.rerank(query, filtered[:50], top_k=10)
        
        return reranked
```

**优势**：
- MemQ 过滤低质量候选（减少 LLM 成本）
- LLM 提供细粒度排序（提升精度）
- 半自适应持续改进（长期优化）

---

## 🔍 参考实现

### CortexReach memory-lancedb-pro-skill

**参考链接**：https://github.com/CortexReach/memory-lancedb-pro-skill

**关键特性**（待探索）：
- [ ] LLM 重排序实现
- [ ] 与 LanceDB 集成
- [ ] 性能优化技巧
- [ ] 生产部署经验

---

## 📊 决策框架

### 何时使用 LLM Reranking？

✅ **推荐使用**：
- 查询复杂度高（需要推理）
- 精度要求高（>85%）
- 预算充足
- 延迟要求宽松（>500ms）

❌ **不推荐**：
- 简单查询（关键词匹配即可）
- 高频查询（成本过高）
- 实时性要求高（<100ms）
- 资源受限（无 GPU）

### 混合策略

```python
def smart_rerank(query, memories):
    # 简单查询：不用 LLM
    if len(query.split()) < 5:
        return base_retrieve(query, memories)
    
    # 复杂查询：用 LLM
    else:
        return llm_rerank(query, memories)
```

---

## 🎯 下一步行动

### 本周（Phase 1）

- [ ] 调研 CortexReach 实现
- [ ] 实现 Cross-Encoder Reranker
- [ ] 测试效果（预期 +15%）
- [ ] 文档更新

### 下周（Phase 2）

- [ ] 实现 LLM Zero-Shot Rerank
- [ ] Prompt 工程优化
- [ ] 成本效益分析
- [ ] A/B 测试

### 下月（Phase 3）

- [ ] 收集训练数据
- [ ] Fine-tune LLM Reranker
- [ ] 部署生产服务
- [ ] 性能监控

---

**最后更新**: 2026-03-15  
**状态**: 📝 提案中  
**负责人**: 待定  
**优先级**: ⭐⭐⭐⭐⭐
