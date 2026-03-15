# Memory Retrieval System Evaluation Plan

## 1. Objective

本实验用于评估 memory retrieval 系统在**数据规模增长条件下的稳定性与检索性能**。

主要验证以下 7 类问题：

| ID | Problem | Metric | Target |
| -- | ----------------------------------- | ------------------------- | -------- |
| P1 | Noise memory interference | Noise Ratio | < 5% |
| P2 | Query-memory semantic inconsistency | Semantic Coherence | > 80% |
| P3 | Memory type imbalance | Entropy | Stable |
| P4 | Hybrid retrieval weight imbalance | Recall@k | Optimal α |
| P5 | Embedding drift | Cosine distribution shift | < 0.05 |
| P6 | Reranker pipeline degradation | Recall difference | < 5% |
| P7 | Duplicate memory entries | Duplicate ratio | < 10% |

---

## 2. Experiment Dataset

### Data Source

```
memory_db/
├── memories.jsonl      # Memory corpus
└── queries.jsonl       # Query set with relevance labels
```

### memories.jsonl Format

```json
{
  "id": "m1",
  "text": "Geralt prefers Arch Linux for daily development.",
  "type": "user_preference",
  "timestamp": "2026-01-10",
  "metadata": {
    "person": "Geralt",
    "project": "Development",
    "tech": "Arch Linux"
  }
}
```

### queries.jsonl Format

```json
{
  "id": "q1",
  "query": "What operating system does Geralt use?",
  "relevant_ids": ["m1"],
  "type": "hybrid",
  "expected_keywords": ["Geralt", "Arch Linux"]
}
```

---

## 3. Experiment Environment

### requirements.txt

```txt
lancedb
sentence-transformers
rank-bm25
scikit-learn
numpy
pandas
tqdm
matplotlib
seaborn
```

### Installation

```bash
pip install -r requirements.txt
```

---

## 4. Experiment Pipeline

```
dataset
   ↓
embedding
   ↓
vector retrieval
   ↓
BM25 retrieval
   ↓
hybrid merge
   ↓
reranker
   ↓
evaluation
```

---

## 5. Experiments

---

### EXPERIMENT 1: Noise Memory Ratio

#### Objective

检测数据库中低质量记忆比例。

#### Command

```bash
python scripts/eval_noise.py \
  --memory memory_db/memories.jsonl
```

#### Method

启发式检测：
- Token 长度 < 5
- 仅含停用词
- 模板占位符

#### Metric

```python
noise_ratio = noisy_memories / total_memories
```

#### Target

- noise_ratio < 5%

---

### EXPERIMENT 2: Duplicate Memory Detection

#### Objective

检测语义重复记忆。

#### Command

```bash
python scripts/eval_duplicates.py \
  --memory memory_db/memories.jsonl
```

#### Method

```python
embedding = model.encode(texts)
similarity = cosine_similarity(embedding)
duplicate = similarity > 0.92
```

#### Output

- `duplicate_ratio`: 重复率
- `duplicate_clusters`: 重复簇数量

#### Target

- duplicate_ratio < 10%

---

### EXPERIMENT 3: Embedding Drift

#### Objective

检测 embedding 分布变化。

#### Command

```bash
python scripts/eval_embedding_drift.py \
  --memory memory_db/memories.jsonl
```

#### Metrics

- **Centroid shift**: 新旧语料中心余弦相似度
- **Norm change**: 平均范数变化率
- **Variance change**: 方差变化

#### Threshold

- centroid cosine drop > 0.05 → 触发 re-embedding

---

### EXPERIMENT 4: Hybrid Retrieval Weight Sweep

#### Objective

找到最优 hybrid 权重 α。

#### Command

```bash
python scripts/eval_hybrid.py \
  --alpha_list 0.2 0.4 0.6 0.8 \
  --queries benchmark/datasets/queries.jsonl
```

#### Metric

- Recall@1
- Recall@5
- MRR

#### Output

```
alpha,Recall@5,MRR
0.2,0.68,0.52
0.4,0.72,0.56
0.6,0.74,0.58
0.8,0.71,0.55
```

#### Target

- 找到 stable α 或设计 adaptive 策略

---

### EXPERIMENT 5: Reranker Impact

#### Objective

比较 reranker 与 baseline。

#### Command

```bash
python scripts/eval_reranker.py
```

#### Comparison

- retriever_only
- retriever + reranker

#### Metric

- Recall@5 difference
- Latency overhead

#### Target

- Recall 提升 > 5%
- Latency 增加 < 100ms

---

### EXPERIMENT 6: Semantic Coherence

#### Objective

检测检索结果是否属于同一语义簇。

#### Command

```bash
python scripts/eval_coherence.py
```

#### Method

对 top-10 结果进行 topic 聚类，计算熵。

#### Metric

```python
topic_entropy = -sum(p * log(p))
```

#### Target

- entropy < 2.0 (结果语义一致)

---

## 6. Evaluation Metrics

| Metric | Description | Formula |
| --------------- | ----------------------------- | -------- |
| Recall@1 | top1 accuracy | hits@1 / total |
| Recall@5 | retrieval recall | hits@5 / total |
| MRR | mean reciprocal rank | avg(1/rank) |
| NDCG@k | normalized discounted gain | DCG/IDCG |
| Duplicate Ratio | memory redundancy | dup / total |
| Noise Ratio | low quality memory | noise / total |
| Entropy | memory type distribution | -Σp*log(p) |
| Drift Score | embedding distribution change | centroid cosine |

---

## 7. Expected Outputs

每次实验生成：

```
results/
├── noise.json
├── duplicates.json
├── hybrid_results.csv
├── reranker_comparison.csv
└── embedding_drift.json
```

---

## 8. Visualization

```bash
python scripts/plot_results.py
```

生成图表：

- recall vs alpha (折线图)
- embedding drift (分布图)
- duplicate clusters (热力图)
- noise ratio over time (时间序列)

---

## 9. Success Criteria

系统优化目标：

```python
noise_ratio < 5%
duplicate_ratio < 10%
Recall@5 > baseline + 10%
embedding_drift < 0.05
```

---

## 10. Experiment Schedule

| Week | Task | Deliverable |
| ---- | ---------------------- | ------------------- |
| W1 | baseline evaluation | metrics_report.json |
| W2 | hybrid tuning | optimal_alpha.json |
| W3 | reranker training | reranker_model.pkl |
| W4 | deduplication pipeline | dedup_script.py |

---

## 11. Reproducibility

### Random Seed

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

### Version Control

```bash
# Record environment
pip freeze > requirements.lock.txt

# Record git commit
git rev-parse HEAD > .git_commit.txt
```

### Data Versioning

```bash
# Use DVC or git-lfs for large datasets
dvc add memory_db/memories.jsonl
```

---

## 12. Reporting

实验完成后生成报告：

```bash
python scripts/generate_report.py
```

输出：
- `experiment_report.md` - 完整实验报告
- `results_summary.csv` - 结果汇总
- `plots/` - 所有图表

---

**Last Updated:** 2026-03-15

**Status:** 🧪 Experimental
