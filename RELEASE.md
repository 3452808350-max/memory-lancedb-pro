# MemQ v1.0.0 - Quality-Aware Memory Retrieval

## 🎯 核心特性

MemQ 是一个质量感知的记忆检索系统，专为 LLM Agent 设计。

**核心创新**：
- ✅ 零样本质量评分（无需训练）
- ✅ 主动噪声抑制（不是删除，是降权）
- ✅ 完美分离度（noise 0.198 vs knowledge 1.0）
- ✅ 预期提升 +7-12% Recall@5

## 📦 安装

### Python 依赖

```bash
pip install -r requirements.txt
```

### Node.js 依赖（OpenClaw 插件）

```bash
npm install
```

## 🚀 快速开始

### 1. 质量评分

```bash
python scripts/quality_scorer.py \
  --input memory_db/memories.jsonl \
  --output memory_db/memories_scored.jsonl
```

### 2. A/B 测试

```bash
python scripts/final_ab_test.py \
  --memory memory_db/memories_scored.jsonl \
  --queries memory_db/queries.jsonl \
  --top-k 5
```

### 3. 使用 OpenClaw 插件

```javascript
// OpenClaw 配置中启用
{
  "plugins": {
    "memq": {
      "enabled": true,
      "dbPath": "/path/to/lancedb"
    }
  }
}
```

## 📊 预期结果

### 质量评分分布

| 类型 | 平均分 |
|------|--------|
| knowledge | 1.000 |
| event | 0.926 |
| code | 0.891 |
| conversation | 0.838 |
| noise | 0.198 |

### A/B 测试预期

| 方法 | Recall@5 | 提升 |
|------|----------|------|
| Baseline | ~0.63 | - |
| Quality-Aware | ~0.70-0.75 | +7-12% |

## 📁 项目结构

```
MemQ/
├── benchmark/          # 基准测试套件
├── scripts/            # 分析脚本
├── docs/               # 文档
│   └── PROOF.md        # 数学证明
├── experiments/        # 实验计划
├── memory_db/          # 记忆数据库
├── results/            # 测试结果
├── README.md           # 英文说明
├── README_CN.md        # 中文说明
└── requirements.txt    # Python 依赖
```

## 🔬 技术细节

### 质量评分公式

```python
quality_score = type_weight × length_factor × entity_factor × stopwords_factor × template_factor × metadata_factor
```

### 检索流程

```
Query → Embedding → Vector Retrieval → BM25 → RRF Fusion → Quality Scoring → Final Results
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- LanceDB - 向量数据库
- Sentence Transformers - Embedding 模型
- OpenClaw - Agent 框架
