#!/usr/bin/env python3
"""
快速优化验证脚本
测试索引参数和混合权重的组合效果
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

MODEL_NAME = 'Qwen3-Embedding-4B'

def load_data():
    with open('/home/kyj/.openclaw/workspace/synthetic_perltqa/memories_small.json') as f:
        memories = json.load(f)[:500]
    with open('/home/kyj/.openclaw/workspace/synthetic_perltqa/queries_small.json') as f:
        queries = json.load(f)[:100]  # 只测 100 个查询
    return memories, queries

def tokenize(text):
    return list(text.replace(' ', ''))

def test_config(memories, queries, model, bm25, memory_embs, alpha=0.7, k=60):
    """测试特定配置"""
    hits = 0
    for q in queries:
        # Vector 检索
        q_emb = model.encode([q['query']])[0]
        vector_scores = np.dot(memory_embs, q_emb)
        vector_ranks = np.argsort(np.argsort(-vector_scores)) + 1
        
        # BM25 检索
        q_tokens = tokenize(q['query'])
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1
        
        # 混合检索
        rrf_scores = alpha / (k + vector_ranks) + (1 - alpha) / (k + bm25_ranks)
        top5_indices = np.argsort(-rrf_scores)[:5]
        
        # 检查命中
        target_idx = next(i for i, m in enumerate(memories) if m['id'] == q['target_memory_id'])
        if target_idx in top5_indices:
            hits += 1
    
    return hits / len(queries)

def main():
    print("="*60)
    print("快速优化验证测试")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    memories, queries = load_data()
    print(f"  记忆：{len(memories)} 条")
    print(f"  查询：{len(queries)} 个")
    
    # 加载模型
    print(f"\n加载模型：{MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 生成 embedding
    print("生成记忆 embedding...")
    memory_texts = [m['content'] for m in memories]
    memory_embs = model.encode(memory_texts, batch_size=64)
    
    # 构建 BM25 索引
    print("构建 BM25 索引...")
    tokenized_docs = [tokenize(text) for text in memory_texts]
    bm25 = BM25Okapi(tokenized_docs)
    
    # 测试不同配置
    print("\n测试不同配置...\n")
    configs = [
        {'alpha': 0.5, 'k': 60, 'name': 'Baseline (RRF k=60)'},
        {'alpha': 0.7, 'k': 60, 'name': 'Vector-heavy (α=0.7)'},
        {'alpha': 0.8, 'k': 40, 'name': 'Vector-heavy + low k (α=0.8, k=40)'},
        {'alpha': 0.9, 'k': 40, 'name': 'Vector-dominant (α=0.9, k=40)'},
    ]
    
    results = {}
    for config in configs:
        recall = test_config(memories, queries, model, bm25, memory_embs, 
                           alpha=config['alpha'], k=config['k'])
        results[config['name']] = recall
        print(f"  {config['name']}: Recall@5 = {recall:.3f}")
    
    # 汇总
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    for name, recall in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {recall:.3f}")
    
    best = max(results, key=results.get)
    print(f"\n🏆 最佳配置：{best}")
    print(f"   Recall@5: {results[best]:.3f}")

if __name__ == '__main__':
    main()
