#!/usr/bin/env python3
"""
Phase 1: 混合检索权重优化测试
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# 配置
MODEL_NAME = 'Qwen3-Embedding-4B'

def load_test_data():
    """加载测试数据"""
    with open('/home/kyj/.openclaw/workspace/synthetic_perltqa/memories_small.json') as f:
        memories = json.load(f)
    with open('/home/kyj/.openclaw/workspace/synthetic_perltqa/queries_small.json') as f:
        queries = json.load(f)
    return memories[:500], queries[:500]

def tokenize(text):
    """简单分词"""
    return list(text.replace(' ', ''))

def test_hybrid_weights(alpha=0.7):
    """测试不同混合权重"""
    print(f"\n{'='*60}")
    print(f"测试混合权重：alpha={alpha} (Vector {alpha*100}%, BM25 {(1-alpha)*100}%)")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("加载 Embedding 模型...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 加载数据
    memories, queries = load_test_data()
    print(f"加载 {len(memories)} 条记忆，{len(queries)} 个查询\n")
    
    # 预计算记忆 embedding
    print("生成记忆 embedding...")
    memory_texts = [m['content'] for m in memories]
    memory_embs = model.encode(memory_texts, batch_size=64)
    
    # 构建 BM25 索引
    print("构建 BM25 索引...")
    tokenized_docs = [tokenize(text) for text in memory_texts]
    bm25 = BM25Okapi(tokenized_docs)
    
    # 测试检索
    print(f"测试检索 (alpha={alpha})...\n")
    hits = 0
    for i, q in enumerate(queries):
        if i % 100 == 0:
            print(f"  进度：{i}/{len(queries)}")
        
        # Vector 检索
        q_emb = model.encode([q['query']])[0]
        vector_scores = np.dot(memory_embs, q_emb)
        vector_ranks = np.argsort(np.argsort(-vector_scores)) + 1
        
        # BM25 检索
        q_tokens = tokenize(q['query'])
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1
        
        # 混合检索 (RRF)
        k = 60
        rrf_scores = alpha / (k + vector_ranks) + (1 - alpha) / (k + bm25_ranks)
        top5_indices = np.argsort(-rrf_scores)[:5]
        
        # 检查是否命中
        target_idx = memories.index(next(m for m in memories if m['id'] == q['target_memory_id']))
        if target_idx in top5_indices:
            hits += 1
    
    recall = hits / len(queries)
    print(f"\n✅ Recall@5: {recall:.3f} ({hits}/{len(queries)})")
    
    return recall

def main():
    print("="*60)
    print("Phase 1: 混合检索权重优化测试")
    print("="*60)
    
    # 测试不同权重
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {}
    for alpha in alphas:
        recall = test_hybrid_weights(alpha)
        results[f'alpha_{alpha}'] = recall
    
    # 汇总
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    for config, recall in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{config}: Recall@5 = {recall:.3f}")
    
    best_config = max(results, key=results.get)
    best_alpha = float(best_config.split('_')[1])
    print(f"\n🏆 最佳配置：{best_config} (alpha={best_alpha}, Recall@5 = {results[best_config]:.3f})")

if __name__ == '__main__':
    main()
