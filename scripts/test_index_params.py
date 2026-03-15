#!/usr/bin/env python3
"""
Phase 1: 索引参数优化测试
"""

import json
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

# 配置
DB_PATH = '/home/kyj/.openclaw/workspace/lancedb'
MODEL_NAME = 'Qwen3-Embedding-4B'

def load_test_data():
    """加载测试数据"""
    with open('/home/kyj/.openclaw/workspace/synthetic_perltqa/memories_small.json') as f:
        memories = json.load(f)
    with open('/home/kyj/.openclaw/workspace/synthetic_perltqa/queries_small.json') as f:
        queries = json.load(f)
    return memories[:500], queries[:500]

def test_index_params(num_partitions=64, num_sub_vectors=32):
    """测试不同索引参数"""
    print(f"\n{'='*60}")
    print(f"测试索引参数：num_partitions={num_partitions}, num_sub_vectors={num_sub_vectors}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("加载 Embedding 模型...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 加载数据
    memories, queries = load_test_data()
    print(f"加载 {len(memories)} 条记忆，{len(queries)} 个查询\n")
    
    # 生成 embedding
    print("生成记忆 embedding...")
    memory_texts = [m['content'] for m in memories]
    memory_embs = model.encode(memory_texts, batch_size=64, show_progress_bar=True)
    
    # 连接到 LanceDB
    db = lancedb.connect(DB_PATH)
    
    # 删除旧表
    table_name = f'test_index_p{num_partitions}_sv{num_sub_vectors}'
    try:
        db.drop_table(table_name)
    except:
        pass
    
    # 创建表
    print(f"创建表：{table_name}...")
    data = [{
        'id': m['id'],
        'text': m['content'],
        'vector': emb.tolist()
    } for m, emb in zip(memories, memory_embs)]
    
    table = db.create_table(table_name, data)
    
    # 创建索引
    print(f"创建索引 (partitions={num_partitions}, sub_vectors={num_sub_vectors})...")
    table.create_index(
        metric="cosine",
        index_type="IVF_PQ",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors
    )
    
    # 测试检索
    print("测试检索...\n")
    hits = 0
    for i, q in enumerate(queries):
        if i % 100 == 0:
            print(f"  进度：{i}/{len(queries)}")
        
        q_emb = model.encode([q['query']])[0]
        results = table.search(q_emb).limit(5).to_list()
        
        retrieved_ids = [r['id'] for r in results]
        if q['target_memory_id'] in retrieved_ids:
            hits += 1
    
    recall = hits / len(queries)
    print(f"\n✅ Recall@5: {recall:.3f} ({hits}/{len(queries)})")
    
    return recall

def main():
    print("="*60)
    print("Phase 1: 索引参数优化测试")
    print("="*60)
    
    # 测试不同参数组合
    configs = [
        (16, 16),   # 默认
        (32, 16),   # 中等
        (64, 32),   # 推荐
        (128, 64),  # 高精度
    ]
    
    results = {}
    for partitions, sub_vectors in configs:
        recall = test_index_params(partitions, sub_vectors)
        results[f'p{partitions}_sv{sub_vectors}'] = recall
    
    # 汇总
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    for config, recall in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{config}: Recall@5 = {recall:.3f}")
    
    best_config = max(results, key=results.get)
    print(f"\n🏆 最佳配置：{best_config} (Recall@5 = {results[best_config]:.3f})")

if __name__ == '__main__':
    main()
