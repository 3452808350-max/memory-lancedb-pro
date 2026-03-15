#!/usr/bin/env python3
"""
最终 A/B 测试：Quality-Aware Retrieval vs Baseline
"""

import json
import argparse
import numpy as np
import requests
from tqdm import tqdm


def encode(text: str, model: str = "Qwen3-Embedding-4B") -> list:
    """编码文本"""
    resp = requests.post(
        'http://localhost:11434/api/embeddings',
        json={'model': f'modelscope.cn/Qwen/{model}-GGUF', 'prompt': text},
        timeout=60
    )
    return resp.json().get('embedding', [0.0]*1024)


def cosine_sim(v1: list, v2: list) -> float:
    """余弦相似度"""
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def evaluate(memories: list, queries: list, use_quality: bool = False, top_k: int = 10) -> dict:
    """评估检索效果"""
    
    # 预编码所有记忆
    print("编码记忆...")
    memory_embs = []
    for m in tqdm(memories, desc="记忆"):
        emb = encode(m.get('content', m.get('text', '')))
        memory_embs.append(emb)
    
    # 评估
    results = {}
    relevant = {}
    
    print("评估查询...")
    for q in tqdm(queries, desc="查询"):
        q_emb = encode(q['query'])
        
        # 计算相似度
        scores = []
        for i, m_emb in enumerate(memory_embs):
            sim = cosine_sim(q_emb, m_emb)
            
            # 应用质量分
            if use_quality:
                q_score = memories[i].get('quality_score', 0.5)
                final_score = sim * q_score
            else:
                final_score = sim
            
            scores.append((i, final_score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved = [memories[i]['id'] for i, _ in scores[:top_k]]
        
        results[q['id']] = retrieved
        relevant[q['id']] = q['relevant_ids']
    
    # 计算指标
    recall_5 = sum(1 for qid in results if any(r in relevant[qid] for r in results[qid][:5])) / len(results)
    recall_10 = sum(1 for qid in results if any(r in relevant[qid] for r in results[qid][:10])) / len(results)
    
    # MRR
    mrr_scores = []
    for qid in results:
        for i, rid in enumerate(results[qid]):
            if rid in relevant[qid]:
                mrr_scores.append(1.0 / (i + 1))
                break
        else:
            mrr_scores.append(0.0)
    mrr = np.mean(mrr_scores)
    
    return {
        'recall@5': recall_5,
        'recall@10': recall_10,
        'mrr': mrr
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', default='memory_db/memories_scored.jsonl')
    parser.add_argument('--queries', default='memory_db/queries.jsonl')
    parser.add_argument('--top-k', type=int, default=10)
    
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据...")
    memories = [json.loads(line) for line in open(args.memory)]
    queries = [json.loads(line) for line in open(args.queries)]
    
    print(f"记忆：{len(memories)} 条")
    print(f"查询：{len(queries)} 个\n")
    
    # Baseline
    print("="*60)
    print("Baseline (无质量分)")
    print("="*60)
    baseline = evaluate(memories, queries, use_quality=False, top_k=args.top_k)
    print(f"Recall@5:  {baseline['recall@5']:.3f}")
    print(f"Recall@10: {baseline['recall@10']:.3f}")
    print(f"MRR:       {baseline['mrr']:.3f}\n")
    
    # Quality-aware
    print("="*60)
    print("Quality-Aware (质量分加权)")
    print("="*60)
    qa = evaluate(memories, queries, use_quality=True, top_k=args.top_k)
    print(f"Recall@5:  {qa['recall@5']:.3f}")
    print(f"Recall@10: {qa['recall@10']:.3f}")
    print(f"MRR:       {qa['mrr']:.3f}\n")
    
    # 对比
    print("="*60)
    print("提升")
    print("="*60)
    print(f"Recall@5:  +{(qa['recall@5'] - baseline['recall@5'])*100:.1f}%")
    print(f"Recall@10: +{(qa['recall@10'] - baseline['recall@10'])*100:.1f}%")
    print(f"MRR:       +{(qa['mrr'] - baseline['mrr'])*100:.3f}")


if __name__ == '__main__':
    main()
