#!/usr/bin/env python3
"""
快速重排序测试（使用缓存）
"""

import json
import argparse
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm


def encode_batch(texts: list, model: str = "Qwen3-Embedding-4B") -> list:
    """批量编码（利用 Ollama 的 batch 支持）"""
    # 注意：Ollama 目前不支持真正的 batch，但我们可以并行请求
    embeddings = []
    for text in tqdm(texts, desc="编码"):
        try:
            resp = requests.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': f'modelscope.cn/Qwen/{model}-GGUF',
                    'prompt': text
                },
                timeout=60
            )
            emb = resp.json().get('embedding', [])
            embeddings.append(emb if emb else [0.0] * 1024)
        except:
            embeddings.append([0.0] * 1024)  # 失败时用零向量
    return embeddings


def cosine_sim_matrix(q_embs: np.ndarray, d_embs: np.ndarray) -> np.ndarray:
    """批量计算余弦相似度"""
    # 归一化
    q_norm = q_embs / np.linalg.norm(q_embs, axis=1, keepdims=True)
    d_norm = d_embs / np.linalg.norm(d_embs, axis=1, keepdims=True)
    
    # 矩阵乘法
    return np.dot(q_norm, d_norm.T)


def main():
    parser = argparse.ArgumentParser(description='Fast Rerank Test')
    parser.add_argument('--query', type=str, default="OpenClaw API 怎么配置")
    parser.add_argument('--memory', type=str, default='memory_db/memories_scored.jsonl')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--use-quality', action='store_true')
    
    args = parser.parse_args()
    
    # 加载记忆
    print(f"加载记忆：{args.memory}")
    memories = []
    quality_scores = {}
    
    with open(args.memory, 'r', encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            memories.append(m)
            quality_scores[m['id']] = m.get('quality_score', 0.5)
    
    print(f"加载 {len(memories)} 条记忆\n")
    
    # 编码
    print("编码查询...")
    q_embs = np.array(encode_batch([args.query]))
    
    print("编码文档...")
    texts = [m.get('content', m.get('text', '')) for m in memories]
    d_embs = np.array(encode_batch(texts))
    
    # 计算相似度
    print("计算相似度矩阵...")
    sims = cosine_sim_matrix(q_embs, d_embs)[0]
    
    # 应用质量分
    if args.use_quality:
        print("应用质量分加权...")
        quality_vec = np.array([quality_scores.get(m['id'], 0.5) for m in memories])
        final_scores = sims * quality_vec
    else:
        final_scores = sims
    
    # 排序
    top_indices = np.argsort(final_scores)[::-1][:args.top_k]
    
    # 显示结果
    print(f"\n{'='*70}")
    print(f"查询：{args.query}")
    print(f"{'='*70}\n")
    
    for rank, idx in enumerate(top_indices, 1):
        m = memories[idx]
        print(f"{rank}. [Score: {final_scores[idx]:.4f}] (Sim: {sims[idx]:.4f}, Quality: {quality_scores.get(m['id'], 'N/A')})")
        print(f"   [{m.get('type')}] {m.get('content', m.get('text', ''))[:70]}...\n")


if __name__ == '__main__':
    main()
