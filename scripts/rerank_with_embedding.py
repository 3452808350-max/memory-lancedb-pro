#!/usr/bin/env python3
"""
使用 Embedding 相似度实现重排序
（Ollama 暂不支持 /api/rerank 的替代方案）
"""

import json
import argparse
import numpy as np
from pathlib import Path
import requests


def encode_text(text: str, model: str = "Qwen3-Embedding-4B") -> list:
    """调用 Ollama API 编码文本"""
    resp = requests.post(
        'http://localhost:11434/api/embeddings',
        json={
            'model': f'modelscope.cn/Qwen/{model}-GGUF',
            'prompt': text
        },
        timeout=60
    )
    return resp.json().get('embedding', [])


def cosine_sim(v1: list, v2: list) -> float:
    """计算余弦相似度"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def rerank(query: str, documents: list, quality_scores: dict = None, 
           model: str = "Qwen3-Embedding-4B", top_k: int = 10) -> list:
    """
    重排序文档
    
    Args:
        query: 查询文本
        documents: 文档列表 [{'id': ..., 'text': ..., 'quality_score': ...}]
        quality_scores: 质量分数 {doc_id: score}
        model: embedding 模型
        top_k: 返回数量
    
    Returns:
        重排序后的文档列表
    """
    print(f"编码查询...")
    q_emb = encode_text(query, model)
    
    print(f"编码 {len(documents)} 个文档...")
    scored_docs = []
    for i, doc in enumerate(documents):
        if i % 50 == 0:
            print(f"  进度：{i}/{len(documents)}")
        
        d_emb = encode_text(doc.get('content', doc.get('text', '')), model)
        sim = cosine_sim(q_emb, d_emb)
        
        # 应用质量分
        if quality_scores:
            q_score = quality_scores.get(doc['id'], 0.5)
            final_score = sim * q_score
        else:
            final_score = sim
        
        scored_docs.append({
            **doc,
            'similarity': sim,
            'final_score': final_score
        })
    
    # 排序
    scored_docs.sort(key=lambda x: x['final_score'], reverse=True)
    
    return scored_docs[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Embedding-based Reranker')
    parser.add_argument('--query', type=str, required=True,
                        help='查询文本')
    parser.add_argument('--memory', type=str,
                        default='memory_db/memories_scored.jsonl',
                        help='记忆文件（带质量分）')
    parser.add_argument('--top-k', type=int, default=10,
                        help='返回数量')
    parser.add_argument('--model', type=str,
                        default='Qwen3-Embedding-4B',
                        choices=['Qwen3-Embedding-4B', 'Qwen3-Embedding-8B', 'Qwen3-Embedding-0.6B'],
                        help='Embedding 模型')
    parser.add_argument('--use-quality', action='store_true',
                        help='使用质量分加权')
    
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
    
    # 重排序
    results = rerank(
        query=args.query,
        documents=memories,
        quality_scores=quality_scores if args.use_quality else None,
        model=args.model,
        top_k=args.top_k
    )
    
    # 显示结果
    print(f"\n{'='*70}")
    print(f"查询：{args.query}")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. [Score: {r['final_score']:.4f}] (Sim: {r['similarity']:.4f}, Quality: {r.get('quality_score', 'N/A')})")
        print(f"   [{r['type']}] {r['text'][:70]}...\n")


if __name__ == '__main__':
    main()
