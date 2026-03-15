#!/usr/bin/env python3
"""
Duplicate Memory Detection
检测语义重复记忆
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_memories(path: str) -> List[Dict]:
    """加载记忆"""
    memories = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            memories.append(json.loads(line))
    return memories


def detect_duplicates(memories: List[Dict], 
                      model_name: str = "BAAI/bge-m3",
                      threshold: float = 0.92,
                      batch_size: int = 64) -> Dict:
    """
    检测重复记忆
    
    Args:
        memories: 记忆列表
        model_name: embedding 模型
        threshold: 相似度阈值
        batch_size: 批处理大小
    
    Returns:
        包含重复检测结果的字典
    """
    print(f"加载模型：{model_name}...")
    model = SentenceTransformer(model_name)
    
    texts = [m['text'] for m in memories]
    ids = [m['id'] for m in memories]
    
    print(f"生成 embeddings: {len(texts)} 条记忆...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    print(f"计算相似度矩阵...")
    sim_matrix = cosine_similarity(embeddings)
    
    # 检测重复对
    duplicate_pairs = []
    n = len(memories)
    
    for i in tqdm(range(n), desc="检测重复"):
        for j in range(i + 1, n):
            if sim_matrix[i][j] > threshold:
                duplicate_pairs.append({
                    'id1': ids[i],
                    'id2': ids[j],
                    'text1': texts[i][:100],
                    'text2': texts[j][:100],
                    'similarity': float(sim_matrix[i][j])
                })
    
    # 计算重复率
    # 一个记忆如果与任何其他记忆重复，就算作重复
    duplicate_ids = set()
    for pair in duplicate_pairs:
        duplicate_ids.add(pair['id1'])
        duplicate_ids.add(pair['id2'])
    
    duplicate_ratio = len(duplicate_ids) / n if n > 0 else 0
    
    return {
        'total_memories': n,
        'duplicate_memories': len(duplicate_ids),
        'duplicate_ratio': duplicate_ratio,
        'duplicate_ratio_percent': f"{duplicate_ratio * 100:.2f}%",
        'duplicate_pairs_count': len(duplicate_pairs),
        'duplicate_pairs': duplicate_pairs[:20],  # 只保存前 20 个示例
        'threshold': threshold
    }


def main():
    parser = argparse.ArgumentParser(description='Duplicate Memory Detection')
    parser.add_argument('--memory', type=str, required=True,
                        help='Path to memories.jsonl')
    parser.add_argument('--model', type=str,
                        default='BAAI/bge-m3',
                        help='Embedding model name')
    parser.add_argument('--threshold', type=float,
                        default=0.92,
                        help='Similarity threshold')
    parser.add_argument('--output', type=str,
                        default='results/duplicates.json',
                        help='Output path')
    
    args = parser.parse_args()
    
    print(f"检测重复记忆：{args.memory}")
    memories = load_memories(args.memory)
    
    results = detect_duplicates(
        memories=memories,
        model_name=args.model,
        threshold=args.threshold
    )
    
    print(f"\n{'='*60}")
    print("重复记忆检测结果")
    print(f"{'='*60}")
    print(f"总记忆数：{results['total_memories']}")
    print(f"重复记忆数：{results['duplicate_memories']}")
    print(f"重复比例：{results['duplicate_ratio_percent']}")
    print(f"重复对数量：{results['duplicate_pairs_count']}")
    
    if results['duplicate_pairs']:
        print(f"\n重复示例 (阈值 > {results['threshold']}):")
        for pair in results['duplicate_pairs'][:5]:
            print(f"  {pair['id1']} <-> {pair['id2']} (相似度：{pair['similarity']:.3f})")
            print(f"    1: {pair['text1']}...")
            print(f"    2: {pair['text2']}...")
    
    print(f"{'='*60}\n")
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到：{args.output}")


if __name__ == '__main__':
    main()
