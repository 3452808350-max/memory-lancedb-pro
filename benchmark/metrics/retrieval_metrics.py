#!/usr/bin/env python3
"""
Memory Retrieval Benchmark Metrics
评估召回率、MRR、NDCG 等指标
"""

import numpy as np
from typing import Dict, List, Set


def recall_at_k(results: Dict[str, List[str]], relevant: Dict[str, List[str]], k: int = 5) -> float:
    """
    计算 Recall@k
    
    Args:
        results: {query_id: [retrieved_doc_ids]}
        relevant: {query_id: [relevant_doc_ids]}
        k: 截断位置
    
    Returns:
        Recall@k 分数 (0-1)
    """
    hit = 0
    
    for qid in results:
        retrieved = results[qid][:k]
        rel = set(relevant.get(qid, []))
        
        if any(r in rel for r in retrieved):
            hit += 1
    
    return hit / len(results) if results else 0.0


def mrr(results: Dict[str, List[str]], relevant: Dict[str, List[str]]) -> float:
    """
    计算 Mean Reciprocal Rank (MRR)
    
    Returns:
        MRR 分数 (0-1)
    """
    scores = []
    
    for qid in results:
        rel = set(relevant.get(qid, []))
        retrieved = results[qid]
        
        rank = None
        for i, r in enumerate(retrieved):
            if r in rel:
                rank = i + 1
                break
        
        if rank:
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
    
    return np.mean(scores) if scores else 0.0


def ndcg_at_k(results: Dict[str, List[str]], relevant: Dict[str, List[str]], k: int = 5) -> float:
    """
    计算 NDCG@k (Normalized Discounted Cumulative Gain)
    
    Returns:
        NDCG@k 分数 (0-1)
    """
    def dcg(rels, k):
        rels = rels[:k]
        gains = [1.0 / np.log2(i + 2) if r else 0.0 for i, r in enumerate(rels)]
        return np.sum(gains)
    
    def idcg(rels, k):
        ideal = sorted(rels, reverse=True)
        return dcg(ideal, k)
    
    ndcg_scores = []
    
    for qid in results:
        rel_set = set(relevant.get(qid, []))
        retrieved = results[qid][:k]
        
        rels = [1.0 if r in rel_set else 0.0 for r in retrieved]
        
        dcg_score = dcg(rels, k)
        idcg_score = idcg(rels, k)
        
        if idcg_score > 0:
            ndcg_scores.append(dcg_score / idcg_score)
        else:
            ndcg_scores.append(0.0)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def precision_at_k(results: Dict[str, List[str]], relevant: Dict[str, List[str]], k: int = 5) -> float:
    """
    计算 Precision@k
    
    Returns:
        Precision@k 分数 (0-1)
    """
    precisions = []
    
    for qid in results:
        retrieved = results[qid][:k]
        rel = set(relevant.get(qid, []))
        
        hits = sum(1 for r in retrieved if r in rel)
        precisions.append(hits / k)
    
    return np.mean(precisions) if precisions else 0.0


def evaluate_all(results: Dict[str, List[str]], relevant: Dict[str, List[str]]) -> dict:
    """
    计算所有指标
    
    Returns:
        包含所有指标的字典
    """
    return {
        'recall@1': recall_at_k(results, relevant, k=1),
        'recall@5': recall_at_k(results, relevant, k=5),
        'recall@10': recall_at_k(results, relevant, k=10),
        'precision@5': precision_at_k(results, relevant, k=5),
        'precision@10': precision_at_k(results, relevant, k=10),
        'mrr': mrr(results, relevant),
        'ndcg@5': ndcg_at_k(results, relevant, k=5),
        'ndcg@10': ndcg_at_k(results, relevant, k=10)
    }


if __name__ == '__main__':
    # 测试
    test_results = {
        'q1': ['m1', 'm2', 'm3', 'm4', 'm5'],
        'q2': ['m10', 'm11', 'm12', 'm13', 'm14']
    }
    
    test_relevant = {
        'q1': ['m1', 'm3'],
        'q2': ['m10']
    }
    
    metrics = evaluate_all(test_results, test_relevant)
    
    print("测试指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
