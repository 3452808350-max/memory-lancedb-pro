#!/usr/bin/env python3
"""
A/B Test: Quality-aware Retrieval vs Baseline
对比实验：质量感知检索 vs 基线
"""

import json
import argparse
import sys
sys.path.insert(0, 'benchmark')

from tasks.retrieval_task import HybridRetrieval
from metrics.retrieval_metrics import evaluate_all


def load_memories_with_scores(path: str):
    """加载带质量分数的记忆"""
    memories = []
    quality_scores = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            memories.append(m)
            quality_scores[m['id']] = m.get('quality_score', 0.5)
    
    return memories, quality_scores


def load_queries(path: str):
    """加载查询"""
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def run_retrieval(corpus, queries, use_quality: bool = False, quality_scores: dict = None):
    """运行检索"""
    retriever = HybridRetrieval(corpus)
    
    results = {}
    relevant = {}
    
    for q in queries:
        qid = q['id']
        query_text = q['query']
        relevant_ids = q['relevant_ids']
        
        # 检索
        search_results = retriever.search(query_text, k=10, alpha=0.5)
        retrieved_ids = [doc_id for doc_id, _ in search_results]
        
        results[qid] = retrieved_ids
        relevant[qid] = relevant_ids
    
    return evaluate_all(results, relevant)


def main():
    parser = argparse.ArgumentParser(description='Quality-aware Retrieval A/B Test')
    parser.add_argument('--memory', type=str, 
                        default='memory_db/memories_scored.jsonl',
                        help='带质量分数的记忆文件')
    parser.add_argument('--queries', type=str,
                        default='memory_db/queries.jsonl',
                        help='查询文件')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("A/B Test: 质量感知检索 vs 基线")
    print(f"{'='*60}\n")
    
    # 加载数据
    print("加载数据...")
    corpus, quality_scores = load_memories_with_scores(args.memory)
    queries = load_queries(args.queries)
    
    print(f"记忆：{len(corpus)} 条")
    print(f"查询：{len(queries)} 个\n")
    
    # Baseline（不使用质量分）
    print("运行 Baseline（无质量分）...")
    baseline_metrics = run_retrieval(corpus, queries, use_quality=False)
    
    print(f"\nBaseline 指标:")
    print(f"  Recall@5: {baseline_metrics['recall@5']:.3f}")
    print(f"  MRR:      {baseline_metrics['mrr']:.3f}\n")
    
    # Quality-aware（使用质量分）
    print("运行 Quality-aware Retrieval...")
    # 这里需要修改 retrieval_task.py 来支持 quality_scores
    # 暂时跳过，等修改完成后再运行
    print("  ⚠️  需要先修改 retrieval_task.py 支持 quality_scores\n")
    
    # 对比
    print(f"{'='*60}")
    print("实验设计")
    print(f"{'='*60}")
    print("""
方案：
1. Baseline: 标准 hybrid retrieval
2. Quality-aware: 在 RRF 融合时乘以 quality_score

预期：
- Recall@5 提升 5-10%
- MRR 提升 0.05-0.1

下一步：
修改 benchmark/tasks/retrieval_task.py 的 _rrf_fusion 方法，
添加 quality_scores 参数并在融合时应用降权。
    """)


if __name__ == '__main__':
    main()
