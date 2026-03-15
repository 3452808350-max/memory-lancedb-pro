#!/usr/bin/env python3
"""
Memory Retrieval Benchmark Runner
统一运行所有基准测试
"""

import json
import argparse
import time
from pathlib import Path
from typing import Dict, List

from tasks.retrieval_task import VectorRetrieval, BM25Retrieval, HybridRetrieval
from metrics.retrieval_metrics import evaluate_all


def load_corpus(path: str) -> List[Dict]:
    """加载记忆语料库"""
    corpus = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"加载 {len(corpus)} 条记忆")
    return corpus


def load_queries(path: str) -> List[Dict]:
    """加载查询集"""
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"加载 {len(queries)} 个查询")
    return queries


def run_benchmark(corpus: List[Dict], queries: List[Dict], 
                  retrieval_type: str = 'hybrid',
                  alpha: float = 0.5,
                  k: int = 10) -> Dict:
    """
    运行基准测试
    
    Args:
        corpus: 记忆语料库
        queries: 查询集
        retrieval_type: 检索类型 ('vector', 'bm25', 'hybrid')
        alpha: 混合检索权重
        k: 返回结果数
    """
    print(f"\n{'='*60}")
    print(f"基准测试配置")
    print(f"{'='*60}")
    print(f"检索类型：{retrieval_type}")
    print(f"混合权重 alpha: {alpha}")
    print(f"返回结果数：{k}")
    print(f"{'='*60}\n")
    
    # 初始化检索器
    if retrieval_type == 'vector':
        retriever = VectorRetrieval(corpus)
    elif retrieval_type == 'bm25':
        retriever = BM25Retrieval(corpus)
    elif retrieval_type == 'hybrid':
        retriever = HybridRetrieval(corpus)
    else:
        raise ValueError(f"未知检索类型：{retrieval_type}")
    
    # 运行检索
    results = {}
    relevant = {}
    
    print(f"\n开始检索测试...")
    start_time = time.time()
    
    for i, query in enumerate(queries):
        qid = query['id']
        query_text = query['query']
        relevant_ids = query['relevant_ids']
        
        # 执行检索
        if retrieval_type == 'hybrid':
            search_results = retriever.search(query_text, k=k, alpha=alpha)
        else:
            search_results = retriever.search(query_text, k=k)
        
        # 提取文档 ID
        retrieved_ids = [doc_id for doc_id, _ in search_results]
        results[qid] = retrieved_ids
        relevant[qid] = relevant_ids
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            qps = (i + 1) / elapsed
            print(f"  进度：{i+1}/{len(queries)} ({qps:.1f} queries/s)")
    
    elapsed = time.time() - start_time
    qps = len(queries) / elapsed
    
    print(f"\n检索完成！总耗时：{elapsed:.1f}s, 速度：{qps:.1f} queries/s")
    
    # 计算指标
    print(f"\n计算评估指标...")
    metrics = evaluate_all(results, relevant)
    
    return {
        'config': {
            'retrieval_type': retrieval_type,
            'alpha': alpha,
            'k': k,
            'corpus_size': len(corpus),
            'query_size': len(queries)
        },
        'metrics': metrics,
        'performance': {
            'total_time': elapsed,
            'queries_per_second': qps
        }
    }


def save_results(results: Dict, output_path: str):
    """保存结果"""
    import json
    from datetime import datetime
    
    results['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到：{output_path}")


def print_results(results: Dict):
    """打印结果"""
    print(f"\n{'='*60}")
    print("基准测试结果")
    print(f"{'='*60}")
    
    config = results['config']
    print(f"\n配置:")
    print(f"  语料库大小：{config['corpus_size']}")
    print(f"  查询数量：{config['query_size']}")
    print(f"  检索类型：{config['retrieval_type']}")
    print(f"  混合权重：{config.get('alpha', 'N/A')}")
    
    metrics = results['metrics']
    print(f"\n检索指标:")
    print(f"  Recall@1:  {metrics['recall@1']:.3f}")
    print(f"  Recall@5:  {metrics['recall@5']:.3f}")
    print(f"  Recall@10: {metrics['recall@10']:.3f}")
    print(f"  Precision@5:  {metrics['precision@5']:.3f}")
    print(f"  Precision@10: {metrics['precision@10']:.3f}")
    print(f"  MRR:     {metrics['mrr']:.3f}")
    print(f"  NDCG@5:  {metrics['ndcg@5']:.3f}")
    print(f"  NDCG@10: {metrics['ndcg@10']:.3f}")
    
    perf = results['performance']
    print(f"\n性能:")
    print(f"  总耗时：{perf['total_time']:.1f}s")
    print(f"  速度：{perf['queries_per_second']:.1f} queries/s")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Memory Retrieval Benchmark')
    
    parser.add_argument('--corpus', type=str, 
                        default='datasets/memory_corpus.jsonl',
                        help='记忆语料库路径')
    parser.add_argument('--queries', type=str,
                        default='datasets/queries.jsonl',
                        help='查询集路径')
    parser.add_argument('--type', type=str,
                        choices=['vector', 'bm25', 'hybrid'],
                        default='hybrid',
                        help='检索类型')
    parser.add_argument('--alpha', type=float,
                        default=0.5,
                        help='混合检索权重 (0=纯 BM25, 1=纯向量)')
    parser.add_argument('--k', type=int,
                        default=10,
                        help='返回结果数')
    parser.add_argument('--output', type=str,
                        default='results/benchmark_results.json',
                        help='结果输出路径')
    
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据集...")
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    
    # 运行基准测试
    results = run_benchmark(
        corpus=corpus,
        queries=queries,
        retrieval_type=args.type,
        alpha=args.alpha,
        k=args.k
    )
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    save_results(results, args.output)


if __name__ == '__main__':
    main()
