#!/usr/bin/env python3
"""
Memory Inspector
快速查看和分析记忆数据
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def load_memories(path: str) -> list:
    """加载记忆"""
    memories = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            memories.append(json.loads(line))
    return memories


def analyze(memories: list) -> dict:
    """分析记忆数据"""
    stats = {
        'total': len(memories),
        'types': Counter(m.get('type', 'unknown') for m in memories),
        'persons': Counter(m.get('metadata', {}).get('person', 'unknown') for m in memories),
        'projects': Counter(m.get('metadata', {}).get('project', 'unknown') for m in memories),
        'avg_length': sum(len(m.get('text', '')) for m in memories) / len(memories) if memories else 0
    }
    return stats


def print_stats(stats: dict):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print("记忆数据统计")
    print(f"{'='*60}")
    print(f"总数量：{stats['total']}")
    print(f"平均长度：{stats['avg_length']:.1f} 字符")
    
    print(f"\n类型分布:")
    for t, c in stats['types'].most_common():
        pct = c / stats['total'] * 100
        print(f"  {t}: {c} ({pct:.1f}%)")
    
    print(f"\n人物分布 (Top 5):")
    for p, c in stats['persons'].most_common(5):
        print(f"  {p}: {c}")
    
    print(f"\n项目分布 (Top 5):")
    for proj, c in stats['projects'].most_common(5):
        print(f"  {proj}: {c}")
    
    print(f"{'='*60}\n")


def search(memories: list, keyword: str, limit: int = 10):
    """搜索记忆"""
    results = []
    for m in memories:
        text = m.get('text', '')
        if keyword.lower() in text.lower():
            results.append(m)
            if len(results) >= limit:
                break
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Memory Inspector')
    parser.add_argument('--memory', type=str, 
                        default='memory_db/memories.jsonl',
                        help='记忆文件路径')
    parser.add_argument('--search', type=str,
                        help='搜索关键词')
    parser.add_argument('--limit', type=int,
                        default=10,
                        help='显示数量限制')
    
    args = parser.parse_args()
    
    print(f"加载记忆：{args.memory}")
    memories = load_memories(args.memory)
    
    # 分析
    stats = analyze(memories)
    print_stats(stats)
    
    # 搜索
    if args.search:
        print(f"搜索关键词：'{args.search}'")
        results = search(memories, args.search, args.limit)
        
        if results:
            print(f"找到 {len(results)} 条相关记忆:\n")
            for i, m in enumerate(results, 1):
                print(f"{i}. [{m.get('id')}] ({m.get('type')})")
                print(f"   {m.get('text')}\n")
        else:
            print("未找到匹配的记忆")


if __name__ == '__main__':
    main()
