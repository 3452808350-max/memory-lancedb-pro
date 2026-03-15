#!/usr/bin/env python3
"""
Memory Quality Scorer
主动识别和抑制噪声记忆
"""

import json
import re
import argparse
from typing import Dict, List
from collections import Counter


# 中文停用词（简化版）
STOPWORDS_ZH = set('的了吗是和在了有就这以个不我们要可以后上为也你他都吧嘛呢没还又着那从好没自己吗到最她他让用能做对没小可大没太')

# 类型权重（基于信息量）
TYPE_WEIGHTS = {
    'code': 1.2,        # 高信息量
    'knowledge': 1.2,   # 高信息量
    'event': 0.9,       # 中等
    'conversation': 0.9, # 中等
    'noise': 0.3        # 低质量 - 关键！
}


def count_entities(text: str) -> int:
    """统计文本中的实体数量（人名、项目名等）"""
    # 简单启发式：大写字母、引号内容、特定模式
    entities = 0
    
    # 人名模式（中文 2-4 字）
    entities += len(re.findall(r'[A-Z][a-z]+', text))
    
    # 项目名模式（包含英文或特定关键词）
    project_keywords = ['API', 'AI', 'ML', 'DB', 'System', '系统', '项目', '平台']
    for kw in project_keywords:
        if kw in text:
            entities += 1
    
    return entities


def quality_score(memory: Dict) -> float:
    """
    计算记忆质量分数（0-1）
    
    特征：
    1. 类型权重
    2. 文本长度
    3. 实体密度
    4. 停用词比例
    5. 模板检测
    """
    text = memory.get('content', '') or memory.get('text', '')
    mem_type = memory.get('type', 'unknown')
    metadata = memory.get('metadata', {})
    
    score = 1.0
    
    # 1. 类型权重（最重要）
    score *= TYPE_WEIGHTS.get(mem_type, 1.0)
    
    # 2. 长度惩罚
    if len(text) < 10:
        score *= 0.5
    elif len(text) < 20:
        score *= 0.8
    elif len(text) > 100:
        score *= 1.1  # 长文本更有信息量
    
    # 3. 实体密度
    entity_count = count_entities(text)
    if entity_count >= 2:
        score *= 1.2
    elif entity_count == 0:
        score *= 0.8
    
    # 4. 停用词比例
    tokens = [c for c in text if c not in ' \n\t']
    if tokens:
        stopwords_ratio = sum(1 for t in tokens if t in STOPWORDS_ZH) / len(tokens)
        if stopwords_ratio > 0.5:
            score *= 0.7
    
    # 5. 模板检测
    template_patterns = ['有人提到', '但不是用于', '类似的', '方案']
    if any(p in text for p in template_patterns):
        score *= 0.6  # 很可能是噪声模板
    
    # 6. 元数据完整性
    if metadata:
        score *= 1.1  # 有元数据的记忆更可靠
    
    # 归一化到 0-1
    return min(1.0, max(0.0, score))


def score_all_memories(input_path: str, output_path: str):
    """为所有记忆打分"""
    memories = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            memories.append(json.loads(line))
    
    print(f"加载 {len(memories)} 条记忆")
    
    # 计算分数
    for m in memories:
        m['quality_score'] = quality_score(m)
    
    # 统计
    scores = [m['quality_score'] for m in memories]
    print(f"\n质量分数统计:")
    print(f"  最小：{min(scores):.3f}")
    print(f"  最大：{max(scores):.3f}")
    print(f"  平均：{sum(scores)/len(scores):.3f}")
    
    # 按类型统计
    type_scores = {}
    for m in memories:
        t = m.get('type', 'unknown')
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(m['quality_score'])
    
    print(f"\n按类型统计:")
    for t, s in sorted(type_scores.items()):
        avg = sum(s) / len(s)
        print(f"  {t}: {avg:.3f} ({len(s)}条)")
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for m in memories:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    
    print(f"\n结果已保存到：{output_path}")
    
    return memories


def filter_by_threshold(input_path: str, output_path: str, threshold: float = 0.5):
    """按阈值过滤低质量记忆"""
    memories = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            memories.append(json.loads(line))
    
    filtered = [m for m in memories if m.get('quality_score', 0) >= threshold]
    
    print(f"原始：{len(memories)} 条")
    print(f"过滤后：{len(filtered)} 条 (阈值 >= {threshold})")
    print(f"保留率：{len(filtered)/len(memories)*100:.1f}%")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for m in filtered:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    
    print(f"结果已保存到：{output_path}")


def main():
    parser = argparse.ArgumentParser(description='Memory Quality Scorer')
    parser.add_argument('--input', type=str, required=True,
                        help='输入记忆文件')
    parser.add_argument('--output', type=str, required=True,
                        help='输出文件（带分数）')
    parser.add_argument('--filter-threshold', type=float,
                        help='过滤阈值（可选）')
    
    args = parser.parse_args()
    
    # 计算分数
    score_all_memories(args.input, args.output)
    
    # 可选：过滤
    if args.filter_threshold:
        filter_output = args.output.replace('.jsonl', f'_filtered{args.filter_threshold}.jsonl')
        filter_by_threshold(args.output, filter_output, args.filter_threshold)


if __name__ == '__main__':
    main()
