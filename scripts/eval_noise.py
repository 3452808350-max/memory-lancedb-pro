#!/usr/bin/env python3
"""
Noise Memory Detection
检测低质量记忆比例
"""

import json
import argparse
from typing import List, Dict


STOPWORDS_ZH = set('的了吗是和在了有就这以个不我们要可以后上为也你他都吧嘛呢没还又着那从好没自己吗到最她他让用能做对没小可大没太')
STOPWORDS_EN = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although', 'though', 'after', 'before', 'when', 'whenever', 'where', 'wherever', 'whether', 'which', 'while', 'who', 'whoever', 'whom', 'whose', 'what', 'whatever', 'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}


def is_noise(text: str, min_tokens: int = 5, stopwords_ratio: float = 0.8) -> bool:
    """
    判断文本是否为噪声
    
    Args:
        text: 输入文本
        min_tokens: 最小 token 数
        stopwords_ratio: 停用词比例阈值
    
    Returns:
        True 如果是噪声
    """
    # 分词（简单按空格和标点）
    import re
    tokens_zh = list(text.replace(' ', ''))
    tokens_en = text.lower().split()
    
    # 检查长度
    if len(tokens_zh) < min_tokens and len(tokens_en) < min_tokens:
        return True
    
    # 检查停用词比例
    all_tokens = tokens_zh + tokens_en
    if not all_tokens:
        return True
    
    stopwords = STOPWORDS_ZH | STOPWORDS_EN
    stopwords_count = sum(1 for t in all_tokens if t in stopwords)
    ratio = stopwords_count / len(all_tokens)
    
    if ratio > stopwords_ratio:
        return True
    
    # 检查模板占位符
    template_patterns = ['{', '}', 'XXX', '___', '***', '...']
    if any(p in text for p in template_patterns):
        return True
    
    return False


def eval_noise(memory_path: str) -> dict:
    """
    评估噪声比例
    
    Returns:
        包含统计信息的字典
    """
    noise_count = 0
    total_count = 0
    noise_examples = []
    
    with open(memory_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            memory = json.loads(line)
            text = memory.get('text', '')
            
            if is_noise(text):
                noise_count += 1
                if len(noise_examples) < 10:
                    noise_examples.append({
                        'id': memory.get('id'),
                        'text': text,
                        'type': memory.get('type')
                    })
    
    noise_ratio = noise_count / total_count if total_count > 0 else 0
    
    return {
        'total_memories': total_count,
        'noise_count': noise_count,
        'noise_ratio': noise_ratio,
        'noise_ratio_percent': f"{noise_ratio * 100:.2f}%",
        'noise_examples': noise_examples
    }


def main():
    parser = argparse.ArgumentParser(description='Noise Memory Detection')
    parser.add_argument('--memory', type=str, required=True,
                        help='Path to memories.jsonl')
    parser.add_argument('--output', type=str,
                        default='results/noise.json',
                        help='Output path')
    
    args = parser.parse_args()
    
    print(f"评估噪声记忆：{args.memory}")
    results = eval_noise(args.memory)
    
    print(f"\n{'='*60}")
    print("噪声记忆评估结果")
    print(f"{'='*60}")
    print(f"总记忆数：{results['total_memories']}")
    print(f"噪声数量：{results['noise_count']}")
    print(f"噪声比例：{results['noise_ratio_percent']}")
    
    if results['noise_examples']:
        print(f"\n噪声示例:")
        for ex in results['noise_examples'][:5]:
            print(f"  [{ex['id']}] ({ex['type']}): {ex['text'][:50]}...")
    
    print(f"{'='*60}\n")
    
    # 保存结果
    import json
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到：{args.output}")


if __name__ == '__main__':
    main()
