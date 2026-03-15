#!/usr/bin/env python3
"""
使用 Ollama 的 reranker（通过 generate API）
"""

import json
import argparse
import requests


def rerank_with_ollama(query: str, documents: list, model: str = "Qwen3-Reranker-0.6B") -> list:
    """
    使用 Ollama rerank
    
    由于 Ollama 没有原生 /api/rerank，我们用 prompt 工程实现
    """
    
    # 构造 prompt
    prompt = f"""Rank these documents for the query: "{query}"

Return ONLY a JSON array of indices in ranked order (most relevant first).

Documents:
"""
    
    for i, doc in enumerate(documents):
        prompt += f"[{i}] {doc}\n"
    
    prompt += "\nRanked indices (JSON array):"
    
    # 调用 Ollama
    resp = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': f'modelscope.cn/dengcao/{model}-GGUF',
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': 0.0,  # 确定性输出
                'num_predict': 100
            }
        },
        timeout=120
    )
    
    result = resp.json()
    response_text = result.get('response', '')
    
    # 解析 JSON
    try:
        # 提取 JSON 数组
        import re
        match = re.search(r'\[[\d,\s]+\]', response_text)
        if match:
            ranked_indices = json.loads(match.group())
            return ranked_indices
        else:
            print(f"无法解析响应：{response_text}")
            return list(range(len(documents)))
    except:
        print(f"JSON 解析失败：{response_text}")
        return list(range(len(documents)))


def main():
    parser = argparse.ArgumentParser(description='Ollama Reranker')
    parser.add_argument('--query', type=str, default="OpenClaw API 怎么配置")
    parser.add_argument('--memory', type=str, default='memory_db/memories_scored.jsonl')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--model', type=str, default='Qwen3-Reranker-0.6B')
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
    
    # 采样（rerank 太多文档太慢）
    if len(memories) > 50:
        print(f"文档太多，采样前 50 条进行 rerank 演示...")
        sample_memories = memories[:50]
    else:
        sample_memories = memories
    
    texts = [m.get('content', m.get('text', '')) for m in sample_memories]
    
    # Rerank
    print(f"Rerank {len(texts)} 个文档...")
    ranked_indices = rerank_with_ollama(args.query, texts, args.model)
    
    print(f"\n{'='*70}")
    print(f"查询：{args.query}")
    print(f"{'='*70}\n")
    
    for rank, idx in enumerate(ranked_indices[:args.top_k], 1):
        if idx < len(sample_memories):
            m = sample_memories[idx]
            print(f"{rank}. [{m.get('type')}] {m.get('content', m.get('text', ''))[:70]}...")
            print(f"   Quality: {quality_scores.get(m['id'], 'N/A')}\n")


if __name__ == '__main__':
    main()
