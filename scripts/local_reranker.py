#!/usr/bin/env python3
"""
本地 Reranker 测试（直接加载 transformers 模型）
"""

import json
import argparse
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def load_reranker(model_name: str = "dengcao/Qwen3-Reranker"):
    """加载本地 reranker 模型"""
    print(f"加载模型：{model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir='/home/kyj/.cache/huggingface'
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir='/home/kyj/.cache/huggingface'
    )
    
    # 使用 GPU（如果有）
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用 GPU")
    else:
        print("使用 CPU")
    
    model.eval()
    return tokenizer, model


def rerank(tokenizer, model, query: str, documents: list, top_k: int = 10) -> list:
    """
    重排序文档
    
    Returns:
        [(doc, score), ...] 按分数降序
    """
    # 构造输入
    pairs = [[query, doc] for doc in documents]
    
    # 批量预测
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 移到 GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 预测
        scores = model(**inputs, return_dict=True).logits.view(-1,).float()
    
    # 排序
    scored_docs = list(zip(documents, scores.cpu().numpy()))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_docs[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Local Reranker Test')
    parser.add_argument('--query', type=str, default="OpenClaw API 怎么配置")
    parser.add_argument('--memory', type=str, default='memory_db/memories_scored.jsonl')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--model', type=str, default='dengcao/Qwen3-Reranker')
    parser.add_argument('--use-quality', action='store_true', help='结合质量分加权')
    
    args = parser.parse_args()
    
    # 加载模型
    tokenizer, model = load_reranker(args.model)
    
    # 加载记忆
    print(f"\n加载记忆：{args.memory}")
    memories = []
    quality_scores = {}
    
    with open(args.memory, 'r', encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            memories.append(m)
            quality_scores[m['id']] = m.get('quality_score', 0.5)
    
    print(f"加载 {len(memories)} 条记忆\n")
    
    # 重排序
    print(f"重排序 {len(memories)} 个文档...")
    texts = [m.get('content', m.get('text', '')) for m in memories]
    ranked = rerank(tokenizer, model, args.query, texts, top_k=min(args.top_k * 3, 50))
    
    # 应用质量分（可选）
    if args.use_quality:
        print("应用质量分加权...")
        scored_results = []
        for doc, score in ranked:
            # 找到对应的记忆
            for m in memories:
                if m.get('content', m.get('text', '')) == doc:
                    q_score = quality_scores.get(m['id'], 0.5)
                    final_score = float(score) * q_score
                    scored_results.append((m, final_score, float(score)))
                    break
        
        # 重新排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        ranked = scored_results[:args.top_k]
    
    # 显示结果
    print(f"\n{'='*70}")
    print(f"查询：{args.query}")
    print(f"{'='*70}\n")
    
    for i, item in enumerate(ranked, 1):
        if args.use_quality:
            m, final_score, raw_score = item
            print(f"{i}. [Final: {final_score:.4f}] (Rerank: {raw_score:.4f}, Quality: {quality_scores.get(m['id'], 'N/A')})")
        else:
            m, score = item
            print(f"{i}. [Score: {score:.4f}]")
        
        print(f"   [{m.get('type')}] {m.get('content', m.get('text', ''))[:70]}...\n")


if __name__ == '__main__':
    main()
