#!/usr/bin/env python3
"""
Qwen3-Reranker 本地测试
"""

import json
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def load_reranker(model_path: str = "/home/kyj/.ollama/models/blobs/"):
    """从 Ollama 缓存加载模型"""
    print("加载 Qwen3-Reranker...")
    
    # 尝试从 HuggingFace 加载
    model_name = "dengcao/Qwen3-Reranker"
    
    try:
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
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("✓ 使用 GPU")
        else:
            print("✓ 使用 CPU")
        
        return tokenizer, model
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        print("\n请手动下载模型：")
        print("  git clone https://huggingface.co/dengcao/Qwen3-Reranker")
        return None, None


def rerank(tokenizer, model, query: str, documents: list, top_k: int = 10) -> list:
    """
    重排序文档
    
    Returns:
        [(doc_index, score), ...] 按分数降序
    """
    # 构造输入对
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
        outputs = model(**inputs, return_dict=True)
        scores = outputs.logits.view(-1,).float().cpu().numpy()
    
    # 排序
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    return indexed_scores[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Qwen3-Reranker Test')
    parser.add_argument('--query', type=str, default="OpenClaw API 怎么配置")
    parser.add_argument('--memory', type=str, default='memory_db/memories.jsonl')
    parser.add_argument('--top-k', type=int, default=10)
    
    args = parser.parse_args()
    
    # 加载模型
    tokenizer, model = load_reranker()
    if not model:
        return
    
    # 加载记忆
    print(f"\n加载记忆：{args.memory}")
    with open(args.memory) as f:
        memories = [json.loads(line) for line in f][:20]  # 只测前 20 条
    
    print(f"测试样本：{len(memories)} 条\n")
    
    # 重排序
    print(f"重排序中...")
    documents = [m.get('content', m.get('text', '')) for m in memories]
    ranked = rerank(tokenizer, model, args.query, documents, top_k=args.top_k)
    
    # 显示结果
    print(f"\n{'='*70}")
    print(f"查询：{args.query}")
    print(f"{'='*70}\n")
    
    for rank, (idx, score) in enumerate(ranked, 1):
        m = memories[idx]
        print(f"{rank}. [Score: {score:.4f}] [{m.get('type')}]")
        print(f"   {m.get('content', m.get('text', ''))[:70]}...\n")


if __name__ == '__main__':
    main()
