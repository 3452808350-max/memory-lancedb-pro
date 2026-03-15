#!/usr/bin/env python3
"""
测试 Ollama Reranker 模型
"""

import requests
import json

def test_rerank():
    """测试重排序"""
    
    query = "OpenClaw API 怎么配置"
    documents = [
        "在 OpenClaw 项目中，我们讨论了 API 的实现方案",
        "明天要和吴博士讨论 DSS 选股系统的进度",
        "K 喜欢用向量检索来处理 MemQ 相关任务",
        "有人提到过类似的 BM25 方案但不是用于 Kimi Remote API"
    ]
    
    # 尝试 1: /api/rerank
    print("尝试 1: /api/rerank")
    try:
        resp = requests.post(
            'http://localhost:11434/api/rerank',
            json={
                'model': 'modelscope.cn/dengcao/Qwen3-Reranker-0.6B-GGUF',
                'query': query,
                'documents': documents
            },
            timeout=30
        )
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # 尝试 2: /api/embeddings (带 query 和 documents)
    print("尝试 2: /api/embeddings")
    try:
        resp = requests.post(
            'http://localhost:11434/api/embeddings',
            json={
                'model': 'modelscope.cn/dengcao/Qwen3-Reranker-0.6B-GGUF',
                'query': query,
                'documents': documents
            },
            timeout=30
        )
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # 尝试 3: 手动计算相似度
    print("尝试 3: 分别编码后计算相似度")
    try:
        # 编码查询
        q_resp = requests.post(
            'http://localhost:11434/api/embeddings',
            json={
                'model': 'modelscope.cn/Qwen/Qwen3-Embedding-4B-GGUF',
                'prompt': query
            },
            timeout=30
        )
        q_emb = q_resp.json().get('embedding', [])
        
        # 编码文档
        d_embs = []
        for doc in documents:
            d_resp = requests.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': 'modelscope.cn/Qwen/Qwen3-Embedding-4B-GGUF',
                    'prompt': doc
                },
                timeout=30
            )
            d_embs.append(d_resp.json().get('embedding', []))
        
        # 计算余弦相似度
        import numpy as np
        from numpy import dot
        from numpy.linalg import norm
        
        q_vec = np.array(q_emb)
        scores = []
        for i, d_vec in enumerate(d_embs):
            d_vec = np.array(d_vec)
            sim = dot(q_vec, d_vec) / (norm(q_vec) * norm(d_vec))
            scores.append(sim)
            print(f"Doc{i+1}: {sim:.4f} - {documents[i][:40]}...")
        
        # 排序
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        print("\n重排序结果:")
        for i, (doc, score) in enumerate(ranked, 1):
            print(f"{i}. [Score: {score:.4f}] {doc[:50]}...")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    test_rerank()
