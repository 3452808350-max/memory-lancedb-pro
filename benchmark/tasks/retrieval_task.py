#!/usr/bin/env python3
"""
Memory Retrieval Task
实现向量检索、BM25 检索和混合检索
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re


class VectorRetrieval:
    """向量检索器"""
    
    def __init__(self, corpus: List[Dict], model_name: str = "BAAI/bge-m3"):
        self.corpus = corpus
        self.texts = [doc['text'] for doc in corpus]
        self.ids = [doc['id'] for doc in corpus]
        
        print(f"加载模型：{model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print("生成 embeddings...")
        self.embeddings = self.model.encode(self.texts, batch_size=64, show_progress_bar=True)
        print(f"完成！{len(self.texts)} 条记忆，维度：{self.embeddings.shape[1]}")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        搜索最相关的记忆
        
        Returns:
            [(doc_id, score), ...] 按相似度降序排列
        """
        q_emb = self.model.encode([query])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        
        # 按相似度排序
        idx = sims.argsort()[::-1][:k]
        
        return [(self.ids[i], float(sims[i])) for i in idx]


class BM25Retrieval:
    """BM25 检索器"""
    
    def __init__(self, corpus: List[Dict]):
        self.corpus = corpus
        self.texts = [doc['text'] for doc in corpus]
        self.ids = [doc['id'] for doc in corpus]
        
        # 中文分词（简单按字符分割）
        print("构建 BM25 索引...")
        self.tokenized = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)
        print(f"完成！{len(self.texts)} 条记忆")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单中文分词（按字符）"""
        # 移除标点，保留中英文
        text = re.sub(r'[^\w\s]', ' ', text)
        return list(text.replace(' ', ''))
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """搜索最相关的记忆"""
        q_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        
        # 按分数排序
        idx = scores.argsort()[::-1][:k]
        
        return [(self.ids[i], float(scores[i])) for i in idx]


class HybridRetrieval:
    """混合检索器 (RRF - Reciprocal Rank Fusion)"""
    
    def __init__(self, corpus: List[Dict], model_name: str = "BAAI/bge-m3"):
        self.corpus = corpus
        self.vector = VectorRetrieval(corpus, model_name)
        self.bm25 = BM25Retrieval(corpus)
    
    def search(self, query: str, k: int = 10, 
               alpha: float = 0.5,
               use_rrf: bool = True) -> List[Tuple[str, float]]:
        """
        混合检索
        
        Args:
            query: 查询文本
            k: 返回结果数
            alpha: 向量检索权重 (0=纯 BM25, 1=纯向量)
            use_rrf: 是否使用 RRF 融合
        """
        # 获取两个检索结果
        vector_results = self.vector.search(query, k=50)
        bm25_results = self.bm25.search(query, k=50)
        
        if use_rrf:
            # RRF 融合
            return self._rrf_fusion(vector_results, bm25_results, k)
        else:
            # 线性融合
            return self._linear_fusion(vector_results, bm25_results, k, alpha)
    
    def _rrf_fusion(self, vector_results: List[Tuple], bm25_results: List[Tuple], k: int) -> List[Tuple]:
        """RRF (Reciprocal Rank Fusion) 融合"""
        scores = {}
        
        # Vector scores
        for i, (doc_id, _) in enumerate(vector_results):
            score = scores.get(doc_id, 0)
            scores[doc_id] = score + 1.0 / (60 + i)
        
        # BM25 scores
        for i, (doc_id, _) in enumerate(bm25_results):
            score = scores.get(doc_id, 0)
            scores[doc_id] = score + 1.0 / (60 + i)
        
        # 排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return sorted_docs
    
    def _linear_fusion(self, vector_results: List[Tuple], bm25_results: List[Tuple], 
                       k: int, alpha: float) -> List[Tuple]:
        """线性融合"""
        # 归一化分数
        def normalize(results):
            if not results:
                return {}
            scores = {doc_id: score for doc_id, score in results}
            mean = np.mean(list(scores.values()))
            std = np.std(list(scores.values())) + 1e-9
            return {doc_id: (score - mean) / std for doc_id, score in scores.items()}
        
        vector_norm = normalize(vector_results)
        bm25_norm = normalize(bm25_results)
        
        # 合并
        all_docs = set(vector_norm.keys()) | set(bm25_norm.keys())
        
        final_scores = {}
        for doc_id in all_docs:
            v_score = vector_norm.get(doc_id, 0)
            b_score = bm25_norm.get(doc_id, 0)
            final_scores[doc_id] = alpha * v_score + (1 - alpha) * b_score
        
        # 排序
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return sorted_docs


if __name__ == '__main__':
    # 测试
    test_corpus = [
        {'id': 'm1', 'text': 'OpenClaw 项目中讨论了 API 的实现方案'},
        {'id': 'm2', 'text': '明天要和吴博士讨论 DSS 选股系统的进度'},
        {'id': 'm3', 'text': 'K 喜欢用向量检索来处理 MemQ 相关任务'}
    ]
    
    print("=" * 60)
    print("向量检索测试")
    print("=" * 60)
    vector = VectorRetrieval(test_corpus)
    results = vector.search("OpenClaw API 怎么配置", k=3)
    print(f"结果：{results}")
    
    print("\n" + "=" * 60)
    print("BM25 检索测试")
    print("=" * 60)
    bm25 = BM25Retrieval(test_corpus)
    results = bm25.search("OpenClaw API 配置", k=3)
    print(f"结果：{results}")
    
    print("\n" + "=" * 60)
    print("混合检索测试")
    print("=" * 60)
    hybrid = HybridRetrieval(test_corpus)
    results = hybrid.search("OpenClaw API 怎么配置", k=3, alpha=0.5)
    print(f"结果：{results}")
