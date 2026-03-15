#!/usr/bin/env python3
"""
Cross-Encoder Reranking 测试（small 数据集）

测试配置：
- 数据集：synthetic_perltqa small (500 条记忆，500 个查询)
- 模型：cross-encoder/ms-marco-MiniLM-L-6-v2
- 对比：MemQ vs Cross-Encoder Rerank

预期效果：
- MemQ: Recall@5 ~63%
- Cross-Encoder: Recall@5 ~78% (+15%)
"""

import json
import time
import numpy as np
from typing import List, Dict, Tuple

# 配置
DATA_DIR = '/home/kyj/.openclaw/workspace/synthetic_perltqa'
MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


def load_data():
    """加载 small 数据集"""
    with open(f'{DATA_DIR}/memories_small.json') as f:
        memories = json.load(f)
    
    with open(f'{DATA_DIR}/queries_small.json') as f:
        queries = json.load(f)
    
    return memories[:500], queries[:500]


class MemQRetriever:
    """原始 MemQ 检索器（用于对比）"""
    
    def __init__(self, memories: List[Dict]):
        self.memories = memories
        self.weights = {
            'type': 1.0,
            'length': 1.0,
            'entity': 1.0,
            'stopwords': 1.0,
            'template': 1.0,
            'metadata': 1.0,
        }
    
    def _extract_features(self, memory: Dict) -> Dict[str, float]:
        """提取特征"""
        text = memory.get('content', '')
        metadata = memory.get('metadata', {})
        mem_type = memory.get('type', 'unknown')
        
        # 类型特征
        type_scores = {
            'code': 1.2,
            'knowledge': 1.2,
            'event': 0.9,
            'conversation': 0.9,
            'noise': 0.3,
        }
        type_score = type_scores.get(mem_type, 1.0)
        
        # 长度特征
        length = len(text)
        if length < 10:
            length_score = 0.5
        elif length < 20:
            length_score = 0.8
        elif length > 100:
            length_score = 1.1
        else:
            length_score = 1.0
        
        # 实体特征
        entities = metadata.get('person', '') + metadata.get('project', '') + metadata.get('tech', '')
        entity_score = 1.2 if len(entities) > 0 else 0.8
        
        # 停用词特征
        stopwords_ratio = sum(1 for c in text if c in '的了吗是和在了有') / max(len(text), 1)
        stopwords_score = 0.7 if stopwords_ratio > 0.5 else 1.0
        
        # 模板特征
        noise_patterns = ['有人提到', '但不是用于', '类似的']
        template_score = 0.6 if any(p in text for p in noise_patterns) else 1.0
        
        # 元数据特征
        metadata_score = 1.1 if metadata else 1.0
        
        return {
            'type': type_score,
            'length': length_score,
            'entity': entity_score,
            'stopwords': stopwords_score,
            'template': template_score,
            'metadata': metadata_score,
        }
    
    def predict(self, memory: Dict) -> float:
        """预测质量分数"""
        features = self._extract_features(memory)
        
        quality = 1.0
        for feature_name, feature_value in features.items():
            w = self.weights.get(feature_name, 1.0)
            quality *= w * feature_value
        
        return min(1.0, max(0.0, quality))
    
    def retrieve(self, query: str, k: int = 50) -> List[Tuple[Dict, float]]:
        """检索 Top-K 记忆（基于关键词匹配简化版）"""
        # 简化：使用 TF-IDF 相似度
        scored = []
        for mem in self.memories:
            # 简单的关键词重叠度
            query_words = set(query.lower().replace(' ', ''))
            mem_words = set(mem.get('content', '').lower().replace(' ', ''))
            overlap = len(query_words & mem_words) / max(len(query_words), 1)
            
            # 结合 MemQ 质量分
            memq_score = self.predict(mem)
            final_score = 0.7 * overlap + 0.3 * memq_score
            
            scored.append((mem, final_score))
        
        # 排序
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:k]


class CrossEncoderReranker:
    """Cross-Encoder Reranker"""
    
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型（支持 GPU）"""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            # 自动检测 GPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = CrossEncoder(self.model_name, device=device)
            
            print(f"✅ Cross-Encoder 模型已加载：{self.model_name}")
            print(f"   设备：{device}")
            
            if device == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   GPU: {gpu_name}")
                
                if 'AMD' in gpu_name or 'Radeon' in gpu_name:
                    print(f"   🎯 检测到 AMD GPU (ROCm)")
                    print(f"   💡 提示：ROCm 可能不支持 FP16，使用 FP32 更稳定")
                else:
                    print(f"   🎯 检测到 NVIDIA GPU (CUDA)")
            else:
                print(f"   ⚠️  未检测到 GPU，使用 CPU 模式")
            
        except ImportError:
            print("⚠️  需要安装：pip install sentence-transformers")
            self.model = None
    
    def rerank(self, query: str, memories: List[Dict], 
               top_k: int = 10) -> List[Tuple[Dict, float]]:
        """重排序"""
        if not self.model or len(memories) == 0:
            return [(m, 0.0) for m in memories[:top_k]]
        
        # 限制数量
        memories = memories[:50]
        
        # 构造 pairs
        pairs = [[query, m.get('content', '')] for m in memories]
        
        # 预测
        scores = self.model.predict(pairs)
        
        # 排序
        ranked = sorted(zip(memories, scores), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]


def evaluate(retriever_name, retrieve_func, queries, ground_truth, k=5):
    """评估 Recall@K"""
    hits = 0
    
    for i, query in enumerate(queries):
        if i % 100 == 0:
            print(f"  进度：{i}/{len(queries)}")
        
        # 检索
        results = retrieve_func(query)
        retrieved_ids = [m['id'] for m, _ in results[:k]]
        
        # 检查是否命中
        if query['target_memory_id'] in retrieved_ids:
            hits += 1
    
    recall = hits / len(queries)
    print(f"\n{retriever_name}:")
    print(f"  Recall@{k}: {recall:.3f} ({hits}/{len(queries)})")
    
    return recall


def main():
    print("="*60)
    print("Cross-Encoder Reranking 测试（small 数据集）")
    print("="*60)
    
    # 加载数据
    print("\n1. 加载数据...")
    memories, queries = load_data()
    print(f"   记忆：{len(memories)} 条")
    print(f"   查询：{len(queries)} 个")
    
    # 初始化检索器
    print("\n2. 初始化检索器...")
    memq = MemQRetriever(memories)
    ce_reranker = CrossEncoderReranker()
    
    # 测试 MemQ
    print("\n3. 测试 MemQ...")
    start = time.time()
    
    def memq_retrieve(query):
        return memq.retrieve(query['query'], k=50)
    
    memq_recall = evaluate("MemQ", memq_retrieve, queries, None, k=5)
    memq_time = time.time() - start
    print(f"   耗时：{memq_time:.1f}s")
    print(f"   平均：{memq_time*1000/len(queries):.1f}ms/query")
    
    # 测试 Cross-Encoder Rerank
    print("\n4. 测试 Cross-Encoder Rerank (两阶段)...")
    start = time.time()
    
    def ce_retrieve(query):
        # Stage 1: MemQ 检索 Top-50
        candidates = memq.retrieve(query['query'], k=50)
        
        # Stage 2: Cross-Encoder 重排序 Top-10
        if ce_reranker.model:
            reranked = ce_reranker.rerank(query['query'], 
                                         [m for m, _ in candidates], 
                                         top_k=10)
            return reranked
        else:
            return candidates[:10]
    
    ce_recall = evaluate("Cross-Encoder", ce_retrieve, queries, None, k=5)
    ce_time = time.time() - start
    print(f"   耗时：{ce_time:.1f}s")
    print(f"   平均：{ce_time*1000/len(queries):.1f}ms/query")
    
    # 对比总结
    print("\n" + "="*60)
    print("对比总结")
    print("="*60)
    
    improvement = (ce_recall - memq_recall) / memq_recall * 100
    overhead = (ce_time - memq_time) / memq_time * 100
    
    print(f"Recall@5 提升：{improvement:+.1f}%")
    print(f"延迟增加：{overhead:+.1f}%")
    
    if ce_recall > memq_recall:
        print(f"\n✅ Cross-Encoder 有效！精度提升 {improvement:.1f}%")
        print(f"   代价：延迟增加 {overhead:.1f}%")
    else:
        print(f"\n⚠️  Cross-Encoder 未带来提升")
    
    # 保存结果
    results = {
        'dataset': 'small',
        'memories': len(memories),
        'queries': len(queries),
        'memq_recall': memq_recall,
        'memq_time': memq_time,
        'ce_recall': ce_recall,
        'ce_time': ce_time,
        'improvement': improvement,
        'overhead': overhead,
    }
    
    with open('/home/kyj/.openclaw/workspace/memory-lancedb-pro/results/cross_encoder_small_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存：results/cross_encoder_small_test.json")


if __name__ == '__main__':
    main()
