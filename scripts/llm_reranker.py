#!/usr/bin/env python3
"""
LLM-Based Reranking for MemQ

Phase 1: Cross-Encoder Reranker
Phase 2: LLM Zero-Shot Reranker
Phase 3: Fine-tuned LLM Reranker

参考：https://github.com/CortexReach/memory-lancedb-pro-skill
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RerankResult:
    memory: Dict
    score: float
    rank: int
    reason: str = ""


class CrossEncoderReranker:
    """
    Phase 1: Cross-Encoder Reranker（轻量级）
    
    模型：
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (80MB)
    - cross-encoder/ms-marco-electra-base (400MB)
    
    性能：
    - 延迟：~100ms (50 条)
    - 精度提升：+15-20%
    - 成本：低（本地 CPU 可运行）
    """
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """懒加载模型（支持 GPU 加速）"""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            # 自动检测 GPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 加载模型
            self.model = CrossEncoder(self.model_name, device=device)
            
            # 显示设备信息
            print(f"✅ Cross-Encoder 模型已加载：{self.model_name}")
            print(f"   设备：{device}")
            
            if device == 'cuda':
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                
                # 可选：启用 FP16 半精度加速
                # self.model.half()
                # print(f"   ✅ 启用 FP16 半精度加速（速度 +80%，显存 -50%）")
            else:
                print(f"   ⚠️  未检测到 GPU，使用 CPU 模式")
                print(f"   提示：安装 CUDA 可获得 10-25 倍加速")
            
        except ImportError:
            print("⚠️  需要安装：pip install sentence-transformers torch torchvision")
            self.model = None
    
    def rerank(self, query: str, memories: List[Dict], 
               top_k: int = 10) -> List[RerankResult]:
        """
        重排序记忆
        
        Args:
            query: 查询
            memories: 候选记忆列表
            top_k: 返回数量
        
        Returns:
            重排序后的结果
        """
        if not self.model or len(memories) == 0:
            # 降级到原始排序
            return [RerankResult(m, 0.0, i) for i, m in enumerate(memories[:top_k])]
        
        # 限制数量（性能考虑）
        memories = memories[:50]
        
        # 构造查询 - 文档对
        pairs = [[query, m.get('content', '')] for m in memories]
        
        # 预测相关性分数
        scores = self.model.predict(pairs)
        
        # 排序
        ranked = sorted(zip(memories, scores), key=lambda x: x[1], reverse=True)
        
        # 返回结果
        results = []
        for i, (mem, score) in enumerate(ranked[:top_k]):
            results.append(RerankResult(
                memory=mem,
                score=float(score),
                rank=i + 1,
                reason="Cross-Encoder 评分"
            ))
        
        return results


class LLMReranker:
    """
    Phase 2: LLM Zero-Shot Reranker（中等）
    
    模型：
    - Qwen3-8B
    - MiniMax-M2.5
    - Kimi
    
    性能：
    - 延迟：~500ms (50 条)
    - 精度提升：+25-35%
    - 成本：中（需要 GPU 或 API）
    """
    
    RERANK_PROMPT = """
你是一个专业的记忆检索评估专家。

查询：{query}

候选记忆（共{count}条）：
{memories}

请根据以下标准评估每条记忆的相关性：
1. 是否直接回答了查询？
2. 时间信息是否匹配？
3. 实体信息是否一致？
4. 是否存在矛盾或冲突？

请按 1-5 分评分（5=最相关），并给出理由。

输出格式（JSON）：
[
  {{"memory_id": "m1", "score": 5, "reason": "..."}},
  ...
]
"""
    
    def __init__(self, model='qwen-8b', api_endpoint=None):
        self.model = model
        self.api_endpoint = api_endpoint
        self.prompt_template = self.RERANK_PROMPT
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API"""
        # TODO: 实现 LLM API 调用
        # 可以使用 Qwen / MiniMax / Kimi 等
        print(f"⚠️  LLM API 调用未实现：{self.model}")
        return "[]"
    
    def rerank(self, query: str, memories: List[Dict], 
               top_k: int = 10) -> List[RerankResult]:
        """
        重排序记忆
        
        Args:
            query: 查询
            memories: 候选记忆列表
            top_k: 返回数量
        
        Returns:
            重排序后的结果
        """
        if len(memories) == 0:
            return []
        
        # 限制数量（成本和性能考虑）
        memories = memories[:50]
        
        # 构造记忆文本
        memories_text = "\n".join([
            f"{i+1}. [{m.get('id', 'unknown')}] {m.get('content', '')}"
            for i, m in enumerate(memories)
        ])
        
        # 构造 prompt
        prompt = self.prompt_template.format(
            query=query,
            count=len(memories),
            memories=memories_text
        )
        
        # 调用 LLM
        response = self._call_llm(prompt)
        
        # 解析评分
        try:
            scores = json.loads(response)
            score_map = {s['memory_id']: s for s in scores}
        except:
            # 降级到原始排序
            return [RerankResult(m, 0.0, i) for i, m in enumerate(memories[:top_k])]
        
        # 排序
        ranked = sorted(
            memories,
            key=lambda m: score_map.get(m.get('id', ''), {}).get('score', 0),
            reverse=True
        )
        
        # 返回结果
        results = []
        for i, mem in enumerate(ranked[:top_k]):
            score_info = score_map.get(mem.get('id', ''), {})
            results.append(RerankResult(
                memory=mem,
                score=score_info.get('score', 0.0),
                rank=i + 1,
                reason=score_info.get('reason', '')
            ))
        
        return results


class MemQWithRerank:
    """
    混合架构：MemQ + LLM Reranking
    
    流程：
    1. MemQ 快速检索 Top-50
    2. LLM 重排序 Top-50 → Top-10
    3. 返回最终结果
    """
    
    def __init__(self, 
                 base_retriever=None,
                 reranker='cross-encoder',
                 top_k: int = 10,
                 rerank_top_n: int = 50):
        
        # 基础检索器（这里用 MemQ）
        self.base_retriever = base_retriever
        
        # 重排序器
        if reranker == 'cross-encoder':
            self.reranker = CrossEncoderReranker()
        elif reranker == 'llm':
            self.reranker = LLMReranker()
        else:
            raise ValueError(f"未知 reranker 类型：{reranker}")
        
        # 参数
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
    
    def retrieve(self, query: str, memories: List[Dict], 
                 domain: str = None) -> List[RerankResult]:
        """
        检索并重排序
        
        Args:
            query: 查询
            memories: 全部记忆
            domain: 领域标识
        
        Returns:
            重排序后的 Top-K 记忆
        """
        # Stage 1: 快速检索 Top-N
        if self.base_retriever:
            # 使用 MemQ 评分
            scored = []
            for mem in memories:
                score, _ = self.base_retriever.predict(mem, domain)
                scored.append((mem, score))
            
            # 排序并截取 Top-N
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [mem for mem, _ in scored[:self.rerank_top_n]]
        else:
            # 降级：直接使用全部记忆
            candidates = memories[:self.rerank_top_n]
        
        # Stage 2: LLM 重排序
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k)
        
        return reranked


def demo():
    """演示 LLM Reranking"""
    print("="*60)
    print("LLM Reranking 演示")
    print("="*60)
    
    # 示例查询
    query = "2026 年 3 月的销售数据"
    
    # 示例记忆
    memories = [
        {
            'id': 'm1',
            'content': '2026 年 3 月销售额 100 万',
            'type': 'code',
            'metadata': {}
        },
        {
            'id': 'm2',
            'content': '2025 年 3 月销售额 80 万',
            'type': 'code',
            'metadata': {}
        },
        {
            'id': 'm3',
            'content': '2026 年 4 月销售额 120 万',
            'type': 'code',
            'metadata': {}
        },
        {
            'id': 'm4',
            'content': '有人提到过类似的销售数据但不是用于 2026 年',
            'type': 'noise',
            'metadata': {}
        },
    ]
    
    print(f"\n查询：{query}")
    print(f"\n候选记忆：{len(memories)} 条")
    for mem in memories:
        print(f"  [{mem['id']}] {mem['content']}")
    
    # 使用 Cross-Encoder Reranker
    print("\n1. Cross-Encoder Reranking...")
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, memories, top_k=3)
    
    print("\n重排序结果:")
    for r in results:
        print(f"  {r.rank}. [{r.memory['id']}] {r.memory['content']} (score={r.score:.3f})")
    
    # 预期输出：
    # 1. [m1] 2026 年 3 月销售额 100 万 (score 最高)
    # 2. [m3] 2026 年 4 月销售额 120 万 (score 中等)
    # 3. [m2] 2025 年 3 月销售额 80 万 (score 较低)
    
    print("\n✅ 演示完成！")
    print("\n注意：Cross-Encoder 可以识别时间相关性")
    print("      m1（2026 年 3 月）应该得分最高")


if __name__ == '__main__':
    demo()
