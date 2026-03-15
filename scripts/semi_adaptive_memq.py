#!/usr/bin/env python3
"""
半自适应 MemQ：保守的在线学习方案

核心原则：
1. 保留基础权重（固定）
2. 只做微调（±20% 范围内）
3. 引入人工审核闸口
4. 保持可解释性

适用场景：
- 需要一定适应性，但要可控
- 可以接受人工审核流程
- 需要平衡稳定性和灵活性
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class AdjustmentLog:
    """权重调整日志（用于审计）"""
    timestamp: str
    feature: str
    old_value: float
    new_value: float
    change_percent: float
    reason: str
    feedback_count: int
    requires_review: bool  # 超过 10% 变化需要人工审核


class SemiAdaptiveMemQ:
    """
    半自适应质量评分器
    
    设计原则：
    - 基础权重固定（保证稳定性）
    - 调整因子可学习（保证适应性）
    - 调整范围限制（防止漂移）
    - 完整日志记录（保证可解释性）
    """
    
    # 调整范围限制
    MIN_ADJUSTMENT = 0.8   # 最低调整到 80%
    MAX_ADJUSTMENT = 1.2   # 最高调整到 120%
    REVIEW_THRESHOLD = 0.1 # 超过 10% 变化需要审核
    
    def __init__(self, base_weights: Dict[str, float] = None):
        # 基础权重（固定，领域专家设定）
        self.base_weights = base_weights or {
            'type': 1.0,
            'length': 1.0,
            'entity': 1.0,
            'stopwords': 1.0,
            'template': 1.0,
            'metadata': 1.0,
        }
        
        # 调整因子（可学习，初始为 1.0）
        self.adjustment_factors = {k: 1.0 for k in self.base_weights}
        
        # 学习率（保守）
        self.learning_rate = 0.05
        
        # 反馈计数（用于统计显著性）
        self.feedback_counts = {k: 0 for k in self.base_weights}
        
        # 调整日志（用于审计）
        self.adjustment_log: List[AdjustmentLog] = []
        
        # 领域特定调整（可选）
        self.domain_adjustments: Dict[str, Dict[str, float]] = {}
    
    def get_effective_weights(self, domain: str = None) -> Dict[str, float]:
        """
        获取有效权重 = 基础权重 × 调整因子
        
        Returns:
            {feature: effective_weight, ...}
        """
        if domain and domain in self.domain_adjustments:
            adjustments = self.domain_adjustments[domain]
        else:
            adjustments = self.adjustment_factors
        
        return {
            k: v * adjustments.get(k, 1.0)
            for k, v in self.base_weights.items()
        }
    
    def predict(self, memory: Dict, domain: str = None) -> Tuple[float, Dict]:
        """
        预测质量分数（保持与原始 MemQ 兼容）
        
        Returns:
            (quality_score, feature_breakdown)
        """
        # 这里复用原始 MemQ 的特征提取逻辑
        features = self._extract_features(memory)
        weights = self.get_effective_weights(domain)
        
        # 乘积模型
        quality = 1.0
        weighted_features = {}
        
        for feature_name, feature_value in features.items():
            w = weights.get(feature_name, 1.0)
            weighted_features[feature_name] = w * feature_value
            quality *= weighted_features[feature_name]
        
        # 归一化
        quality = min(1.0, max(0.0, quality))
        
        return quality, weighted_features
    
    def update(self, memory: Dict, feedback: float, 
               domain: str = None, reason: str = "") -> Optional[AdjustmentLog]:
        """
        在线更新调整因子（保守更新）
        
        Args:
            memory: 记忆
            feedback: 用户反馈 (0.0-1.0)
            domain: 领域标识
            reason: 更新原因（用于审计）
        
        Returns:
            AdjustmentLog if change > threshold, else None
        """
        # 提取特征
        features = self._extract_features(memory)
        prediction, _ = self.predict(memory, domain)
        
        # 计算误差
        error = feedback - prediction
        
        # 只有统计显著时才更新（至少 10 个反馈）
        for feature_name in features:
            self.feedback_counts[feature_name] += 1
        
        if any(c < 10 for c in self.feedback_counts.values()):
            return None  # 数据不足，不更新
        
        # 计算建议调整（梯度下降）
        suggested_adjustments = {}
        for feature_name, feature_value in features.items():
            old_adj = self.adjustment_factors.get(feature_name, 1.0)
            
            # 梯度更新（保守学习率）
            gradient = -error * feature_value
            new_adj = old_adj - self.learning_rate * gradient
            
            # 限制调整范围（防止漂移）
            new_adj = max(self.MIN_ADJUSTMENT, min(self.MAX_ADJUSTMENT, new_adj))
            
            suggested_adjustments[feature_name] = new_adj
        
        # 应用调整
        if domain:
            if domain not in self.domain_adjustments:
                self.domain_adjustments[domain] = {}
            current = self.domain_adjustments[domain]
        else:
            current = self.adjustment_factors
        
        # 记录日志
        log_entry = None
        for feature_name, new_adj in suggested_adjustments.items():
            old_adj = current.get(feature_name, 1.0)
            change = (new_adj - old_adj) / old_adj if old_adj != 0 else 0
            
            # 只记录显著变化
            if abs(change) > 0.01:  # 1% 变化就记录
                log_entry = AdjustmentLog(
                    timestamp=datetime.now().isoformat(),
                    feature=feature_name,
                    old_value=old_adj,
                    new_value=new_adj,
                    change_percent=change * 100,
                    reason=reason or f"feedback: {feedback:.2f}, prediction: {prediction:.2f}",
                    feedback_count=self.feedback_counts[feature_name],
                    requires_review=abs(change) > self.REVIEW_THRESHOLD
                )
                self.adjustment_log.append(log_entry)
            
            current[feature_name] = new_adj
        
        return log_entry
    
    def get_pending_reviews(self) -> List[AdjustmentLog]:
        """获取需要人工审核的调整"""
        return [
            log for log in self.adjustment_log[-100:]  # 最近 100 条
            if log.requires_review
        ]
    
    def approve_adjustment(self, log: AdjustmentLog, approved: bool):
        """
        人工审核调整
        
        Args:
            log: 调整日志
            approved: 是否批准
        """
        if not approved:
            # 回滚调整
            feature = log.feature
            old_value = log.old_value
            
            if feature in self.adjustment_factors:
                self.adjustment_factors[feature] = old_value
            
            # 记录回滚
            log.reason += " [回滚：未通过审核]"
    
    def export_report(self) -> Dict:
        """导出可解释性报告"""
        return {
            'base_weights': self.base_weights,
            'adjustment_factors': self.adjustment_factors,
            'effective_weights': self.get_effective_weights(),
            'feedback_counts': self.feedback_counts,
            'recent_adjustments': [
                asdict(log) for log in self.adjustment_log[-20:]
            ],
            'pending_reviews': [
                asdict(log) for log in self.get_pending_reviews()
            ],
        }
    
    def save(self, path: str):
        """保存模型（包括日志）"""
        data = {
            'base_weights': self.base_weights,
            'adjustment_factors': self.adjustment_factors,
            'domain_adjustments': self.domain_adjustments,
            'feedback_counts': self.feedback_counts,
            'adjustment_log': [asdict(log) for log in self.adjustment_log],
            'learning_rate': self.learning_rate,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.base_weights = data['base_weights']
        self.adjustment_factors = data['adjustment_factors']
        self.domain_adjustments = data.get('domain_adjustments', {})
        self.feedback_counts = data.get('feedback_counts', {})
        self.adjustment_log = [
            AdjustmentLog(**log) for log in data.get('adjustment_log', [])
        ]
        self.learning_rate = data.get('learning_rate', 0.05)
    
    # ========== 内部方法 ==========
    
    def _extract_features(self, memory: Dict) -> Dict[str, float]:
        """提取特征（与原始 MemQ 相同）"""
        text = memory.get('content', '')
        metadata = memory.get('metadata', {})
        mem_type = memory.get('type', 'unknown')
        
        # 1. 类型特征
        type_scores = {
            'code': 1.2,
            'knowledge': 1.2,
            'event': 0.9,
            'conversation': 0.9,
            'noise': 0.3,
        }
        type_score = type_scores.get(mem_type, 1.0)
        
        # 2. 长度特征
        length = len(text)
        if length < 10:
            length_score = 0.5
        elif length < 20:
            length_score = 0.8
        elif length > 100:
            length_score = 1.1
        else:
            length_score = 1.0
        
        # 3. 实体特征
        entities = metadata.get('person', '') + metadata.get('project', '') + metadata.get('tech', '')
        entity_score = 1.2 if len(entities) > 0 else 0.8
        
        # 4. 停用词特征
        stopwords_ratio = sum(1 for c in text if c in '的了吗是和在了有') / max(len(text), 1)
        stopwords_score = 0.7 if stopwords_ratio > 0.5 else 1.0
        
        # 5. 模板特征
        noise_patterns = ['有人提到', '但不是用于', '类似的']
        template_score = 0.6 if any(p in text for p in noise_patterns) else 1.0
        
        # 6. 元数据特征
        metadata_score = 1.1 if metadata else 1.0
        
        return {
            'type': type_score,
            'length': length_score,
            'entity': entity_score,
            'stopwords': stopwords_score,
            'template': template_score,
            'metadata': metadata_score,
        }


def demo():
    """演示半自适应 MemQ"""
    print("="*60)
    print("半自适应 MemQ 演示")
    print("="*60)
    
    # 创建评分器
    memq = SemiAdaptiveMemQ()
    
    # 示例记忆
    memories = [
        {
            'id': 'm1',
            'content': '在 OpenClaw 项目中，我们讨论了 API 的实现方案',
            'type': 'code',
            'metadata': {'person': 'K', 'project': 'OpenClaw', 'tech': 'API'}
        },
        {
            'id': 'm2',
            'content': '有人提到过类似的 API 方案但不是用于 OpenClaw',
            'type': 'noise',
            'metadata': {}
        },
    ]
    
    # 初始预测
    print("\n1. 初始预测（基础权重）:")
    for mem in memories:
        score, _ = memq.predict(mem)
        print(f"   {mem['id']}: {score:.3f}")
    
    # 显示有效权重
    print("\n2. 初始有效权重:")
    for k, v in memq.get_effective_weights().items():
        print(f"   {k}: {v:.3f}")
    
    # 模拟反馈（需要至少 10 个才能触发更新）
    print("\n3. 模拟 20 个反馈...")
    for i in range(20):
        memq.update(memories[0], feedback=1.0, reason=f"模拟反馈 #{i+1}")
        memq.update(memories[1], feedback=0.0, reason=f"模拟反馈 #{i+1}")
    
    # 学习后预测
    print("\n4. 学习后预测（调整因子生效）:")
    for mem in memories:
        score, _ = memq.predict(mem)
        print(f"   {mem['id']}: {score:.3f}")
    
    # 显示调整后的权重
    print("\n5. 调整后有效权重:")
    for k, v in memq.get_effective_weights().items():
        base = memq.base_weights[k]
        adj = memq.adjustment_factors[k]
        print(f"   {k}: {v:.3f} (base={base:.3f}, adj={adj:.3f})")
    
    # 导出报告
    print("\n6. 可解释性报告:")
    report = memq.export_report()
    print(f"   基础权重：{report['base_weights']}")
    print(f"   调整因子：{report['adjustment_factors']}")
    print(f"   有效权重：{report['effective_weights']}")
    print(f"   反馈计数：{report['feedback_counts']}")
    print(f"   待审核调整：{len(report['pending_reviews'])} 条")
    
    print("\n✅ 演示完成！")


if __name__ == '__main__':
    demo()
