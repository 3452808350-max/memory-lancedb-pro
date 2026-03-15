#!/usr/bin/env python3
"""
自适应 MemQ：从固定权重 → 在线学习权重

核心思想：
- 保持乘积模型的可解释性
- 权重 w_i 从用户反馈中在线学习
- 根据领域/用户自动调整
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class AdaptiveMemQ:
    """
    自适应质量评分器
    
    传统 MemQ:
        quality = w1 × f1 × w2 × f2 × ... (固定权重)
    
    自适应 MemQ:
        quality = w1(t) × f1 × w2(t) × f2 × ... (时变权重)
        
        w_i(t+1) = w_i(t) + η × (feedback - prediction) × f_i
    """
    
    def __init__(self, initial_weights: Dict[str, float] = None):
        # 默认初始权重（可以领域特定）
        self.weights = initial_weights or {
            'type': 1.0,
            'length': 1.0,
            'entity': 1.0,
            'stopwords': 1.0,
            'template': 1.0,
            'metadata': 1.0,
        }
        
        # 学习率
        self.learning_rate = 0.1
        
        # 反馈历史
        self.feedback_history = []
        
        # 领域特定权重（可选）
        self.domain_weights = {}
    
    def extract_features(self, memory: Dict) -> Dict[str, float]:
        """提取特征（与传统 MemQ 相同）"""
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
        
        # 4. 停用词特征（简化版）
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
    
    def predict(self, memory: Dict, domain: str = None) -> Tuple[float, Dict]:
        """
        预测质量分数
        
        Returns:
            (quality_score, feature_breakdown)
        """
        features = self.extract_features(memory)
        
        # 使用领域特定权重（如果有）
        weights = self.domain_weights.get(domain, self.weights)
        
        # 自适应乘积模型
        quality = 1.0
        weighted_features = {}
        
        for feature_name, feature_value in features.items():
            w = weights.get(feature_name, 1.0)
            weighted_features[feature_name] = w * feature_value
            quality *= weighted_features[feature_name]
        
        # 归一化到 [0, 1]
        quality = min(1.0, max(0.0, quality))
        
        return quality, weighted_features
    
    def update(self, memory: Dict, feedback: float, domain: str = None):
        """
        在线更新权重
        
        Args:
            memory: 记忆
            feedback: 用户反馈 (0.0-1.0, 1.0=有用, 0.0=无用)
            domain: 领域标识
        """
        features = self.extract_features(memory)
        prediction, _ = self.predict(memory, domain)
        
        # 计算误差
        error = feedback - prediction
        
        # 更新权重（在线梯度下降）
        weights = self.domain_weights.get(domain, self.weights)
        
        for feature_name, feature_value in features.items():
            # 梯度：∂loss/∂w_i = -2 × error × f_i × (Π_{j≠i} w_j × f_j)
            # 简化：∂loss/∂w_i ≈ -error × f_i
            
            gradient = -error * feature_value
            
            # 更新权重
            w_old = weights.get(feature_name, 1.0)
            w_new = w_old - self.learning_rate * gradient
            
            # 约束权重在合理范围
            w_new = max(0.1, min(2.0, w_new))
            
            weights[feature_name] = w_new
        
        # 保存反馈历史
        self.feedback_history.append({
            'timestamp': datetime.now().isoformat(),
            'memory_id': memory.get('id'),
            'prediction': prediction,
            'feedback': feedback,
            'error': error,
            'domain': domain,
        })
        
        # 更新领域权重
        if domain:
            self.domain_weights[domain] = weights
    
    def batch_update(self, feedback_data: List[Dict], domain: str = None):
        """
        批量更新权重
        
        Args:
            feedback_data: [{memory_id, feedback, ...}, ...]
            domain: 领域标识
        """
        for item in feedback_data:
            memory = {'id': item['memory_id'], 'content': item.get('content', '')}
            feedback = item['feedback']
            self.update(memory, feedback, domain)
    
    def get_weights(self, domain: str = None) -> Dict[str, float]:
        """获取当前权重"""
        if domain:
            return self.domain_weights.get(domain, self.weights)
        return self.weights
    
    def save(self, path: str):
        """保存模型"""
        data = {
            'weights': self.weights,
            'domain_weights': self.domain_weights,
            'feedback_count': len(self.feedback_history),
            'learning_rate': self.learning_rate,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.weights = data['weights']
        self.domain_weights = data.get('domain_weights', {})
        self.learning_rate = data.get('learning_rate', 0.1)


def demo():
    """演示自适应 MemQ"""
    print("="*60)
    print("自适应 MemQ 演示")
    print("="*60)
    
    # 创建评分器
    memq = AdaptiveMemQ()
    
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
    print("\n1. 初始预测（固定权重）:")
    for mem in memories:
        score, _ = memq.predict(mem)
        print(f"   {mem['id']}: {score:.3f}")
    
    # 模拟用户反馈
    print("\n2. 模拟用户反馈:")
    print("   m1: 有用 (feedback=1.0)")
    print("   m2: 无用 (feedback=0.0)")
    
    # 在线学习
    memq.update(memories[0], feedback=1.0)  # m1 有用
    memq.update(memories[1], feedback=0.0)  # m2 无用
    
    # 学习后预测
    print("\n3. 学习后预测（自适应权重）:")
    for mem in memories:
        score, _ = memq.predict(mem)
        print(f"   {mem['id']}: {score:.3f}")
    
    # 显示学习到的权重
    print("\n4. 学习到的权重:")
    weights = memq.get_weights()
    for feature, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"   {feature}: {weight:.3f}")
    
    # 领域自适应演示
    print("\n5. 领域自适应演示:")
    
    # 医学领域反馈
    medical_mem = {
        'id': 'm3',
        'content': '患者可能有 X 症状，但不一定是 Y 病',
        'type': 'knowledge',
        'metadata': {'person': '医生', 'project': '诊断', 'tech': '症状'}
    }
    
    print("   医学领域：谨慎诊断是有用的 (feedback=1.0)")
    memq.update(medical_mem, feedback=1.0, domain='medical')
    
    print("   电商领域：免责声明是无用的 (feedback=0.0)")
    memq.update(medical_mem, feedback=0.0, domain='ecommerce')
    
    # 比较不同领域的权重
    print("\n6. 不同领域的权重对比:")
    print(f"   通用：type={memq.weights['type']:.3f}")
    print(f"   医学：type={memq.domain_weights.get('medical', {}).get('type', 'N/A'):.3f}")
    print(f"   电商：type={memq.domain_weights.get('ecommerce', {}).get('type', 'N/A'):.3f}")
    
    print("\n✅ 演示完成！")


if __name__ == '__main__':
    demo()
