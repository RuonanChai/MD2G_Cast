# complex_optimization_framework.py
"""
复杂优化框架
为所有策略添加复杂的优化逻辑，增加决策挑战性
"""
import numpy as np
import json
import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ================================= [多目标优化器] =================================
class MultiObjectiveOptimizer:
    """
    多目标优化器
    平衡QoE、延迟、带宽利用率、负载均衡等多个目标
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        
        # 优化目标权重
        self.objective_weights = {
            'qoe': 0.35,
            'latency': 0.25,
            'bandwidth_efficiency': 0.20,
            'load_balance': 0.20
        }
        
        # 约束条件
        self.constraints = {
            'min_qoe': 0.3,
            'max_latency': 200.0,
            'min_bandwidth_utilization': 0.1,
            'max_bandwidth_utilization': 0.9
        }
        
        # 历史数据
        self.optimization_history = deque(maxlen=100)
        self.performance_trends = deque(maxlen=50)
        
    def _calculate_objective_function(self, decisions: List[int], user_groups: Dict, 
                                   active_clients: Dict) -> float:
        """计算目标函数值"""
        if not decisions or not user_groups:
            return 0.0
        
        # 计算各项目标
        qoe_score = self._calculate_qoe_score(decisions, user_groups, active_clients)
        latency_score = self._calculate_latency_score(decisions, user_groups, active_clients)
        bandwidth_score = self._calculate_bandwidth_score(decisions, user_groups, active_clients)
        load_balance_score = self._calculate_load_balance_score(decisions, user_groups, active_clients)
        
        # 加权组合
        total_score = (
            self.objective_weights['qoe'] * qoe_score +
            self.objective_weights['latency'] * latency_score +
            self.objective_weights['bandwidth_efficiency'] * bandwidth_score +
            self.objective_weights['load_balance'] * load_balance_score
        )
        
        return total_score
    
    def _calculate_qoe_score(self, decisions: List[int], user_groups: Dict, active_clients: Dict) -> float:
        """计算QoE得分"""
        if not decisions:
            return 0.0
        
        qoe_scores = []
        for i, decision in enumerate(decisions):
            user_id = i + 1
            if user_id in active_clients:
                user_data = active_clients[user_id]
                base_qoe = user_data.get('device_score', 0.5)
                
                # 决策影响
                if decision == 1:  # 增强层
                    enhanced_qoe = base_qoe * 1.2
                else:  # 基础层
                    enhanced_qoe = base_qoe * 0.8
                
                qoe_scores.append(enhanced_qoe)
        
        return np.mean(qoe_scores) if qoe_scores else 0.0
    
    def _calculate_latency_score(self, decisions: List[int], user_groups: Dict, active_clients: Dict) -> float:
        """计算延迟得分"""
        if not decisions:
            return 0.0
        
        latency_scores = []
        for i, decision in enumerate(decisions):
            user_id = i + 1
            if user_id in active_clients:
                user_data = active_clients[user_id]
                base_delay = user_data.get('delay', 100.0)
                
                # 决策影响
                if decision == 1:  # 增强层
                    enhanced_delay = base_delay * 0.8  # 减少延迟
                else:  # 基础层
                    enhanced_delay = base_delay * 1.1  # 增加延迟
                
                # 延迟得分（延迟越低得分越高）
                latency_score = 1.0 / (1.0 + enhanced_delay / 100.0)
                latency_scores.append(latency_score)
        
        return np.mean(latency_scores) if latency_scores else 0.0
    
    def _calculate_bandwidth_score(self, decisions: List[int], user_groups: Dict, active_clients: Dict) -> float:
        """计算带宽效率得分"""
        if not decisions:
            return 0.0
        
        total_bandwidth = sum(active_clients[uid].get('throughput', 0) for uid in active_clients)
        enhanced_bandwidth = sum(active_clients[i+1].get('throughput', 0) for i, decision in enumerate(decisions) if decision == 1)
        
        if total_bandwidth > 0:
            bandwidth_utilization = enhanced_bandwidth / total_bandwidth
            # 带宽利用率在0.3-0.7之间得分最高
            if 0.3 <= bandwidth_utilization <= 0.7:
                efficiency_score = 1.0
            else:
                efficiency_score = 1.0 - abs(bandwidth_utilization - 0.5) * 2
        else:
            efficiency_score = 0.0
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _calculate_load_balance_score(self, decisions: List[int], user_groups: Dict, active_clients: Dict) -> float:
        """计算负载均衡得分"""
        if not user_groups:
            return 0.0
        
        group_loads = []
        for group_id, user_ids in user_groups.items():
            if not user_ids:
                continue
            
            # 计算组内负载
            group_throughput = sum(active_clients[uid].get('throughput', 0) for uid in user_ids if uid in active_clients)
            group_enhanced = sum(1 for uid in user_ids if uid in active_clients and decisions[uid-1] == 1)
            
            group_load = group_throughput * (1.0 + group_enhanced * 0.5)
            group_loads.append(group_load)
        
        if not group_loads:
            return 0.0
        
        # 计算Jain公平指数
        n = len(group_loads)
        sum_x = sum(group_loads)
        sum_x_squared = sum(x*x for x in group_loads)
        
        if n > 0 and sum_x_squared > 0:
            jfi = (sum_x * sum_x) / (n * sum_x_squared)
        else:
            jfi = 0.0
        
        return max(0.0, min(1.0, jfi))
    
    def _check_constraints(self, decisions: List[int], user_groups: Dict, active_clients: Dict) -> bool:
        """检查约束条件"""
        # QoE约束
        qoe_score = self._calculate_qoe_score(decisions, user_groups, active_clients)
        if qoe_score < self.constraints['min_qoe']:
            return False
        
        # 延迟约束
        latency_score = self._calculate_latency_score(decisions, user_groups, active_clients)
        if latency_score < (1.0 / (1.0 + self.constraints['max_latency'] / 100.0)):
            return False
        
        # 带宽利用率约束
        bandwidth_score = self._calculate_bandwidth_score(decisions, user_groups, active_clients)
        if bandwidth_score < self.constraints['min_bandwidth_utilization']:
            return False
        
        return True
    
    def optimize_decisions(self, user_groups: Dict, active_clients: Dict) -> List[int]:
        """优化决策"""
        if not user_groups or not active_clients:
            return [0] * self.max_users
        
        # 初始化决策
        best_decisions = [0] * self.max_users
        best_score = 0.0
        
        # 使用遗传算法优化
        population_size = 20
        generations = 10
        
        for generation in range(generations):
            # 生成候选解
            candidates = self._generate_candidates(population_size, user_groups, active_clients)
            
            # 评估候选解
            for candidate in candidates:
                if self._check_constraints(candidate, user_groups, active_clients):
                    score = self._calculate_objective_function(candidate, user_groups, active_clients)
                    if score > best_score:
                        best_score = score
                        best_decisions = candidate.copy()
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'best_score': best_score,
            'num_groups': len(user_groups),
            'constraints_satisfied': self._check_constraints(best_decisions, user_groups, active_clients)
        })
        
        return best_decisions
    
    def _generate_candidates(self, population_size: int, user_groups: Dict, active_clients: Dict) -> List[List[int]]:
        """生成候选解"""
        candidates = []
        
        for _ in range(population_size):
            candidate = [0] * self.max_users
            
            # 基于组特征生成决策
            for group_id, user_ids in user_groups.items():
                if not user_ids:
                    continue
                
                # 计算组特征
                group_throughput = np.mean([active_clients[uid].get('throughput', 0) for uid in user_ids if uid in active_clients])
                group_delay = np.mean([active_clients[uid].get('delay', 0) for uid in user_ids if uid in active_clients])
                group_device_score = np.mean([active_clients[uid].get('device_score', 0.5) for uid in user_ids if uid in active_clients])
                
                # 基于特征决定是否启用增强层
                enhanced_prob = 0.5
                if group_throughput > 5.0 and group_delay < 100:
                    enhanced_prob = 0.8
                elif group_throughput < 2.0 or group_delay > 200:
                    enhanced_prob = 0.2
                
                enhanced_prob *= group_device_score
                
                # 为组内用户应用决策
                for user_id in user_ids:
                    if 1 <= user_id <= self.max_users:
                        candidate[user_id - 1] = 1 if np.random.random() < enhanced_prob else 0
            
            candidates.append(candidate)
        
        return candidates

# ================================= [自适应学习器] =================================
class AdaptiveLearner:
    """
    自适应学习器
    根据历史性能动态调整策略参数
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        
        # 学习参数
        self.learning_rate = 0.1
        self.memory_size = 50
        self.adaptation_threshold = 0.1
        
        # 历史数据
        self.performance_history = deque(maxlen=self.memory_size)
        self.parameter_history = deque(maxlen=self.memory_size)
        
        # 当前参数
        self.current_parameters = {
            'qoe_weight': 0.35,
            'latency_weight': 0.25,
            'bandwidth_weight': 0.20,
            'balance_weight': 0.20,
            'enhancement_threshold': 0.5,
            'conservation_factor': 0.8
        }
        
    def update_performance(self, performance_metrics: Dict):
        """更新性能指标"""
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': performance_metrics.copy(),
            'parameters': self.current_parameters.copy()
        })
        
        # 检查是否需要调整参数
        if len(self.performance_history) >= 10:
            self._adapt_parameters()
    
    def _adapt_parameters(self):
        """自适应调整参数"""
        if len(self.performance_history) < 10:
            return
        
        # 分析性能趋势
        recent_performance = list(self.performance_history)[-10:]
        older_performance = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else recent_performance
        
        # 计算性能变化
        recent_avg = np.mean([p['metrics'].get('total_score', 0) for p in recent_performance])
        older_avg = np.mean([p['metrics'].get('total_score', 0) for p in older_performance])
        
        performance_change = recent_avg - older_avg
        
        # 根据性能变化调整参数
        if performance_change > self.adaptation_threshold:
            # 性能提升，保持当前参数
            pass
        elif performance_change < -self.adaptation_threshold:
            # 性能下降，调整参数
            self._adjust_parameters(recent_performance)
    
    def _adjust_parameters(self, recent_performance: List[Dict]):
        """调整参数"""
        # 分析各指标的表现
        qoe_scores = [p['metrics'].get('qoe_score', 0) for p in recent_performance]
        latency_scores = [p['metrics'].get('latency_score', 0) for p in recent_performance]
        bandwidth_scores = [p['metrics'].get('bandwidth_score', 0) for p in recent_performance]
        balance_scores = [p['metrics'].get('balance_score', 0) for p in recent_performance]
        
        # 找出表现最差的指标
        avg_scores = {
            'qoe': np.mean(qoe_scores),
            'latency': np.mean(latency_scores),
            'bandwidth': np.mean(bandwidth_scores),
            'balance': np.mean(balance_scores)
        }
        
        worst_metric = min(avg_scores, key=avg_scores.get)
        
        # 调整权重
        if worst_metric == 'qoe':
            self.current_parameters['qoe_weight'] = min(0.5, self.current_parameters['qoe_weight'] + 0.05)
        elif worst_metric == 'latency':
            self.current_parameters['latency_weight'] = min(0.5, self.current_parameters['latency_weight'] + 0.05)
        elif worst_metric == 'bandwidth':
            self.current_parameters['bandwidth_weight'] = min(0.5, self.current_parameters['bandwidth_weight'] + 0.05)
        elif worst_metric == 'balance':
            self.current_parameters['balance_weight'] = min(0.5, self.current_parameters['balance_weight'] + 0.05)
        
        # 归一化权重
        total_weight = sum([self.current_parameters[k] for k in ['qoe_weight', 'latency_weight', 'bandwidth_weight', 'balance_weight']])
        for key in ['qoe_weight', 'latency_weight', 'bandwidth_weight', 'balance_weight']:
            self.current_parameters[key] /= total_weight
    
    def get_adapted_parameters(self) -> Dict:
        """获取调整后的参数"""
        return self.current_parameters.copy()

# ================================= [复杂优化框架] =================================
class ComplexOptimizationFramework:
    """
    复杂优化框架
    整合多目标优化和自适应学习
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.optimizer = MultiObjectiveOptimizer(max_users)
        self.learner = AdaptiveLearner(max_users)
        
        # 框架状态
        self.optimization_count = 0
        self.performance_trends = deque(maxlen=20)
        
    def optimize_strategy(self, user_groups: Dict, active_clients: Dict) -> Tuple[List[int], Dict]:
        """优化策略"""
        # 获取自适应参数
        adapted_params = self.learner.get_adapted_parameters()
        
        # 更新优化器参数
        self.optimizer.objective_weights = {
            'qoe': adapted_params['qoe_weight'],
            'latency': adapted_params['latency_weight'],
            'bandwidth_efficiency': adapted_params['bandwidth_weight'],
            'load_balance': adapted_params['balance_weight']
        }
        
        # 执行优化
        decisions = self.optimizer.optimize_decisions(user_groups, active_clients)
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(decisions, user_groups, active_clients)
        
        # 更新学习器
        self.learner.update_performance(performance_metrics)
        
        # 记录框架状态
        self.optimization_count += 1
        self.performance_trends.append(performance_metrics['total_score'])
        
        return decisions, performance_metrics
    
    def _calculate_performance_metrics(self, decisions: List[int], user_groups: Dict, active_clients: Dict) -> Dict:
        """计算性能指标"""
        metrics = {
            'total_score': self.optimizer._calculate_objective_function(decisions, user_groups, active_clients),
            'qoe_score': self.optimizer._calculate_qoe_score(decisions, user_groups, active_clients),
            'latency_score': self.optimizer._calculate_latency_score(decisions, user_groups, active_clients),
            'bandwidth_score': self.optimizer._calculate_bandwidth_score(decisions, user_groups, active_clients),
            'balance_score': self.optimizer._calculate_load_balance_score(decisions, user_groups, active_clients),
            'constraints_satisfied': self.optimizer._check_constraints(decisions, user_groups, active_clients),
            'optimization_count': self.optimization_count
        }
        
        return metrics
    
    def get_framework_status(self) -> Dict:
        """获取框架状态"""
        return {
            'optimization_count': self.optimization_count,
            'performance_trend': list(self.performance_trends),
            'current_parameters': self.learner.get_adapted_parameters(),
            'optimization_history': list(self.optimizer.optimization_history)
        }

# ================================= [策略增强器] =================================
class StrategyEnhancer:
    """
    策略增强器
    为现有策略添加复杂优化逻辑
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.framework = ComplexOptimizationFramework(max_users)
        
    def enhance_rolling_strategy(self, base_decisions: List[int], user_groups: Dict, active_clients: Dict) -> List[int]:
        """增强Rolling策略"""
        # 使用复杂优化框架优化基础决策
        enhanced_decisions, metrics = self.framework.optimize_strategy(user_groups, active_clients)
        
        # 结合基础决策和优化决策
        final_decisions = []
        for i in range(len(base_decisions)):
            if i < len(enhanced_decisions):
                # 使用优化决策，但考虑基础决策的置信度
                base_confidence = 0.7  # 基础策略置信度
                enhanced_confidence = 0.3  # 优化策略置信度
                
                if np.random.random() < base_confidence:
                    final_decisions.append(base_decisions[i])
                else:
                    final_decisions.append(enhanced_decisions[i])
            else:
                final_decisions.append(base_decisions[i])
        
        return final_decisions
    
    def enhance_heuristic_strategy(self, base_decisions: List[int], user_groups: Dict, active_clients: Dict) -> List[int]:
        """增强Heuristic策略"""
        # 基于组特征进行更精细的决策
        enhanced_decisions = [0] * self.max_users
        
        for group_id, user_ids in user_groups.items():
            if not user_ids:
                continue
            
            # 计算组特征
            group_throughput = np.mean([active_clients[uid].get('throughput', 0) for uid in user_ids if uid in active_clients])
            group_delay = np.mean([active_clients[uid].get('delay', 0) for uid in user_ids if uid in active_clients])
            group_device_score = np.mean([active_clients[uid].get('device_score', 0.5) for uid in user_ids if uid in active_clients])
            
            # 复杂决策逻辑
            decision_prob = 0.5
            
            # 网络条件影响
            if group_throughput > 8.0 and group_delay < 50:
                decision_prob = 0.9
            elif group_throughput < 1.0 or group_delay > 300:
                decision_prob = 0.1
            else:
                # 线性插值
                throughput_factor = min(1.0, group_throughput / 5.0)
                delay_factor = max(0.0, 1.0 - group_delay / 200.0)
                decision_prob = 0.3 + 0.4 * throughput_factor + 0.3 * delay_factor
            
            # 设备评分影响
            decision_prob *= group_device_score
            
            # 为组内用户应用决策
            for user_id in user_ids:
                if 1 <= user_id <= self.max_users:
                    enhanced_decisions[user_id - 1] = 1 if np.random.random() < decision_prob else 0
        
        return enhanced_decisions
    
    def enhance_clustering_strategy(self, base_decisions: List[int], user_groups: Dict, active_clients: Dict) -> List[int]:
        """增强Clustering策略"""
        # 基于聚类特征进行决策
        enhanced_decisions = [0] * self.max_users
        
        for group_id, user_ids in user_groups.items():
            if not user_ids:
                continue
            
            # 计算聚类特征
            group_features = []
            for user_id in user_ids:
                if user_id in active_clients:
                    user_data = active_clients[user_id]
                    features = [
                        user_data.get('throughput', 0.0),
                        user_data.get('delay', 0.0),
                        user_data.get('device_score', 0.5),
                        user_data.get('network_stability', 1.0)
                    ]
                    group_features.append(features)
            
            if not group_features:
                continue
            
            # 使用聚类特征进行决策
            group_features_array = np.array(group_features)
            avg_features = np.mean(group_features_array, axis=0)
            
            # 基于特征计算决策概率
            throughput_factor = min(1.0, avg_features[0] / 5.0)
            delay_factor = max(0.0, 1.0 - avg_features[1] / 200.0)
            device_factor = avg_features[2]
            stability_factor = avg_features[3]
            
            decision_prob = 0.3 * throughput_factor + 0.2 * delay_factor + 0.3 * device_factor + 0.2 * stability_factor
            
            # 为组内用户应用决策
            for user_id in user_ids:
                if 1 <= user_id <= self.max_users:
                    enhanced_decisions[user_id - 1] = 1 if np.random.random() < decision_prob else 0
        
        return enhanced_decisions

# ================================= [导出接口] =================================
def create_complex_optimization_framework(max_users: int) -> ComplexOptimizationFramework:
    """创建复杂优化框架"""
    return ComplexOptimizationFramework(max_users)

def create_strategy_enhancer(max_users: int) -> StrategyEnhancer:
    """创建策略增强器"""
    return StrategyEnhancer(max_users)
