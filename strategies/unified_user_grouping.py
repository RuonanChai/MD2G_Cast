# unified_user_grouping.py
"""
统一用户分组机制
确保所有策略使用相同的分组逻辑，保证公平比较
"""
import numpy as np
import json
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import math

class UnifiedUserGrouping:
    """
    统一用户分组机制
    为所有策略提供一致的用户分组逻辑
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        
        # 分组参数
        self.min_group_size = 2
        self.max_groups = 8
        self.similarity_threshold = 0.6
        
        # 特征权重
        self.feature_weights = {
            'throughput': 0.25,
            'delay': 0.20,
            'device_score': 0.20,
            'network_stability': 0.15,
            'user_preference': 0.10,
            'bandwidth_variance': 0.05,
            'latency_jitter': 0.05
        }
        
        # 历史数据
        self.grouping_history = deque(maxlen=100)
        self.performance_metrics = deque(maxlen=50)
        
        # 特征标准化
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_user_features(self, user_data: Dict) -> np.ndarray:
        """提取用户特征向量"""
        features = [
            user_data.get('throughput', 0.0),
            user_data.get('delay', 0.0),
            user_data.get('device_score', 0.5),
            user_data.get('network_stability', 1.0),
            user_data.get('user_preference', 0.5),
            user_data.get('bandwidth_variance', 0.0),
            user_data.get('latency_jitter', 0.0)
        ]
        return np.array(features, dtype=np.float32)
    
    def _calculate_user_similarity(self, user1_data: Dict, user2_data: Dict) -> float:
        """计算用户相似度"""
        features1 = self._extract_user_features(user1_data)
        features2 = self._extract_user_features(user2_data)
        
        # 加权欧几里得距离
        weights = np.array(list(self.feature_weights.values()))
        weighted_diff = np.sqrt(np.sum(weights * (features1 - features2) ** 2))
        
        # 转换为相似度
        similarity = 1.0 / (1.0 + weighted_diff)
        return max(0.0, min(1.0, similarity))
    
    def _determine_optimal_groups(self, user_features: np.ndarray, user_ids: List[int]) -> int:
        """确定最优分组数"""
        if len(user_features) <= 2:
            return 1
        
        max_groups = min(self.max_groups, len(user_features) // self.min_group_size)
        if max_groups < 2:
            return 1
        
        best_k = 2
        best_score = -1
        
        for k in range(2, max_groups + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(user_features)
                
                if len(set(labels)) > 1:
                    # 轮廓系数
                    silhouette = silhouette_score(user_features, labels)
                    
                    # 组内紧密度
                    intra_cluster_distances = []
                    for cluster_id in range(k):
                        cluster_points = user_features[labels == cluster_id]
                        if len(cluster_points) > 1:
                            center = np.mean(cluster_points, axis=0)
                            distances = np.linalg.norm(cluster_points - center, axis=1)
                            intra_cluster_distances.extend(distances)
                    
                    intra_cluster_avg = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
                    
                    # 综合评分
                    score = 0.7 * silhouette + 0.3 * (1.0 / (1.0 + intra_cluster_avg))
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        return best_k
    
    def _adaptive_grouping(self, active_clients: Dict) -> Dict[int, List[int]]:
        """自适应分组算法"""
        if len(active_clients) < self.min_group_size:
            return {0: list(active_clients.keys())}
        
        # 准备特征数据
        user_ids = list(active_clients.keys())
        user_features = np.array([self._extract_user_features(active_clients[uid]) for uid in user_ids])
        
        # 标准化特征
        if not self.is_fitted:
            user_features_scaled = self.scaler.fit_transform(user_features)
            self.is_fitted = True
        else:
            user_features_scaled = self.scaler.transform(user_features)
        
        # 确定最优分组数
        optimal_k = self._determine_optimal_groups(user_features_scaled, user_ids)
        
        # 执行K-means分组
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        group_labels = kmeans.fit_predict(user_features_scaled)
        
        # 组织分组结果
        groups = defaultdict(list)
        for i, label in enumerate(group_labels):
            groups[label].append(user_ids[i])
        
        return dict(groups)
    
    def _calculate_group_metrics(self, groups: Dict[int, List[int]], active_clients: Dict) -> Dict:
        """计算分组指标"""
        metrics = {
            'num_groups': len(groups),
            'group_sizes': [len(group) for group in groups.values()],
            'group_diversity': [],
            'group_efficiency': [],
            'overall_balance': 0.0
        }
        
        for group_id, user_ids in groups.items():
            if not user_ids:
                continue
            
            # 组内多样性
            group_features = [self._extract_user_features(active_clients[uid]) for uid in user_ids if uid in active_clients]
            if group_features:
                group_features_array = np.array(group_features)
                diversity = np.std(group_features_array, axis=0).mean()
                metrics['group_diversity'].append(diversity)
                
                # 组内效率
                throughputs = [active_clients[uid].get('throughput', 0) for uid in user_ids if uid in active_clients]
                delays = [active_clients[uid].get('delay', 0) for uid in user_ids if uid in active_clients]
                
                if throughputs and delays:
                    efficiency = np.mean(throughputs) / max(1.0, np.mean(delays) / 100.0)
                    metrics['group_efficiency'].append(efficiency)
        
        # 计算整体平衡性
        if metrics['group_sizes']:
            size_variance = np.var(metrics['group_sizes'])
            size_mean = np.mean(metrics['group_sizes'])
            metrics['overall_balance'] = 1.0 / (1.0 + size_variance / max(1.0, size_mean))
        
        return metrics
    
    def group_users(self, active_clients: Dict) -> Tuple[Dict[int, List[int]], Dict]:
        """执行统一用户分组"""
        if not active_clients:
            return {}, {}
        
        # 执行自适应分组
        groups = self._adaptive_grouping(active_clients)
        
        # 计算分组指标
        metrics = self._calculate_group_metrics(groups, active_clients)
        
        # 记录分组历史
        self.grouping_history.append({
            'timestamp': time.time(),
            'num_users': len(active_clients),
            'num_groups': len(groups),
            'group_sizes': metrics['group_sizes'],
            'overall_balance': metrics['overall_balance']
        })
        
        return groups, metrics
    
    def get_grouping_statistics(self) -> Dict:
        """获取分组统计信息"""
        if not self.grouping_history:
            return {}
        
        recent_groupings = list(self.grouping_history)[-10:]  # 最近10次分组
        
        stats = {
            'total_groupings': len(self.grouping_history),
            'avg_groups_per_grouping': np.mean([g['num_groups'] for g in recent_groupings]),
            'avg_balance': np.mean([g['overall_balance'] for g in recent_groupings]),
            'grouping_consistency': 1.0 - np.std([g['num_groups'] for g in recent_groupings]) / max(1.0, np.mean([g['num_groups'] for g in recent_groupings]))
        }
        
        return stats

# ================================= [策略适配器] =================================
class StrategyAdapter:
    """
    策略适配器，为不同策略提供统一的分组接口
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.grouping = UnifiedUserGrouping(max_users)
        
    def get_user_groups(self, active_clients: Dict) -> Dict[int, List[int]]:
        """获取用户分组"""
        groups, _ = self.grouping.group_users(active_clients)
        return groups
    
    def get_group_features(self, group_id: int, user_ids: List[int], active_clients: Dict) -> Dict:
        """获取组特征"""
        if not user_ids:
            return {}
        
        group_data = [active_clients[uid] for uid in user_ids if uid in active_clients]
        if not group_data:
            return {}
        
        # 计算组统计特征
        throughputs = [user.get('throughput', 0) for user in group_data]
        delays = [user.get('delay', 0) for user in group_data]
        device_scores = [user.get('device_score', 0.5) for user in group_data]
        
        features = {
            'group_id': group_id,
            'size': len(user_ids),
            'avg_throughput': np.mean(throughputs),
            'min_throughput': min(throughputs),
            'max_throughput': max(throughputs),
            'avg_delay': np.mean(delays),
            'max_delay': max(delays),
            'avg_device_score': np.mean(device_scores),
            'min_device_score': min(device_scores),
            'throughput_variance': np.var(throughputs),
            'delay_variance': np.var(delays),
            'device_score_variance': np.var(device_scores)
        }
        
        return features
    
    def get_grouping_recommendations(self, groups: Dict[int, List[int]], active_clients: Dict) -> Dict:
        """获取分组建议"""
        recommendations = {}
        
        for group_id, user_ids in groups.items():
            group_features = self.get_group_features(group_id, user_ids, active_clients)
            
            # 基于组特征生成建议
            recommendations[group_id] = {
                'recommended_strategy': self._recommend_strategy(group_features),
                'priority_level': self._calculate_priority(group_features),
                'optimization_potential': self._calculate_optimization_potential(group_features)
            }
        
        return recommendations
    
    def _recommend_strategy(self, group_features: Dict) -> str:
        """推荐策略"""
        if group_features['avg_throughput'] > 5.0 and group_features['avg_delay'] < 100:
            return 'enhanced_aggressive'
        elif group_features['avg_throughput'] < 2.0 or group_features['avg_delay'] > 200:
            return 'conservative'
        else:
            return 'balanced'
    
    def _calculate_priority(self, group_features: Dict) -> float:
        """计算优先级"""
        size_weight = min(1.0, group_features['size'] / 10.0)
        performance_weight = min(1.0, group_features['avg_throughput'] / 10.0)
        stability_weight = 1.0 - (group_features['throughput_variance'] / 100.0)
        
        priority = 0.4 * size_weight + 0.4 * performance_weight + 0.2 * stability_weight
        return max(0.0, min(1.0, priority))
    
    def _calculate_optimization_potential(self, group_features: Dict) -> float:
        """计算优化潜力"""
        # 基于组内差异计算优化潜力
        throughput_potential = group_features['throughput_variance'] / max(1.0, group_features['avg_throughput'])
        delay_potential = group_features['delay_variance'] / max(1.0, group_features['avg_delay'])
        device_potential = group_features['device_score_variance']
        
        potential = (throughput_potential + delay_potential + device_potential) / 3.0
        return max(0.0, min(1.0, potential))

# ================================= [全局分组管理器] =================================
class GlobalGroupingManager:
    """
    全局分组管理器，确保所有策略使用相同的分组
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.adapter = StrategyAdapter(max_users)
        self.current_groups = {}
        self.current_metrics = {}
        self.last_update_time = 0
        self.update_interval = 5.0  # 5秒更新一次分组
        
    def get_current_groups(self, active_clients: Dict) -> Dict[int, List[int]]:
        """获取当前分组"""
        current_time = time.time()
        
        # 检查是否需要更新分组
        if (current_time - self.last_update_time > self.update_interval or 
            not self.current_groups or 
            len(active_clients) != len(self.current_groups.get('all_users', []))):
            
            # 更新分组
            self.current_groups, self.current_metrics = self.adapter.grouping.group_users(active_clients)
            self.current_groups['all_users'] = list(active_clients.keys())
            self.last_update_time = current_time
        
        return self.current_groups
    
    def get_grouping_info(self) -> Dict:
        """获取分组信息"""
        return {
            'current_groups': self.current_groups,
            'metrics': self.current_metrics,
            'statistics': self.adapter.grouping.get_grouping_statistics(),
            'last_update': self.last_update_time
        }

# ================================= [导出接口] =================================
def create_unified_grouping_manager(max_users: int) -> GlobalGroupingManager:
    """创建统一分组管理器"""
    return GlobalGroupingManager(max_users)

def get_user_groups(manager: GlobalGroupingManager, active_clients: Dict) -> Dict[int, List[int]]:
    """获取用户分组"""
    return manager.get_current_groups(active_clients)

def get_grouping_statistics(manager: GlobalGroupingManager) -> Dict:
    """获取分组统计"""
    return manager.get_grouping_info()
