# predictive_controller_v2.py
import argparse
import os
import json
import time
import traceback
import glob
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================================= [深度学习模型] =================================
class QoEPredictor(nn.Module):
    """
    QoE预测网络，用于预测用户QoE
    """
    def __init__(self, input_dim=8, hidden_dim=256):
        super(QoEPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class MulticastOptimizer(nn.Module):
    """
    多播优化网络，用于优化多播策略
    """
    def __init__(self, input_dim=10, hidden_dim=512):
        super(MulticastOptimizer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 多播决策头
        self.multicast_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 延迟优化头
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        multicast_output = self.multicast_head(encoded)
        latency_output = self.latency_head(encoded)
        return multicast_output, latency_output

# ================================= [QoE感知聚类算法] =================================
class QoEAwareClustering:
    """
    实现论文中的QoE感知聚类算法
    "Taming the latency in multi-user VR 360°: A QoE-aware deep learning-aided multicast framework"
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.qoe_predictor = QoEPredictor()
        self.multicast_optimizer = MulticastOptimizer()
        
        # 聚类参数 - 轻微降敏调整
        self.min_cluster_size = 2
        self.max_clusters = 5
        self.similarity_threshold = 0.7
        self.window_size = 5  # 原为10，缩短预测窗口
        self.stability_alpha = 0.7  # EMA平滑系数
        
        # 历史数据
        self.clustering_history = deque(maxlen=100)
        self.qoe_history = deque(maxlen=50)
        self.performance_metrics = deque(maxlen=20)
        
        # 特征标准化
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_user_features(self, user_data: Dict) -> np.ndarray:
        """提取用户特征向量"""
        features = [
            user_data.get('throughput', 0.0),
            user_data.get('delay', 0.0),
            user_data.get('device_score', 0.5),
            user_data.get('viewpoint', 0.0) if isinstance(user_data.get('viewpoint'), (int, float)) else 0.0,
            user_data.get('network_stability', 1.0),
            user_data.get('user_preference', 0.5),
            user_data.get('bandwidth_variance', 0.0),
            user_data.get('latency_jitter', 0.0)
        ]
        return np.array(features, dtype=np.float32)
    
    def _predict_user_qoe(self, user_features: np.ndarray) -> float:
        """预测用户QoE"""
        # ⚠️ 已改为简单启发式公式，不使用随机初始化的深度学习模型
        # 原论文使用训练好的DL模型，但为了公平对比，这里使用启发式公式
        # 这样更符合"baseline"定位，不依赖训练数据质量
        
        # 提取关键特征
        throughput = user_features[0] if len(user_features) > 0 else 0.0
        delay = user_features[1] if len(user_features) > 1 else 0.0
        device_score = user_features[2] if len(user_features) > 2 else 0.5
        
        # 简单QoE预测公式（与MD2G的QoE公式类似，但更简单）
        # QoE = device_score * (1 - delay_penalty) * bandwidth_factor
        delay_penalty = min(1.0, delay / 200.0)  # 延迟惩罚，200ms为阈值
        bandwidth_factor = min(1.0, throughput / 10.0)  # 带宽因子，10Mbps为阈值
        
        qoe_prediction = device_score * (1.0 - delay_penalty * 0.3) * bandwidth_factor
        return max(0.0, min(1.0, qoe_prediction))
        
        # 原代码（已禁用）：
        # with torch.no_grad():
        #     features_tensor = torch.from_numpy(user_features).float().unsqueeze(0)
        #     qoe_prediction = self.qoe_predictor(features_tensor).item()
        # return qoe_prediction
    
    def _calculate_qoe_similarity(self, user1_features: np.ndarray, user2_features: np.ndarray) -> float:
        """计算基于QoE的用户相似度"""
        # 预测两个用户的QoE
        qoe1 = self._predict_user_qoe(user1_features)
        qoe2 = self._predict_user_qoe(user2_features)
        
        # QoE相似度
        qoe_sim = 1.0 - abs(qoe1 - qoe2)
        
        # 特征相似度
        feature_sim = 1.0 - np.linalg.norm(user1_features - user2_features) / np.linalg.norm(user1_features + user2_features + 1e-8)
        
        # 综合相似度
        similarity = 0.6 * qoe_sim + 0.4 * feature_sim
        return max(0.0, min(1.0, similarity))
    
    def _adaptive_clustering(self, user_features: Dict[int, np.ndarray]) -> Dict[int, List[int]]:
        """自适应聚类算法"""
        if len(user_features) < self.min_cluster_size:
            return {0: list(user_features.keys())}
        
        # 准备特征矩阵
        user_ids = list(user_features.keys())
        feature_matrix = np.array([user_features[uid] for uid in user_ids])
        
        # 标准化特征
        if not self.is_fitted:
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            self.is_fitted = True
        else:
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
        
        # 确定最优聚类数
        optimal_k = self._determine_optimal_clusters(feature_matrix_scaled, user_ids)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # 组织聚类结果
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(user_ids[i])
        
        return dict(clusters)
    
    def _determine_optimal_clusters(self, features: np.ndarray, user_ids: List[int]) -> int:
        """确定最优聚类数"""
        if len(features) <= 2:
            return 1
        
        # 使用肘部法则和轮廓系数
        max_k = min(self.max_clusters, len(features) // 2)
        if max_k < 2:
            return 1
        
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
                # 计算轮廓系数
                from sklearn.metrics import silhouette_score
                if len(set(labels)) > 1:
                    score = silhouette_score(features, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        return best_k
    
    def cluster_users(self, active_clients: Dict) -> Dict[int, List[int]]:
        """执行QoE感知聚类"""
        if not active_clients:
            return {}
        
        # 提取用户特征
        user_features = {}
        for user_id, user_data in active_clients.items():
            features = self._extract_user_features(user_data)
            user_features[user_id] = features
        
        # 执行自适应聚类
        clusters = self._adaptive_clustering(user_features)
        
        # 记录聚类历史
        self.clustering_history.append({
            'timestamp': time.time(),
            'num_clusters': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters.values()],
            'total_users': len(active_clients)
        })
        
        return clusters

# ================================= [多播优化框架] =================================
class MulticastFramework:
    """
    实现论文中的多播优化框架
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.clustering = QoEAwareClustering(max_users)
        self.multicast_optimizer = MulticastOptimizer()
        
        # 多播参数
        self.bandwidth_efficiency_weight = 0.4
        self.latency_weight = 0.3
        self.qoe_weight = 0.3
        
        # 优化历史
        self.optimization_history = deque(maxlen=50)
        self.performance_trends = deque(maxlen=20)
        
    def _calculate_multicast_efficiency(self, cluster: List[int], active_clients: Dict) -> float:
        """计算多播效率"""
        if not cluster:
            return 0.0
        
        # 计算组内用户特征
        throughputs = [active_clients[uid].get('throughput', 0) for uid in cluster if uid in active_clients]
        delays = [active_clients[uid].get('delay', 0) for uid in cluster if uid in active_clients]
        
        if not throughputs:
            return 0.0
        
        # 多播效率 = 最小吞吐量 / 平均延迟
        min_throughput = min(throughputs)
        avg_delay = np.mean(delays)
        
        efficiency = min_throughput / max(1.0, avg_delay / 100.0)
        return min(1.0, efficiency)
    
    def _optimize_multicast_strategy(self, clusters: Dict[int, List[int]], 
                                   active_clients: Dict) -> Dict[int, Dict]:
        """优化多播策略"""
        multicast_strategies = {}
        
        for cluster_id, user_ids in clusters.items():
            if not user_ids:
                continue
            
            # 计算集群特征
            cluster_features = []
            for user_id in user_ids:
                if user_id in active_clients:
                    user_data = active_clients[user_id]
                    features = [
                        user_data.get('throughput', 0.0),
                        user_data.get('delay', 0.0),
                        user_data.get('device_score', 0.5),
                        user_data.get('network_stability', 1.0),
                        len(user_ids),  # 集群大小
                        self._calculate_multicast_efficiency(user_ids, active_clients),
                        user_data.get('bandwidth_variance', 0.0),
                        user_data.get('latency_jitter', 0.0),
                        user_data.get('user_preference', 0.5),
                        user_data.get('viewpoint', 0.0) if isinstance(user_data.get('viewpoint'), (int, float)) else 0.0
                    ]
                    cluster_features.append(features)
            
            if not cluster_features:
                continue
            
            # 使用深度学习模型优化多播策略
            cluster_features_array = np.array(cluster_features)
            avg_features = np.mean(cluster_features_array, axis=0)
            
            with torch.no_grad():
                features_tensor = torch.from_numpy(avg_features).float().unsqueeze(0)
                multicast_output, latency_output = self.multicast_optimizer(features_tensor)
                
                multicast_prob = multicast_output.item()
                latency_optimization = latency_output.item()
            
            # 生成多播策略
            strategy = {
                'use_multicast': multicast_prob > 0.5,
                'multicast_probability': multicast_prob,
                'latency_optimization': latency_optimization,
                'cluster_size': len(user_ids),
                'efficiency': self._calculate_multicast_efficiency(user_ids, active_clients),
                'qoe_prediction': np.mean([self.clustering._predict_user_qoe(
                    self.clustering._extract_user_features(active_clients[uid])
                ) for uid in user_ids if uid in active_clients])
            }
            
            multicast_strategies[cluster_id] = strategy
        
        return multicast_strategies
    
    def optimize_multicast_framework(self, active_clients: Dict) -> Tuple[Dict, Dict]:
        """优化多播框架"""
        # 1. QoE感知聚类
        clusters = self.clustering.cluster_users(active_clients)
        
        # 2. 多播策略优化
        multicast_strategies = self._optimize_multicast_strategy(clusters, active_clients)
        
        # 3. 记录优化历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'num_clusters': len(clusters),
            'multicast_strategies': len(multicast_strategies),
            'total_users': len(active_clients)
        })
        
        return clusters, multicast_strategies

# ================================= [增强的预测控制器] =================================
class EnhancedPredictiveController:
    """
    增强版预测控制器，实现QoE感知的深度学习辅助多播框架
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.multicast_framework = MulticastFramework(max_users)
        self.decision_history = deque(maxlen=100)
        
        # 自适应参数
        self.adaptation_rate = 0.05
        self.performance_window = 10
        
    def make_decision(self, active_clients: Dict) -> List[int]:
        """基于QoE感知多播框架的决策函数"""
        if not active_clients:
            return [0] * self.max_users
        
        # 执行多播框架优化
        clusters, multicast_strategies = self.multicast_framework.optimize_multicast_framework(active_clients)
        
        # 生成决策
        decisions = [0] * self.max_users
        
        for cluster_id, user_ids in clusters.items():
            if cluster_id not in multicast_strategies:
                continue
            
            strategy = multicast_strategies[cluster_id]
            
            # 基于多播策略和QoE预测做决策
            if strategy['use_multicast']:
                # 多播模式：基于QoE预测和延迟优化
                qoe_threshold = 0.6
                latency_threshold = 0.7
                
                use_enhanced = (strategy['qoe_prediction'] > qoe_threshold and 
                              strategy['latency_optimization'] > latency_threshold)
            else:
                # 单播模式：基于用户个体特征
                use_enhanced = strategy['multicast_probability'] > 0.5
            
            # 为集群内所有用户应用决策
            for user_id in user_ids:
                if 1 <= user_id <= self.max_users:
                    decisions[user_id - 1] = 1 if use_enhanced else 0
        
        # 记录决策历史
        self.decision_history.append({
            'timestamp': time.time(),
            'decisions': decisions.copy(),
            'num_clusters': len(clusters),
            'multicast_strategies': len(multicast_strategies),
            'total_qoe': sum(strategy['qoe_prediction'] for strategy in multicast_strategies.values())
        })
        
        return decisions

# ================================= [主控制器] =================================
def run_enhanced_predictive_controller(args):
    """运行增强版预测控制器"""
    # ✅ 【关键修复】兼容 max_users 和 num_users 参数
    num_users = getattr(args, 'max_users', None) or getattr(args, 'num_users', 20)
    user_offset = getattr(args, 'user_offset', 0)
    
    info_prefix = f"[EnhancedPredictiveController (Users for h{user_offset+1}-h{user_offset+num_users})]"
    print(f"{info_prefix} Initializing with QoE-Aware Deep Learning Multicast Framework...")
    print(f"{info_prefix} Implementing: QoE Prediction + Adaptive Clustering + Multicast Optimization")
    
    # 初始化控制器
    controller = EnhancedPredictiveController(num_users)
    
    print(f"{info_prefix} Starting enhanced decision loop. Outputting to {args.decision_file}")
    
    # ✅ 【关键修复】等待客户端启动完成（大量用户需要更长时间）
    # 客户端分批启动：batch_size=5，每批间隔2秒
    # 90用户需要：18批 × 2秒 = 36秒
    if num_users > 50:
        wait_time = max(30, int(num_users * 0.4))
        print(f"{info_prefix} ⏳ Waiting {wait_time} seconds for clients to start ({num_users} users)...", flush=True)
        time.sleep(wait_time)
    elif num_users > 20:
        wait_time = 20
        print(f"{info_prefix} ⏳ Waiting {wait_time} seconds for clients to start...", flush=True)
        time.sleep(wait_time)
    else:
        wait_time = 10
        print(f"{info_prefix} ⏳ Waiting {wait_time} seconds for clients to start...", flush=True)
        time.sleep(wait_time)
    
    while True:
        try:
            # 1. 数据采集
            active_clients = aggregate_client_states(num_users)
            
            if not active_clients:
                time.sleep(2)
                continue
            
            # 2. QoE感知多播优化决策
            decisions = controller.make_decision(active_clients)
            
            # 3. 生成决策文件
            decision_payload = {
                "layers": decisions,
                "timestamp": time.time(),
                "used_model": "EnhancedQoEAwareMulticast_v2",
                "framework_info": {
                    "num_clusters": len(controller.multicast_framework.clustering.clustering_history),
                    "multicast_efficiency": np.mean([h.get('efficiency', 0) for h in controller.multicast_framework.optimization_history]),
                    "qoe_predictions": len(controller.multicast_framework.clustering.qoe_history)
                }
            }
            
            write_decision_file(args.decision_file, decision_payload)
            
            # 4. 打印调试信息
            active_count = len(active_clients)
            subscribed_count = sum(decisions)
            num_clusters = len(controller.multicast_framework.clustering.clustering_history)
            
            print(f"{info_prefix} QoE-aware multicast optimization: {active_count} users, {num_clusters} clusters, {subscribed_count} enhanced subscriptions.")
            
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ {info_prefix} Error in enhanced decision loop: {e}\n{traceback.format_exc()}")
            time.sleep(5)

# ================================= [辅助函数] =================================
def aggregate_client_states(max_users):
    """聚合客户端状态"""
    client_states = {}
    state_files = glob.glob("/tmp/client_h*_state.json")
    current_time = time.time()
    # ✅ 【关键修复】状态文件超时时间 - 适应大量用户启动时间
    # 客户端分批启动：batch_size=5，每批间隔2秒
    # 90用户需要：18批 × 2秒 = 36秒
    # 因此需要更长的超时时间
    STATE_TIMEOUT_S = 30  # 从10秒增加到30秒，适应大量用户启动时间

    for file_path in state_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if current_time - data.get("timestamp", 0) > STATE_TIMEOUT_S:
                continue

            host_id = data.get("host_id")
            if host_id is not None and 1 <= host_id <= max_users:
                client_states[host_id] = {
                    "throughput": data.get("throughput_mbps", 0.0),
                    "delay": data.get("delay_ms", 0.0),
                    "device_score": data.get("device_score", 0.5),
                    "viewpoint": data.get("viewpoint", "unknown"),
                    "network_stability": data.get("network_stability", 1.0),
                    "user_preference": data.get("user_preference", 0.5),
                    "bandwidth_variance": data.get("bandwidth_variance", 0.0),
                    "latency_jitter": data.get("latency_jitter", 0.0)
                }
        except (IOError, json.JSONDecodeError):
            continue
    return client_states

def write_decision_file(file_path, decision_data):
    """原子方式写入决策文件"""
    temp_path = file_path + ".tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(decision_data, f, indent=2)
        os.rename(temp_path, file_path)
    except IOError as e:
        print(f"[EnhancedPredictiveController] Failed to write file {file_path}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Predictive Controller with QoE-Aware Multicast Framework")
    parser.add_argument("--num_users", type=int, required=True, help="Number of users")
    parser.add_argument("--user_offset", type=int, required=True, help="User offset (starting user ID)")
    parser.add_argument("--decision_file", type=str, required=True, help="Path to decision file")
    parser.add_argument("--model_path", type=str, help="Path to model file (optional)")
    args = parser.parse_args()
    
    # ✅ 【验证】确保参数正确接收
    print(f"[EnhancedPredictiveController] 参数验证:")
    print(f"  --num_users: {args.num_users}")
    print(f"  --user_offset: {args.user_offset}")
    print(f"  --decision_file: {args.decision_file}")
    print(f"  --model_path: {args.model_path if args.model_path else 'None'}")
    
    if args.num_users <= 0:
        raise ValueError(f"❌ --num_users 必须大于0，当前值: {args.num_users}")
    if args.user_offset < 0:
        raise ValueError(f"❌ --user_offset 必须大于等于0，当前值: {args.user_offset}")
    if not args.decision_file:
        raise ValueError(f"❌ --decision_file 不能为空")

    run_enhanced_predictive_controller(args)
