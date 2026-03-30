# heuristic_controller_v2.py
import argparse
import os
import json
import time
import traceback
import glob
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
import math

# ================================= [联合优化算法] =================================
class JointOptimizationSolver:
    """
    实现论文中的联合用户分组、版本选择、带宽分配算法
    "Joint user grouping, version selection, and bandwidth allocation for live video multicasting"
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.user_groups = {}  # 用户分组
        self.version_selections = {}  # 版本选择
        self.bandwidth_allocations = {}  # 带宽分配
        
        # 优化参数（优先级2修改：降低迭代次数，模拟计算资源限制）
        self.optimization_iterations = 5  # 从10降到5
        self.convergence_threshold = 0.01
        self.learning_rate = 0.1
        
        # 历史数据
        self.performance_history = deque(maxlen=20)
        self.optimization_history = deque(maxlen=10)
        
    def _calculate_user_similarity(self, user1: Dict, user2: Dict) -> float:
        """计算用户相似度"""
        # 基于吞吐量、延迟、设备评分的相似度
        throughput_sim = 1.0 - abs(user1.get('throughput', 0) - user2.get('throughput', 0)) / max(user1.get('throughput', 1), user2.get('throughput', 1), 1)
        delay_sim = 1.0 - abs(user1.get('delay', 0) - user2.get('delay', 0)) / 200.0
        device_sim = 1.0 - abs(user1.get('device_score', 0) - user2.get('device_score', 0))
        
        # 加权相似度
        similarity = 0.4 * throughput_sim + 0.3 * delay_sim + 0.3 * device_sim
        return max(0.0, min(1.0, similarity))
    
    def _optimize_user_grouping(self, active_clients: Dict) -> Dict:
        """
        联合用户分组算法
        基于用户特征进行智能分组
        """
        if len(active_clients) < 2:
            return {0: list(active_clients.keys())}
        
        # 初始化分组
        groups = {}
        group_id = 0
        
        # 基于相似度的聚类算法
        unassigned_users = list(active_clients.keys())
        similarity_threshold = 0.6  # 相似度阈值
        
        while unassigned_users:
            # 选择第一个未分配用户作为种子
            seed_user = unassigned_users[0]
            current_group = [seed_user]
            unassigned_users.remove(seed_user)
            
            # 寻找相似用户
            to_remove = []
            for user_id in unassigned_users:
                similarity = self._calculate_user_similarity(
                    active_clients[seed_user], 
                    active_clients[user_id]
                )
                
                if similarity >= similarity_threshold:
                    current_group.append(user_id)
                    to_remove.append(user_id)
            
            # 移除已分配用户
            for user_id in to_remove:
                unassigned_users.remove(user_id)
            
            # 添加到分组
            groups[group_id] = current_group
            group_id += 1
        
        return groups
    
    def _optimize_version_selection(self, groups: Dict, active_clients: Dict) -> Dict:
        """
        版本选择优化算法
        为每个组选择最优的版本组合
        """
        version_selections = {}
        
        for group_id, user_ids in groups.items():
            if not user_ids:
                continue
            
            # 计算组的特征
            group_throughput = np.mean([active_clients[uid].get('throughput', 0) for uid in user_ids])
            group_delay = np.mean([active_clients[uid].get('delay', 0) for uid in user_ids])
            group_device_score = np.mean([active_clients[uid].get('device_score', 0) for uid in user_ids])
            
            # 版本选择策略
            base_version_prob = 0.5
            enhanced_version_prob = 0.5
            
            # 基于网络条件的版本选择
            if group_throughput > 5.0 and group_delay < 100:
                # 高带宽低延迟：优先增强版本
                enhanced_version_prob = min(0.7, 0.5 + (group_throughput - 5.0) / 20.0)  # 上限从0.9降到0.7
                base_version_prob = 1.0 - enhanced_version_prob
            elif group_throughput < 2.0 or group_delay > 200:
                # 低带宽高延迟：优先基础版本
                base_version_prob = min(0.9, 0.5 + (2.0 - group_throughput) / 4.0)
                enhanced_version_prob = 1.0 - base_version_prob
            
            # 基于设备评分的调整
            device_factor = group_device_score
            enhanced_version_prob *= device_factor
            # 再次限制上限（优先级2修改：避免过度激进）
            enhanced_version_prob = min(0.7, enhanced_version_prob)
            base_version_prob = 1.0 - enhanced_version_prob
            
            version_selections[group_id] = {
                'base_version_prob': base_version_prob,
                'enhanced_version_prob': enhanced_version_prob,
                'group_size': len(user_ids),
                'group_throughput': group_throughput,
                'group_delay': group_delay
            }
        
        return version_selections
    
    def _optimize_bandwidth_allocation(self, groups: Dict, version_selections: Dict, 
                                     total_bandwidth: float) -> Dict:
        """
        带宽分配优化算法
        为每个组分配最优带宽
        """
        bandwidth_allocations = {}
        
        # 计算每个组的权重
        group_weights = {}
        total_weight = 0.0
        
        for group_id, version_info in version_selections.items():
            # 权重基于组大小、吞吐量需求和版本选择
            size_weight = version_info['group_size']
            throughput_weight = version_info['group_throughput']
            version_weight = version_info['enhanced_version_prob']
            
            weight = size_weight * (1.0 + throughput_weight / 10.0) * (1.0 + version_weight)
            group_weights[group_id] = weight
            total_weight += weight
        
        # 分配带宽
        for group_id, weight in group_weights.items():
            if total_weight > 0:
                allocated_bandwidth = (weight / total_weight) * total_bandwidth
                bandwidth_allocations[group_id] = {
                    'allocated_bandwidth': allocated_bandwidth,
                    'weight': weight,
                    'efficiency': allocated_bandwidth / max(1.0, weight)
                }
        
        return bandwidth_allocations
    
    def _calculate_group_qoe(self, group_id: int, user_ids: List[int], 
                           active_clients: Dict, version_selection: Dict,
                           bandwidth_allocation: Dict) -> float:
        """计算组的QoE"""
        if not user_ids or group_id not in version_selection or group_id not in bandwidth_allocation:
            return 0.0
        
        # 组内用户QoE
        group_qoes = []
        for user_id in user_ids:
            if user_id not in active_clients:
                continue
            
            user_data = active_clients[user_id]
            throughput = user_data.get('throughput', 0)
            delay = user_data.get('delay', 0)
            device_score = user_data.get('device_score', 0)
            
            # 基础QoE计算
            base_qoe = device_score * (1.0 - delay / 200.0) * min(1.0, throughput / 10.0)
            
            # 版本选择影响
            version_factor = (version_selection['base_version_prob'] * 0.8 + 
                            version_selection['enhanced_version_prob'] * 1.2)
            
            # 带宽分配影响
            allocated_bw = bandwidth_allocation['allocated_bandwidth']
            bandwidth_factor = min(1.0, allocated_bw / max(1.0, throughput))
            
            user_qoe = base_qoe * version_factor * bandwidth_factor
            group_qoes.append(user_qoe)
        
        return np.mean(group_qoes) if group_qoes else 0.0
    
    # ✅ 【关键修复】带宽容量对齐 - 与Mininet拓扑瓶颈链路一致
    # Mininet拓扑瓶颈：区域->边缘 = 400 Mbps (BW_R1_R3等)
    # 原默认值100.0会导致baseline策略"自我设限"，即使物理带宽充裕也只发Base层
    def solve_joint_optimization(self, active_clients: Dict, total_bandwidth: float = 400.0) -> Tuple[Dict, Dict, Dict]:
        """
        联合优化求解器
        返回: (用户分组, 版本选择, 带宽分配)
        """
        # 迭代优化
        best_groups = {}
        best_versions = {}
        best_bandwidth = {}
        best_total_qoe = 0.0
        
        for iteration in range(self.optimization_iterations):
            # 1. 用户分组优化
            current_groups = self._optimize_user_grouping(active_clients)
            
            # 2. 版本选择优化
            current_versions = self._optimize_version_selection(current_groups, active_clients)
            
            # 3. 带宽分配优化
            current_bandwidth = self._optimize_bandwidth_allocation(
                current_groups, current_versions, total_bandwidth)
            
            # 4. 计算总QoE
            total_qoe = 0.0
            for group_id in current_groups:
                group_qoe = self._calculate_group_qoe(
                    group_id, current_groups[group_id], active_clients,
                    current_versions, current_bandwidth)
                total_qoe += group_qoe
            
            # 5. 更新最佳解
            if total_qoe > best_total_qoe:
                best_total_qoe = total_qoe
                best_groups = current_groups.copy()
                best_versions = current_versions.copy()
                best_bandwidth = current_bandwidth.copy()
            
            # 6. 检查收敛
            if iteration > 0:
                qoe_improvement = total_qoe - (self.performance_history[-1] if self.performance_history else 0)
                if abs(qoe_improvement) < self.convergence_threshold:
                    break
        
        # 记录性能
        self.performance_history.append(best_total_qoe)
        self.optimization_history.append({
            'iteration': iteration + 1,
            'total_qoe': best_total_qoe,
            'num_groups': len(best_groups)
        })
        
        return best_groups, best_versions, best_bandwidth

# ================================= [增强的启发式控制器] =================================
class EnhancedHeuristicController:
    """
    增强版启发式控制器，实现联合优化算法
    """
    
    def __init__(self, max_users: int):
        self.max_users = max_users
        self.solver = JointOptimizationSolver(max_users)
        self.decision_history = deque(maxlen=50)
        
        # 自适应参数 - 轻微降敏调整
        self.adaptation_rate = 0.05  # 原为 0.1，降低适应速度
        self.performance_threshold = 0.7
        self.dynamic_threshold = False  # 禁止动态阈值更新
        self.bandwidth_threshold = 1.15  # 固定阈值
        
    # ✅ 【关键修复】带宽容量对齐 - 与Mininet拓扑瓶颈链路一致
    def make_decision(self, active_clients: Dict, total_bandwidth: float = 400.0) -> List[int]:
        """
        基于联合优化的决策函数
        """
        if not active_clients:
            return [0] * self.max_users
        
        # 执行联合优化
        groups, versions, bandwidth = self.solver.solve_joint_optimization(
            active_clients, total_bandwidth)
        
        # 生成决策
        decisions = [0] * self.max_users
        
        for group_id, user_ids in groups.items():
            if group_id not in versions:
                continue
            
            # 基于版本选择概率决定是否启用增强层
            enhanced_prob = versions[group_id]['enhanced_version_prob']
            use_enhanced = np.random.random() < enhanced_prob
            
            # 为组内所有用户应用相同决策
            for user_id in user_ids:
                if 1 <= user_id <= self.max_users:
                    decisions[user_id - 1] = 1 if use_enhanced else 0
        
        # 记录决策
        self.decision_history.append({
            'timestamp': time.time(),
            'decisions': decisions.copy(),
            'num_groups': len(groups),
            'total_qoe': sum(versions[gid]['enhanced_version_prob'] for gid in versions)
        })
        
        return decisions

# ================================= [主控制器] =================================
def run_enhanced_heuristic_controller(args):
    """
    运行增强版启发式控制器
    """
    # ✅ 【关键修复】兼容 max_users 和 num_users 参数
    num_users = getattr(args, 'max_users', None) or getattr(args, 'num_users', 20)
    user_offset = getattr(args, 'user_offset', 0)
    
    info_prefix = f"[EnhancedHeuristicController (Users for h{user_offset+1}-h{user_offset+num_users})]"
    print(f"{info_prefix} Initializing with Joint Optimization Algorithm...")
    print(f"{info_prefix} Implementing: User Grouping + Version Selection + Bandwidth Allocation")
    
    # 初始化控制器
    controller = EnhancedHeuristicController(num_users)
    
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
            # 1. 数据采集：获取所有客户端的最新状态
            active_clients = aggregate_client_states(num_users)
            
            if not active_clients:
                time.sleep(2)
                continue
            
            # 2. 联合优化决策
            decisions = controller.make_decision(active_clients)
            
            # 3. 生成决策文件
            decision_payload = {
                "layers": decisions,
                "timestamp": time.time(),
                "used_model": "EnhancedJointOptimization_v2",
                "optimization_info": {
                    "num_groups": len(controller.solver.user_groups),
                    "total_qoe": controller.solver.performance_history[-1] if controller.solver.performance_history else 0.0,
                    "convergence": len(controller.solver.optimization_history)
                }
            }
            
            write_decision_file(args.decision_file, decision_payload)
            
            # 4. 打印调试信息
            active_count = len(active_clients)
            subscribed_count = sum(decisions)
            num_groups = len(controller.solver.user_groups)
            
            print(f"{info_prefix} Joint optimization completed: {active_count} users, {num_groups} groups, {subscribed_count} enhanced subscriptions.")
            
            time.sleep(5)  # 控制器决策频率（优先级2修改：从2秒改为5秒，模拟信令开销）
            
        except Exception as e:
            print(f"❌ {info_prefix} Error in enhanced decision loop: {e}\n{traceback.format_exc()}")
            time.sleep(5)

# ================================= [辅助函数] =================================
def aggregate_client_states(max_users):
    """聚合客户端状态（保持原有接口）"""
    client_states = {}
    state_files = glob.glob("/tmp/client_h*_state.json")
    current_time = time.time()
    # ✅ 【关键修复】状态文件超时时间 - 适应大量用户启动时间
    # 客户端分批启动：batch_size=5，每批间隔2秒
    # 90用户需要：18批 × 2秒 = 36秒
    # 因此需要更长的超时时间
    STATE_TIMEOUT_S = 30  # 从15秒增加到30秒，适应大量用户启动时间

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
                    "viewpoint": data.get("viewpoint", "unknown")
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
        print(f"[EnhancedHeuristicController] Failed to write file {file_path}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Heuristic Controller with Joint Optimization")
    parser.add_argument("--num_users", type=int, required=True)
    parser.add_argument("--user_offset", type=int, required=True)
    parser.add_argument("--decision_file", type=str, required=True)
    parser.add_argument("--model_path", type=str)  # 保留以兼容
    args = parser.parse_args()

    run_enhanced_heuristic_controller(args)
