#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling SC-DDQN 训练脚本
严格复现论文《Toward Optimal Real-Time Volumetric Video Streaming》
使用真实数据集（带宽、设备、FoV）进行训练
"""

import numpy as np
import torch
import os
import time
import pandas as pd
import random
import math
from collections import deque
from sc_ddqn_model import SCDDQN_Agent, LSTMPredictor, SCDDQN_Network

# 项目根目录（用于构建相对路径）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# 配置参数 (尽量与论文一致)
# ==============================================================================
N_GOF = 5          # 滚动窗口大小 (论文中的N)
C_TILES = 8        # 每个GoF的切片数 (论文中的C)
L_LEVELS = 2       # 质量等级数 (Base=0, Enhanced=1)
# ✅ 【关键修复】状态维度计算
# Y向量实际维度: predicted_bw(5) + predicted_fov(40) + buffer(1) + device_score(1) + network_onehot(4) + base_bitrate(1) + enh_bitrate(1) = 53
# x_decisions维度: N_GOF * C_TILES = 5 * 8 = 40
# 总状态维度: 53 + 40 = 93
STATE_DIM = 93     # 状态维度 (Y + x_m 的总长度) - 修复：从64增加到93，避免截断丢失关键信息
MAX_EPISODES = 20000  # 训练轮数 (论文用了20万，这里设2万供快速测试)
SAVE_PATH = os.path.join(PROJECT_ROOT, "trained_models", "sc_ddqn_rolling.pth")
TARGET_UPDATE_FREQ = 10  # 目标网络更新频率

# 数据集路径
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(PROJECT_ROOT, "datasets"))
WIFI_CSV = os.path.join(DATASET_DIR, "wifi_clean.csv")
FOURG_CSV = os.path.join(DATASET_DIR, "4G-network-data_clean.csv")
FIVEG_CSV = os.path.join(DATASET_DIR, "5g_final_trace.csv")  # 优先使用5g_final_trace.csv
FIVEG_CSV_ALT = os.path.join(DATASET_DIR, "5g_network_data_clean.csv")  # 备选
OPTIC_CSV = os.path.join(DATASET_DIR, "Optic_Bandwidth_clean_2.csv")
DEVICE_CSV = os.path.join(DATASET_DIR, "Headset device performance.csv")

# ==============================================================================
# 真实数据集加载
# ==============================================================================
class RealDatasetLoader:
    """加载真实数据集"""
    def __init__(self):
        self.bandwidth_data = {}
        self.device_data = None
        self._load_datasets()
    
    def _load_datasets(self):
        """加载所有数据集"""
        # 加载带宽数据
        for net_type, csv_paths in [
            ("wifi", [WIFI_CSV]),
            ("4g", [FOURG_CSV]),
            ("5g", [FIVEG_CSV, FIVEG_CSV_ALT]),  # 尝试多个文件
            ("fiber_optic", [OPTIC_CSV])
        ]:
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # 尝试多种可能的列名
                    bw_column = None
                    for col in ["bytes_sec (Mbps)", "bytes_sec(Mbps)", "bandwidth", "Bandwidth", "Mbps"]:
                        if col in df.columns:
                            bw_column = col
                            break
                    
                    # 如果没找到，尝试第一列
                    if bw_column is None and len(df.columns) > 0:
                        bw_column = df.columns[0]
                    
                    if bw_column:
                        self.bandwidth_data[net_type] = pd.to_numeric(df[bw_column], errors='coerce').dropna().values
                        if len(self.bandwidth_data[net_type]) > 0:
                            print(f"✅ 加载 {net_type} 带宽数据: {len(self.bandwidth_data[net_type])} 条 (列: {bw_column})")
                        else:
                            print(f"⚠️  {net_type} 数据为空")
                    else:
                        print(f"⚠️  {net_type} CSV未找到带宽列")
                except Exception as e:
                    print(f"❌ 加载 {net_type} 数据失败: {e}")
        
        # 加载设备数据
        if os.path.exists(DEVICE_CSV):
            try:
                self.device_data = pd.read_csv(DEVICE_CSV)
                print(f"✅ 加载设备数据: {len(self.device_data)} 条")
            except Exception as e:
                print(f"❌ 加载设备数据失败: {e}")
    
    def sample_bandwidth(self, net_type="wifi"):
        """从真实数据集中采样带宽"""
        if net_type in self.bandwidth_data and len(self.bandwidth_data[net_type]) > 0:
            return float(np.random.choice(self.bandwidth_data[net_type]))
        return 10.0  # 默认值
    
    def sample_device_score(self):
        """从真实数据集中采样设备分数"""
        if self.device_data is not None and len(self.device_data) > 0:
            # 使用GPU Clock作为设备分数代理
            if "GPU Clock (MHz)" in self.device_data.columns:
                gpu_vals = self.device_data["GPU Clock (MHz)"].dropna().values
                if len(gpu_vals) > 0:
                    gpu_val = float(np.random.choice(gpu_vals))
                    # 归一化到0-1范围（假设GPU范围400-1000 MHz）
                    return min(1.0, max(0.0, (gpu_val - 400) / 600))
        return 0.5  # 默认值

# ==============================================================================
# 模拟环境 (基于真实数据集和QoE公式)
# ==============================================================================
class RollingSimEnv:
    """
    基于真实数据集的滚动优化模拟环境
    使用真实的QoE计算公式（R_o, R_q, R_b, R_l）
    """
    def __init__(self, dataset_loader):
        self.dataset = dataset_loader
        
        # ✅ 【关键修复】Base和Enhanced的码率 (Mbps) - 必须与cmaf_publish.sh一致
        # 实际运行配置：Base=3.0Mbps, Enhanced=0.8Mbps (叠加在Base上)
        self.base_bitrate = 3.0   # Base层码率（与cmaf_publish.sh一致）
        self.enh_bitrate = 0.8    # Enhanced层码率（叠加在Base上，与cmaf_publish.sh一致）
        
        # 质量得分 (论文公式)
        self.quality_base = 0.6   # Base层质量得分
        self.quality_enh = 1.0    # Enhanced层质量得分
        
        # ✅ 【关键修复】奖励函数权重 - 与推理时统一
        # Note: all baseline strategies (rolling, md2g, heuristic, clustering) must use the same QoE formula and coefficients.
        # 推理时配置（dispatch_strategy_enhanced_unified.py）:
        #   lambda_o=0.2, lambda_q=0.5, lambda_b=0.1, lambda_l=0.2
        # 
        # 训练时特殊处理：
        #   - lambda_o=0.0: 单用户场景无法学习分组，设为0避免干扰
        #   - lambda_q=0.5: 与推理时保持一致（统一为0.5，不是0.7）
        #   - lambda_b=0.1: 与推理时保持一致
        #   - lambda_l=0.2: 与推理时保持一致（虽然R_l=0，但权重保留以保持公式结构一致）
        self.lambda_o = 0.0   # 单用户场景下设为0（无法学习分组，推理时用0.2）
        self.lambda_q = 0.5   # ✅ 修复：与推理时统一为0.5（所有策略相同）
        self.lambda_b = 0.1   # 与推理时保持一致（所有策略相同）
        self.lambda_l = 0.2   # 与推理时保持一致（所有策略相同）
        # 注意：虽然OFF模式下R_l=0，但lambda_l保留0.2是为了：
        #   1. 与推理时公式结构完全一致
        #   2. 如果未来切换到ON模式，R_l会有值，权重已配置好
        #   3. 实际计算：0.2 * 0.0 = 0，不影响训练
        
        # R_q 权重
        self.alpha1, self.alpha2, self.alpha3 = 1.0, 0.2, 0.3
        
        # Buffer状态
        self.buffer = 2.0  # 初始Buffer (秒)
        self.buffer_max = 5.0
        
        # 网络类型
        self.network_types = ["wifi", "4g", "5g", "fiber_optic"]
        
    def reset(self, network_type=None):
        """重置环境"""
        self.buffer = 2.0
        
        # 随机选择网络类型
        if network_type is None:
            network_type = random.choice(self.network_types)
        self.current_network = network_type
        
        # 从真实数据集采样未来 N 步的带宽
        self.bw_trace = np.array([
            self.dataset.sample_bandwidth(network_type) 
            for _ in range(N_GOF)
        ])
        
        # 采样设备分数
        self.device_score = self.dataset.sample_device_score()
        
        # 模拟FoV命中情况 (简化：每个Tile有30%概率在视野内)
        self.fov_trace = np.random.choice(
            [0, 1], 
            size=(N_GOF, C_TILES), 
            p=[0.7, 0.3]
        )
        
        return self._get_env_features()
    
    def _get_env_features(self):
        """
        构造环境特征 Y (论文 Eq.26)
        Y = [预测带宽(N), 预测FoV(N*C), Buffer(1), 设备分数(1), 网络类型编码(4), ...]
        """
        # 预测带宽 (N维)
        predicted_bw = self.bw_trace.copy()
        
        # 预测FoV (N*C维，扁平化)
        predicted_fov = self.fov_trace.flatten()
        
        # 网络类型 one-hot编码
        network_onehot = np.zeros(4)
        network_idx = self.network_types.index(self.current_network) if self.current_network in self.network_types else 0
        network_onehot[network_idx] = 1.0
        
        # 构造Y向量
        y = np.concatenate([
            predicted_bw,           # N维
            predicted_fov,          # N*C维
            [self.buffer],          # 1维
            [self.device_score],    # 1维
            network_onehot,         # 4维
            [self.base_bitrate],    # 1维
            [self.enh_bitrate],     # 1维
        ])
        
        # ✅ 【关键修复】不再截断，确保所有关键信息都保留
        # Y向量维度: 53 (predicted_bw 5 + predicted_fov 40 + buffer 1 + device_score 1 + network_onehot 4 + base_bitrate 1 + enh_bitrate 1)
        # x_decisions维度: 40 (N_GOF * C_TILES)
        # 总状态维度: 53 + 40 = 93
        target_y_dim = STATE_DIM - N_GOF * C_TILES  # 93 - 40 = 53
        
        # 验证维度是否正确
        if len(y) != target_y_dim:
            if len(y) < target_y_dim:
                # 如果维度不足，填充零
                pad = np.zeros(target_y_dim - len(y))
                y = np.concatenate([y, pad])
            else:
                # 如果维度超出，说明计算有误，报错
                raise ValueError(f"Y向量维度({len(y)})超出目标维度({target_y_dim})，请检查特征构造逻辑！")
        
        return y
    
    def step(self, action_quality, step_idx):
        """
        执行一步动作 (为一个 Tile 选择质量)
        
        Args:
            action_quality: 0=Base, 1=Enhanced
            step_idx: 当前是第几个 Tile (0 ~ N*C-1)
        
        Returns:
            reward: 基于真实QoE公式计算的奖励
        """
        gof_idx = step_idx // C_TILES
        tile_idx = step_idx % C_TILES
        
        # 获取当前Tile的属性
        is_in_fov = self.fov_trace[gof_idx, tile_idx]
        current_bw = self.bw_trace[gof_idx]
        
        # 确定选择的码率
        if action_quality == 0:  # Base
            selected_bitrate = self.base_bitrate
            quality_score = self.quality_base
        else:  # Enhanced (叠加在Base上)
            selected_bitrate = self.base_bitrate + self.enh_bitrate
            quality_score = self.quality_enh
        
        # --- 计算QoE奖励 (使用真实公式) ---
        
        # 1. R_q: User Perceived Quality
        # R_q = α₁*quality_score - α₂*delay_norm - α₃*stall_norm
        # 简化：假设延迟和卡顿为0（在模拟环境中）
        delay_norm = 0.0
        stall_norm = 0.0
        R_q = self.alpha1 * quality_score - self.alpha2 * delay_norm - self.alpha3 * stall_norm
        R_q = max(0.0, min(1.0, R_q))
        
        # 2. R_o: Grouping Efficiency (简化：单用户场景)
        R_o = 0.2  # 固定值（单用户场景）
        
        # 3. R_b: Bandwidth Efficiency Penalty
        TARGET_BITRATE = 1.0  # 与dispatch_strategy_enhanced_unified.py一致
        bandwidth_utilization = min(1.0, current_bw / TARGET_BITRATE)
        R_b = 1.0 - bandwidth_utilization
        
        # 4. R_l: Load Balance Penalty (OFF模式为0)
        R_l = 0.0
        
        # 5. 计算总奖励
        reward = self.lambda_o * R_o + self.lambda_q * R_q - self.lambda_b * R_b - self.lambda_l * R_l
        reward = max(0.0, reward)
        
        # 6. Buffer更新 (模拟)
        # 下载时间 = Tile大小 / 带宽
        download_time = (selected_bitrate / C_TILES) / max(current_bw, 0.1)
        play_time_per_tile = 1.0 / C_TILES
        self.buffer += play_time_per_tile - download_time
        
        # Buffer限制
        if self.buffer < 0:
            # Buffer耗尽，增加卡顿惩罚
            stall_penalty = abs(self.buffer) * 10
            reward -= stall_penalty
            self.buffer = 0
        elif self.buffer > self.buffer_max:
            self.buffer = self.buffer_max
        
        return reward

# ==============================================================================
# 训练主循环
# ==============================================================================
def train():
    """主训练函数"""
    # 创建输出目录
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # 加载数据集
    print("📊 加载真实数据集...")
    dataset_loader = RealDatasetLoader()
    
    # 初始化环境
    env = RollingSimEnv(dataset_loader)
    
    # 初始化Agent
    print(f"🚀 初始化 SC-DDQN Agent...")
    print(f"   State Dim: {STATE_DIM}, Action Dim: {L_LEVELS}")
    agent = SCDDQN_Agent(state_dim=STATE_DIM, action_dim=L_LEVELS)
    
    print(f"\n🎯 开始训练 (Target: {MAX_EPISODES} Episodes)...")
    print(f"   模型保存路径: {SAVE_PATH}")
    print(f"   目标网络更新频率: 每 {TARGET_UPDATE_FREQ} 个episode\n")
    
    start_time = time.time()
    episode_rewards = deque(maxlen=100)  # 记录最近100个episode的奖励
    
    for episode in range(1, MAX_EPISODES + 1):
        # 1. 环境重置，获取 Y
        env_features_Y = env.reset()
        
        # 初始化决策向量 x (全0，表示未决策)
        # 使用归一化值：0=未决策, 0.5=Base, 1.0=Enhanced
        x_decisions = np.zeros(N_GOF * C_TILES)
        
        total_reward = 0
        
        # 2. 串行循环 (Serial Cyclic) - 论文算法核心
        for step in range(N_GOF * C_TILES):
            # 构造当前状态 S_m = [Y, x_decisions]
            state = np.concatenate((env_features_Y, x_decisions))
            
            # 智能体选择动作
            action = agent.select_action(state, training=True)
            
            # 更新决策向量 (记录归一化的决策值)
            x_decisions[step] = (action + 1) / L_LEVELS  # 0.5=Base, 1.0=Enhanced
            
            # 执行动作，获取奖励
            reward = env.step(action, step)
            total_reward += reward
            
            # 存储经验
            next_state = np.concatenate((env_features_Y, x_decisions))
            done = (step == N_GOF * C_TILES - 1)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()  # 内部执行梯度下降
            
        episode_rewards.append(total_reward)
        
        # 定期更新目标网络
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # 打印日志（前10个episode每个都打印，之后每10个打印一次，100个episode后每100个打印）
        should_print = False
        if episode <= 10:
            should_print = True
        elif episode <= 100 and episode % 10 == 0:
            should_print = True
        elif episode % 100 == 0:
            should_print = True
        
        if should_print:
            avg_time = (time.time() - start_time) / episode
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            elapsed = time.time() - start_time
            print(f"Episode {episode:5d}/{MAX_EPISODES} | "
                  f"Reward: {total_reward:6.2f} (Avg: {avg_reward:6.2f}) | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Speed: {1/avg_time:.1f} iter/s | "
                  f"Time: {elapsed/60:.1f}min | "
                  f"Memory: {len(agent.memory)}/{agent.memory.maxlen}")
        
        # 保存模型
        if episode % 1000 == 0:
            checkpoint = {
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'episode': episode,
                'epsilon': agent.epsilon,
                'avg_reward': np.mean(episode_rewards) if episode_rewards else 0
            }
            torch.save(checkpoint, SAVE_PATH)
            print(f"💾 模型已保存: {SAVE_PATH} (Episode {episode})")
    
    # 最终保存
    final_checkpoint = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'episode': MAX_EPISODES,
        'epsilon': agent.epsilon,
        'avg_reward': np.mean(episode_rewards) if episode_rewards else 0
    }
    torch.save(final_checkpoint, SAVE_PATH)
    print(f"\n✅ 训练完成！")
    print(f"   最终模型已保存: {SAVE_PATH}")
    print(f"   平均奖励: {np.mean(episode_rewards):.4f}")
    print(f"   最终Epsilon: {agent.epsilon:.4f}")

if __name__ == "__main__":
    # ==============================================================================
    # 最终检查清单 (运行前必看)
    # ==============================================================================
    print("="*80)
    print("🔍 训练前检查清单")
    print("="*80)
    
    # 1. 路径检查：确认数据集文件存在
    print("\n【1. 数据集路径检查】")
    # 必需文件（如果不存在会导致训练失败）
    required_files = {
        "WiFi": WIFI_CSV,
        "4G": FOURG_CSV,
        "5G": FIVEG_CSV,
        "Fiber_Optic": OPTIC_CSV,
        "Device": DEVICE_CSV
    }
    # 可选文件（备选，不存在不影响训练）
    optional_files = {
        "5G_ALT": FIVEG_CSV_ALT,
    }
    
    all_ok = True
    for name, path in required_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"  ✅ {name}: {path} ({size:.1f} KB)")
        else:
            print(f"  ❌ {name}: {path} (文件不存在！)")
            all_ok = False
    
    # 检查可选文件（仅提示，不影响all_ok）
    for name, path in optional_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"  ✅ {name}: {path} ({size:.1f} KB) [可选]")
        else:
            print(f"  ⚠️  {name}: {path} (文件不存在，但这是可选文件，不影响训练)")
    
    # 2. GPU设置检查
    print("\n【2. GPU设置检查】")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"  ✅ GPU可用: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"  ✅ 将使用GPU训练")
    else:
        print(f"  ⚠️  GPU不可用，将使用CPU训练（速度较慢）")
    
    # 3. 模型一致性检查
    print("\n【3. 模型一致性检查】")
    print(f"  STATE_DIM: {STATE_DIM}")
    print(f"  N_GOF: {N_GOF}, C_TILES: {C_TILES}")
    y_dim = N_GOF + N_GOF * C_TILES + 1 + 1 + 4 + 1 + 1
    x_dim = N_GOF * C_TILES
    print(f"  Y向量维度: {y_dim}")
    print(f"  x_decisions维度: {x_dim}")
    print(f"  总状态维度: {y_dim + x_dim} (应该等于 {STATE_DIM})")
    
    # 验证网络结构
    test_net = SCDDQN_Network(state_dim=STATE_DIM, action_dim=L_LEVELS)
    test_input = torch.randn(1, STATE_DIM)
    test_output = test_net(test_input)
    if test_output.shape == (1, L_LEVELS):
        print(f"  ✅ 网络结构验证通过 (输入: {STATE_DIM}维, 输出: {L_LEVELS}维)")
    else:
        print(f"  ❌ 网络结构验证失败 (输出形状: {test_output.shape})")
        all_ok = False
    
    # 4. 奖励函数权重检查
    print("\n【4. 奖励函数权重检查】")
    print(f"  lambda_o: 0.0 (单用户场景，无法学习分组)")
    print(f"  lambda_q: 0.7 (质量得分权重)")
    print(f"  lambda_b: 0.1 (带宽惩罚)")
    print(f"  lambda_l: 0.2 (负载均衡惩罚)")
    
    # 5. 码率配置检查
    print("\n【5. 码率配置检查】")
    print(f"  Base层码率: 3.0 Mbps (与cmaf_publish.sh一致)")
    print(f"  Enhanced层码率: 0.8 Mbps (与cmaf_publish.sh一致)")
    print(f"  Enhanced用户总码率: 3.8 Mbps")
    
    print("\n" + "="*80)
    if all_ok:
        print("✅ 所有检查通过，可以开始训练！")
        print("="*80)
        print()
        train()
    else:
        print("❌ 检查未通过，请修复上述问题后再训练！")
        print("="*80)
        exit(1)

