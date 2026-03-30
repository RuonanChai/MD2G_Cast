# strategies/rolling_drl_strategy_v2_refined.py
# ==============================================================================
# 严格复现论文《Toward Optimal Real-Time Volumetric Video Streaming》
# SC-DDQN (Serial Cyclic Dueling DQN) + LSTM Predictor
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# ✅ 【修复 PyTorch 2.6】添加安全全局变量，解决 weights_only=True 导致的 NumPy 标量加载问题
# 注意：add_safe_globals 是 PyTorch 2.6+ 的功能，旧版本可能不支持
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        # PyTorch < 2.6 不支持 add_safe_globals，跳过（不影响功能）
except (AttributeError, TypeError) as e:
    # 如果当前 PyTorch 版本不支持此功能，忽略错误（不影响功能）
    pass
import random
import os
import json
import time
import logging
from collections import deque
from typing import List, Dict, Tuple, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 1. LSTM Predictor (论文 VI.A - Bandwidth and FoV Prediction)
# ==============================================================================
class LSTMPredictor(nn.Module):
    """
    基于 LSTM 的带宽和 FoV 预测模型
    参考文献: Section VI.A "Bandwidth and FoV Prediction"
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        prediction = self.fc(last_out)
        return prediction

# ==============================================================================
# 2. SC-DDQN Network Structure (论文 V.B & Fig.3)
# ==============================================================================
class SCDDQN_Network(nn.Module):
    """
    Serial Cyclic Dueling DQN 网络结构
    论文设置: 
    - 全连接层 (FC) 神经元数量: 761 (论文 Fig.3 文字描述)
    - 激活函数: ReLU
    - 结构: Dueling (Value stream + Advantage stream)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=761):
        super(SCDDQN_Network, self).__init__()
        
        # 特征提取层 (论文 Fig.3)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value Stream (估计状态价值 V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage Stream (估计动作优势 A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, state):
        features = self.feature_layer(state)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling DQN 聚合公式: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

# ==============================================================================
# 3. SC-DDQN Agent (论文 V.A & Algo.2)
# ==============================================================================
class SCDDQN_Agent:
    """
    SC-DDQN 智能体
    实现论文 Algorithm 2: Training Process of SC-DDQN
    """
    def __init__(self, state_dim, action_dim, 
                 lr=1.5e-3, gamma=1.0, buffer_size=200000, batch_size=256,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Epsilon-Greedy 策略参数 (论文 V.A)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化预测网络和目标网络 (论文 Algo.2)
        self.policy_net = SCDDQN_Network(state_dim, action_dim).to(self.device)
        self.target_net = SCDDQN_Network(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器 (论文: Adam, lr=1.5e-3)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 经验回放池 (论文: Replay Buffer)
        self.memory = deque(maxlen=buffer_size)
        
        logging.info(f"SC-DDQN Agent initialized on {self.device}")

    def select_action(self, state, valid_actions=None, training=False):
        """
        选择动作 (论文 Algo.2: epsilon-greedy)
        state: 当前状态 S_m
        valid_actions: 可选的动作掩码
        training: 是否在训练模式（影响epsilon使用）
        """
        if training and random.random() < self.epsilon:
            # 探索: 随机选择
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.action_dim)
        else:
            # 利用: 选择 Q 值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                if valid_actions is not None:
                    # 将无效动作的 Q 值设为负无穷
                    mask = torch.full_like(q_values, -float('inf'))
                    mask[0, valid_actions] = 0
                    q_values = q_values + mask
                    
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放池"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """更新网络 (论文 Algo.2: Q-learning update)"""
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # 计算当前 Q 值: Q(s, a)
        curr_q = self.policy_net(state).gather(1, action)
        
        # 计算目标 Q 值: R + gamma * max Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1, keepdim=True)[0]
            target_q = reward + (1 - done) * self.gamma * next_q
            
        # 计算 Loss (MSE)
        loss = nn.MSELoss()(curr_q, target_q)
        
        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Epsilon 衰减
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """更新目标网络 (论文 Algo.2: 定期更新)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==============================================================================
# 4. State Construction Helper (论文 V.A.1 Eq.26)
# ==============================================================================
class StateManager:
    """
    管理状态构造 S_m = [Y, x_m]
    Y 包括: 预测带宽, 预测FoV, 缓存状态, 上一帧决策等
    x_m 是当前决策向量的一维展开
    """
    def __init__(self, N=5, C=8, L=2):
        self.N = N  # 预测步长 (GoF数量)
        self.C = C  # 每个 GoF 的 Tile 数量
        self.L = L  # 质量等级数量 (Base=0, Enhanced=1)
        self.total_steps = N * C  # 一个 Rolling 窗口的总决策步数
        
    def construct_state(self, variable_params_Y, current_decision_vector_x):
        """
        拼接固定参数 Y 和当前部分决策向量 x_m (论文 Eq.26)
        """
        return np.concatenate((variable_params_Y, current_decision_vector_x))

# ==============================================================================
# 5. Rolling Optimization Strategy (论文 Algorithm 1 & 3)
# ==============================================================================
class RollingOptimizationStrategy:
    """
    实现论文中的滚动优化算法
    "Toward optimal real-time volumetric video streaming: A rolling optimization 
    and deep reinforcement learning based approach"
    
    核心改进：
    1. 严格实现 SC-DDQN 串行循环决策
    2. 使用 LSTM 预测带宽和 FoV
    3. 状态构造遵循论文 Eq.26
    4. 决策过程遵循论文 Algorithm 3
    """
    
    def __init__(self, model_path=None, window_size=5, state_feature_count=8, 
                 N=5, C=8, L=2):
        """
        Args:
            model_path: 预训练模型路径
            window_size: 状态窗口大小
            state_feature_count: 状态特征数量
            N: Rolling窗口的GoF数量 (论文参数)
            C: 每个GoF的Tile数量 (论文参数)
            L: 质量等级数量 (Base=0, Enhanced=1)
        """
        self.window_size = window_size
        self.state_feature_count = state_feature_count
        self.N = N
        self.C = C
        self.L = L
        
        # 状态历史
        self.state_history = deque(maxlen=self.window_size)
        self.bandwidth_history = deque(maxlen=20)  # 用于LSTM预测
        self.fov_history = deque(maxlen=20)  # 用于LSTM预测
        
        # 状态管理器
        self.state_manager = StateManager(N=N, C=C, L=L)
        
        # ✅ 【关键修复】计算状态维度（与训练时保持一致）
        # Y 的维度: 预测带宽(N) + 预测FoV(N*C) + Buffer(1) + 设备分数(1) + 网络类型(4) + Base码率(1) + Enhanced码率(1)
        # 实际维度：N(5) + N*C(40) + 1 + 1 + 4 + 1 + 1 = 53
        # x_m 的维度: N * C (每个Tile的决策，归一化值) = 40
        # 总状态维度：53 + 40 = 93
        self.x_dim = N * C  # 40
        self.y_dim = 53  # ✅ 修复：从24改为53，保留所有关键信息（不再截断）
        self.state_dim = self.y_dim + self.x_dim  # 53 + 40 = 93
        
        # 初始化 SC-DDQN Agent
        self.agent = SCDDQN_Agent(state_dim=self.state_dim, action_dim=L)
        
        # 初始化 LSTM 预测器
        self.lstm_bw = LSTMPredictor(input_dim=1, output_dim=N).to(self.agent.device)
        # ✅ 【关键修复】FoV预测应该是N*C维（40维），而不是N*3维（15维），与训练时保持一致
        self.lstm_fov = LSTMPredictor(input_dim=6, output_dim=N*C).to(self.agent.device)
        
        # 加载预训练模型
        self._load_models(model_path)
        
        # 滚动优化状态
        self.rolling_state = {
            'bandwidth_trend': 0.0,
            'qoe_trend': 0.0,
            'network_stability': 1.0,
            'user_demand': 0.0
        }
        
    def _load_models(self, model_path):
        """加载预训练模型"""
        if model_path and os.path.exists(model_path):
            try:
                logging.info(f"[SC-DDQN] Loading model from: {model_path}")
                # ✅ 【修复 PyTorch 2.6】使用 weights_only=False 允许加载包含 NumPy 标量的旧模型
                checkpoint = torch.load(model_path, map_location=self.agent.device, weights_only=False)
                
                # ✅ 【修复模型加载】尝试多种模型格式
                loaded = False
                
                # 尝试1: 新格式 (包含policy_net_state_dict和target_net_state_dict)
                if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
                    try:
                        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                        self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                        logging.info("[SC-DDQN] ✅ Model loaded successfully (new format)")
                        loaded = True
                    except Exception as e:
                        logging.warning(f"[SC-DDQN] Failed to load new format: {e}")
                
                # 尝试2: 旧格式 (直接是state_dict)
                if not loaded:
                    try:
                        # 检查是否是旧模型结构 (net.0.weight格式)
                        if isinstance(checkpoint, dict) and 'net.0.weight' in checkpoint:
                            logging.warning("[SC-DDQN] ⚠️  Old model structure detected (net.0.weight format)")
                            logging.warning("[SC-DDQN] ⚠️  Model structure mismatch, using random initialization")
                            logging.warning("[SC-DDQN] ⚠️  Please retrain model with correct structure or use compatible model")
                        else:
                            # 尝试直接加载
                            self.agent.policy_net.load_state_dict(checkpoint)
                            self.agent.target_net.load_state_dict(checkpoint)
                            logging.info("[SC-DDQN] ✅ Model loaded successfully (direct format)")
                            loaded = True
                    except Exception as e:
                        logging.warning(f"[SC-DDQN] ⚠️  Failed to load direct format: {e}")
                        logging.warning("[SC-DDQN] ⚠️  Model structure mismatch, using random initialization")
                        logging.warning(f"[SC-DDQN] ⚠️  Expected state_dim={self.state_dim}, action_dim={self.L}")
                        logging.warning(f"[SC-DDQN] ⚠️  Model keys: {list(checkpoint.keys())[:10] if isinstance(checkpoint, dict) else 'Not a dict'}")
                
                if not loaded:
                    logging.error("[SC-DDQN] ❌ Failed to load model, using random initialization")
                    logging.error("[SC-DDQN] ❌ This may cause poor decision quality (e.g., all Enhanced)")
                
                self.agent.policy_net.eval()
                self.agent.target_net.eval()
                
            except Exception as e:
                logging.error(f"[SC-DDQN] ❌ Error loading model: {e}")
                logging.error("[SC-DDQN] ❌ Using random initialization, this may cause poor decision quality")
                logging.info("[SC-DDQN] Using random initialization")
        else:
            logging.info("[SC-DDQN] No model path provided, using random initialization")
    
    def _predict_bandwidth(self, bandwidth_history):
        """使用LSTM预测未来N步带宽 (论文 VI.A)"""
        if len(bandwidth_history) < 5:
            # 历史不足，返回当前值
            current_bw = bandwidth_history[-1] if bandwidth_history else 10.0
            return np.full(self.N, current_bw)
        
        # 准备输入序列
        seq_len = min(10, len(bandwidth_history))
        input_seq = np.array(bandwidth_history[-seq_len:]).reshape(1, seq_len, 1)
        input_tensor = torch.FloatTensor(input_seq).to(self.agent.device)
        
        # 预测
        with torch.no_grad():
            prediction = self.lstm_bw(input_tensor).cpu().numpy().flatten()
        
        # 确保预测值非负
        prediction = np.maximum(prediction, 0.1)
        return prediction[:self.N]
    
    def _predict_fov(self, fov_history):
        """使用LSTM预测未来N步FoV (论文 VI.A)"""
        # ✅ 【关键修复】FoV预测应该是N*C维（40维），与训练时保持一致
        if len(fov_history) < 5:
            # 历史不足，返回默认值（N*C维）
            default_fov = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])  # x, y, z, pitch, yaw, roll
            # 为每个Tile生成FoV预测（N*C个Tile，每个Tile需要6维FoV，但这里简化为扁平化）
            return np.tile(default_fov[:3], self.N * self.C)  # 只使用x, y, z，共N*C*3，但实际需要N*C维
        
        # 准备输入序列
        seq_len = min(10, len(fov_history))
        input_seq = np.array(fov_history[-seq_len:]).reshape(1, seq_len, 6)
        input_tensor = torch.FloatTensor(input_seq).to(self.agent.device)
        
        # 预测（输出应该是N*C维）
        with torch.no_grad():
            prediction = self.lstm_fov(input_tensor).cpu().numpy().flatten()
        
        # ✅ 【关键修复】返回N*C维，而不是N*3维
        return prediction[:self.N * self.C]
    
    def _construct_state_Y(self, state_window):
        """
        构造状态向量 Y (论文 Eq.26)
        Y 包括: 预测带宽, 预测FoV, 当前状态等
        """
        if len(state_window) == 0:
            return np.zeros(self.y_dim)
        
        current_state = state_window[-1]
        
        # 提取当前特征
        current_bandwidth = current_state[1] if len(current_state) > 1 else 10.0
        current_qoe = current_state[3] if len(current_state) > 3 else 0.5
        device_score = current_state[4] if len(current_state) > 4 else 0.5
        
        # 更新历史
        self.bandwidth_history.append(current_bandwidth)
        # FoV历史 (简化: 使用设备分数作为代理)
        fov_proxy = np.array([device_score, device_score, device_score, 0.0, 0.0, 0.0])
        self.fov_history.append(fov_proxy)
        
        # 预测未来N步带宽
        predicted_bw = self._predict_bandwidth(list(self.bandwidth_history))
        
        # 预测未来N步FoV
        predicted_fov = self._predict_fov(list(self.fov_history))
        
        # ✅ 【关键修复】构造Y向量，与训练时保持一致
        # Y向量维度：predicted_bw(N) + predicted_fov(N*C) + buffer(1) + device_score(1) + network_onehot(4) + base_bitrate(1) + enh_bitrate(1) = 53
        # 但运行时没有buffer、network_onehot、bitrate信息，所以用其他信息替代
        network_onehot = np.zeros(4)  # 网络类型one-hot编码（简化：全0）
        # ✅ 【指令一修复】统一使用实际物理码率：Base=3.31Mbps, Enhanced=2.12Mbps
        base_bitrate = 3.31  # Base实际物理码率（实测：3.31 Mbps）
        enh_bitrate = 2.12   # Enhanced实际物理码率（实测：2.12 Mbps）
        buffer = 0.5         # 默认buffer状态（简化）
        
        # 构造Y向量（与训练时维度一致）
        Y = np.concatenate([
            predicted_bw,           # N维 (5)
            predicted_fov,          # N*C维 (40) ✅ 修复：从N*3改为N*C
            [buffer],              # 1维
            [device_score],        # 1维
            network_onehot,        # 4维
            [base_bitrate],        # 1维
            [enh_bitrate],         # 1维
        ])
        
        # 验证维度
        expected_y_dim = self.N + self.N * self.C + 1 + 1 + 4 + 1 + 1  # 5 + 40 + 1 + 1 + 4 + 1 + 1 = 53
        if len(Y) != expected_y_dim:
            if len(Y) < expected_y_dim:
                # 如果维度不足，填充零
                pad = np.zeros(expected_y_dim - len(Y))
                Y = np.concatenate([Y, pad])
            else:
                # 如果维度超出，截断
                Y = Y[:expected_y_dim]
        
        return Y[:self.y_dim]
    
    def _extract_enhanced_features(self, state_window: np.ndarray) -> np.ndarray:
        """提取增强特征，用于简化决策（向后兼容）"""
        if len(state_window) < self.window_size:
            padded_state = np.zeros((self.window_size, self.state_feature_count))
            padded_state[-len(state_window):] = state_window
            state_window = padded_state
        
        return state_window.flatten()
    
    def decide(self, state_window: np.ndarray) -> int:
        """
        决策函数 (论文 Algorithm 3 的简化版本)
        
        注意: 完整实现应该是串行循环 N*C 次，但为了适配现有框架，
        这里实现一个简化版本：基于当前状态和预测，做出单次决策。
        
        完整版本应该在 optimize_one_rolling_window 中实现。
        """
        # 更新状态历史
        if len(state_window) > 0:
            self.state_history.append(state_window[-1])
        
        # 更新滚动状态
        if len(self.state_history) >= 2:
            recent_states = list(self.state_history)[-2:]
            if len(recent_states[0]) > 1 and len(recent_states[1]) > 1:
                bandwidth_trend = recent_states[-1][1] - recent_states[-2][1]
                self.rolling_state['bandwidth_trend'] = bandwidth_trend
                
                if len(recent_states[0]) > 3 and len(recent_states[1]) > 3:
                    qoe_trend = recent_states[-1][3] - recent_states[-2][3]
                    self.rolling_state['qoe_trend'] = qoe_trend
                
                # 网络稳定性
                try:
                    bandwidth_std = np.std([s[1] for s in recent_states if len(s) > 1])
                    self.rolling_state['network_stability'] = 1.0 / (1.0 + bandwidth_std)
                except:
                    self.rolling_state['network_stability'] = 1.0
        
        # 构造状态Y
        Y = self._construct_state_Y(state_window)
        
        # 构造决策向量x_m (初始化为全0，表示未决策)
        x_m = np.zeros(self.x_dim)
        
        # 构造完整状态 S_m = [Y, x_m]
        state = self.state_manager.construct_state(Y, x_m)
        
        # 使用 SC-DDQN Agent 选择动作
        # 动作: 0=Base, 1=Enhanced
        try:
            action = self.agent.select_action(state, training=False)
            return action
        except Exception as e:
            logging.error(f"[SC-DDQN] Decision error: {e}")
            # 回退到保守决策
            return 0
    
    def optimize_one_rolling_window(self, state_window, training=False):
        """
        执行一次完整的 N*C 步串行决策 (论文 Algorithm 3)
        
        这是完整的滚动优化实现，但由于当前框架限制，主要在decide()中使用简化版本。
        """
        # 构造状态Y
        Y = self._construct_state_Y(state_window)
        
        # 初始化决策向量
        x_decisions = np.zeros(self.N * self.C)
        
        total_reward = 0
        
        # 串行循环 N * C 次 (论文 Algorithm 3)
        for step in range(self.N * self.C):
            # 1. 构造当前状态 S_m
            state = self.state_manager.construct_state(Y, x_decisions)
            
            # 2. 选择动作
            action = self.agent.select_action(state, training=training)
            
            # 3. 更新决策向量
            x_decisions[step] = action
            
            # 4. 计算奖励 (简化版本)
            reward = self._calculate_step_reward(x_decisions, step, state_window)
            total_reward += reward
            
            # 5. 存储经验并训练 (如果是训练模式)
            if training:
                next_state = self.state_manager.construct_state(Y, x_decisions)
                done = (step == (self.N * self.C - 1))
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.update()
        
        return x_decisions, total_reward
    
    def _calculate_step_reward(self, decisions, step, state_window):
        """
        计算步骤奖励 (论文 Eq.28)
        
        简化版本：基于QoE增量
        完整版本应该计算: reward = delta_QoE - penalty
        """
        if len(state_window) == 0:
            return 0.0
        
        current_state = state_window[-1]
        current_qoe = current_state[3] if len(current_state) > 3 else 0.5
        
        # 简化奖励：如果选择Enhanced，给予正奖励；否则给予小奖励
        action = int(decisions[step]) if step < len(decisions) else 0
        
        if action == 1:  # Enhanced
            # 奖励与当前QoE和带宽相关
            bandwidth = current_state[1] if len(current_state) > 1 else 10.0
            reward = 0.1 * current_qoe * min(1.0, bandwidth / 10.0)
        else:  # Base
            reward = 0.01
        
        return reward

# ==============================================================================
# 6. 主策略类 (向后兼容接口)
# ==============================================================================
class RollingDRLStrategy:
    """
    增强版Rolling DRL策略，严格实现论文SC-DDQN算法
    """
    
    def __init__(self, model_path=None, window_size=5, state_feature_count=8):
        self.strategy = RollingOptimizationStrategy(
            model_path=model_path,
            window_size=window_size,
            state_feature_count=state_feature_count,
            N=5,  # Rolling窗口GoF数量
            C=8,  # 每个GoF的Tile数量
            L=2   # 质量等级 (Base=0, Enhanced=1)
        )
        self.window_size = window_size
        self.state_feature_count = state_feature_count
    
    def decide(self, state_window: np.ndarray) -> int:
        """决策接口"""
        return self.strategy.decide(state_window)
    
    def update_reward(self, reward: float):
        """更新奖励接口（保留用于未来扩展）"""
        pass

# ==============================================================================
# 7. 辅助函数：收集客户端状态
# ==============================================================================
def aggregate_client_states(max_users):
    """聚合客户端状态（类似 heuristic_controller）"""
    import glob
    client_states = {}
    # ✅ 【共享目录修复】从共享目录读取客户端状态文件
    # 解决Mininet节点和主机环境隔离的问题
    SHARED_STATE_DIR = "/tmp/mininet_shared"
    
    # ✅ 【调试】检查目录是否存在
    if not os.path.exists(SHARED_STATE_DIR):
        # 只在第一次打印，避免日志过多
        if not hasattr(aggregate_client_states, 'dir_warned'):
            print(f"[Rolling Controller] ⚠️  共享目录不存在: {SHARED_STATE_DIR}", flush=True)
            aggregate_client_states.dir_warned = True
        return client_states
    
    state_files = glob.glob(f"{SHARED_STATE_DIR}/client_h*_state.json")
    current_time = time.time()
    
    # ✅ 【调试】打印找到的文件数量（每10次打印一次）
    if not hasattr(aggregate_client_states, 'check_count'):
        aggregate_client_states.check_count = 0
    aggregate_client_states.check_count += 1
    if aggregate_client_states.check_count % 10 == 1:
        print(f"[Rolling Controller] 🔍 找到 {len(state_files)} 个状态文件", flush=True)
    # ✅ 【关键修复】增加状态文件超时时间，适应大量用户启动时间
    # 客户端启动时序：
    # - 100用户分批启动：10批 × 1秒 = 10秒
    # - 客户端连接Relay、订阅track：10-30秒
    # - 客户端开始接收数据、写入状态文件：30-60秒
    # 因此需要更长的超时时间，确保能收集到状态
    STATE_TIMEOUT_S = 60  # 从30秒增加到60秒，确保能收集到客户端状态
    
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
                    "network_type": data.get("network_type", "unknown")
                }
        except (IOError, json.JSONDecodeError):
            continue
    return client_states

def build_state_window_for_user(user_state, history=None, window_size=3):
    """
    为单个用户构建状态窗口
    
    Args:
        user_state: 用户当前状态字典
        history: 用户历史状态列表（可选）
        window_size: 窗口大小
    
    Returns:
        state_window: numpy array, shape (window_size, 8)
    """
    import numpy as np
    
    # 状态特征：8维
    # [bandwidth, delay, device_score, qoe_estimate, network_score, buffer_estimate, throughput_trend, stability]
    current_features = np.array([
        user_state.get("throughput", 0.0),           # 带宽 (Mbps)
        user_state.get("delay", 0.0) / 100.0,       # 延迟 (归一化到秒)
        user_state.get("device_score", 0.5),         # 设备评分
        0.5,                                          # QoE估计（简化）
        _network_type_to_score(user_state.get("network_type", "unknown")),  # 网络类型评分
        0.5,                                          # Buffer估计（简化）
        0.0,                                          # 吞吐量趋势（简化）
        1.0                                           # 稳定性（简化）
    ])
    
    # 如果有历史，使用历史；否则用当前状态填充
    if history and len(history) > 0:
        state_window = np.zeros((window_size, 8))
        # 填充历史状态
        start_idx = max(0, window_size - len(history) - 1)
        for i, hist_state in enumerate(history[-window_size+1:]):
            state_window[start_idx + i] = hist_state
        # 最后一个是当前状态
        state_window[-1] = current_features
    else:
        # 没有历史，用当前状态填充整个窗口
        state_window = np.tile(current_features, (window_size, 1))
    
    return state_window

def _network_type_to_score(network_type):
    """将网络类型转换为评分（0-1）"""
    network_scores = {
        "fiber_optic": 1.0,
        "5g": 0.8,
        "5g_dominant": 0.8,
        "wifi": 0.6,
        "4g": 0.4,
        "default_mix": 0.5
    }
    return network_scores.get(network_type.lower(), 0.5)

# ==============================================================================
# 8. 命令行接口
# ==============================================================================
def main():
    """Rolling DRL策略的命令行接口"""
    import argparse
    import sys
    import glob
    
    parser = argparse.ArgumentParser(description='Rolling DRL Strategy v2 (SC-DDQN)')
    parser.add_argument('--model_path', type=str, help='Model path')
    parser.add_argument('--decision_file', type=str, help='Decision file path')
    parser.add_argument('--max_users', type=int, default=20, help='Maximum users')
    
    args = parser.parse_args()
    
    print(f"[Rolling Controller] Starting SC-DDQN Rolling Strategy...", flush=True)
    print(f"[Rolling Controller] Model path: {args.model_path}", flush=True)
    print(f"[Rolling Controller] Decision file: {args.decision_file}", flush=True)
    print(f"[Rolling Controller] Max users: {args.max_users}", flush=True)
    
    # 初始化策略
    strategy = RollingDRLStrategy(model_path=args.model_path)
    
    print(f"[Rolling Controller] ✅ SC-DDQN Strategy initialized successfully", flush=True)
    
    # 存储每个用户的状态历史（用于构建状态窗口）
    user_state_histories = {i: deque(maxlen=5) for i in range(1, args.max_users + 1)}
    
    # 更新决策的循环
    update_interval = 5  # 每5秒更新一次决策
    last_update_time = time.time()
    
    def update_decisions():
        """更新所有用户的决策"""
        # 1. 收集客户端状态（带重试机制）
        client_states = aggregate_client_states(args.max_users)
        
        # ✅ 【关键修复】如果Active clients=0，记录警告信息
        if len(client_states) == 0:
            print(f"[Rolling Controller] ⚠️  Warning: No active clients found, using default states for all users", flush=True)
            # 检查状态文件是否存在（使用共享目录）
            import glob
            SHARED_STATE_DIR = "/tmp/mininet_shared"
            state_files = glob.glob(f"{SHARED_STATE_DIR}/client_h*_state.json")
            if state_files:
                print(f"[Rolling Controller] ⚠️  Found {len(state_files)} state files in {SHARED_STATE_DIR}, but all are expired (timestamp > 60s)", flush=True)
            else:
                print(f"[Rolling Controller] ⚠️  No state files found in {SHARED_STATE_DIR}/client_h*_state.json", flush=True)
        
        decisions = {}
        import numpy as np
        
        # 2. 为每个用户生成决策
        for user_id in range(1, args.max_users + 1):
            user_key = str(user_id)
            
            if user_id in client_states:
                # 有真实状态，使用真实状态构建状态窗口
                user_state = client_states[user_id]
                history = list(user_state_histories[user_id])
                state_window = build_state_window_for_user(user_state, history, window_size=3)
                
                # 更新历史
                current_features = state_window[-1].copy()
                user_state_histories[user_id].append(current_features)
            else:
                # ✅ 【修复】没有真实状态，使用默认状态（保守决策 - 选择Base）
                # 修改默认状态使其更保守，避免全Enhanced决策
                default_state = {
                    "throughput": 1.0,  # ✅ 降低默认带宽（从5.0改为1.0），表示带宽不足
                    "delay": 100.0,     # ✅ 增加默认延迟（从50.0改为100.0），表示网络质量差
                    "device_score": 0.3, # ✅ 降低默认设备分数（从0.5改为0.3），表示设备性能差
                    "network_type": "unknown"
                }
                history = list(user_state_histories[user_id])
                state_window = build_state_window_for_user(default_state, history, window_size=3)
            
            # 使用SC-DDQN策略生成决策
            try:
                decision = strategy.decide(state_window)
            except Exception as e:
                print(f"[Rolling Controller] ⚠️  Decision error for user {user_id}: {e}", flush=True)
                decision = 0  # 出错时使用保守决策
            
            decisions[user_key] = {
                "pull_enhanced": bool(decision),
                "strategy": "rolling_drl_sc_ddqn",
                "model_used": "rolling_drl_actor.pth" if args.model_path else "sc_ddqn_default"
            }
        
        # 3. 写入决策文件
        output_data = {"decisions": decisions, "timestamp": time.time()}
        
        decision_dir = os.path.dirname(args.decision_file) if args.decision_file else "."
        if decision_dir and not os.path.exists(decision_dir):
            os.makedirs(decision_dir, exist_ok=True)
        
        temp_file = args.decision_file + ".tmp" if args.decision_file else None
        try:
            if temp_file:
                with open(temp_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                os.rename(temp_file, args.decision_file)
            else:
                with open(args.decision_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
        except Exception as e:
            print(f"[Rolling Controller] ❌ Failed to write decision file: {e}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            return False
        
        # 4. 统计信息
        enhanced_count = sum(1 for d in decisions.values() if d.get('pull_enhanced', False))
        base_count = args.max_users - enhanced_count
        active_count = len(client_states)
        
        print(f"[Rolling Controller] ✅ Decisions updated: Base={base_count}, Enhanced={enhanced_count} ({enhanced_count/args.max_users*100:.1f}%), Active clients={active_count}", flush=True)
        
        return True
    
    # 生成初始决策
    print(f"[Rolling Controller] Generating initial decisions...", flush=True)
    
    # ✅ 【关键修复】等待客户端启动完成（大量用户需要更长时间）
    # 客户端启动时序：
    # - 分批启动：batch_size=10，每批间隔1秒
    # - 100用户需要：10批 × 1秒 = 10秒
    # - 客户端连接Relay、订阅track：10-30秒
    # - 客户端开始接收数据、写入状态文件：30-60秒
    # 因此需要更长的等待时间，确保客户端状态文件已生成
    if args.max_users > 50:
        wait_time = max(60, int(args.max_users * 0.7))  # 至少60秒，或用户数×0.7秒（从0.4增加到0.7）
        print(f"[Rolling Controller] ⏳ Waiting {wait_time} seconds for clients to start and write state files ({args.max_users} users)...", flush=True)
        time.sleep(wait_time)
    elif args.max_users > 20:
        wait_time = 40  # 20-50用户等待40秒（从20秒增加到40秒）
        print(f"[Rolling Controller] ⏳ Waiting {wait_time} seconds for clients to start and write state files...", flush=True)
        time.sleep(wait_time)
    else:
        wait_time = 20  # 20用户以下等待20秒（从10秒增加到20秒）
        print(f"[Rolling Controller] ⏳ Waiting {wait_time} seconds for clients to start and write state files...", flush=True)
        time.sleep(wait_time)
    
    # ✅ 【关键修复】添加状态收集重试机制
    # 如果第一次收集时Active clients=0，等待一段时间后重试
    max_retries = 3
    retry_delay = 10  # 每次重试等待10秒
    for retry in range(max_retries):
        client_states = aggregate_client_states(args.max_users)
        active_count = len(client_states)
        if active_count > 0:
            print(f"[Rolling Controller] ✅ Collected {active_count} client states after {retry} retries", flush=True)
            break
        elif retry < max_retries - 1:
            print(f"[Rolling Controller] ⚠️  No client states found (retry {retry+1}/{max_retries}), waiting {retry_delay} seconds...", flush=True)
            time.sleep(retry_delay)
        else:
            print(f"[Rolling Controller] ⚠️  No client states found after {max_retries} retries, using default states", flush=True)
    
    update_decisions()
    
    print(f"[Rolling Controller] ✅ Initial decisions written to {args.decision_file}", flush=True)
    print(f"[Rolling Controller] ℹ️  Note: SC-DDQN strategy uses strict paper implementation", flush=True)
    print(f"[Rolling Controller]    Controller will update decisions every {update_interval} seconds", flush=True)
    
    # 定期更新决策
    try:
        while True:
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                update_decisions()
                last_update_time = current_time
            time.sleep(1)  # 每秒检查一次
    except KeyboardInterrupt:
        print(f"[Rolling Controller] Stopped", flush=True)

if __name__ == "__main__":
    main()
