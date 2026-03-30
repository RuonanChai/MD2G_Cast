# sc_ddqn_model.py
# ==============================================================================
# SC-DDQN 模型核心文件
# 严格复现论文《Toward Optimal Real-Time Volumetric Video Streaming》
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

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
                 lr=1.5e-3, gamma=0.99, buffer_size=200000, batch_size=256,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99995):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Epsilon-Greedy 策略参数 (论文 V.A)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
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

    def select_action(self, state, training=True):
        """
        选择动作 (论文 Algo.2: epsilon-greedy)
        state: 当前状态 S_m
        training: 是否在训练模式（影响epsilon使用）
        """
        if training and random.random() < self.epsilon:
            # 探索: 随机选择
            return random.randrange(self.action_dim)
        else:
            # 利用: 选择 Q 值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """更新目标网络 (论文 Algo.2: 定期更新)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

