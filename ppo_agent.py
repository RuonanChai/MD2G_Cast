# ppo_agent.py (高效率、简化版)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

class ActorNet(nn.Module):
    """
    MD2G Actor Network with dual heads:
    - group_head: outputs group assignment for each user (K classes)
    - enhanced_head: outputs enhanced decision for each user (0/1)
    """
    def __init__(self, state_dim, num_users, num_groups=3, hidden_size=128, freeze_enhanced=False):
        super().__init__()
        self.num_users = num_users
        self.num_groups = num_groups
        self.freeze_enhanced = freeze_enhanced
        
        # 共享特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # 分组head：每用户输出K个logits（K=num_groups）
        self.group_head = nn.Linear(hidden_size, num_users * num_groups)
        
        # Enhanced head：每用户输出1个logit（enhanced决策，0/1）
        self.enhanced_head = nn.Linear(hidden_size, num_users)
    
    def forward(self, state):
        """
        Args:
            state: [B, state_dim]
        Returns:
            group_logits: [B, num_users, num_groups]
            enhanced_logits: [B, num_users]
        """
        shared = self.shared_net(state)
        B = shared.size(0)
        
        # ✅ 【修正】使用shared.size(0)明确batch维度
        group_logits = self.group_head(shared).view(B, self.num_users, self.num_groups)
        enhanced_logits = self.enhanced_head(shared).view(B, self.num_users)
        
        # 如果冻结enhanced head，返回零logits（训练时不会更新）
        if self.freeze_enhanced:
            enhanced_logits = torch.zeros_like(enhanced_logits)
        
        return group_logits, enhanced_logits

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.net(state)
    
class PPOAgent:
    """
    MD2G PPO Agent with grouping support.
    - 支持分组学习（group_id）和增强层决策（enhanced）
    - 动作空间：group_id[u] ∈ {0..K-1}, enhanced[u] ∈ {0,1}
    """
    def __init__(self, obs_dim, num_users, num_groups=3,
             hidden_size=128, freeze_enhanced=False,
             lr_actor=3e-5, lr_critic=1e-4,
             device="cpu", gamma=0.99, lam=0.95,
             clip_ratio=0.2, ppo_epochs=10, entropy_coef=0.01,
             vf_coef=0.5, distill_coef=0.1):

        self.device = torch.device(device)
        self.num_users = num_users
        self.num_groups = num_groups
        self.gamma, self.lam, self.clip_ratio = gamma, lam, clip_ratio
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.distill_coef = distill_coef  # 知识蒸馏系数

        self.actor = ActorNet(obs_dim, num_users, num_groups, hidden_size, freeze_enhanced).to(self.device)
        self.critic = CriticNet(obs_dim, hidden_size).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.clear_memory()

    def clear_memory(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []
    

    def store_transition(self, state, action, log_prob, reward, done, value):
        """
        存储转换（支持dict格式的action）
        
        Args:
            state: 状态向量
            action: dict with 'group_id' and 'enhanced', 或 tuple of tensors
            log_prob: 联合log概率
            reward: 奖励
            done: 是否结束
            value: 状态价值
        """
        # ✅ 【修复】确保 state 是 tensor（如果是 numpy 数组则转换）
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        self.states.append(state)
        
        # ✅ 【支持dict格式action】如果是dict，转换为tuple存储；如果是tuple，直接存储
        if isinstance(action, dict):
            # 转换为tuple格式存储（tensor格式）
            group_action = torch.LongTensor(action['group_id']).to(self.device)
            enhanced_action = torch.FloatTensor(action['enhanced']).to(self.device)
            self.actions.append((group_action, enhanced_action))
        else:
            # 已经是tuple格式
            self.actions.append(action)
        
        # log_prob应该是标量
        if isinstance(log_prob, torch.Tensor):
            self.log_probs.append(log_prob)
        else:
            self.log_probs.append(torch.tensor(log_prob, dtype=torch.float32, device=self.device))
        
        self.rewards.append(torch.tensor([reward], dtype=torch.float32, device=self.device))
        self.dones.append(torch.tensor([done], dtype=torch.float32, device=self.device))
        self.values.append(value)

    def select_action(self, obs, deterministic=False):
        """
        选择动作：分组 + 增强层决策
        
        Returns:
            action: dict with 'group_id' [num_users] and 'enhanced' [num_users]
            action_tensor: tuple of (group_action_tensor, enhanced_action_tensor) for buffer storage
            log_prob: 联合log概率（group + enhanced）
            value: 状态价值
        """
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            group_logits, enhanced_logits = self.actor(state)
            value = self.critic(state)
        
        # ✅ 【修正】统一shape处理
        gl = group_logits.squeeze(0)  # [U, K]
        el = enhanced_logits.squeeze(0)  # [U]
        
        # 分组：每用户K类Categorical分布
        group_dist = Categorical(logits=gl)  # [U, K]
        if deterministic:
            group_action = gl.argmax(dim=-1)  # [U]
        else:
            group_action = group_dist.sample()  # [U]
        group_log_prob = group_dist.log_prob(group_action).sum()  # 标量
        
        # Enhanced：每用户Bernoulli分布（0/1）
        enhanced_dist = Bernoulli(logits=el)  # [U]
        if deterministic:
            # ✅ 【修正】使用sigmoid > 0.5判定
            enhanced_action = (torch.sigmoid(el) > 0.5).float()  # [U]
        else:
            enhanced_action = enhanced_dist.sample()  # [U]
        enhanced_log_prob = enhanced_dist.log_prob(enhanced_action).sum()  # 标量
        
        # 联合log概率
        total_log_prob = group_log_prob + enhanced_log_prob
        
        # 返回dict格式（用于环境交互）
        action_dict = {
            'group_id': group_action.cpu().numpy(),  # [U], 范围[0..K-1]
            'enhanced': enhanced_action.cpu().numpy()  # [U], 范围[0,1]
        }
        
        # 返回tensor格式（用于buffer存储）
        action_tensor = (group_action, enhanced_action)
        
        return action_dict, action_tensor, total_log_prob, value

    def _compute_advantages(self, last_value, last_done):
        rewards = torch.cat(self.rewards)
        dones = torch.cat(self.dones)
        values = torch.cat(self.values).squeeze() # <-- 在这里添加 .squeeze()
        
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            
        returns = advantages + values
        # 标准化Advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, teacher_logits=None):
        """
        更新策略网络
        Args:
            teacher_logits: Teacher 模型的 logits，用于知识蒸馏（可选）
        """
        if not self.states: return

        # 计算 GAE 和 Returns
        with torch.no_grad():
            # ✅ states[-1] 现在已经是 tensor（在 store_transition 中已转换）
            last_state = self.states[-1]
            if not isinstance(last_state, torch.Tensor):
                last_state = torch.FloatTensor(last_state).to(self.device)
            last_value = self.critic(last_state.unsqueeze(0).to(self.device))
        advantages, returns = self._compute_advantages(last_value, self.dones[-1])
        
        # 准备批量数据
        b_states = torch.stack(self.states).to(self.device)
        b_old_log_probs = torch.stack(self.log_probs).to(self.device)
        
        # ✅ 【支持联合动作】从buffer恢复action tuple
        b_group_actions = torch.stack([a[0] for a in self.actions]).to(self.device)  # [T, U]
        b_enhanced_actions = torch.stack([a[1] for a in self.actions]).to(self.device)  # [T, U]
        
        # PPO 迭代更新
        for _ in range(self.ppo_epochs):
            group_logits, enhanced_logits = self.actor(b_states)  # group: [T, U, K], enhanced: [T, U]
            
            # 计算新的log_prob
            group_dist = Categorical(logits=group_logits)  # [T, U, K]
            enhanced_dist = Bernoulli(logits=enhanced_logits)  # [T, U]
            
            # ✅ 计算联合log_prob
            new_group_log_prob = group_dist.log_prob(b_group_actions).sum(dim=-1)  # [T]
            new_enhanced_log_prob = enhanced_dist.log_prob(b_enhanced_actions).sum(dim=-1)  # [T]
            new_log_probs = new_group_log_prob + new_enhanced_log_prob  # [T]
            
            # 计算 Policy Ratio
            prob_ratio = torch.exp(new_log_probs - b_old_log_probs)
            
            # 计算 PPO-Clip Loss
            surr1 = prob_ratio * advantages
            surr2 = torch.clamp(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # ✅ 【知识蒸馏】如果提供了 Teacher logits，计算蒸馏损失
            distill_loss = None
            if teacher_logits is not None and self.distill_coef > 0:
                # 使用 KL divergence 或 MSE loss
                # 方法1: KL divergence (更常用)
                # ✅ 【修复】teacher_logits 应该是 enhanced_logits 的格式 [T, U]
                teacher_probs = torch.sigmoid(teacher_logits)
                student_probs = torch.sigmoid(enhanced_logits)  # ✅ 修复：使用 enhanced_logits 而不是 logits
                # KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
                # 为了避免数值不稳定，使用 log_softmax 和 softmax
                # 对于二分类，使用简化的 KL divergence
                eps = 1e-8
                teacher_probs = torch.clamp(teacher_probs, eps, 1 - eps)
                student_probs = torch.clamp(student_probs, eps, 1 - eps)
                kl_div = teacher_probs * torch.log(teacher_probs / student_probs) + \
                         (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs))
                distill_loss = kl_div.mean()
                
                # 方法2: MSE loss (备选，更简单)
                # distill_loss = F.mse_loss(logits, teacher_logits)
            
            # 计算 Value Loss
            new_values = self.critic(b_states).squeeze()
            value_loss = F.mse_loss(new_values, returns)
            
            # 计算 Entropy Loss（分组和增强的联合熵）
            group_entropy = group_dist.entropy().mean()  # 分组熵
            enhanced_entropy = enhanced_dist.entropy().mean()  # 增强熵
            entropy_loss = group_entropy + enhanced_entropy  # 联合熵
            
            # 更新 Actor（包含蒸馏损失）
            actor_loss = policy_loss - self.entropy_coef * entropy_loss
            if distill_loss is not None:
                actor_loss = actor_loss + self.distill_coef * distill_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
            
            # 更新 Critic
            critic_loss = self.vf_coef * value_loss
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.optimizer_critic.step()
        
        MD2GAgent = PPOAgent

def canonicalize_groups(group_assignments, device_scores, num_groups):
    """
    Canonicalization: 按组平均device_score重排组号，避免标签交换问题
    
    Args:
        group_assignments: [num_users] - 原始组分配（0..K-1）
        device_scores: [num_users] - 设备性能分数
        num_groups: 组数K
    
    Returns:
        canonical_groups: [num_users] - 规范化后的组分配（0..K-1）
        remap_dict: {old_group: new_group} - 组号重映射字典
    """
    if isinstance(group_assignments, torch.Tensor):
        group_assignments = group_assignments.cpu().numpy()
    if isinstance(device_scores, torch.Tensor):
        device_scores = device_scores.cpu().numpy()
    
    num_users = len(group_assignments)
    num_device_scores = len(device_scores)
    
    # ✅ 【关键修复】确保索引不越界：只使用有效的用户索引
    # 如果group_assignments的长度大于device_scores，只处理前num_device_scores个用户
    valid_users = min(num_users, num_device_scores)
    
    # 计算每组平均device_score
    group_avg_scores = {}
    for g in range(num_groups):
        group_members = [u for u in range(valid_users) if group_assignments[u] == g]
        if group_members:
            # ✅ 【关键修复】确保u在device_scores范围内
            valid_members = [u for u in group_members if u < num_device_scores]
            if valid_members:
                group_avg_scores[g] = np.mean([device_scores[u] for u in valid_members])
            else:
                group_avg_scores[g] = -1.0  # 没有有效成员，排到最后
        else:
            group_avg_scores[g] = -1.0  # 空组，排到最后
    
    # 按平均score排序，重映射组号
    sorted_groups = sorted(group_avg_scores.items(), key=lambda x: x[1])
    remap_dict = {old_g: new_g for new_g, (old_g, _) in enumerate(sorted_groups)}
    
    # 重映射
    canonical_groups = np.array([remap_dict[g] for g in group_assignments], dtype=np.int32)
    
    return canonical_groups, remap_dict

def interpret_md2g_action(action_dict, num_users):
    """
    解释MD2G动作（向后兼容）
    
    Args:
        action_dict: dict with 'group_id' and 'enhanced'
        num_users: 用户数
    
    Returns:
        dict with 'groups' and 'layers' (向后兼容格式)
    """
    if isinstance(action_dict, dict):
        return {
            "groups": action_dict['group_id'].astype(int),
            "layers": action_dict['enhanced'].astype(int)
        }
    else:
        # 向后兼容：如果是旧格式，返回默认值
        return {
            "groups": np.zeros(num_users, dtype=int),
            "layers": np.zeros(num_users, dtype=int)
        }
