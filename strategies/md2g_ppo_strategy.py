#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MD2G PPO 策略
基于现有的 PPO agent 实现
"""
import torch
import numpy as np
from typing import List, Dict, Any
from .strategy_base import StrategyBase
from ppo_agent import ActorNet
from state_builder import (
    MAX_USERS
)
from collections import Counter


class MD2G_PPO_Strategy(StrategyBase):
    """
    MD2G PPO 策略
    使用训练好的 PPO 模型进行决策
    """
    
    def __init__(self, max_users: int, model_path: str, device: str = "cpu"):
        """
        Args:
            max_users: 最大用户数
            model_path: PPO 模型文件路径
            device: 计算设备（"cpu" 或 "cuda"）
        """
        super().__init__(max_users)
        self.model_path = model_path
        self.device = torch.device(device)
        
        # 状态维度：训练时使用的是 max_users * 3 + 1 = 301（旧格式）
        # 但为了兼容，我们使用新的状态构建方式，然后转换为旧格式
        self.state_dim = MAX_USERS * 3 + 1  # 301（训练时的维度）
        
        # 加载模型
        self.actor = self._load_model()
    
    def _load_model(self):
        """加载 PPO 模型"""
        import os
        
        if not os.path.exists(self.model_path):
            print(f"⚠️ Warning: Model file not found at {self.model_path}")
            return None
        
        try:
            print(f"[MD2G_PPO] Loading model from: {self.model_path}")
            
            # 加载模型文件
            state = torch.load(self.model_path, map_location=self.device)
            
            # 根据模型键名判断是 Student (128) 还是 Teacher (512)
            first_key = list(state.keys())[0]
            if first_key.startswith("net."):
                hidden_size = 512  # Teacher 模型
                print(f"[MD2G_PPO] Detected Teacher model (hidden_size=512)")
            else:
                hidden_size = 128  # Student 模型
                print(f"[MD2G_PPO] Detected Student model (hidden_size=128)")
            
            actor = ActorNet(self.state_dim, MAX_USERS, hidden_size).to(self.device)
            
            try:
                actor.load_state_dict(state)
            except RuntimeError as e:
                if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                    from collections import OrderedDict
                    fixed = OrderedDict()
                    
                    # 判断需要转换的方向
                    needs_prefix = any(not k.startswith("net.") and k[0].isdigit() for k in state.keys())
                    has_prefix = any(k.startswith("net.") for k in state.keys())
                    
                    if needs_prefix and not has_prefix:
                        # 添加 "net." 前缀
                        for k, v in state.items():
                            if k[0].isdigit():
                                fixed[f"net.{k}"] = v
                            else:
                                fixed[k] = v
                        actor.load_state_dict(fixed)
                        print("[MD2G_PPO] Auto-fixed state_dict keys (added net. prefix)")
                    elif has_prefix:
                        # 移除 "net." 前缀
                        for k, v in state.items():
                            if k.startswith("net."):
                                fixed[k[4:]] = v
                            else:
                                fixed[k] = v
                        actor.load_state_dict(fixed)
                        print("[MD2G_PPO] Auto-fixed state_dict keys (removed net. prefix)")
                    else:
                        raise RuntimeError(f"Cannot map model keys")
                else:
                    raise
            
            actor.eval()
            print(f"✅ [MD2G_PPO] Model loaded successfully")
            return actor
            
        except Exception as e:
            print(f"❌ [MD2G_PPO] Error loading model: {e}")
            return None
    
    def _determine_dominant_network(self, user_metrics: List[Dict]) -> str:
        """确定主导网络类型"""
        if not user_metrics:
            return "wifi"
        network_types = [u.get("network_type", "wifi") for u in user_metrics]
        most_common = Counter(network_types).most_common(1)
        return most_common[0][0] if most_common else "wifi"
    
    def _build_legacy_state(self, user_metrics: List[Dict]) -> np.ndarray:
        """
        构建旧格式的状态向量（训练时使用的格式）
        格式：每个用户 3 个特征 [priority, last_layer, device_score] + 1 个全局特征 [capacity]
        """
        training_max_users = MAX_USERS  # 100
        
        # 构建用户特征
        user_features = np.zeros(training_max_users * 3)
        all_throughputs = []
        
        for i, user in enumerate(user_metrics):
            if i >= training_max_users:
                break
            base_idx = i * 3
            priority = 0.5  # 默认优先级（旧格式需要）
            last_layer = 0.0  # 默认（旧格式需要）
            device_score = float(user.get("device_score", 0.5))
            user_features[base_idx] = priority
            user_features[base_idx + 1] = last_layer
            user_features[base_idx + 2] = device_score
            all_throughputs.append(user.get("bandwidth", 0.0))
        
        # 全局特征：平均带宽（capacity），归一化
        capacity = np.mean(all_throughputs) if all_throughputs else 0.0
        normalized_capacity = capacity / 2000.0
        
        # 组合状态向量：user_features (300) + normalized_capacity (1) = 301
        obs = np.concatenate([user_features, [normalized_capacity]]).astype(np.float32)
        
        return obs
    
    def compute_actions(self, state: Dict[str, Any]) -> List[int]:
        """
        使用 PPO 模型计算动作
        """
        if self.actor is None:
            print("⚠️ [MD2G_PPO] Model not loaded, returning default actions")
            return [0] * self.max_users
        
        user_metrics = state.get("user_metrics", [])
        if not user_metrics:
            return [0] * self.max_users
        
        # 构建旧格式的状态向量（匹配训练时的维度）
        obs = self._build_legacy_state(user_metrics)
        
        # 验证维度
        if obs.shape[0] != self.state_dim:
            print(f"❌ [MD2G_PPO] Obs dimension mismatch: expected {self.state_dim}, got {obs.shape[0]}")
            return [0] * self.max_users
        
        # PPO 推理
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.actor(obs_tensor)
        
        # 动作解码：logits -> action
        action_np_full = (logits > 0).float().squeeze(0).cpu().numpy()
        
        # 只使用前 len(user_metrics) 个动作（实际用户数）
        num_actual_users = len(user_metrics)
        action_np = action_np_full[:num_actual_users] if num_actual_users <= len(action_np_full) else action_np_full
        
        # 扩展到 max_users
        actions = [0] * self.max_users
        for i in range(min(len(action_np), self.max_users)):
            actions[i] = int(action_np[i])
        
        return actions

