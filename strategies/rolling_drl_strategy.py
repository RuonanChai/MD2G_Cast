#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RollingDRL 策略（简化版）
基于 Li et al. TCSVT 2023 的滚动优化和深度强化学习方法
"""
import os
import sys
import numpy as np
from typing import List, Dict, Any

# ✅ 【关键修复】支持相对导入和绝对导入（兼容直接运行和模块导入）
try:
    from .strategy_base import StrategyBase
except ImportError:
    # 如果相对导入失败（文件被直接运行），尝试绝对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from strategies.strategy_base import StrategyBase


class RollingDRLStrategy(StrategyBase):
    """
    RollingDRL 策略（简化版）
    根据带宽预测和滚动优化决定是否拉取 enhanced 流
    """
    
    def __init__(self, max_users: int, bandwidth_threshold: float = 8.0):
        """
        Args:
            max_users: 最大用户数
            bandwidth_threshold: 带宽阈值（Mbps），超过此值才拉取 enhanced
        """
        super().__init__(max_users)
        self.bandwidth_threshold = bandwidth_threshold
    
    def compute_actions(self, state: Dict[str, Any]) -> List[int]:
        """
        基于带宽的简单决策策略
        """
        user_metrics = state.get("user_metrics", [])
        if not user_metrics:
            return [0] * self.max_users
        
        actions = [0] * self.max_users
        
        for i, user in enumerate(user_metrics):
            if i >= self.max_users:
                break
            
            bandwidth = user.get("bandwidth", 0.0)  # Mbps
            device_score = user.get("device_score", 0.5)
            
            # 决策逻辑：带宽 > 阈值 且 设备评分 > 0.5 时拉取 enhanced
            if bandwidth > self.bandwidth_threshold and device_score > 0.5:
                actions[i] = 1
            else:
                actions[i] = 0
        
        return actions

