#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwoStageHeuristic 策略（简化版）
基于 Chen et al. TCOM 2021 的两阶段启发式方法
"""
import numpy as np
from typing import List, Dict, Any
from .strategy_base import StrategyBase


class TwoStageHeuristicStrategy(StrategyBase):
    """
    TwoStageHeuristic 策略（简化版）
    两阶段决策：
    1. 用户分组（基于设备评分）
    2. 每组中找最低能力用户作为主导，决定整组的版本选择
    """
    
    def __init__(self, max_users: int, device_threshold: float = 0.5, bandwidth_threshold: float = 10.0):
        """
        Args:
            max_users: 最大用户数
            device_threshold: 设备评分阈值
            bandwidth_threshold: 带宽阈值（Mbps）
        """
        super().__init__(max_users)
        self.device_threshold = device_threshold
        self.bandwidth_threshold = bandwidth_threshold
    
    def compute_actions(self, state: Dict[str, Any]) -> List[int]:
        """
        两阶段启发式决策
        """
        user_metrics = state.get("user_metrics", [])
        if not user_metrics:
            return [0] * self.max_users
        
        actions = [0] * self.max_users
        
        # 阶段1：用户分组（简化版：直接按设备评分分组）
        # 阶段2：为每个用户独立决策（简化版）
        for i, user in enumerate(user_metrics):
            if i >= self.max_users:
                break
            
            device_score = user.get("device_score", 0.5)
            bandwidth = user.get("bandwidth", 0.0)  # Mbps
            
            # 决策逻辑：
            # - 如果设备评分 < 阈值，拉取 base
            # - 如果设备评分 >= 阈值 且 带宽 > 阈值，拉取 enhanced
            if device_score < self.device_threshold:
                actions[i] = 0
            else:
                if bandwidth > self.bandwidth_threshold:
                    actions[i] = 1
                else:
                    actions[i] = 0
        
        return actions

