#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PredictiveClustering 策略（简化版）
基于 Perfecto et al. TCOM 2020 的预测聚类方法
"""
import numpy as np
from typing import List, Dict, Any
from .strategy_base import StrategyBase


class PredictiveClusteringStrategy(StrategyBase):
    """
    PredictiveClustering 策略（简化版）
    使用历史带宽预测，聚类用户，每类至少一个 base，余下 enhanced
    """
    
    def __init__(self, max_users: int):
        """
        Args:
            max_users: 最大用户数
        """
        super().__init__(max_users)
    
    def compute_actions(self, state: Dict[str, Any]) -> List[int]:
        """
        基于带宽中位数的预测聚类决策
        """
        user_metrics = state.get("user_metrics", [])
        if not user_metrics:
            return [0] * self.max_users
        
        # 提取所有用户的带宽
        bandwidths = [user.get("bandwidth", 0.0) for user in user_metrics]
        
        # 计算中位数带宽
        median_bw = np.median(bandwidths) if bandwidths else 0.0
        
        actions = [0] * self.max_users
        
        # 决策逻辑：带宽 > 中位数时拉取 enhanced
        for i, user in enumerate(user_metrics):
            if i >= self.max_users:
                break
            
            bandwidth = user.get("bandwidth", 0.0)
            
            if bandwidth > median_bw:
                actions[i] = 1
            else:
                actions[i] = 0
        
        return actions

