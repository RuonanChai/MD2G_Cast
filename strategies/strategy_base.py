#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一策略接口
所有策略（MD2G_PPO、RollingDRL、TwoStageHeuristic、PredictiveClustering）都继承此基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class StrategyBase(ABC):
    """
    策略基类
    所有策略必须实现 compute_actions 方法
    """
    
    def __init__(self, max_users: int):
        """
        Args:
            max_users: 最大用户数
        """
        self.max_users = max_users
    
    @abstractmethod
    def compute_actions(self, state: Dict[str, Any]) -> List[int]:
        """
        根据状态计算动作（决策）
        
        Args:
            state: 状态字典，包含：
                - user_metrics: List[Dict] - 用户指标列表，每个元素包含：
                    {
                        "host_id": int,
                        "bandwidth": float,      # Mbps
                        "device_score": float,   # [0, 1]
                        "network_type": str,     # "4g"/"5g"/"wifi"/"fiber_optic"
                        "group_id": int,         # 分组编号/区域 (0-9)
                        "qoe_smooth": float,     # 可选：平滑 QoE
                    }
                - relay_load: Dict - relay 负载信息（可选）
                    {
                        "avg_load": float,
                        "var_load": float,
                        "loads": List[float]
                    }
                - federation_mode: str - "on" 或 "off"（可选）
                - cache_stats: Dict - 缓存统计信息（可选，Federation ON 时）
        
        Returns:
            List[int]: 动作列表，长度为 max_users
                - 0: 拉取 base 流
                - 1: 拉取 enhanced 流
        """
        pass
    
    def get_strategy_name(self) -> str:
        """返回策略名称"""
        return self.__class__.__name__

