# strategies/__init__.py
"""
策略模块
包含统一策略系统和增强版策略
"""

# ==============================================================================
# 新统一策略系统（推荐使用）
# ==============================================================================
from .strategy_base import StrategyBase
from .md2g_ppo_strategy import MD2G_PPO_Strategy
from .rolling_drl_strategy import RollingDRLStrategy
from .two_stage_heuristic_strategy import TwoStageHeuristicStrategy
from .predictive_clustering_strategy import PredictiveClusteringStrategy

# ==============================================================================
# 旧增强版策略（向后兼容）
# ==============================================================================
try:
    from .rolling_drl_strategy_v2_refined import RollingDRLStrategy as RollingDRLStrategyV2
    from .heuristic_controller_v2_refined import run_enhanced_heuristic_controller
    from .predictive_controller_v2_refined import run_enhanced_predictive_controller
    _has_old_strategies = True
except ImportError:
    _has_old_strategies = False

# ==============================================================================
# 统一分组和优化框架（可选）
# ==============================================================================
try:
    from .unified_user_grouping import create_unified_grouping_manager, get_user_groups, get_grouping_statistics
    from .complex_optimization_framework import create_complex_optimization_framework, create_strategy_enhancer
    _has_frameworks = True
except ImportError:
    _has_frameworks = False

# ==============================================================================
# 导出列表
# ==============================================================================
__all__ = [
    # 新统一策略系统
    'StrategyBase',
    'MD2G_PPO_Strategy',
    'RollingDRLStrategy',
    'TwoStageHeuristicStrategy',
    'PredictiveClusteringStrategy',
]

# 添加旧策略（如果存在）
if _has_old_strategies:
    __all__.extend([
        'RollingDRLStrategyV2',
        'run_enhanced_heuristic_controller',
        'run_enhanced_predictive_controller',
    ])

# 添加框架（如果存在）
if _has_frameworks:
    __all__.extend([
        'create_unified_grouping_manager',
        'get_user_groups',
        'get_grouping_statistics',
        'create_complex_optimization_framework',
        'create_strategy_enhancer',
    ])
