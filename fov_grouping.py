#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FOV-Aware Greedy Clustering (FAGC) - 基于视口重叠的用户分组算法
Inspired by Pano's viewport-driven design philosophy

核心设计理念:
1. 视口驱动 (Viewport-driven): 基于可见内容（Visible Tiles）的重叠进行分组
2. 效用最大化 (Utility Maximization): 最大化多播增益，降低带宽消耗
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class FOVGroupManager:
    """
    实现基于FOV重叠的用户分组算法
    
    目标: 将视口重叠度高的用户分组，最大化Multicast效用
    """
    
    def __init__(self, overlap_threshold: float = 0.5):
        """
        Args:
            overlap_threshold (float): IoU阈值 (0.0 到 1.0)
                                     重叠度 > threshold 的用户将被分组
                                     值越高 = 分组越严格（每组用户数越少）
        """
        self.overlap_threshold = overlap_threshold
    
    def calculate_iou(self, tiles_a: Set[int], tiles_b: Set[int]) -> float:
        """
        计算两个Tile集合的IoU (Intersection over Union / Jaccard Index)
        
        Args:
            tiles_a: 用户A的可见Tile集合
            tiles_b: 用户B的可见Tile集合
            
        Returns:
            IoU值 (0.0 到 1.0)
        """
        if not tiles_a and not tiles_b:
            return 0.0
        
        intersection = len(tiles_a.intersection(tiles_b))
        union = len(tiles_a.union(tiles_b))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def group_users(self, user_tiles_map: Dict[str, List[int]]) -> List[List[str]]:
        """
        基于Tile重叠进行用户分组的主算法
        
        Args:
            user_tiles_map: 字典，映射 'user_id' -> [tile_id1, tile_id2, ...]
            
        Returns:
            分组列表，每个组是一个用户ID列表
            例如: [['1', '3'], ['2'], ['4', '5']]
        """
        # 1. 预处理：将列表转换为集合，便于O(1)查找
        user_sets = {uid: set(tiles) for uid, tiles in user_tiles_map.items()}
        active_users = list(user_sets.keys())
        
        # 如果只有0或1个用户，无需分组
        if len(active_users) <= 1:
            return [active_users] if active_users else []
        
        groups = []
        visited_users = set()
        
        # 2. 贪心聚类
        # 可选启发式：按可见Tile数量排序（密集视口优先）
        # active_users.sort(key=lambda u: len(user_sets[u]), reverse=True)
        
        while len(visited_users) < len(active_users):
            # 找到一个未访问的用户作为新组的种子
            seed_user = None
            for u in active_users:
                if u not in visited_users:
                    seed_user = u
                    break
            
            if seed_user is None:
                break
            
            # 用种子用户创建新组
            current_group = [seed_user]
            visited_users.add(seed_user)
            
            # 组的"代表视口"初始化为种子用户的视口
            # 选项A: 交集（保守 - 只分组匹配核心内容的用户）
            # 选项B: 并集（激进 - 分组匹配任何部分的用户）
            # 这里使用种子用户的Tile作为锚点（Pano风格的锚点）
            group_tiles = user_sets[seed_user].copy()
            
            # 尝试将其他未访问用户加入该组
            potential_members = []
            for candidate in active_users:
                if candidate not in visited_users:
                    # 计算与组锚点的重叠度
                    score = self.calculate_iou(group_tiles, user_sets[candidate])
                    if score >= self.overlap_threshold:
                        potential_members.append((candidate, score))
            
            # 按相似度排序（重叠度高的优先）
            potential_members.sort(key=lambda x: x[1], reverse=True)
            
            # 将它们加入组
            for candidate, score in potential_members:
                current_group.append(candidate)
                visited_users.add(candidate)
                # 注意：这里不动态更新group_tiles，保持"星型拓扑"分组
                # （每个人都必须与种子相似）
                # 或者，你可以更新 group_tiles = group_tiles.intersection(...)
            
            groups.append(current_group)
        
        return groups
    
    def group_users_adaptive(self, user_tiles_map: Dict[str, List[int]], 
                             strategy: str = "intersection") -> List[List[str]]:
        """
        自适应分组算法：支持动态更新组视口
        
        Args:
            user_tiles_map: 用户Tile映射
            strategy: "intersection" (保守) 或 "union" (激进)
            
        Returns:
            分组列表
        """
        user_sets = {uid: set(tiles) for uid, tiles in user_tiles_map.items()}
        active_users = list(user_sets.keys())
        
        if len(active_users) <= 1:
            return [active_users] if active_users else []
        
        groups = []
        visited_users = set()
        
        while len(visited_users) < len(active_users):
            seed_user = None
            for u in active_users:
                if u not in visited_users:
                    seed_user = u
                    break
            
            if seed_user is None:
                break
            
            current_group = [seed_user]
            visited_users.add(seed_user)
            group_tiles = user_sets[seed_user].copy()
            
            # 动态更新组视口
            changed = True
            while changed:
                changed = False
                for candidate in active_users:
                    if candidate not in visited_users:
                        score = self.calculate_iou(group_tiles, user_sets[candidate])
                        if score >= self.overlap_threshold:
                            current_group.append(candidate)
                            visited_users.add(candidate)
                            
                            # 根据策略更新组视口
                            if strategy == "intersection":
                                group_tiles = group_tiles.intersection(user_sets[candidate])
                            elif strategy == "union":
                                group_tiles = group_tiles.union(user_sets[candidate])
                            
                            changed = True
                            break
            
            groups.append(current_group)
        
        return groups
    
    def get_group_stats(self, groups: List[List[str]], 
                       user_tiles_map: Dict[str, List[int]]) -> Dict:
        """
        获取分组统计信息
        
        Args:
            groups: 分组列表
            user_tiles_map: 用户Tile映射
            
        Returns:
            统计信息字典
        """
        if not groups:
            return {
                "total_groups": 0,
                "avg_group_size": 0.0,
                "max_group_size": 0,
                "min_group_size": 0,
                "multicast_gain_estimate": 0.0
            }
        
        group_sizes = [len(g) for g in groups]
        total_users = sum(group_sizes)
        
        # 估算多播增益（简化版）
        # 假设：单播需要 N 份传输，多播需要 1 份传输
        # 增益 = (单播开销 - 多播开销) / 单播开销
        unicast_cost = total_users  # 每个用户一份
        multicast_cost = len(groups)  # 每组一份
        multicast_gain = (unicast_cost - multicast_cost) / unicast_cost if unicast_cost > 0 else 0.0
        
        stats = {
            "total_groups": len(groups),
            "avg_group_size": sum(group_sizes) / len(groups),
            "max_group_size": max(group_sizes),
            "min_group_size": min(group_sizes),
            "multicast_gain_estimate": multicast_gain,
            "total_users": total_users
        }
        
        return stats
    
    def calculate_group_overlap(self, group: List[str], 
                               user_tiles_map: Dict[str, List[int]]) -> float:
        """
        计算组内用户的平均重叠度
        
        Args:
            group: 用户组
            user_tiles_map: 用户Tile映射
            
        Returns:
            平均重叠度
        """
        if len(group) <= 1:
            return 1.0
        
        user_sets = {uid: set(tiles) for uid, tiles in user_tiles_map.items()}
        overlaps = []
        
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                uid_i, uid_j = group[i], group[j]
                if uid_i in user_sets and uid_j in user_sets:
                    overlap = self.calculate_iou(user_sets[uid_i], user_sets[uid_j])
                    overlaps.append(overlap)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0


def load_user_tiles_from_trace(trace_csv_path: str, frame_id: int = None) -> Dict[str, List[int]]:
    """
    从trace CSV文件中加载指定帧的所有用户可见Tile
    
    Args:
        trace_csv_path: trace CSV文件路径（train_tiles.csv 或 test_tiles.csv）
        frame_id: 帧ID（如果为None，则使用第一帧）
        
    Returns:
        用户Tile映射: {'user_id': [tile_id1, tile_id2, ...]}
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(trace_csv_path)
        
        # 如果frame_id为None，使用第一帧
        if frame_id is None:
            frame_id = df['frame_id'].min()
        
        # 筛选指定帧的数据
        frame_data = df[df['frame_id'] == frame_id]
        
        if len(frame_data) == 0:
            # 如果指定帧不存在，尝试使用第一帧
            frame_id = df['frame_id'].min()
            frame_data = df[df['frame_id'] == frame_id]
            print(f"⚠️  指定帧不存在，使用第一帧 (frame_id={frame_id})")
        
        user_tiles_map = {}
        for _, row in frame_data.iterrows():
            user_id = str(row['user_id'])
            visible_tiles_str = row['visible_tiles']
            
            # 解析visible_tiles字符串（格式: "[1, 2, 3]" 或 "1,2,3"）
            if isinstance(visible_tiles_str, str):
                # 移除方括号和空格
                visible_tiles_str = visible_tiles_str.strip('[]').strip()
                if visible_tiles_str:
                    tiles = [int(t.strip()) for t in visible_tiles_str.split(',') if t.strip()]
                else:
                    tiles = []
            else:
                tiles = []
            
            user_tiles_map[user_id] = tiles
        
        return user_tiles_map
    
    except Exception as e:
        print(f"⚠️ 加载trace数据失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _parse_vec3(s: str) -> Optional[np.ndarray]:
    """解析3D向量字符串，返回归一化的向量"""
    if s is None:
        return None
    try:
        t = str(s).strip().strip("()[]")
        parts = [float(x.strip()) for x in t.split(",")]
        if len(parts) < 3:
            return None
        v = np.array(parts[:3], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / (n + 1e-8)
    except:
        return None


def _rotate_yaw_pitch(v: np.ndarray, yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """
    对3D向量应用yaw和pitch旋转
    
    Args:
        v: 归一化的3D向量
        yaw_rad: yaw角（弧度），绕y轴旋转
        pitch_rad: pitch角（弧度），绕x轴旋转
    
    Returns:
        旋转后的归一化向量
    """
    # yaw: around y-axis, pitch: around x-axis (simple approximation)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cx, sx = np.cos(pitch_rad), np.sin(pitch_rad)

    Ry = np.array([[ cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cx, -sx],
                   [0.0,  sx,  cx]], dtype=np.float32)

    vv = (Ry @ (Rx @ v.reshape(3,1))).reshape(3)
    vv = vv / (np.linalg.norm(vv) + 1e-8)
    return vv


def load_fov_from_head_movement(
    head_movement_csv_path: str,
    num_users: int,
    timestamp: Optional[float] = None,
    time_window_sec: float = 0.3,         # 平滑窗口（0.2-0.5s）
    per_user_time_shift_sec: float = 0.2, # 用户时间偏移步长（0.1-0.5s）
    jitter_yaw_deg: float = 6.0,          # 朝向扰动幅度（3-10度，不超过15度）
    jitter_pitch_deg: float = 3.0,        # pitch扰动（通常比yaw小）
    fov_half_angle_deg: float = 45.0,     # 角相似度尺度
    default: float = 0.5,
    seed: int = 7
) -> Dict[int, float]:
    """
    ✅ 【学术可接受方法】从单条Head_movement轨迹生成多用户FOV重叠度
    
    核心思想：通过时间偏移 + 轻微角度扰动，从一条真实轨迹生成多个"不同用户"的视角，
    既保留真实运动统计特征，又引入用户差异性。
    
    论文说明（英文）：
    "When only a single head-movement trace is available, we synthesize multiple users 
    by applying per-user time offsets and small orientation perturbations to the original 
    trace, which preserves realistic motion dynamics while introducing inter-user diversity."
    
    论文说明（中文）：
    "当仅有单条头部运动轨迹时，本文通过对原始轨迹施加用户特定的时间偏移与小幅朝向扰动
    生成多用户视角序列，从而在保留真实运动动态的同时引入用户差异性。"
    
    Args:
        head_movement_csv_path: Head_movement CSV文件路径
        num_users: 用户数量
        timestamp: 基准时间戳（如果为None，使用中间段避免开头静止）
        time_window_sec: 平滑窗口大小（秒），用于平均headForw向量
        per_user_time_shift_sec: 每个用户的时间偏移步长（秒）
        jitter_yaw_deg: yaw角扰动幅度（度），模拟个体差异
        jitter_pitch_deg: pitch角扰动幅度（度）
        fov_half_angle_deg: FOV半角（度），用于计算角相似度
        default: 默认FOV重叠度（当无法计算时使用）
        seed: 随机种子，确保可重复性
    
    Returns:
        fov_overlap_dict: {user_id: fov_overlap_score}
        fov_overlap_score 是基于角相似度的重叠度 [0, 1]
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(head_movement_csv_path)
        
        if df.empty or "timestamp" not in df.columns:
            print(f"⚠️ Head_movement文件为空或缺少timestamp列: {head_movement_csv_path}")
            return {u: default for u in range(1, num_users + 1)}
        
        if "headForw" not in df.columns:
            print(f"⚠️ Head_movement文件缺少headForw列: {head_movement_csv_path}")
            return {u: default for u in range(1, num_users + 1)}

        # 基准时间：外部给 timestamp，否则用中间段避免开头静止
        if timestamp is None:
            t0 = float(df["timestamp"].iloc[len(df)//2])
        else:
            t0 = float(timestamp)

        rng = np.random.default_rng(seed)
        sigma = np.deg2rad(fov_half_angle_deg)

        user_dir: Dict[int, Optional[np.ndarray]] = {}

        for u in range(1, num_users + 1):
            # 1) 时间偏移：每个用户取不同时间点
            tu = t0 + (u - 1) * per_user_time_shift_sec

            # 2) 取窗口并平均 headForw
            sub = df[(df["timestamp"] >= tu - time_window_sec) & (df["timestamp"] <= tu)]
            if sub.empty:
                # 如果窗口内没有数据，找最接近的时间点
                idx = (df["timestamp"] - tu).abs().idxmin()
                sub = df.loc[[idx]]

            vs = []
            for s in sub["headForw"].tolist():
                v = _parse_vec3(s)
                if v is not None:
                    vs.append(v)
            
            if not vs:
                user_dir[u] = None
                continue

            # 计算平均方向向量
            vmean = np.mean(np.stack(vs, axis=0), axis=0)
            vmean = vmean / (np.linalg.norm(vmean) + 1e-8)

            # 3) 朝向扰动：模拟不同用户差异（小扰动，保持"同一类行为"）
            yaw_j = np.deg2rad(rng.uniform(-jitter_yaw_deg, jitter_yaw_deg))
            pit_j = np.deg2rad(rng.uniform(-jitter_pitch_deg, jitter_pitch_deg))
            vj = _rotate_yaw_pitch(vmean, yaw_j, pit_j)

            user_dir[u] = vj

        # 4) pairwise 角相似度 -> per-user mean overlap summary
        out: Dict[int, float] = {}
        for i in range(1, num_users + 1):
            vi = user_dir.get(i)
            if vi is None:
                out[i] = default
                continue
            
            sims = []
            for j in range(1, num_users + 1):
                if j == i:
                    continue
                vj = user_dir.get(j)
                if vj is None:
                    continue
                
                # 计算角相似度（使用高斯核）
                cosang = float(np.clip(np.dot(vi, vj), -1.0, 1.0))
                theta = np.arccos(cosang)
                sim = float(np.exp(-(theta * theta) / (2 * sigma * sigma)))
                sims.append(sim)
            
            out[i] = float(np.mean(sims)) if sims else default

        return out
    
    except Exception as e:
        print(f"⚠️ 加载Head_movement数据失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认值
        return {i: default for i in range(1, num_users + 1)}


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("FOV Grouping Algorithm - 测试")
    print("=" * 70)
    
    # 示例数据
    user_tiles_map = {
        '1': [1, 2, 3, 4, 5],
        '2': [1, 2, 3, 6, 7],  # 与用户1重叠度高
        '3': [10, 11, 12, 13, 14],  # 与用户1、2不重叠
        '4': [1, 2, 8, 9],  # 与用户1、2部分重叠
        '5': [15, 16, 17, 18, 19],  # 独立
    }
    
    manager = FOVGroupManager(overlap_threshold=0.3)
    groups = manager.group_users(user_tiles_map)
    
    print(f"\n分组结果: {groups}")
    
    stats = manager.get_group_stats(groups, user_tiles_map)
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)

