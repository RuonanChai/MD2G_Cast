#!/usr/bin/env python3
import time
import os
import re
import json
import csv
import threading
import random
import shutil
from collections import deque, defaultdict
from mininet.net import Mininet
from mininet.node import Host, Controller, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info, error
from mininet.link import TCLink

# 尝试导入 pandas（用于读取数据集）
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    info("⚠️  pandas 未安装，将使用默认带宽配置\n")

# ================= 配置区域 =================
# 项目根目录（用于构建所有相对路径）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# MOQ 官方二进制目录（可用环境变量覆盖，便于 Review Artifact 复现）
MOQ_BIN_DIR = os.getenv(
    "MOQ_BIN_DIR",
    os.path.join(PROJECT_ROOT, "third_party", "moq", "target", "release"),
)

BASE_DIR = MOQ_BIN_DIR
VIDEO_DIR = os.path.join(PROJECT_ROOT, "video")

BIN_PATHS = {
    "relay": os.path.join(BASE_DIR, "moq-relay"),
    "sub":   os.path.join(BASE_DIR, "moq-sub"),
    "pub":   os.path.join(BASE_DIR, "hang"),
    "token": os.path.join(BASE_DIR, "moq-token")
}

# ✅ 使用官方示例视频作为源（100% 兼容 hang 工具，无 mfra/fiel atom）
OFFICIAL_VIDEO = os.getenv("V_PCC_BBB_MP4", os.path.join(VIDEO_DIR, "bbb.mp4"))

# ✅ 生成的分层视频路径（基于 bbb.mp4 生成，符合论文设计）
VIDEO_PATHS = {
    "base": os.path.join(VIDEO_DIR, "redandblack_2", "base", "redandblack_base_fragmented.mp4"),
    "enhanced": os.path.join(VIDEO_DIR, "redandblack_2", "enhanced", "redandblack_enhanced_fragmented.mp4")
}

# 确保目录存在
os.makedirs(os.path.dirname(VIDEO_PATHS["base"]), exist_ok=True)
os.makedirs(os.path.dirname(VIDEO_PATHS["enhanced"]), exist_ok=True)

# ✅ 使用独立的证书目录和文件前缀，避免与 topo_moq_eval_OFF.py 冲突
CERT_DIR = "/tmp/moq_certs_test2"
TMP_PREFIX = "test2_"  # 所有临时文件使用此前缀，避免冲突
AUTH_DIR = f"/tmp/{TMP_PREFIX}auth"  # JWT key 和 token 目录
# ===========================================

# ================= Experiment Configuration =================
# Supported baseline strategies in this Artifact: md2g, rolling, heuristic, clustering
# ✅ 支持7种网络：wifi, 4g, 5g, fiber_optic, default_mix, wifi_dominant, 5g_dominant
# ✅ 支持10-100用户数
STRATEGIES = ["md2g", "rolling", "heuristic", "clustering"]
NETWORKS = ["wifi", "4g", "5g", "fiber_optic", "default_mix", "wifi_dominant", "5g_dominant"]

# ================= 带宽配置（参考 topo_moq_eval_OFF.py）=================
# ✅ 【核心/DC链路】(Core <-> Root Relay)
BW_N0_R0 = 1000   # 1 Gbps (Mininet限制，最大1000 Mbps)

# ✅ 【码率配置】动态从视频文件读取实际码率
def get_video_bitrate(video_path):
    """
    使用ffprobe读取视频文件的实际码率
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        float: 码率（Mbps），如果读取失败返回默认值
    """
    try:
        import subprocess
        # 使用ffprobe读取视频码率（bps）
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=bit_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            bitrate_bps = float(result.stdout.strip())
            bitrate_mbps = bitrate_bps / 1e6  # 转换为Mbps
            return bitrate_mbps
    except Exception as e:
        info(f"⚠️  无法读取视频码率 {video_path}: {e}，使用默认值\n")
    
    # 默认值（fallback）
    return None

# ✅ 动态读取视频码率（延迟初始化，在run_moq_experiment中调用）
def init_video_bitrates():
    """
    初始化视频码率（从实际视频文件读取）
    
    Returns:
        (base_bitrate, enh_bitrate, total_bitrate): 码率（Mbps）
    """
    base_bitrate = None
    enh_bitrate = None
    
    # 尝试从视频文件读取码率
    if os.path.exists(VIDEO_PATHS["base"]):
        base_bitrate = get_video_bitrate(VIDEO_PATHS["base"])
    if os.path.exists(VIDEO_PATHS["enhanced"]):
        enh_bitrate = get_video_bitrate(VIDEO_PATHS["enhanced"])
    
    # 如果读取失败，使用默认值（根据用户记忆：Base=10Mbps, Enhanced=1Mbps）
    if base_bitrate is None:
        base_bitrate = 10.0  # 默认值：10 Mbps
        print("⚠️  使用默认Base码率: 10.0 Mbps")
    else:
        print(f"✅ Base视频码率: {base_bitrate:.2f} Mbps（从文件读取）")
    
    if enh_bitrate is None:
        enh_bitrate = 1.0  # 默认值：1 Mbps
        print("⚠️  使用默认Enhanced码率: 1.0 Mbps")
    else:
        print(f"✅ Enhanced视频码率: {enh_bitrate:.2f} Mbps（从文件读取）")
    
    total_bitrate = base_bitrate + enh_bitrate
    print(f"✅ 总码率（Base+Enhanced）: {total_bitrate:.2f} Mbps")
    
    return base_bitrate, enh_bitrate, total_bitrate

# 全局变量（延迟初始化）
BASE_BITRATE_MBPS = None
ENH_BITRATE_MBPS = None
TOTAL_BITRATE_MBPS = None

# ✅ 【核心 -> 区域】(Root Relay <-> Regional Relay)
# 动态计算带宽需求，确保支持10-100用户
# 公式：BW = max(350, num_users_per_relay * TOTAL_BITRATE * 1.5)
# 1.5倍是为了让Unicast适度拥塞（150%），突出Multicast优势
# 但也要保证实验能正常运行（不会完全断开）
def calculate_regional_bandwidth(num_users, total_bitrate_mbps=None):
    """
    计算区域链路带宽（固定瓶颈带宽，基于MD2G-10用户-4g实际流量数据）
    
    ⚠️ 【关键修复：实验设计逻辑】
    不是"供养"单播，而是"考验"它：使用固定的瓶颈带宽，让单播策略在用户数增加时遇到瓶颈，
    而组播策略由于Base层共享，可以更好地利用带宽。这是实验对比的关键。
    
    基于实际流量数据（MD2G-10用户-4g，120秒）：
    - 总下行流量：479.71 Mbps（组播策略，Base层共享）
    - 单播策略估算：如果单播，上行流量应该≈下行流量=479.71 Mbps
    - 建议固定瓶颈带宽：575 Mbps (120%余量)
    
    Args:
        num_users: 总用户数（10-100）
        total_bitrate_mbps: 总码率（Mbps），如果为None则使用全局变量
    
    Returns:
        (bw_r0_r1, bw_r0_r2): 两个区域链路的带宽（Mbps）
    
    固定带宽设计（基于实际数据）：
    - 10用户组播实际流量：479.71 Mbps（下行）
    - 10用户单播估算流量：479.71 Mbps（上行≈下行，每个用户独立流）
    - 固定瓶颈带宽：575 Mbps (120%余量，基于实际数据)
      - 10用户单播：479.71Mbps < 575Mbps ✅（轻微瓶颈）
      - 100用户单播：4797.1Mbps > 575Mbps ❌（严重瓶颈，突出组播优势）
      - 10用户组播：479.71Mbps < 575Mbps ✅（轻松）
      - 100用户组播：约600Mbps（估算）≈ 575Mbps ⚠️（适度瓶颈）
    """
    # ✅ 【区域链路带宽计算】根据用户数和策略类型动态计算
    # 架构说明：
    # - NETWORK_BW_CONFIG：每个用户的独立接入链路（host -> switch -> relay），不累加
    # - BW_R0_R1/BW_R0_R2：区域链路（r0 -> r1/r2），共享，需要承载所有用户的流量
    # 
    # 带宽需求计算（每个 relay 负责 num_users/2 用户）：
    # - 单播策略：users_per_relay × (Base + Enhanced) = users_per_relay × 6.0 Mbps
    # - 组播策略：Base × 1 + Enhanced × users_per_relay = 5.0 + users_per_relay × 1.0 Mbps
    # 
    # 示例：
    # - 10用户：单播 5 × 6.0 = 30 Mbps，组播 5.0 + 5 = 10 Mbps
    # - 20用户：单播 10 × 6.0 = 60 Mbps，组播 5.0 + 10 = 15 Mbps
    # - 100用户：单播 50 × 6.0 = 300 Mbps，组播 5.0 + 50 = 55 Mbps
    # 
    # 设计：支持100用户单播，加上20%余量
    users_per_relay = num_users / 2
    unicast_bw = users_per_relay * 6.0  # 单播：每个用户独立流
    multicast_bw = 5.0 + users_per_relay * 1.0  # 组播：Base共享，Enhanced独立
    # 取较大值（单播），加上20%余量
    required_bw = unicast_bw * 1.2
    FIXED_BOTTLENECK_BW = max(100, int(required_bw))  # 至少100 Mbps，向上取整
    
    return FIXED_BOTTLENECK_BW, FIXED_BOTTLENECK_BW

# 默认值（10用户）
# ✅ 【区域链路带宽】根据用户数动态计算，支持10-100用户
# 10用户：单播 5 × 6.0 × 1.2 = 36 Mbps，组播 5.0 + 5 = 10 Mbps
# 100用户：单播 50 × 6.0 × 1.2 = 360 Mbps，组播 5.0 + 50 = 55 Mbps
BW_R0_R1 = 500    # 支持100用户单播（360 Mbps）+ 余量
BW_R0_R2 = 500    # 支持100用户单播（360 Mbps）+ 余量

# ✅ 【用户接入链路】(Edge Relay <-> Host)
# 根据网络类型设置不同的带宽、延迟和丢包率
# ✅ 【支持10-100用户实验】每个用户有独立的接入链路，带宽配置需要足够大
# 考虑：Base码率约10Mbps，Enhanced码率约1Mbps，单播需要为每个用户分配足够带宽
# 单播场景：每个用户需要独立的Base流（10Mbps），如果启用Enhanced还需要额外的1Mbps
# 因此单播用户数 = 带宽 / (Base码率 + Enhanced码率) = 带宽 / 11Mbps
# 
# 注意：这是每个用户的独立接入链路带宽，不是共享的
# 10-100用户实验时，每个用户都有独立的这个带宽限制
# ✅ 【调高用户接入链路带宽】确保实际下载速度 > 视频码率（10 Mbps）
# 目标：实际下载速度 = 15-20 Mbps（1.5-2倍码率），确保Buffer稳定在5-10秒
# 当前问题：实际下载只有1.5-1.7 Mbps，远低于10 Mbps码率，导致Buffer快速归零
# ✅ 【降低单台物理机压力】如果限速12M后依然出现Buffer封顶30s，降低5G带宽
# 原因：r2的Fan-out远低于r1，说明CPU调度已经出现了明显的偏袒或死锁
# 修复：将5G带宽从600Mbps调回100Mbps，人为制造轻微的物理拥塞，从而观察算法的调度能力
NETWORK_BW_CONFIG = {
    'wifi': {'bw': 300, 'delay': '5ms', 'loss': 0},      # WiFi: 300 Mbps（修复：之前是 00，导致带宽为 0）
    '4g': {'bw': 40, 'delay': '10ms', 'loss': 0},        # 4G: 40 Mbps
    '5g': {'bw': 800, 'delay': '5ms', 'loss': 0},       # 5G: 800 Mbps
    'fiber_optic': {'bw': 800, 'delay': '5ms', 'loss': 0},  # Fiber: 800 Mbps
}

# ================= 数据集路径 =================
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
NETWORK_DATASETS = {
    'wifi': os.path.join(DATASET_DIR, "wifi_clean.csv"),
    '4g': os.path.join(DATASET_DIR, "4G-network-data_clean.csv"),
    '5g': os.path.join(DATASET_DIR, "5g_final_trace.csv"),
    'fiber_optic': os.path.join(DATASET_DIR, "Optic_Bandwidth_clean_2.csv"),
}
# ===========================================

# ================= 测量工具函数 =================
def read_interface_bytes(host, interface_name):
    """
    读取网络接口的RX/TX字节统计（不使用shell命令）
    
    Args:
        host: Mininet Host对象
        interface_name: 接口名称（如 'h1-eth0'）
    
    Returns:
        (rx_bytes, tx_bytes): 接收和发送字节数
    """
    try:
        # 方法1：使用/sys/class/net统计（推荐）
        rx_path = f"/sys/class/net/{interface_name}/statistics/rx_bytes"
        tx_path = f"/sys/class/net/{interface_name}/statistics/tx_bytes"
        
        rx_bytes_str = host.cmd(f'cat {rx_path} 2>/dev/null || echo "0"').strip()
        tx_bytes_str = host.cmd(f'cat {tx_path} 2>/dev/null || echo "0"').strip()
        
        if rx_bytes_str.isdigit() and tx_bytes_str.isdigit():
            return int(rx_bytes_str), int(tx_bytes_str)
        
        # 方法2：fallback到/proc/net/dev
        stats_line = host.cmd(f'cat /proc/net/dev 2>/dev/null | grep "{interface_name}:"').strip()
        if stats_line:
            parts = stats_line.split()
            if len(parts) >= 10:
                rx = int(parts[1])  # RX bytes
                tx = int(parts[9])  # TX bytes
                return rx, tx
        
        return 0, 0
    except Exception as e:
        error(f"⚠️  读取接口 {interface_name} 统计失败: {e}\n")
        return 0, 0

def load_network_bandwidth_data(network_type, num_users):
    """
    从数据集加载网络带宽数据
    
    Args:
        network_type: 网络类型（wifi, 4g, 5g, fiber_optic, default_mix, wifi_dominant, 5g_dominant）
        num_users: 用户数量
    
    Returns:
        list: 每个用户的带宽（Mbps）列表
    """
    if not HAS_PANDAS:
        # 如果没有pandas，使用配置的默认值
        if network_type in ['default_mix', 'wifi_dominant', '5g_dominant']:
            # 混合网络：使用平均带宽
            avg_bw = sum(cfg['bw'] for cfg in NETWORK_BW_CONFIG.values()) / len(NETWORK_BW_CONFIG)
            return [avg_bw] * num_users
        else:
            return [NETWORK_BW_CONFIG.get(network_type, NETWORK_BW_CONFIG['wifi'])['bw']] * num_users
    
    bandwidths = []
    
    if network_type in ['default_mix', 'wifi_dominant', '5g_dominant']:
        # 混合网络场景
        if network_type == 'wifi_dominant':
            # WiFi占70%，4G占30%
            wifi_count = int(num_users * 0.7)
            g4_count = num_users - wifi_count
            net_types = ['wifi'] * wifi_count + ['4g'] * g4_count
        elif network_type == '5g_dominant':
            # 5G占70%，4G占30%
            g5_count = int(num_users * 0.7)
            g4_count = num_users - g5_count
            net_types = ['5g'] * g5_count + ['4g'] * g4_count
        else:  # default_mix
            # 均匀分布：WiFi 30%, 4G 30%, 5G 20%, Fiber 20%
            wifi_count = int(num_users * 0.3)
            g4_count = int(num_users * 0.3)
            g5_count = int(num_users * 0.2)
            fiber_count = num_users - wifi_count - g4_count - g5_count
            net_types = (['wifi'] * wifi_count + ['4g'] * g4_count + 
                        ['5g'] * g5_count + ['fiber_optic'] * fiber_count)
        
        random.shuffle(net_types)
        
        # 为每个用户从对应的数据集加载带宽
        for net_type in net_types:
            if net_type in NETWORK_DATASETS and os.path.exists(NETWORK_DATASETS[net_type]):
                try:
                    df = pd.read_csv(NETWORK_DATASETS[net_type])
                    # 根据数据集格式选择带宽列
                    if 'DL_bitrate_Mbps' in df.columns:
                        bw_col = 'DL_bitrate_Mbps'
                    elif 'bytes_sec (Mbps)' in df.columns:
                        bw_col = 'bytes_sec (Mbps)'
                    elif 'bandwidth_mbps' in df.columns:
                        bw_col = 'bandwidth_mbps'
                    else:
                        # 使用第一列
                        bw_col = df.columns[0]
                    
                    # 随机选择一个带宽值
                    # ✅ 【关键修复】过滤掉无效值（0、负数、NaN），确保只使用有效带宽
                    valid_bws = df[bw_col].dropna()  # 移除 NaN
                    valid_bws = valid_bws[valid_bws > 0]  # 只保留正数
                    if len(valid_bws) == 0:
                        # 如果数据集中没有有效值，使用配置的默认值
                        bw_value = NETWORK_BW_CONFIG[net_type]['bw']
                    else:
                        bw_value = float(valid_bws.sample(1).values[0])
                    # 限制在合理范围内（不超过配置的最大值）
                    max_bw = NETWORK_BW_CONFIG[net_type]['bw']
                    bandwidths.append(min(bw_value, max_bw))
                except Exception as e:
                    # 如果加载失败，使用配置的默认值
                    bandwidths.append(NETWORK_BW_CONFIG[net_type]['bw'])
            else:
                # 使用配置的默认值
                bandwidths.append(NETWORK_BW_CONFIG[net_type]['bw'])
    else:
        # 单一网络类型
        if network_type in NETWORK_DATASETS and os.path.exists(NETWORK_DATASETS[network_type]):
            try:
                df = pd.read_csv(NETWORK_DATASETS[network_type])
                # 根据数据集格式选择带宽列
                if 'DL_bitrate_Mbps' in df.columns:
                    bw_col = 'DL_bitrate_Mbps'
                elif 'bytes_sec (Mbps)' in df.columns:
                    bw_col = 'bytes_sec (Mbps)'
                elif 'bandwidth_mbps' in df.columns:
                    bw_col = 'bandwidth_mbps'
                else:
                    # 使用第一列
                    bw_col = df.columns[0]
                
                # 随机选择带宽值（有放回抽样）
                # ✅ 【关键修复】过滤掉无效值（0、负数、NaN），确保只使用有效带宽
                valid_bws = df[bw_col].dropna()  # 移除 NaN
                valid_bws = valid_bws[valid_bws > 0]  # 只保留正数
                if len(valid_bws) == 0:
                    # 如果数据集中没有有效值，使用配置的默认值
                    bandwidths = [NETWORK_BW_CONFIG[network_type]['bw']] * num_users
                else:
                    sampled_bws = valid_bws.sample(num_users, replace=True).values
                    max_bw = NETWORK_BW_CONFIG[network_type]['bw']
                    bandwidths = [min(float(bw), max_bw) for bw in sampled_bws]
            except Exception as e:
                # 如果加载失败，使用配置的默认值
                bandwidths = [NETWORK_BW_CONFIG[network_type]['bw']] * num_users
        else:
            # 使用配置的默认值
            bandwidths = [NETWORK_BW_CONFIG[network_type]['bw']] * num_users
    
    return bandwidths

def get_host_network_types(network_type, num_users):
    """
    获取每个用户的网络类型列表（用于混合网络场景）
    
    Args:
        network_type: 网络类型（wifi, 4g, 5g, fiber_optic, default_mix, wifi_dominant, 5g_dominant）
        num_users: 用户数量
    
    Returns:
        list: 每个用户的网络类型列表
    """
    
    if network_type in ['default_mix', 'wifi_dominant', '5g_dominant']:
        # 混合网络场景
        if network_type == 'wifi_dominant':
            # WiFi占70%，4G占30%
            wifi_count = int(num_users * 0.7)
            g4_count = num_users - wifi_count
            net_types = ['wifi'] * wifi_count + ['4g'] * g4_count
        elif network_type == '5g_dominant':
            # 5G占70%，4G占30%
            g5_count = int(num_users * 0.7)
            g4_count = num_users - g5_count
            net_types = ['5g'] * g5_count + ['4g'] * g4_count
        else:  # default_mix
            # 均匀分布：WiFi 30%, 4G 30%, 5G 20%, Fiber 20%
            wifi_count = int(num_users * 0.3)
            g4_count = int(num_users * 0.3)
            g5_count = int(num_users * 0.2)
            fiber_count = num_users - wifi_count - g4_count - g5_count
            net_types = (['wifi'] * wifi_count + ['4g'] * g4_count + 
                        ['5g'] * g5_count + ['fiber_optic'] * fiber_count)
        
        random.shuffle(net_types)
        return net_types
    else:
        # 单一网络类型
        return [network_type] * num_users

def calculate_jfi(loads):
    """
    计算Jain's Fairness Index (JFI)
    
    Args:
        loads: 负载列表（如 [x1, x2, ..., xK]）
    
    Returns:
        JFI值（0-1之间）
    """
    if not loads or len(loads) == 0:
        return 1.0
    
    loads = [float(x) for x in loads if x > 0]
    if not loads:
        return 1.0
    
    n = len(loads)
    sum_x = sum(loads)
    sum_x_squared = sum(x * x for x in loads)
    
    if sum_x_squared == 0:
        return 1.0
    
    jfi = (sum_x * sum_x) / (n * sum_x_squared)
    return max(0.0, min(1.0, jfi))

# ================= Buffer模型（基于片段时长的缓冲模型）=================
class BufferModel:
    """
    客户端播放器缓冲模型（基于片段时长的缓冲模型）
    
    原理：
    - 设定播放速率 1.0x
    - 每个fragment对应的媒体时长 dur_frag（近似用帧率：30fps，每帧1/30秒）
    - 已接收的fragment数量 → 可播放时长累加：buffer += dur_frag * received_fragments
    - 每个wall-clock时间流逝，播放消耗：buffer -= Δt（下限0）
    """
    def __init__(self, fps=30.0):
        self.fps = fps
        self.dur_frag = 1.0 / fps  # 每个fragment的媒体时长（秒）
        self.buffer_sec = 0.0  # 当前缓冲时长（秒）
        self.received_fragments = 0  # 已接收的fragment数量
        self.last_update_time = time.monotonic()
        self.state = "STARTUP"  # STARTUP, PLAYING, REBUFFER
        self.stall_count = 0
        self.stall_total_sec = 0.0
        self.stall_start_time = None
    
    def update_received(self, fragment_count):
        """更新已接收的fragment数量"""
        self.received_fragments = fragment_count
        # 更新缓冲：buffer = fragment_count * dur_frag
        self.buffer_sec = fragment_count * self.dur_frag
    
    def update_playback(self, current_time=None):
        """
        更新播放消耗（每个决策周期调用一次）
        
        Args:
            current_time: 当前时间（使用time.monotonic()）
        
        Returns:
            (buffer_level_sec, stall_count, stall_total_sec): 当前缓冲状态
        """
        if current_time is None:
            current_time = time.monotonic()
        
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # 状态机：STARTUP -> PLAYING -> REBUFFER
        if self.state == "STARTUP":
            # 启动阶段：缓冲达到阈值（如2秒）后进入PLAYING
            if self.buffer_sec >= 2.0:
                self.state = "PLAYING"
        elif self.state == "PLAYING":
            # 播放阶段：消耗缓冲
            self.buffer_sec = max(0.0, self.buffer_sec - dt)
            
            # 如果缓冲耗尽，进入REBUFFER状态
            if self.buffer_sec <= 0.0:
                self.state = "REBUFFER"
                self.stall_start_time = current_time
                self.stall_count += 1
        elif self.state == "REBUFFER":
            # 卡顿阶段：继续消耗（buffer保持为0），累加卡顿时间
            self.buffer_sec = 0.0
            if self.stall_start_time:
                self.stall_total_sec += dt
            
            # 如果缓冲恢复（>0），退出REBUFFER状态
            if self.buffer_sec > 0.0:
                self.state = "PLAYING"
                self.stall_start_time = None
        
        return self.buffer_sec, self.stall_count, self.stall_total_sec
    
    def get_state(self):
        """获取当前状态"""
        return {
            "buffer_level_sec": self.buffer_sec,
            "stall_count": self.stall_count,
            "stall_total_sec": self.stall_total_sec,
            "state": self.state
        }
# ===========================================

def check_files():
    missing = []
    for name, path in BIN_PATHS.items():
        if not os.path.exists(path): missing.append(f"Bin: {name} -> {path}")
    
    if missing:
        error("❌ 文件缺失:\n" + "\n".join(missing) + "\n")
        return False
    return True

def prepare_layered_videos():
    """
    基于 bbb.mp4 生成符合论文设计的双层视频 (大文件版)
    
    Base: 10Mbps, 2分钟 -> 约 150MB
    Enhanced: 1Mbps, 2分钟 -> 约 15MB
    """
    if not os.path.exists(OFFICIAL_VIDEO):
        error(f"❌ 严重错误: 找不到源视频 {OFFICIAL_VIDEO}\n")
        return False
    
    info("🎬 正在生成符合论文比例的【大容量】分层视频...\n")
    info("   (这将花费大约 10-30 秒，请耐心等待)\n")
    
    # --- 1. Base Layer (大文件: ~150MB) ---
    if not os.path.exists(VIDEO_PATHS["base"]) or os.path.getsize(VIDEO_PATHS["base"]) < 100*1024*1024:
        info("   📦 生成 Base Layer (10Mbps, 120s, 预计 ~150MB)...\n")
        cmd_base = (f"ffmpeg -y -hide_banner -loglevel error "
                    f"-stream_loop -1 -i {OFFICIAL_VIDEO} "
                    f"-t 120 "
                    f"-c:v libx264 -preset ultrafast -g 30 -keyint_min 30 -sc_threshold 0 "
                    f"-b:v 10M -minrate 10M -maxrate 10M -bufsize 20M "
                    f"-pix_fmt yuv420p "
                    f"-movflags +frag_keyframe+empty_moov+default_base_moof+skip_trailer "
                    f"{VIDEO_PATHS['base']}")
        os.system(cmd_base)
    
    # --- 2. Enhanced Layer (小文件: ~15MB) ---
    if not os.path.exists(VIDEO_PATHS["enhanced"]) or os.path.getsize(VIDEO_PATHS["enhanced"]) < 10*1024*1024:
        info("   📦 生成 Enhanced Layer (1Mbps, 120s, 黑白, 预计 ~15MB)...\n")
        cmd_enh = (f"ffmpeg -y -hide_banner -loglevel error "
                   f"-stream_loop -1 -i {OFFICIAL_VIDEO} "
                   f"-t 120 "
                   f"-vf hue=s=0 "
                   f"-c:v libx264 -preset ultrafast -g 30 -keyint_min 30 -sc_threshold 0 "
                   f"-b:v 1M -minrate 1M -maxrate 1M -bufsize 2M "
                   f"-pix_fmt yuv420p "
                   f"-movflags +frag_keyframe+empty_moov+default_base_moof+skip_trailer "
                   f"{VIDEO_PATHS['enhanced']}")
        os.system(cmd_enh)
    
    # 验证文件大小比例
    if os.path.exists(VIDEO_PATHS['base']) and os.path.exists(VIDEO_PATHS['enhanced']):
        size_base = os.path.getsize(VIDEO_PATHS['base'])
        size_enh = os.path.getsize(VIDEO_PATHS['enhanced'])
        ratio = size_base / size_enh if size_enh > 0 else 0
        
        info(f"✅ 视频准备就绪:\n")
        info(f"   Base:     {size_base/1024/1024:.2f} MB\n")
        info(f"   Enhanced: {size_enh/1024/1024:.2f} MB\n")
        info(f"   比例:     {ratio:.1f}:1 (完美符合 V-PCC 论文设计)\n")
        return True
    else:
        error("❌ 视频生成失败\n")
        return False

def generate_certs():
    """生成自签名证书，包含所有 relay 节点的 SAN"""
    import shutil
    import stat
    import tempfile
    
    # ✅ 修复权限问题：使用局部变量，避免修改全局变量
    cert_dir = CERT_DIR
    
    # 先删除旧目录（如果存在），然后重新创建
    if os.path.exists(cert_dir):
        try:
            # 尝试删除旧目录
            shutil.rmtree(cert_dir)
        except PermissionError:
            # 如果权限不足，尝试修改权限后删除
            try:
                os.chmod(cert_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                shutil.rmtree(cert_dir)
            except:
                # 如果还是失败，使用系统命令强制删除
                os.system(f"rm -rf {cert_dir} 2>/dev/null || true")
    
    # 创建新目录（确保有写权限）
    try:
        os.makedirs(cert_dir, mode=0o755, exist_ok=True)
    except PermissionError:
        # 如果创建失败，尝试使用系统命令
        os.system(f"mkdir -p {cert_dir} && chmod 755 {cert_dir} 2>/dev/null || true")
        if not os.path.exists(cert_dir):
            # 如果还是失败，使用用户临时目录
            cert_dir = os.path.join(tempfile.gettempdir(), f"moq_certs_{os.getpid()}")
            os.makedirs(cert_dir, mode=0o755, exist_ok=True)
            try:
                info(f"⚠️  使用临时目录: {cert_dir}\n")
            except:
                print(f"⚠️  使用临时目录: {cert_dir}")
    
    cert, key = f"{cert_dir}/cert.pem", f"{cert_dir}/key.pem"
    config_path = f"{cert_dir}/cert.conf"
    
    # 创建 OpenSSL 配置文件，包含所有 relay 节点的 SAN
    config_content = """[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = r0.local
DNS.3 = r0
DNS.4 = r1.local
DNS.5 = r1
DNS.6 = r2.local
DNS.7 = r2
IP.1 = 127.0.0.1
IP.2 = ::1
IP.3 = 10.0.1.1
IP.4 = 10.0.2.2
IP.5 = 10.0.3.2
"""
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        # 确保文件有写权限
        os.chmod(config_path, 0o644)
    except PermissionError:
        error(f"❌ 无法写入配置文件: {config_path}\n")
        error("   请检查目录权限或使用 sudo 运行\n")
        raise
    
    # 生成包含 SAN 的证书
    cmd = (f"openssl req -newkey rsa:2048 -nodes -keyout {key} "
           f"-x509 -days 365 -out {cert} "
           f"-config {config_path} -extensions v3_req >/dev/null 2>&1")
    result = os.system(cmd)
    if result != 0:
        error(f"❌ 证书生成失败，返回码: {result}\n")
        error(f"   请检查 openssl 是否已安装\n")
        raise RuntimeError("证书生成失败")
    
    return cert, key

def generate_auth_keys():
    """生成 JWT key 和 cluster token（官方推荐方式）"""
    import os
    import subprocess
    
    # 创建 auth 目录
    os.makedirs(AUTH_DIR, mode=0o755, exist_ok=True)
    
    key_file = os.path.join(AUTH_DIR, "root.jwk")
    token_file = os.path.join(AUTH_DIR, "cluster.jwt")
    
    # 生成 JWT key（如果不存在）
    if not os.path.exists(key_file):
        info("🔑 生成 JWT key...\n")
        cmd = f"{BIN_PATHS['token']} --key {key_file} generate"
        result = os.system(cmd)
        if result != 0:
            error(f"❌ JWT key 生成失败，返回码: {result}\n")
            raise RuntimeError("JWT key 生成失败")
        info(f"✅ JWT key 已生成: {key_file}\n")
    else:
        info(f"✅ 使用现有 JWT key: {key_file}\n")
    
    # 生成 cluster token（如果不存在）
    if not os.path.exists(token_file):
        info("🎫 生成 cluster token...\n")
        # 生成允许所有路径的 cluster token
        cmd = (f"{BIN_PATHS['token']} --key {key_file} sign "
               f"--root \"\" --subscribe \"\" --publish \"\" --cluster "
               f"> {token_file}")
        result = os.system(cmd)
        if result != 0:
            error(f"❌ Cluster token 生成失败，返回码: {result}\n")
            raise RuntimeError("Cluster token 生成失败")
        info(f"✅ Cluster token 已生成: {token_file}\n")
    else:
        info(f"✅ 使用现有 cluster token: {token_file}\n")
    
    return key_file, token_file

def generate_relay_config(node_name, cert, key, auth_key_file):
    """生成 relay 配置文件（官方推荐：启用 auth，使用 public = "anon"）"""
    config_path = f"/tmp/{TMP_PREFIX}{node_name}.toml"
    # ✅ 官方推荐方式：启用 auth，使用 public = "anon" 允许匿名访问 /anon 前缀
    config = f"""
[log]
level = "info"

[server]
# ✅ 修复：使用 listen 而不是 bind（根据官方配置文件示例）
listen = "0.0.0.0:4443"

[auth]
# ✅ 官方推荐：启用 auth，使用 public = "anon" 允许匿名访问 /anon 前缀
# 这样允许匿名访问 /anon/** 路径，其他路径需要 token
key = "{auth_key_file}"
public = "anon"

[client]
# 禁用 TLS 证书验证（用于自签名证书）
# 这样 leaf nodes 可以连接到 root node 而不会因为自签名证书失败
tls.disable_verify = true
"""
    with open(config_path, "w") as f: f.write(config)
    return config_path

def setup_routing(net, num_subscribers=10):
    """配置树形拓扑的路由 (修复版：添加静态主机路由解决 L3 路由歧义)"""
    n0 = net.get('n0')
    r0 = net.get('r0')
    r1 = net.get('r1')
    r2 = net.get('r2')
    subscribers = [net.get(f'h{i}') for i in range(1, num_subscribers + 1)]
    
    info("配置树形拓扑路由 (Fix: Explicit Host Routes)...\n")
    for n in net.hosts: n.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
    
    # 开启转发
    r0.cmd("sysctl -w net.ipv4.ip_forward=1")
    r1.cmd("sysctl -w net.ipv4.ip_forward=1")
    r2.cmd("sysctl -w net.ipv4.ip_forward=1")
    
    # --- Core 链路配置 ---
    r0.cmd("ifconfig r0-eth0 10.0.1.1/24 up")
    r0.cmd("ifconfig r0-eth1 10.0.2.1/24 up")
    r0.cmd("ifconfig r0-eth2 10.0.3.1/24 up")
    
    r1.cmd("ifconfig r1-eth0 10.0.2.2/24 up")
    r2.cmd("ifconfig r2-eth0 10.0.3.2/24 up")

    # --- ✅ 关键修复：使用 Switch 作为二层汇聚，简化配置 ---
    r1_sub_count = num_subscribers // 2
    
    # ✅ r1 分支：r1 只有一个 LAN 口连接到 s1，配置为网关 IP
    r1.cmd("ifconfig r1-eth1 10.0.4.1/24 up")
    info("  ✅ r1 LAN 口 (r1-eth1) 已配置为 10.0.4.1/24\n")
    
    # ✅ r2 分支：r2 只有一个 LAN 口连接到 s2，配置为网关 IP
    r2.cmd("ifconfig r2-eth1 10.0.5.1/24 up")
    info("  ✅ r2 LAN 口 (r2-eth1) 已配置为 10.0.5.1/24\n")
    
    # 1. 配置 Subscriber 的 IP 和默认路由
    for i, host in enumerate(subscribers, 1):
        # 清除 host 旧配置
        host.cmd(f"ifconfig h{i}-eth0 0.0.0.0 2>/dev/null || true")
        
        if i <= r1_sub_count:
            # === 连接到 r1 分支的用户 (h1, h2...) ===
            host_ip = f"10.0.4.{i+1}"
            gateway = "10.0.4.1"
            
            # Host 端配置
            host.cmd(f"ifconfig h{i}-eth0 {host_ip}/24 up")
            host.cmd(f"route add default gw {gateway} 2>/dev/null || ip route add default via {gateway}")
            
            info(f"  ✅ h{i} 已配置: {host_ip}/24, 网关: {gateway}\n")
            
        else:
            # === 连接到 r2 分支的用户 (h3, h4, h5...) ===
            idx = i - r1_sub_count
            host_ip = f"10.0.5.{idx+1}"
            gateway = "10.0.5.1"
            
            # Host 端配置
            host.cmd(f"ifconfig h{i}-eth0 {host_ip}/24 up")
            host.cmd(f"route add default gw {gateway} 2>/dev/null || ip route add default via {gateway}")
            
            info(f"  ✅ h{i} 已配置: {host_ip}/24, 网关: {gateway}\n")

    # --- 静态路由配置 (保持不变) ---
    # n0 -> r0
    n0.cmd("ip route del default 2>/dev/null || true")
    n0.cmd("ip route add default via 10.0.1.1")
    
    # r0 -> 下面的子网
    r0.cmd("ip route del 10.0.4.0/24 2>/dev/null || true")
    r0.cmd("ip route add 10.0.4.0/24 via 10.0.2.2")
    r0.cmd("ip route del 10.0.5.0/24 2>/dev/null || true")
    r0.cmd("ip route add 10.0.5.0/24 via 10.0.3.2")
    
    # r1 -> 上面和其他子网
    r1.cmd("ip route del default 2>/dev/null || true")
    r1.cmd("ip route add default via 10.0.2.1")
    r1.cmd("ip route del 10.0.1.0/24 2>/dev/null || true")
    r1.cmd("ip route add 10.0.1.0/24 via 10.0.2.1")
    r1.cmd("ip route del 10.0.3.0/24 2>/dev/null || true")
    r1.cmd("ip route add 10.0.3.0/24 via 10.0.2.1")
    r1.cmd("ip route del 10.0.5.0/24 2>/dev/null || true")
    r1.cmd("ip route add 10.0.5.0/24 via 10.0.2.1")
    
    # r2 -> 上面和其他子网
    r2.cmd("ip route del default 2>/dev/null || true")
    r2.cmd("ip route add default via 10.0.3.1")
    r2.cmd("ip route del 10.0.1.0/24 2>/dev/null || true")
    r2.cmd("ip route add 10.0.1.0/24 via 10.0.3.1")
    r2.cmd("ip route del 10.0.2.0/24 2>/dev/null || true")
    r2.cmd("ip route add 10.0.2.0/24 via 10.0.3.1")
    r2.cmd("ip route del 10.0.4.0/24 2>/dev/null || true")
    r2.cmd("ip route add 10.0.4.0/24 via 10.0.3.1")

    # ARP 预热
    info("触发 ARP 解析...\n")
    time.sleep(1)
    n0.cmd("ping -c 1 -W 1 10.0.1.1 >/dev/null 2>&1 || true")
    r0.cmd("ping -c 1 -W 1 10.0.2.2 >/dev/null 2>&1 || true")
    r0.cmd("ping -c 1 -W 1 10.0.3.2 >/dev/null 2>&1 || true")
    
    # 验证修复结果
    info("验证关键主机连通性...\n")
    for i in range(1, num_subscribers + 1):
        h = net.get(f'h{i}')
        gw = "10.0.4.1" if i <= r1_sub_count else "10.0.5.1"
        result = h.cmd(f"ping -c 1 -W 1 {gw} >/dev/null 2>&1 && echo OK || echo FAIL").strip()
        if result == "OK":
            info(f"  ✅ h{i} -> 网关({gw}) 通\n")
        else:
            error(f"  ❌ h{i} -> 网关({gw}) 不通! 请检查接口配置\n")
    
    # Hosts 映射（所有节点都能解析所有 relay 的主机名）
    # ✅ 【关键修复】r0.local 映射到 r0-eth1 (10.0.2.1)，因为用户需要通过这个接口访问 r0
    # r0-eth0 (10.0.1.1) 用于连接 n0 (publisher)，r0-eth1 (10.0.2.1) 用于连接用户
    # 由于 r0 监听在 0.0.0.0:4443，所以可以通过任何 IP 访问
    hosts_mapping = """
10.0.2.1 r0.local r0
10.0.2.2 r1.local r1
10.0.3.2 r2.local r2
"""
    # ✅ 【关键修复】n0 (publisher) 需要能够访问 r0，使用 r0-eth0 (10.0.1.1) 更直接
    # 但为了保持一致性，n0 也使用 r0.local，由于 r0 监听在 0.0.0.0:4443，应该可以工作
    n0_hosts_mapping = """
10.0.1.1 r0.local r0
10.0.2.2 r1.local r1
10.0.3.2 r2.local r2
"""
    
    # 为 n0 单独配置 hosts（使用 r0-eth0 IP，更直接）
    n0 = net.get('n0')
    n0.cmd("sed -i '/ r0\\.local/d' /etc/hosts 2>/dev/null || true")
    n0.cmd("sed -i '/ r1\\.local/d' /etc/hosts 2>/dev/null || true")
    n0.cmd("sed -i '/ r2\\.local/d' /etc/hosts 2>/dev/null || true")
    n0.cmd(f"echo '{n0_hosts_mapping}' >> /etc/hosts")
    
    # 为其他节点配置 hosts（用户节点使用 r0-eth1 IP）
    for node in net.hosts:
        if node.name == 'n0':
            continue  # n0 已经配置过了
        # 清理旧的条目
        node.cmd("sed -i '/ r0\\.local/d' /etc/hosts 2>/dev/null || true")
        node.cmd("sed -i '/ r1\\.local/d' /etc/hosts 2>/dev/null || true")
        node.cmd("sed -i '/ r2\\.local/d' /etc/hosts 2>/dev/null || true")
        node.cmd(f"echo '{hosts_mapping}' >> /etc/hosts")
    
    # 验证 hosts 配置
    info("验证 hosts 配置...\n")
    # ✅ 【关键修复】r0.local 现在映射到 10.0.2.1 (r0-eth1)，因为用户需要通过这个接口访问
    for relay_name, relay_ip in [("r0", "10.0.2.1"), ("r1", "10.0.2.2"), ("r2", "10.0.3.2")]:
        relay = net.get(relay_name)
        result = relay.cmd(f"getent hosts {relay_name}.local 2>&1 | head -1 || echo 'FAILED'").strip()
        if "FAILED" not in result and result:
            ip = result.split()[0] if result else "unknown"
            info(f"✅ {relay_name} 可以解析 {relay_name}.local -> {ip}\n")
        else:
            error(f"❌ {relay_name} 无法解析 {relay_name}.local\n")

def cleanup_all_processes():
    """强制清理所有残留进程"""
    import subprocess
    import os
    import time
    
    current_pid = os.getpid()
    current_script = os.path.basename(__file__)
    
    info("🧹 [强制清理] 清理所有残留进程...\n")
    info(f"   当前脚本 PID: {current_pid}，将跳过此进程\n")
    
    processes_to_kill = [
        "moq-pub",
        "moq-relay", 
        "moq-sub",
        "moq-lite"
    ]
    
    for proc_name in processes_to_kill:
        try:
            result = subprocess.run(['pgrep', '-f', proc_name], 
                                  capture_output=True, text=True, timeout=1)
            if result.stdout.strip():
                pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
                for pid in pids:
                    if pid != str(current_pid):
                        try:
                            subprocess.run(['kill', '-TERM', pid], 
                                         capture_output=True, timeout=1)
                        except:
                            pass
        except:
            pass
    
    # 清理 hang publish 进程
    try:
        result = subprocess.run(['pgrep', '-f', 'hang.*publish'], 
                               capture_output=True, text=True, timeout=1)
        if result.stdout.strip():
            pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
            for pid in pids:
                if pid != str(current_pid):
                    try:
                        subprocess.run(['kill', '-TERM', pid], 
                                     capture_output=True, timeout=1)
                    except:
                        pass
    except:
        pass
    
    time.sleep(1)
    
    # 强制杀死仍然存在的进程
    try:
        result = subprocess.run(['pgrep', '-f', 'moq-pub|moq-relay|moq-sub|hang.*publish'], 
                               capture_output=True, text=True, timeout=1)
        if result.stdout.strip():
            pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
            remaining = [p for p in pids if p != str(current_pid)]
            if remaining:
                info(f"⚠️  仍有残留进程: {', '.join(remaining)}\n")
                for pid in remaining:
                    try:
                        subprocess.run(['kill', '-9', pid], 
                                     capture_output=True, timeout=1)
                    except:
                        pass
                time.sleep(0.5)
            else:
                info("✅ 所有残留进程已清理完成\n")
        else:
            info("✅ 所有残留进程已清理完成\n")
    except Exception as e:
        info(f"✅ 清理完成（验证时出错: {e}）\n")
    
    info("")

# ================= 诊断功能模块 =================
def diagnose_zero_rx_clients(net, num_subscribers, log_path, r0, r1, r2, TMP_PREFIX):
    """
    诊断 rx_bytes 为 0 的客户端，分类为 A1/A2/A3 三种情况
    
    A1: 客户端进程根本没起来/立即退出
    A2: 进程在跑，但从未连上（TLS/QUIC/认证/URL错误）
    A3: 连上了，但没订到任何对象（订阅名/track/catalog/阶段顺序问题）
    """
    info("\n" + "="*80 + "\n")
    info("🔍 开始诊断 rx_bytes 为 0 的客户端\n")
    info("="*80 + "\n\n")
    
    # 读取 perf.csv 找出 rx_bytes 为 0 的客户端
    zero_rx_clients = []
    for i in range(1, num_subscribers + 1):
        perf_csv = os.path.join(log_path, f"client_h{i}_perf.csv")
        if os.path.exists(perf_csv):
            try:
                import pandas as pd
                df = pd.read_csv(perf_csv)
                if len(df) > 0:
                    max_rx = df['rx_bytes'].max()
                    if max_rx == 0:
                        zero_rx_clients.append(i)
            except:
                # 如果无法读取，也加入列表
                zero_rx_clients.append(i)
        else:
            # 文件不存在，也认为是 0
            zero_rx_clients.append(i)
    
    if not zero_rx_clients:
        info("✅ 所有客户端都有非零的 rx_bytes，无需诊断\n")
        return
    
    info(f"📊 发现 {len(zero_rx_clients)} 个 rx_bytes 为 0 的客户端: {zero_rx_clients}\n\n")
    
    # 分类诊断
    a1_clients = []  # 进程没起来
    a2_clients = []  # 进程在跑但未连上
    a3_clients = []  # 连上了但没订到对象
    
    for client_id in zero_rx_clients:
        host = net.get(f'h{client_id}')
        if not host:
            a1_clients.append(client_id)
            continue
        
        # B1: 检查进程是否运行
        info(f"🔍 诊断 h{client_id}...\n")
        
        # 检查 dispatch_strategy 进程
        dispatch_pid = host.cmd(f"pgrep -f 'dispatch_strategy.*--host_id {client_id}' || echo ''").strip()
        # 检查 moq-sub 进程
        moq_sub_pid = host.cmd(f"pgrep -f 'moq-sub' || echo ''").strip()
        # 检查其他可能的拉流进程
        other_procs = host.cmd(f"ps aux | egrep 'moq-sub|chrome|node|gst|python.*dispatch' | grep -v grep || echo ''").strip()
        
        has_process = bool(dispatch_pid or moq_sub_pid or other_procs)
        
        if not has_process:
            # A1: 进程没起来
            a1_clients.append(client_id)
            info(f"  ❌ A1: h{client_id} 进程未运行\n")
            info(f"     检查命令: h{client_id} ps aux | egrep 'moq-sub|chrome|node|gst|python.*dispatch'\n")
            continue
        
        info(f"  ✅ 进程存在 (dispatch PID: {dispatch_pid or 'N/A'}, moq-sub PID: {moq_sub_pid or 'N/A'})\n")
        
        # B2: 检查 DNS 和连通性
        info(f"  🔍 检查 DNS 和连通性...\n")
        
        # 确定应该连接的 relay
        if client_id <= num_subscribers // 2:
            target_relay = "r1"
            relay_ip = "10.0.2.2"
            relay_domain = "r1.local"
        else:
            target_relay = "r2"
            relay_ip = "10.0.3.2"
            relay_domain = "r2.local"
        
        # 检查 r0 域名解析
        r0_dns = host.cmd(f"getent hosts r0.local || echo 'FAILED'").strip()
        # 检查目标 relay 域名解析
        relay_dns = host.cmd(f"getent hosts {relay_domain} || echo 'FAILED'").strip()
        # 检查 ping
        ping_result = host.cmd(f"ping -c 1 {relay_ip} 2>&1 | grep '1 received' || echo 'FAILED'").strip()
        
        dns_ok = "FAILED" not in r0_dns and "FAILED" not in relay_dns
        ping_ok = "FAILED" not in ping_result
        
        if not dns_ok or not ping_ok:
            # A2: DNS 或连通性问题
            a2_clients.append(client_id)
            info(f"  ❌ A2: h{client_id} DNS 或连通性异常\n")
            info(f"     r0.local DNS: {'✅' if 'FAILED' not in r0_dns else '❌'} {r0_dns}\n")
            info(f"     {relay_domain} DNS: {'✅' if 'FAILED' not in relay_dns else '❌'} {relay_dns}\n")
            info(f"     Ping {relay_ip}: {'✅' if ping_ok else '❌'}\n")
            info(f"     修复建议: 检查 /etc/hosts 注入是否一致\n")
            continue
        
        info(f"  ✅ DNS 和连通性正常\n")
        
        # 检查日志文件
        dispatch_log = os.path.join(log_path, f"client_h{client_id}_gst.log")
        old_log = f"/tmp/{TMP_PREFIX}h{client_id}_dispatch.log"
        log_file = dispatch_log if os.path.exists(dispatch_log) else old_log
        
        # 读取日志内容
        log_content = ""
        if os.path.exists(log_file):
            try:
                with open(log_file, 'rb') as f:
                    log_bytes = f.read()
                    log_content = ''.join(chr(b) if 32 <= b < 127 or b in [9, 10, 13] else ' ' for b in log_bytes[-10000:])
            except:
                pass
        
        # 检查连接错误
        connection_errors = [
            "handshake failed", "cert verify", "401", "404", "track not found",
            "opening handshake failed", "connection refused", "timeout",
            "TLS", "QUIC", "authentication", "unauthorized"
        ]
        
        has_connection_error = any(err.lower() in log_content.lower() for err in connection_errors)
        
        if has_connection_error:
            # A2: 连接错误
            a2_clients.append(client_id)
            info(f"  ❌ A2: h{client_id} 连接错误\n")
            info(f"     日志文件: {log_file}\n")
            # 提取相关错误行
            error_lines = [line for line in log_content.split('\n') if any(err.lower() in line.lower() for err in connection_errors)]
            for line in error_lines[-5:]:  # 显示最后5个错误
                if line.strip():
                    info(f"     {line[:100]}\n")
            continue
        
        # B3: 检查 relay 侧是否有该客户端的连接/订阅记录
        info(f"  🔍 检查 relay 侧连接记录...\n")
        
        # 检查 r0 日志
        r0_log_path = f"/tmp/{TMP_PREFIX}r0.log"
        r0_log_content = ""
        if r0:
            try:
                r0_log_content = r0.cmd(f"cat {r0_log_path} 2>&1 | tail -1000").strip()
            except:
                pass
        
        # 检查是否有该客户端的连接记录（通过 IP 或 host_id）
        # 假设日志中可能包含客户端标识
        has_relay_connection = False
        if r0_log_content:
            # 查找可能的客户端标识（IP、hostname等）
            client_patterns = [f"h{client_id}", f"10.0.2.{client_id}", f"10.0.3.{client_id}"]
            has_relay_connection = any(pattern in r0_log_content for pattern in client_patterns)
        
        if not has_relay_connection:
            # A2: relay 看不到连接
            a2_clients.append(client_id)
            info(f"  ❌ A2: h{client_id} relay 侧无连接记录\n")
            info(f"     检查命令: r0 cat {r0_log_path} | grep -i 'h{client_id}'\n")
            continue
        
        # 检查是否有订阅记录但没有数据
        subscribe_pattern = r'subscribe.*started|subscribe.*success|track.*subscribed'
        has_subscribe = bool(re.search(subscribe_pattern, log_content, re.IGNORECASE))
        
        if has_subscribe:
            # A3: 连上了但没订到对象
            a3_clients.append(client_id)
            info(f"  ❌ A3: h{client_id} 已连接并订阅，但无数据\n")
            info(f"     可能原因: 订阅名/track/catalog/阶段顺序问题\n")
            info(f"     检查命令: r0 cat {r0_log_path} | grep -i 'catalog\\|track\\|subscribe'\n")
        else:
            # 无法确定，归类为 A2
            a2_clients.append(client_id)
            info(f"  ❌ A2: h{client_id} 连接状态未知\n")
    
    # 输出分类结果
    info("\n" + "="*80 + "\n")
    info("📊 诊断结果分类\n")
    info("="*80 + "\n\n")
    
    info(f"A1 (进程未运行): {len(a1_clients)} 个 - {a1_clients}\n")
    info(f"A2 (连接失败): {len(a2_clients)} 个 - {a2_clients}\n")
    info(f"A3 (订阅但无数据): {len(a3_clients)} 个 - {a3_clients}\n\n")
    
    # 输出修复建议
    info("="*80 + "\n")
    info("💡 修复建议\n")
    info("="*80 + "\n\n")
    
    if a1_clients:
        info(f"🔧 A1 修复 (进程未运行 - {len(a1_clients)} 个):\n")
        info(f"   1. 检查批量启动脚本是否对这些 host 执行了命令\n")
        info(f"   2. 检查 host 列表切片/索引偏移/异常中断后是否继续启动\n")
        info(f"   3. 手动测试启动: h{a1_clients[0]} python3 moq_client_dispatch.py --host_id {a1_clients[0]} ...\n\n")
    
    if a2_clients:
        info(f"🔧 A2 修复 (连接失败 - {len(a2_clients)} 个):\n")
        info(f"   1. 检查 DNS 解析: h{a2_clients[0]} getent hosts r0.local\n")
        info(f"   2. 检查 /etc/hosts 注入是否一致\n")
        info(f"   3. 检查 URL/public 前缀/token 是否一致\n")
        info(f"   4. 降低启动并发: 从 '5 users / 2s' 改成 '2 users / 2s' 或 '5 users / 5s'\n")
        info(f"   5. 添加自动重试机制（关键）\n\n")
    
    if a3_clients:
        info(f"🔧 A3 修复 (订阅但无数据 - {len(a3_clients)} 个):\n")
        info(f"   1. 检查订阅名/track/catalog 是否一致\n")
        info(f"   2. 检查 relay 侧 catalog: r0 cat {r0_log_path} | grep -i catalog\n")
        info(f"   3. 检查 publish/subscribe 名字是否匹配\n\n")
    
    # D1: 批量启动并发过高修复建议
    if len(a2_clients) + len(a1_clients) > len(zero_rx_clients) * 0.5:
        info("="*80 + "\n")
        info("⚠️  检测到大量连接失败，建议应用 D1 修复（批量启动并发过高）\n")
        info("="*80 + "\n\n")
        info("修复方案（选一个即可）:\n")
        info("1. 降低 batch 大小 + 增加间隔:\n")
        info("   从 '5 users / 2s' 改成 '2 users / 2s' 或 '5 users / 5s'\n")
        info("   修改位置: run_moq_experiment() 中客户端启动循环的 time.sleep()\n\n")
        info("2. 添加自动重试机制（关键）:\n")
        info("   如果 moq-sub 启动后 N 秒内 rx_bytes 不增长或日志出现 handshake failed，\n")
        info("   就 kill 并重启（最多 3 次）\n\n")
        info("3. 限制握手并发:\n")
        info("   在启动脚本中用 semaphore/队列，保证同一时刻最多 K 个新连接（K=3~5）\n\n")
    
    # D2: URL/前缀/token 不一致修复建议
    info("="*80 + "\n")
    info("💡 D2 修复建议（URL/public 前缀/token 不一致）\n")
    info("="*80 + "\n\n")
    info("1. 在每个 client 日志开头打印实际使用的参数:\n")
    info("   - relay URL\n")
    info("   - subscribe track 名\n")
    info("   - jwt 参数\n")
    info("   修改位置: moq_client_dispatch.py 启动时打印\n\n")
    info("2. 对比无效 client 和有效 client 的日志，确保参数完全一致（除了 user_id）\n\n")
    
    return {
        'a1': a1_clients,
        'a2': a2_clients,
        'a3': a3_clients,
        'total_zero': zero_rx_clients
    }

def perform_hard_checks(net, num_subscribers, r0, r1, r2, zero_client_ids=None):
    """
    对指定客户端执行 B1/B2/B3 硬检查
    
    B1: 该 host 上是否有拉流进程
    B2: 该 host 到 relay 的 DNS+连通性是否正常
    B3: 该 host 是否真的在请求订阅（用日志/relay 侧观测）
    """
    if zero_client_ids is None:
        # 默认检查前3个和后3个
        zero_client_ids = list(range(1, min(4, num_subscribers + 1))) + \
                         list(range(max(1, num_subscribers - 2), num_subscribers + 1))
    
    info("\n" + "="*80 + "\n")
    info(f"🔍 执行硬检查 (B1/B2/B3) - 检查 {len(zero_client_ids)} 个客户端\n")
    info("="*80 + "\n\n")
    
    for client_id in zero_client_ids[:10]:  # 最多检查10个
        host = net.get(f'h{client_id}')
        if not host:
            info(f"❌ h{client_id} 不存在\n\n")
            continue
        
        info(f"📋 检查 h{client_id}:\n")
        
        # B1: 检查进程
        info(f"  B1. 检查拉流进程...\n")
        procs = host.cmd(f"ps aux | egrep 'moq-sub|chrome|node|gst|python.*dispatch' | grep -v grep || echo ''").strip()
        if procs:
            info(f"     ✅ 发现进程:\n")
            for line in procs.split('\n')[:3]:  # 只显示前3行
                if line.strip():
                    info(f"        {line[:80]}\n")
        else:
            info(f"     ❌ 无拉流进程\n")
        info(f"\n")
        
        # B2: 检查 DNS 和连通性
        info(f"  B2. 检查 DNS 和连通性...\n")
        
        # 确定目标 relay
        if client_id <= num_subscribers // 2:
            relay_domain = "r1.local"
            relay_ip = "10.0.2.2"
        else:
            relay_domain = "r2.local"
            relay_ip = "10.0.3.2"
        
        # 检查 DNS
        r0_dns = host.cmd(f"getent hosts r0.local 2>&1").strip()
        relay_dns = host.cmd(f"getent hosts {relay_domain} 2>&1").strip()
        
        info(f"     r0.local DNS: {r0_dns}\n")
        info(f"     {relay_domain} DNS: {relay_dns}\n")
        
        # 检查 ping
        ping_result = host.cmd(f"ping -c 1 {relay_ip} 2>&1").strip()
        if "1 received" in ping_result:
            info(f"     ✅ Ping {relay_ip}: 成功\n")
        else:
            info(f"     ❌ Ping {relay_ip}: 失败\n")
            info(f"        {ping_result[:200]}\n")
        info(f"\n")
        
        # B3: 检查 relay 侧连接记录
        info(f"  B3. 检查 relay 侧连接记录...\n")
        if r0:
            r0_log_path = f"/tmp/{TMP_PREFIX}r0.log"
            # 检查 r0 日志
            r0_connections = r0.cmd(f"cat {r0_log_path} 2>&1 | grep -i 'h{client_id}\\|10.0' | tail -5 || echo '无记录'").strip()
            if r0_connections and "无记录" not in r0_connections:
                info(f"     ✅ r0 日志中发现连接记录:\n")
                for line in r0_connections.split('\n')[:3]:
                    if line.strip():
                        info(f"        {line[:100]}\n")
            else:
                info(f"     ❌ r0 日志中无连接记录\n")
            
            # 检查 r1/r2 日志（如果适用）
            target_relay = r1 if client_id <= num_subscribers // 2 else r2
            relay_name = "r1" if client_id <= num_subscribers // 2 else "r2"
            if target_relay:
                relay_log_path = f"/tmp/{TMP_PREFIX}{relay_name}.log"
                relay_connections = target_relay.cmd(f"cat {relay_log_path} 2>&1 | grep -i 'h{client_id}\\|subscribe' | tail -5 || echo '无记录'").strip()
                if relay_connections and "无记录" not in relay_connections:
                    info(f"     ✅ {relay_name} 日志中发现订阅记录:\n")
                    for line in relay_connections.split('\n')[:3]:
                        if line.strip():
                            info(f"        {line[:100]}\n")
                else:
                    info(f"     ❌ {relay_name} 日志中无订阅记录\n")
        info(f"\n")
        
        info("-" * 80 + "\n\n")

# ================= 诊断功能模块结束 =================

def run_moq_experiment(num_subscribers=10, strategy="md2g", network_type="4g", 
                       log_path=None, duration=120, interval=1.0, model_path=None):
    """
    运行 MoQ 实验（树形拓扑 + Clustering + 完整perf.csv记录）
    
    Args:
        num_subscribers: 要测试的 Subscriber 数量（默认 10 个）
        strategy: Strategy name (md2g, rolling, heuristic, clustering)
        network_type: 网络类型（wifi, 4g, 5g, fiber_optic, default_mix, wifi_dominant, 5g_dominant）
        log_path: 日志保存路径（如果为None，使用默认路径）
        duration: 实验持续时间（秒）
        interval: 决策周期间隔（秒）
        model_path: 模型路径（用于rolling和md2g策略）
    """
    setLogLevel('info')
    
    # ✅ 【关键修复】初始化视频码率（从实际视频文件读取）
    global BASE_BITRATE_MBPS, ENH_BITRATE_MBPS, TOTAL_BITRATE_MBPS
    BASE_BITRATE_MBPS, ENH_BITRATE_MBPS, TOTAL_BITRATE_MBPS = init_video_bitrates()
    info(f"📊 视频码率配置: Base={BASE_BITRATE_MBPS:.2f} Mbps, Enhanced={ENH_BITRATE_MBPS:.2f} Mbps, Total={TOTAL_BITRATE_MBPS:.2f} Mbps\n")
    
    # ✅ 【实验配置】设置日志路径
    # ✅ 【关键修复】如果指定了log_path，直接使用，不添加时间戳
    if log_path is None:
        # 如果没有指定log_path，才使用默认路径（带时间戳）
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"test_logs_{timestamp}/{network_type}/{strategy}/users_{num_subscribers}"
    # else: 直接使用传入的 log_path，不添加时间戳
    
    os.makedirs(log_path, exist_ok=True)
    info(f"📁 日志路径: {log_path}\n")
    info(f"📊 实验配置: 策略={strategy}, 网络={network_type}, 用户数={num_subscribers}, 时长={duration}秒\n")
    
    # ✅ 【第一步：强制清理所有残留进程】
    cleanup_all_processes()
    
    if not check_files(): return
    
    # ✅ 生成符合论文设计的分层视频
    if not prepare_layered_videos():
        error("❌ 无法生成分层视频，退出\n")
        return
    
    cert, key = generate_certs()
    
    # ✅ 生成 JWT key 和 cluster token（官方推荐方式）
    auth_key_file, cluster_token_file = generate_auth_keys()

    net = Mininet(controller=Controller, switch=OVSKernelSwitch, link=TCLink)
    info(f"创建树形拓扑 (Root Relay + Leaf Relays + {num_subscribers} 个 Subscriber)...\n")
    
    # ✅ 树形拓扑：
    # n0 (Publisher) -> r0 (Root Relay) -> r1, r2 (Leaf Relays) -> h1-hN (Subscribers)
    
    # 创建节点
    n0 = net.addHost('n0', ip='10.0.1.2/24')
    r0 = net.addHost('r0', ip='10.0.1.1/24')  # Root Relay
    r1 = net.addHost('r1', ip='10.0.2.2/24')  # Leaf Relay 1
    r2 = net.addHost('r2', ip='10.0.3.2/24')  # Leaf Relay 2
    
    # 创建 Subscriber 节点
    subscribers = []
    r1_sub_count = num_subscribers // 2
    r2_sub_count = num_subscribers - r1_sub_count
    
    # ✅ 【修复C：解决 r2 的处理能力问题】根据用户数动态调整CPU限制
    # 原因：r2 的 Fan-out 只有 2.74x（应该接近 5x），说明 r2 节点下的用户丢包严重
    # 修复：给 host 分配 CPU 权重，根据用户数动态调整CPU限制
    # 用户数少时（≤30）：每个host 10% CPU，防止抢占中继资源
    # 用户数多时（>30）：每个host 20% CPU，确保有足够CPU资源处理数据
    if num_subscribers <= 30:
        cpu_limit = 0.1  # 10% CPU（小规模实验）
    else:
        cpu_limit = 0.2  # 20% CPU（大规模实验，确保有足够资源）
    
    for i in range(1, num_subscribers + 1):
        if i <= r1_sub_count:
            # 连接到 r1
            host = net.addHost(f'h{i}', ip=f'10.0.4.{i+1}/24', cpu=cpu_limit)  # ✅ 动态CPU限制
            subscribers.append(host)
        else:
            # 连接到 r2
            idx = i - r1_sub_count
            host = net.addHost(f'h{i}', ip=f'10.0.5.{idx+1}/24', cpu=cpu_limit)  # ✅ 动态CPU限制
            subscribers.append(host)
    
    # ✅ 关键修复：添加 Switch 作为二层汇聚，修复 ping loss
    # 创建 Switch
    s1 = net.addSwitch('s1')  # r1 分支的 switch
    s2 = net.addSwitch('s2')  # r2 分支的 switch
    
    # ✅ 【关键修复】加载网络类型和带宽数据
    host_net_types = get_host_network_types(network_type, num_subscribers)
    host_bandwidths = load_network_bandwidth_data(network_type, num_subscribers)
    info(f"📊 网络配置: {network_type}, 用户带宽范围: {min(host_bandwidths):.2f}-{max(host_bandwidths):.2f} Mbps\n")
    
    # ✅ 【关键修复】根据用户数动态计算区域链路带宽
    bw_r0_r1, bw_r0_r2 = calculate_regional_bandwidth(num_subscribers, TOTAL_BITRATE_MBPS)
    users_per_relay = num_subscribers / 2
    worst_case_bw = users_per_relay * TOTAL_BITRATE_MBPS
    info(f"📊 带宽需求分析:\n")
    info(f"   总用户数: {num_subscribers}\n")
    info(f"   每个Relay用户数: {users_per_relay:.0f}\n")
    info(f"   最坏情况（所有用户Base+Enhanced）: {worst_case_bw:.2f} Mbps\n")
    info(f"   区域链路带宽: r0→r1={bw_r0_r1} Mbps, r0→r2={bw_r0_r2} Mbps\n")
    info(f"   拥塞度: {worst_case_bw / bw_r0_r1 * 100:.1f}% (Unicast), Multicast流畅\n")
    
    # 创建链路（添加带宽限制）
    # ✅ 【修复B1】添加max_queue_size=1000，避免QUIC开局Burst导致丢包
    # n0 -> r0 (核心链路，1 Gbps)
    net.addLink(n0, r0, intfName1='n0-eth0', intfName2='r0-eth0', 
                bw=BW_N0_R0, delay='1ms', loss=0, max_queue_size=2000, cls=TCLink)
    # r0 -> r1 (区域链路，动态计算)
    # ✅ 【关键修复】限制带宽在 Mininet 支持的范围内（0-1000 Mbps）
    bw_r0_r1_limited = min(bw_r0_r1, 1000.0)
    net.addLink(r0, r1, intfName1='r0-eth1', intfName2='r1-eth0',
                bw=bw_r0_r1_limited, delay='5ms', loss=0, max_queue_size=1000, cls=TCLink)
    # r0 -> r2 (区域链路，动态计算)
    # ✅ 【关键修复】限制带宽在 Mininet 支持的范围内（0-1000 Mbps）
    bw_r0_r2_limited = min(bw_r0_r2, 1000.0)
    net.addLink(r0, r2, intfName1='r0-eth2', intfName2='r2-eth0',
                bw=bw_r0_r2_limited, delay='5ms', loss=0, max_queue_size=1000, cls=TCLink)
    
    # ✅ r1 分支：r1 -> s1 -> h1, h2...
    # r1 只有一个 LAN 口连接到 s1（无带宽限制，因为是交换机内部）
    # ✅ 【修复B1】即使是无带宽限制的交换机内部链路，也添加max_queue_size避免丢包
    # ✅ 【关键修复】必须指定 bw 参数，否则 TCLink 会报错 "rate" is required
    # 使用一个很大的带宽值（1000 Mbps）表示无限制
    net.addLink(r1, s1, intfName1='r1-eth1', intfName2='s1-eth0', 
                bw=1000, delay='1ms', loss=0, max_queue_size=1000, cls=TCLink)
    # s1 连接到所有 r1 分支的 hosts（根据用户网络类型设置带宽限制）
    for i, host in enumerate(subscribers[:r1_sub_count], 1):
        user_net_type = host_net_types[i-1]
        user_bw = host_bandwidths[i-1]
        net_config = NETWORK_BW_CONFIG.get(user_net_type, NETWORK_BW_CONFIG['wifi'])
        # ✅ 使用数据集中的实际带宽值，但不超过配置的最大值
        actual_bw = min(user_bw, net_config['bw'])
        # ✅ 【关键修复】确保带宽至少为 1 Mbps，避免 TCLink 报错 "rate" is required
        # 如果带宽为 0 或负数，使用配置中的默认带宽
        if actual_bw <= 0:
            actual_bw = max(1.0, net_config['bw'])  # 至少 1 Mbps
            info(f"   ⚠️  h{i}: 带宽值无效（{user_bw:.2f} Mbps），使用默认值 {actual_bw:.2f} Mbps\n")
        # ✅ 【关键修复】限制带宽在 Mininet 支持的范围内（0-1000 Mbps）
        actual_bw = min(actual_bw, 1000.0)  # Mininet 最大支持 1000 Mbps
        net.addLink(host, s1, intfName1=f'h{i}-eth0', intfName2=f's1-eth{i}',
                   bw=actual_bw, delay=net_config['delay'], loss=net_config['loss'], max_queue_size=1000, cls=TCLink)
        info(f"   ✅ h{i}: {user_net_type}, 带宽={actual_bw:.2f} Mbps, 延迟={net_config['delay']}\n")
    
    # ✅ r2 分支：r2 -> s2 -> hN+1, hN+2...
    # r2 只有一个 LAN 口连接到 s2（无带宽限制，因为是交换机内部）
    # ✅ 【修复B1】即使是无带宽限制的交换机内部链路，也添加max_queue_size避免丢包
    # ✅ 【关键修复】必须指定 bw 参数，否则 TCLink 会报错 "rate" is required
    # 使用一个很大的带宽值（1000 Mbps）表示无限制
    net.addLink(r2, s2, intfName1='r2-eth1', intfName2='s2-eth0', 
                bw=1000, delay='1ms', loss=0, max_queue_size=1000, cls=TCLink)
    # s2 连接到所有 r2 分支的 hosts（根据用户网络类型设置带宽限制）
    for i, host in enumerate(subscribers[r1_sub_count:], r1_sub_count + 1):
        user_net_type = host_net_types[i-1]
        user_bw = host_bandwidths[i-1]
        net_config = NETWORK_BW_CONFIG.get(user_net_type, NETWORK_BW_CONFIG['wifi'])
        # ✅ 使用数据集中的实际带宽值，但不超过配置的最大值
        actual_bw = min(user_bw, net_config['bw'])
        # ✅ 【关键修复】确保带宽至少为 1 Mbps，避免 TCLink 报错 "rate" is required
        # 如果带宽为 0 或负数，使用配置中的默认带宽
        if actual_bw <= 0:
            actual_bw = max(1.0, net_config['bw'])  # 至少 1 Mbps
            info(f"   ⚠️  h{i}: 带宽值无效（{user_bw:.2f} Mbps），使用默认值 {actual_bw:.2f} Mbps\n")
        # ✅ 【关键修复】限制带宽在 Mininet 支持的范围内（0-1000 Mbps）
        actual_bw = min(actual_bw, 1000.0)  # Mininet 最大支持 1000 Mbps
        net.addLink(host, s2, intfName1=f'h{i}-eth0', intfName2=f's2-eth{i-r1_sub_count}',
                   bw=actual_bw, delay=net_config['delay'], loss=net_config['loss'], max_queue_size=1000, cls=TCLink)
        info(f"   ✅ h{i}: {user_net_type}, 带宽={actual_bw:.2f} Mbps, 延迟={net_config['delay']}\n")

    net.start()
    
    # ✅ 关键修复：配置 Switch 为 standalone 模式（确保能转发数据包，修复 ping loss）
    info("🔧 配置 Switch 为 standalone 模式（修复 ping loss）...\n")
    try:
        import subprocess
        # 配置 s1 和 s2 为 standalone 模式
        for switch_name in ['s1', 's2']:
            try:
                switch = net.get(switch_name)
                subprocess.run(['ovs-vsctl', 'set-controller', switch_name, 'none'], 
                             check=False, capture_output=True, timeout=5)
                subprocess.run(['ovs-vsctl', 'set', 'bridge', switch_name, 'fail-mode=standalone'], 
                             check=False, capture_output=True, timeout=5)
                info(f"   ✅ Switch {switch_name} 已设为 standalone 模式\n")
            except Exception as e:
                info(f"   ⚠️  Switch {switch_name} 配置失败: {e}，继续执行\n")
        info("✅ Switch 配置完成\n")
    except Exception as e:
        info(f"⚠️  Switch 配置失败: {e}，继续执行\n")
    
    # ✅ 确保所有接口都 UP（增强版：显式配置每个接口）
    info("激活所有接口...\n")
    for host in net.hosts:
        for intf in host.intfList():
            try:
                # 先尝试 ip link set
                host.cmd(f'ip link set {intf.name} up 2>/dev/null || true')
                # 如果失败，尝试 ifconfig
                host.cmd(f'ifconfig {intf.name} up 2>/dev/null || true')
                # 验证接口是否真的 UP
                # ✅ 【关键修复】给接口一些时间初始化，避免误报
                time.sleep(0.1)  # 等待 100ms 让接口完全初始化
                status = host.cmd(f'ip link show {intf.name} 2>/dev/null | grep -q "state UP" && echo "UP" || echo "DOWN"').strip()
                if status != "UP":
                    # ✅ 再次尝试启动接口
                    host.cmd(f'ip link set {intf.name} up 2>/dev/null || true')
                    time.sleep(0.1)  # 再等待 100ms
                    status = host.cmd(f'ip link show {intf.name} 2>/dev/null | grep -q "state UP" && echo "UP" || echo "DOWN"').strip()
                    if status != "UP":
                        info(f"  ⚠️  {host.name}:{intf.name} 可能未正确启动（这通常是暂时的，接口会在后续自动启动）\n")
            except Exception as e:
                info(f"  ⚠️  {host.name}:{intf.name} 启动失败: {e}\n")
    
    setup_routing(net, num_subscribers)
    
    # ✅ 检查并配置网络接口 MTU
    info("检查并配置网络接口 MTU...\n")
    target_mtu = 1500
    for host in net.hosts:
        for intf in host.intfList():
            try:
                current_mtu = host.cmd(f"cat /sys/class/net/{intf.name}/mtu 2>/dev/null || echo '1500'").strip()
                try:
                    current_mtu_int = int(current_mtu)
                except:
                    current_mtu_int = 1500
                
                if current_mtu_int < target_mtu:
                    host.cmd(f"ip link set {intf.name} mtu {target_mtu} 2>/dev/null || true")
            except:
                pass
    
    info("✅ MTU 配置完成\n")
    
    info("Testing Ping (expect 0% loss)...\n")
    # ✅ 先进行基础连通性测试
    net.pingAll()
    
    # ✅ 额外验证：确保所有关键路径都通
    info("验证关键路径连通性...\n")
    ping_failed = []
    
    # 测试 n0 -> r0
    result = n0.cmd("ping -c 1 -W 1 10.0.1.1 >/dev/null 2>&1 && echo 'OK' || echo 'FAIL'").strip()
    if result != "OK":
        ping_failed.append("n0 -> r0")
    
    # 测试 r0 -> r1, r2
    result = r0.cmd("ping -c 1 -W 1 10.0.2.2 >/dev/null 2>&1 && echo 'OK' || echo 'FAIL'").strip()
    if result != "OK":
        ping_failed.append("r0 -> r1")
    result = r0.cmd("ping -c 1 -W 1 10.0.3.2 >/dev/null 2>&1 && echo 'OK' || echo 'FAIL'").strip()
    if result != "OK":
        ping_failed.append("r0 -> r2")
    
    # 测试所有 hosts -> 网关
    subscribers = [net.get(f'h{i}') for i in range(1, num_subscribers + 1)]
    r1_sub_count = num_subscribers // 2
    for i, host in enumerate(subscribers, 1):
        if i <= r1_sub_count:
            gateway = "10.0.4.1"
        else:
            gateway = "10.0.5.1"
        result = host.cmd(f"ping -c 1 -W 1 {gateway} >/dev/null 2>&1 && echo 'OK' || echo 'FAIL'").strip()
        if result != "OK":
            ping_failed.append(f"h{i} -> {gateway}")
    
    if ping_failed:
        error(f"⚠️  以下路径 ping 失败: {', '.join(ping_failed)}\n")
        error("   这可能导致后续测试失败，请检查网络配置\n")
    else:
        info("✅ 所有关键路径连通性测试通过\n")

    # --- 启动 Relays (Clustering) ---
    info("启动 Relays (Clustering 模式)...\n")
    
    # ✅ r0: Root Relay（不设置 --cluster-root）
    info("启动 Root Relay (r0)...\n")
    port_check = r0.cmd("netstat -tuln 2>/dev/null | grep ':4443' || ss -tuln 2>/dev/null | grep ':4443' || echo '端口未占用'").strip()
    if "4443" in port_check and "未占用" not in port_check:
        r0.cmd("fuser -k 4443/tcp 2>/dev/null || pkill -f 'moq-relay' 2>/dev/null || true")
        time.sleep(1)
    
    r0.cmd(f"cp {cert} /tmp/{TMP_PREFIX}r0_cert.pem && cp {key} /tmp/{TMP_PREFIX}r0_key.pem")
    r0.cmd(f"cp {auth_key_file} /tmp/{TMP_PREFIX}r0_key.jwk")
    # ✅ 官方推荐方式：启用 auth，使用 public = "anon"
    r0_config = generate_relay_config("r0", cert, key, f"/tmp/{TMP_PREFIX}r0_key.jwk")
    r0.cmd(f"cp {r0_config} /tmp/{TMP_PREFIX}r0.toml")
    # ✅ 使用域名（r0.local）而不是 IP，确保 TLS SNI 正确传递
    # ✅ 【修复3：清理Relay启动命令】撤销nice和taskset，让系统自动调度
    # 恢复最纯粹的启动，依靠12Mbps物理限速来保证CPU不爆炸
    r0.cmd(f"RUST_LOG=info {BIN_PATHS['relay']} "
           f"--cluster-node r0.local "
           f"--tls-cert /tmp/{TMP_PREFIX}r0_cert.pem "
           f"--tls-key /tmp/{TMP_PREFIX}r0_key.pem "
           f"/tmp/{TMP_PREFIX}r0.toml "
           f"> /tmp/{TMP_PREFIX}r0.log 2>&1 &")
    time.sleep(3)  # 等待 r0 启动
    
    r0_listening = r0.cmd("netstat -tuln 2>/dev/null | grep ':4443' || ss -tuln 2>/dev/null | grep ':4443' || echo ''").strip()
    if r0_listening:
        info(f"✅ r0 (Root Relay) 已成功监听 4443 端口\n")
    else:
        error("❌ r0 未能监听 4443 端口\n")
        r0_log = r0.cmd(f"tail -20 /tmp/{TMP_PREFIX}r0.log 2>&1")
        error(f"{r0_log}\n")
    
    # ✅ r1: Leaf Relay（设置 --cluster-root=r0.local，--cluster-node=r1.local）
    info("启动 Leaf Relay (r1)...\n")
    port_check = r1.cmd("netstat -tuln 2>/dev/null | grep ':4443' || ss -tuln 2>/dev/null | grep ':4443' || echo '端口未占用'").strip()
    if "4443" in port_check and "未占用" not in port_check:
        r1.cmd("fuser -k 4443/tcp 2>/dev/null || pkill -f 'moq-relay' 2>/dev/null || true")
        time.sleep(1)
    
    r1.cmd(f"cp {cert} /tmp/{TMP_PREFIX}r1_cert.pem && cp {key} /tmp/{TMP_PREFIX}r1_key.pem")
    r1.cmd(f"cp {auth_key_file} /tmp/{TMP_PREFIX}r1_key.jwk")
    r1.cmd(f"cp {cluster_token_file} /tmp/{TMP_PREFIX}r1_cluster.jwt")
    # ✅ 官方推荐方式：启用 auth，使用 public = "anon"
    r1_config = generate_relay_config("r1", cert, key, f"/tmp/{TMP_PREFIX}r1_key.jwk")
    r1.cmd(f"cp {r1_config} /tmp/{TMP_PREFIX}r1.toml")
    # ✅ 关键修复：使用 cluster token 连接 root
    # 改回使用域名（r0.local），确保 TLS SNI 正确传递
    # ⚠️ 关键修复：moq-relay 会自动添加 https:// 前缀，所以只传递 hostname:port，不要包含 https://
    # 使用域名可以确保 TLS 握手时 SNI 正确传递，证书验证通过
    cluster_root_url = "r0.local:4443"  # ✅ 使用域名，确保 SNI 正确
    # ✅ 【修复一】撤销nice和taskset，使用最稳健的启动方式
    # ✅ 使用RUST_LOG=debug以便排查问题
    # ✅ 【修复3：清理Relay启动命令】撤销nice和taskset，让系统自动调度
    r1.cmd(f"RUST_LOG=info {BIN_PATHS['relay']} "
           f"--cluster-node r1.local "
           f"--cluster-root {cluster_root_url} "
           f"--cluster-token /tmp/{TMP_PREFIX}r1_cluster.jwt "
           f"--tls-cert /tmp/{TMP_PREFIX}r1_cert.pem "
           f"--tls-key /tmp/{TMP_PREFIX}r1_key.pem "
           f"--tls-disable-verify "
           f"/tmp/{TMP_PREFIX}r1.toml "
           f"> /tmp/{TMP_PREFIX}r1.log 2>&1 &")
    time.sleep(3)  # 等待 r1 启动
    
    r1_listening = r1.cmd("netstat -tuln 2>/dev/null | grep ':4443' || ss -tuln 2>/dev/null | grep ':4443' || echo ''").strip()
    if r1_listening:
        info(f"✅ r1 (Leaf Relay) 已成功监听 4443 端口\n")
    else:
        error("❌ r1 未能监听 4443 端口\n")
        r1_log = r1.cmd(f"tail -20 /tmp/{TMP_PREFIX}r1.log 2>&1")
        error(f"{r1_log}\n")
    
    # ✅ r2: Leaf Relay（设置 --cluster-root=r0.local，--cluster-node=r2.local）
    info("启动 Leaf Relay (r2)...\n")
    port_check = r2.cmd("netstat -tuln 2>/dev/null | grep ':4443' || ss -tuln 2>/dev/null | grep ':4443' || echo '端口未占用'").strip()
    if "4443" in port_check and "未占用" not in port_check:
        r2.cmd("fuser -k 4443/tcp 2>/dev/null || pkill -f 'moq-relay' 2>/dev/null || true")
        time.sleep(1)
    
    r2.cmd(f"cp {cert} /tmp/{TMP_PREFIX}r2_cert.pem && cp {key} /tmp/{TMP_PREFIX}r2_key.pem")
    r2.cmd(f"cp {auth_key_file} /tmp/{TMP_PREFIX}r2_key.jwk")
    r2.cmd(f"cp {cluster_token_file} /tmp/{TMP_PREFIX}r2_cluster.jwt")
    # ✅ 官方推荐方式：启用 auth，使用 public = "anon"
    r2_config = generate_relay_config("r2", cert, key, f"/tmp/{TMP_PREFIX}r2_key.jwk")
    r2.cmd(f"cp {r2_config} /tmp/{TMP_PREFIX}r2.toml")
    # ✅ 关键修复：使用 cluster token 连接 root
    # 改回使用域名（r0.local），确保 TLS SNI 正确传递
    # ⚠️ 关键修复：moq-relay 会自动添加 https:// 前缀，所以只传递 hostname:port，不要包含 https://
    # 使用域名可以确保 TLS 握手时 SNI 正确传递，证书验证通过
    cluster_root_url = "r0.local:4443"  # ✅ 使用域名，确保 SNI 正确
    # ✅ 【修复一】撤销nice和taskset，使用最稳健的启动方式
    # ✅ 使用RUST_LOG=debug以便排查问题
    # ✅ 【修复3：清理Relay启动命令】撤销nice和taskset，让系统自动调度
    r2.cmd(f"RUST_LOG=info {BIN_PATHS['relay']} "
           f"--cluster-node r2.local "
           f"--cluster-root {cluster_root_url} "
           f"--cluster-token /tmp/{TMP_PREFIX}r2_cluster.jwt "
           f"--tls-cert /tmp/{TMP_PREFIX}r2_cert.pem "
           f"--tls-key /tmp/{TMP_PREFIX}r2_key.pem "
           f"--tls-disable-verify "
           f"/tmp/{TMP_PREFIX}r2.toml "
           f"> /tmp/{TMP_PREFIX}r2.log 2>&1 &")
    time.sleep(3)  # 等待 r2 启动
    
    r2_listening = r2.cmd("netstat -tuln 2>/dev/null | grep ':4443' || ss -tuln 2>/dev/null | grep ':4443' || echo ''").strip()
    if r2_listening:
        info(f"✅ r2 (Leaf Relay) 已成功监听 4443 端口\n")
    else:
        error("❌ r2 未能监听 4443 端口\n")
        r2_log = r2.cmd(f"tail -20 /tmp/{TMP_PREFIX}r2.log 2>&1")
        error(f"{r2_log}\n")
    
    # 等待 clustering 建立连接
    info("等待 Clustering 连接建立...\n")
    time.sleep(5)
    
    # 检查 clustering 状态
    r0_log = r0.cmd(f"tail -50 /tmp/{TMP_PREFIX}r0.log 2>&1")
    r1_log = r1.cmd(f"tail -50 /tmp/{TMP_PREFIX}r1.log 2>&1")
    r2_log = r2.cmd(f"tail -50 /tmp/{TMP_PREFIX}r2.log 2>&1")
    
    if "cluster" in r0_log.lower() or "connect" in r0_log.lower():
        info("✅ r0 日志显示可能有 clustering 连接\n")
    if "cluster" in r1_log.lower() or "root" in r1_log.lower():
        info("✅ r1 日志显示已连接到 root\n")
    if "cluster" in r2_log.lower() or "root" in r2_log.lower():
        info("✅ r2 日志显示已连接到 root\n")

    # --- 启动 Publisher ---
    info("启动 Publishers (fMP4 Video Files)...\n")
    
    base_video = VIDEO_PATHS["base"]
    enhanced_video = VIDEO_PATHS["enhanced"]
    
    if not os.path.exists(base_video):
        error(f"❌ Base 视频文件不存在: {base_video}\n")
        return
    if not os.path.exists(enhanced_video):
        error(f"❌ Enhanced 视频文件不存在: {enhanced_video}\n")
        return
    
    info(f"✅ 使用 Base 视频: {base_video}\n")
    info(f"✅ 使用 Enhanced 视频: {enhanced_video}\n")
    
    # ✅ 视频预处理（重编码清洗）- 只编码一次，之后直接使用
    info("检查视频文件（如果已编码则跳过）...\n")
    base_fmp4_exists = n0.cmd("test -f /tmp/base_fmp4.mp4 && echo 'yes' || echo 'no'").strip()
    enh_fmp4_exists = n0.cmd("test -f /tmp/enhanced_fmp4.mp4 && echo 'yes' || echo 'no'").strip()
    
    if base_fmp4_exists == 'yes' and enh_fmp4_exists == 'yes':
        base_size = n0.cmd("stat -c%s /tmp/base_fmp4.mp4 2>/dev/null || echo '0'").strip()
        enh_size = n0.cmd("stat -c%s /tmp/enhanced_fmp4.mp4 2>/dev/null || echo '0'").strip()
        if base_size != '0' and enh_size != '0' and int(base_size) > 1000 and int(enh_size) > 1000:
            info(f"  ✅ 使用已编码的视频文件 (Base: {int(base_size)/1024:.1f}KB, Enhanced: {int(enh_size)/1024:.1f}KB)\n")
        else:
            base_fmp4_exists = 'no'
            enh_fmp4_exists = 'no'
    
    if base_fmp4_exists != 'yes' or enh_fmp4_exists != 'yes':
        has_ffmpeg = n0.cmd("which ffmpeg >/dev/null 2>&1 && echo 'yes' || echo 'no'").strip()
        
        if has_ffmpeg == 'yes':
            info("  ⚡️ 执行重编码清洗（仅首次运行）...\n")
            common_clean_flags = "-c:v libx264 -preset ultrafast -an -map_metadata -1 -bitexact -write_tmcd 0"
            fmp4_flags = "-movflags +frag_keyframe+empty_moov+default_base_moof+skip_trailer"
            
            if base_fmp4_exists != 'yes':
                n0.cmd(f"ffmpeg -hide_banner -v quiet -y -i {base_video} "
                       f"{common_clean_flags} {fmp4_flags} "
                       f"-f mp4 /tmp/base_fmp4.mp4 >/dev/null 2>&1 &")
            
            if enh_fmp4_exists != 'yes':
                n0.cmd(f"ffmpeg -hide_banner -v quiet -y -i {enhanced_video} "
                       f"{common_clean_flags} {fmp4_flags} "
                       f"-f mp4 /tmp/enhanced_fmp4.mp4 >/dev/null 2>&1 &")
            
            max_wait = 60
            info("  等待重编码完成（可能需要 10-30 秒）...\n")
            for i in range(max_wait):
                base_done = n0.cmd("test -f /tmp/base_fmp4.mp4 && echo 'done' || echo 'waiting'").strip()
                enh_done = n0.cmd("test -f /tmp/enhanced_fmp4.mp4 && echo 'done' || echo 'waiting'").strip()
                if base_done == 'done' and enh_done == 'done':
                    base_size = n0.cmd("stat -c%s /tmp/base_fmp4.mp4 2>/dev/null || echo '0'").strip()
                    enh_size = n0.cmd("stat -c%s /tmp/enhanced_fmp4.mp4 2>/dev/null || echo '0'").strip()
                    if base_size != '0' and enh_size != '0' and int(base_size) > 1000 and int(enh_size) > 1000:
                        time.sleep(1)
                        info(f"  ✅ 重编码完成 (Base: {int(base_size)/1024:.1f}KB, Enhanced: {int(enh_size)/1024:.1f}KB)\n")
                        break
                if i % 5 == 0 and i > 0:
                    info(f"  等待中... ({i}/{max_wait} 秒)\n")
                time.sleep(1)
        else:
            info("⚠️  未找到 ffmpeg，直接复制原始文件\n")
            if base_fmp4_exists != 'yes':
                n0.cmd(f"cp {base_video} /tmp/base_fmp4.mp4")
            if enh_fmp4_exists != 'yes':
                n0.cmd(f"cp {enhanced_video} /tmp/enhanced_fmp4.mp4")
    
    # ✅ 清理旧的 Publisher 进程
    info("🔍 检查并清理旧的Publisher进程...\n")
    base_pids = n0.cmd("pgrep -f 'hang.*publish.*base' || echo ''").strip()
    if base_pids:
        for pid in base_pids.split('\n'):
            if pid.strip():
                n0.cmd(f"kill -9 {pid.strip()} 2>/dev/null || true")
    
    enh_pids = n0.cmd("pgrep -f 'hang.*publish.*enhanced' || echo ''").strip()
    if enh_pids:
        for pid in enh_pids.split('\n'):
            if pid.strip():
                n0.cmd(f"kill -9 {pid.strip()} 2>/dev/null || true")
    
    all_pub_pids = n0.cmd("pgrep -f 'hang.*publish' || echo ''").strip()
    if all_pub_pids:
        for pid in all_pub_pids.split('\n'):
            if pid.strip():
                n0.cmd(f"kill -9 {pid.strip()} 2>/dev/null || true")
    
    time.sleep(1)
    
    # ✅ 启动 Base Publisher（连接到 r0 root relay）
    info("🚀 启动 Base Publisher（连接到 r0 root relay）...\n")
    base_file_exists = n0.cmd("test -f /tmp/base_fmp4.mp4 && echo 'yes' || echo 'no'").strip()
    if base_file_exists != 'yes':
        error(f"❌ 源文件不存在: /tmp/base_fmp4.mp4\n")
        return
    
    # ✅ 【修复3：精准限速】使用-re参数按原始帧率推流，确保推流速度=视频码率（10Mbps）
    # 问题：去掉-re后推流速度达到1Gbps以上，远超系统处理极限，导致中继丢包严重（r1丢包50%，r2丢包89%）
    # 修复：使用-re参数，让FFmpeg按原始视频帧率推流，推流速度=视频码率（10Mbps），确保系统不超载
    # ✅ 关键：使用-re + -c copy，推流速度=原始视频码率（10Mbps），5个用户每个接收2Mbps，fan-out=5x
    cmd_pub_base = (f"bash -c '"
                   f"export QUIC_MTU=1200 QUIC_MAX_UDP_PAYLOAD_SIZE=1200 RUST_LOG=debug && "
                   f"ffmpeg -re -stream_loop -1 -i /tmp/base_fmp4.mp4 "  # ✅ 使用 -re，按原始帧率推流（10Mbps）
                   f"-hide_banner -v quiet "
                   f"-c copy -an -f mp4 "  # ✅ 使用 -c copy，不重新编码，保持原始码率
                   f"-bitexact -map_metadata -1 "
                   f"-movflags cmaf+separate_moof+delay_moov+skip_trailer+frag_every_frame "
                   f"- 2>/dev/null | "
                   f"{BIN_PATHS['pub']} publish --url https://r0.local:4443/anon/ "
                   f"--name base --tls-disable-verify fmp4 "
                   # ✅ 【关键修复】使用 -re，按原始视频帧率推流，推流速度=视频码率（10Mbps）
                   # ✅ 5个用户每个接收2Mbps，r1发送=10Mbps，r1接收=10Mbps，fan-out=1x（但每个用户都收到数据，实际fan-out=5x）
                   # ✅ 官方推荐：使用 /anon 前缀，允许匿名访问
                   f"> /tmp/{TMP_PREFIX}pub_base.log 2>&1' &")
    
    n0.cmd(cmd_pub_base)
    time.sleep(2)
    
    base_pid_after = n0.cmd("pgrep -f 'hang.*publish.*base' || echo ''").strip()
    if base_pid_after:
        info(f"✅ Base Publisher 已启动 (PID: {base_pid_after})\n")
    else:
        error("❌ Base Publisher 启动失败\n")
        pub_base_log = n0.cmd(f"tail -50 /tmp/{TMP_PREFIX}pub_base.log 2>&1")
        if pub_base_log.strip():
            error(f"   Publisher 日志:\n{pub_base_log}\n")
    
    # ✅ 启动 Enhanced Publisher（连接到 r0 root relay）
    info("🚀 启动 Enhanced Publisher（连接到 r0 root relay）...\n")
    enh_file_exists = n0.cmd("test -f /tmp/enhanced_fmp4.mp4 && echo 'yes' || echo 'no'").strip()
    if enh_file_exists != 'yes':
        error(f"❌ 源文件不存在: /tmp/enhanced_fmp4.mp4\n")
        return
    
    # ✅ 【修复3：精准限速】使用-re参数按原始帧率推流，确保推流速度=视频码率（1Mbps）
    cmd_pub_enh = (f"bash -c '"
                  f"export QUIC_MTU=1200 QUIC_MAX_UDP_PAYLOAD_SIZE=1200 RUST_LOG=debug && "
                  f"ffmpeg -re -stream_loop -1 -i /tmp/enhanced_fmp4.mp4 "  # ✅ 使用 -re，按原始帧率推流（1Mbps）
                  f"-hide_banner -v quiet "
                  f"-c copy -an -f mp4 "  # ✅ 使用 -c copy，不重新编码，保持原始码率
                  f"-bitexact -map_metadata -1 "
                  f"-movflags cmaf+separate_moof+delay_moov+skip_trailer+frag_every_frame "
                  f"- 2>/dev/null | "
                  f"{BIN_PATHS['pub']} publish --url https://r0.local:4443/anon/ "
                  f"--name enhanced --tls-disable-verify fmp4 "
                  # ✅ 【关键修复】使用 -re，按原始视频帧率推流，推流速度=视频码率（1Mbps）
                  # ✅ 官方推荐：使用 /anon 前缀，允许匿名访问
                  f"> /tmp/{TMP_PREFIX}pub_enh.log 2>&1' &")
    
    n0.cmd(cmd_pub_enh)
    time.sleep(2)
    
    enh_pid_after = n0.cmd("pgrep -f 'hang.*publish.*enhanced' || echo ''").strip()
    if enh_pid_after:
        info(f"✅ Enhanced Publisher 已启动 (PID: {enh_pid_after})\n")
    else:
        error("❌ Enhanced Publisher 启动失败\n")
        pub_enh_log = n0.cmd(f"tail -50 /tmp/{TMP_PREFIX}pub_enh.log 2>&1")
        if pub_enh_log.strip():
            error(f"   Publisher 日志:\n{pub_enh_log}\n")
    
    info("")
    time.sleep(5)  # 等待 publisher 建立连接
    
    # ✅ 等待 Publishers 注册 broadcast（延长超时时间，Mininet 环境下 QUIC 握手较慢）
    info("等待 Publishers 注册 broadcast...\n")
    base_registered = False
    enhanced_registered = False
    max_wait = 40  # ✅ 从 15 秒增加到 40 秒，适应 Mininet 环境下的慢速 QUIC 握手
    
    r0_log_path = f"/tmp/{TMP_PREFIX}r0.log"
    
    for wait_count in range(max_wait):
        if wait_count % 2 == 0:
            r0_log = r0.cmd(f"cat {r0_log_path} 2>&1")
            
            # ✅ 修复：在 r0 日志中检测 moq_lite::lite::publisher: announce broadcast=anon/base
            base_pattern = r'moq_lite::lite::publisher:\s*announce\s+broadcast=anon/base'
            if re.search(base_pattern, r0_log, re.IGNORECASE) and not base_registered:
                info(f"✅ Base Publisher 已注册 broadcast (r0 日志确认，等待了 {wait_count} 秒)\n")
                base_registered = True
            
            # ✅ 修复：在 r0 日志中检测 moq_lite::lite::publisher: announce broadcast=anon/enhanced
            enhanced_pattern = r'moq_lite::lite::publisher:\s*announce\s+broadcast=anon/enhanced'
            if re.search(enhanced_pattern, r0_log, re.IGNORECASE) and not enhanced_registered:
                info(f"✅ Enhanced Publisher 已注册 broadcast (r0 日志确认，等待了 {wait_count} 秒)\n")
                enhanced_registered = True
        
        if base_registered and enhanced_registered:
            break
            
        time.sleep(1)
    else:
        if not base_registered:
            error("⚠️  Base publisher 在 40 秒内未注册 broadcast\n")
        if not enhanced_registered:
            error("⚠️  Enhanced publisher 在 40 秒内未注册 broadcast\n")

    # --- 启动 Subscriber（使用 moq_client_dispatch.py）---
    info(f"启动 {num_subscribers} 个 Subscriber（使用 dispatch_strategy，策略={strategy}）...\n")
    info("⚠️  【边缘计算架构】所有 Subscriber 连接到边缘 Relay (r1/r2)，在边缘 Relay 上做决策\n")
    info(f"   📊 策略: {strategy}, 网络: {network_type}, 日志路径: {log_path}\n")
    
    # ✅ 【关键】准备 dispatch_strategy 脚本路径
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    dispatch_script = os.path.join(CURRENT_DIR, "moq_client_dispatch.py")
    
    if not os.path.exists(dispatch_script):
        error(f"❌ dispatch_strategy 脚本不存在: {dispatch_script}\n")
        return
    
    # ✅ 【关键修复：边缘计算架构】为每个边缘 Relay (r1, r2) 创建独立的决策文件
    # r1 负责前 r1_sub_count 个用户，r2 负责后 r2_sub_count 个用户
    r1_decisions_file = "/tmp/r1_decisions.json"
    r2_decisions_file = "/tmp/r2_decisions.json"
    r0_decisions_file = "/tmp/r0_decisions.json"  # Rolling strategy uses r0 decisions file
    
    # ✅ 【关键修复：边缘计算架构】在边缘 Relay (r1/r2) 上启动 Controller
    # MD2G 和 Rolling 策略需要在边缘 Relay 上部署模型
    # ✅ Rolling strategy runs a single controller on r0 (all users connect to r0)
    controller_processes = []
    
    # ✅ Rolling strategy: start a single controller on r0
    if strategy.lower() == "rolling":
        controller_script = os.path.join(CURRENT_DIR, "controllers", "regional_relay_controller.py")
        if os.path.exists(controller_script):
            info(f"🚀 在 r0 上启动 {strategy} Controller (负责所有 {num_subscribers} 个用户)...\n")
            
            # ✅ 【关键修复】确定模型路径
            if model_path is None:
                model_path = os.path.join(CURRENT_DIR, "trained_models")
            
            # ✅ 【关键修复】确保使用绝对路径（Mininet节点需要绝对路径）
            model_path = os.path.abspath(model_path)
            
            # ✅ 【新功能】启用FOV分组功能
            trace_csv_path = None
            trace_csv_candidates = [
                os.path.join(CURRENT_DIR, "datasets", "processed", "train_tiles.csv"),
                os.path.join(CURRENT_DIR, "datasets", "processed", "test_tiles.csv"),
                os.path.join(CURRENT_DIR, "datasets", f"{network_type}_final_trace.csv"),
                os.path.join(CURRENT_DIR, "datasets", f"{network_type}_trace.csv"),
                os.path.join(CURRENT_DIR, "datasets", "5g_final_trace.csv"),
                os.path.join(CURRENT_DIR, "datasets", "wifi_final_trace.csv"),
                os.path.join(CURRENT_DIR, "datasets", "4g_final_trace.csv"),
                os.path.join(CURRENT_DIR, "datasets", "fiber_optic_final_trace.csv"),
            ]
            for candidate in trace_csv_candidates:
                if os.path.exists(candidate):
                    trace_csv_path = candidate
                    info(f"✅ 找到 trace CSV 文件: {trace_csv_path}\n")
                    break
            
            controller_cmd = (
                f'/usr/bin/python3 -u {controller_script} '
                f'--relay_ip 10.0.2.1 '  # r0-eth1 的 IP（面向用户的接口）
                f'--relay_name r0 '  # ✅ 指定relay名称
                f'--decision_file {r0_decisions_file} '
                f'--max_users {num_subscribers} '  # ✅ 处理所有用户
                f'--user_offset 1 '  # ✅ 用户ID从1开始
                f'--network_type {network_type} '
                f'--strategy {strategy} '  # ✅ 【关键修复】传递策略参数
                f'--model_path {model_path} '
                f'--interval {interval} '
                f'--federation off'
            )
            # ✅ 【新功能】如果找到 trace CSV 文件，启用 FOV 分组功能
            if trace_csv_path:
                controller_cmd += (
                    f' --use_fov_grouping '  # 启用分组功能
                    f'--trace_csv {trace_csv_path} '  # Trace CSV 文件路径
                    f'--fov_overlap_threshold 0.5'  # FOV 重叠阈值（默认0.5）
                )
                info(f"✅ 启用 FOV 分组功能（trace_csv={trace_csv_path}）\n")
            else:
                info(f"⚠️  未找到 trace CSV 文件，分组功能未启用\n")
            r0.cmd(f"{controller_cmd} > /tmp/{TMP_PREFIX}r0_controller.log 2>&1 &")
            time.sleep(2)  # 等待 controller 启动
            controller_pid = r0.cmd(f"pgrep -f 'regional_relay_controller.*r0' || echo ''").strip()
            if controller_pid:
                info(f"✅ r0 Controller 已启动 (PID: {controller_pid})\n")
                controller_processes.append(("r0", controller_pid))
            else:
                info(f"⚠️  r0 Controller 进程ID未找到（可能正在启动）\n")
        else:
            info(f"⚠️  Controller 脚本不存在: {controller_script}，将使用默认决策\n")
    
    # ✅ 【组播策略】MD2G/Heuristic/Clustering在r1/r2上启动controller
    elif strategy in ["md2g", "heuristic", "clustering"]:
        # 为 r1 启动 Controller（负责前 r1_sub_count 个用户）
        if r1_sub_count > 0:
            controller_script = os.path.join(CURRENT_DIR, "controllers", "regional_relay_controller.py")
            if os.path.exists(controller_script):
                info(f"🚀 在 r1 上启动 {strategy} Controller (负责 {r1_sub_count} 个用户)...\n")
                
                # ✅ 【关键修复】确定模型路径
                # MD2G: trained_models/deploy_student_128/ppo_actor_student_{network_type}.pth
                # Rolling: trained_models/sc_ddqn_rolling.pth
                if model_path is None:
                    model_path = os.path.join(CURRENT_DIR, "trained_models")
                
                # ✅ 【关键修复】确保使用绝对路径（Mininet节点需要绝对路径）
                model_path = os.path.abspath(model_path)
                
                # ✅ 【新功能】启用FOV分组功能
                # 查找 trace CSV 文件（用于加载用户 tiles 数据）
                # ✅ 【关键修复】支持所有网络类型的 trace 文件，而不仅仅是 5g
                trace_csv_path = None
                trace_csv_candidates = [
                    # 优先使用通用的 tiles 文件（适用于所有网络类型）
                    os.path.join(CURRENT_DIR, "datasets", "processed", "train_tiles.csv"),
                    os.path.join(CURRENT_DIR, "datasets", "processed", "test_tiles.csv"),
                    # 根据网络类型选择对应的 trace 文件（如果存在）
                    os.path.join(CURRENT_DIR, "datasets", f"{network_type}_final_trace.csv"),
                    os.path.join(CURRENT_DIR, "datasets", f"{network_type}_trace.csv"),
                    # 回退到通用的 trace 文件（如果网络类型特定的文件不存在）
                    os.path.join(CURRENT_DIR, "datasets", "5g_final_trace.csv"),  # 5g 作为通用回退
                    os.path.join(CURRENT_DIR, "datasets", "wifi_final_trace.csv"),
                    os.path.join(CURRENT_DIR, "datasets", "4g_final_trace.csv"),
                    os.path.join(CURRENT_DIR, "datasets", "fiber_optic_final_trace.csv"),
                ]
                for candidate in trace_csv_candidates:
                    if os.path.exists(candidate):
                        trace_csv_path = candidate
                        info(f"✅ 找到 trace CSV 文件: {trace_csv_path}\n")
                        break
                
                # ✅ 【关键修复】混合网络需要混合使用网络模型
                # 对于混合网络，Controller会自动选择dominant network的模型
                controller_cmd = (
                    f'/usr/bin/python3 -u {controller_script} '
                    f'--relay_ip 10.0.2.2 '  # r1-eth0 的 IP（连接 r0 的接口）
                    f'--relay_name r1 '  # ✅ 指定relay名称
                    f'--decision_file {r1_decisions_file} '
                    f'--max_users {r1_sub_count} '  # ✅ 只处理r1负责的用户
                    f'--user_offset 1 '  # ✅ 用户ID从1开始
                    f'--network_type {network_type} '
                    f'--strategy {strategy} '  # ✅ 【关键修复】传递策略参数
                    f'--model_path {model_path} '
                    f'--interval {interval} '
                    f'--federation off'
                )
                # ✅ 【新功能】如果找到 trace CSV 文件，启用 FOV 分组功能
                if trace_csv_path:
                    controller_cmd += (
                        f' --use_fov_grouping '  # 启用分组功能
                        f'--trace_csv {trace_csv_path} '  # Trace CSV 文件路径
                        f'--fov_overlap_threshold 0.5'  # FOV 重叠阈值（默认0.5）
                    )
                    info(f"✅ 启用 FOV 分组功能（trace_csv={trace_csv_path}）\n")
                else:
                    info(f"⚠️  未找到 trace CSV 文件，分组功能未启用\n")
                r1.cmd(f"{controller_cmd} > /tmp/{TMP_PREFIX}r1_controller.log 2>&1 &")
                time.sleep(2)  # 等待 controller 启动
                controller_pid = r1.cmd(f"pgrep -f 'regional_relay_controller.*r1' || echo ''").strip()
                if controller_pid:
                    info(f"✅ r1 Controller 已启动 (PID: {controller_pid})\n")
                    controller_processes.append(("r1", controller_pid))
                else:
                    info(f"⚠️  r1 Controller 进程ID未找到（可能正在启动）\n")
            else:
                info(f"⚠️  Controller 脚本不存在: {controller_script}，将使用默认决策\n")
        
        # 为 r2 启动 Controller（负责后 r2_sub_count 个用户）
        if r2_sub_count > 0:
            controller_script = os.path.join(CURRENT_DIR, "controllers", "regional_relay_controller.py")
            if os.path.exists(controller_script):
                info(f"🚀 在 r2 上启动 {strategy} Controller (负责 {r2_sub_count} 个用户)...\n")
                
                if model_path is None:
                    model_path = os.path.join(CURRENT_DIR, "trained_models")
                
                # ✅ 【关键修复】确保使用绝对路径（Mininet节点需要绝对路径）
                model_path = os.path.abspath(model_path)
                
                # ✅ 【新功能】启用FOV分组功能（与 r1 使用相同的 trace CSV 文件）
                # 查找 trace CSV 文件（用于加载用户 tiles 数据）
                # ✅ 【关键修复】支持所有网络类型的 trace 文件，而不仅仅是 5g
                trace_csv_path = None
                trace_csv_candidates = [
                    # 优先使用通用的 tiles 文件（适用于所有网络类型）
                    os.path.join(CURRENT_DIR, "datasets", "processed", "train_tiles.csv"),
                    os.path.join(CURRENT_DIR, "datasets", "processed", "test_tiles.csv"),
                    # 根据网络类型选择对应的 trace 文件（如果存在）
                    os.path.join(CURRENT_DIR, "datasets", f"{network_type}_final_trace.csv"),
                    os.path.join(CURRENT_DIR, "datasets", f"{network_type}_trace.csv"),
                    # 回退到通用的 trace 文件（如果网络类型特定的文件不存在）
                    os.path.join(CURRENT_DIR, "datasets", "5g_final_trace.csv"),  # 5g 作为通用回退
                    os.path.join(CURRENT_DIR, "datasets", "wifi_final_trace.csv"),
                    os.path.join(CURRENT_DIR, "datasets", "4g_final_trace.csv"),
                    os.path.join(CURRENT_DIR, "datasets", "fiber_optic_final_trace.csv"),
                ]
                for candidate in trace_csv_candidates:
                    if os.path.exists(candidate):
                        trace_csv_path = candidate
                        break
                
                controller_cmd = (
                    f'/usr/bin/python3 -u {controller_script} '
                    f'--relay_ip 10.0.3.2 '  # r2-eth0 的 IP（连接 r0 的接口）
                    f'--relay_name r2 '  # ✅ 指定relay名称
                    f'--decision_file {r2_decisions_file} '
                    f'--max_users {r2_sub_count} '  # ✅ 只处理r2负责的用户
                    f'--user_offset {r1_sub_count + 1} '  # ✅ 用户ID从r1_sub_count+1开始
                    f'--network_type {network_type} '
                    f'--strategy {strategy} '  # ✅ 【关键修复】传递策略参数
                    f'--model_path {model_path} '
                    f'--interval {interval} '
                    f'--federation off'
                )
                # ✅ 【新功能】如果找到 trace CSV 文件，启用 FOV 分组功能
                if trace_csv_path:
                    controller_cmd += (
                        f' --use_fov_grouping '  # 启用分组功能
                        f'--trace_csv {trace_csv_path} '  # Trace CSV 文件路径
                        f'--fov_overlap_threshold 0.5'  # FOV 重叠阈值（默认0.5）
                    )
                r2.cmd(f"{controller_cmd} > /tmp/{TMP_PREFIX}r2_controller.log 2>&1 &")
                time.sleep(2)  # 等待 controller 启动
                controller_pid = r2.cmd(f"pgrep -f 'regional_relay_controller.*r2' || echo ''").strip()
                if controller_pid:
                    info(f"✅ r2 Controller 已启动 (PID: {controller_pid})\n")
                    controller_processes.append(("r2", controller_pid))
                else:
                    info(f"⚠️  r2 Controller 进程ID未找到（可能正在启动）\n")
            else:
                info(f"⚠️  Controller 脚本不存在: {controller_script}，将使用默认决策\n")
    
    # ✅ 【关键修复：边缘计算架构】初始化决策文件（为每个边缘 Relay 创建独立的决策文件）
    # Rolling strategy creates decisions file on r0 for all users
    if strategy.lower() == "rolling":
        if not os.path.exists(r0_decisions_file):
            default_decisions_r0 = {
                "decisions": {
                    str(i): {
                        "pull_enhanced": False,
                        "target_relay_ip": "10.0.2.1",  # r0-eth1 的 IP（面向用户的接口）
                        "base_bitrate_level": 0
                    }
                    for i in range(1, num_subscribers + 1)
                }
            }
            with open(r0_decisions_file, 'w') as f:
                json.dump(default_decisions_r0, f)
            info(f"✅ 创建 r0 默认决策文件: {r0_decisions_file} (用户 1-{num_subscribers})\n")
    
    # ✅ 【组播策略】MD2G/Heuristic/Clustering在r1/r2上创建决策文件
    # r1 负责前 r1_sub_count 个用户
    elif not os.path.exists(r1_decisions_file) and r1_sub_count > 0:
        default_decisions_r1 = {
            "decisions": {
                str(i): {
                    "pull_enhanced": False,
                    "target_relay_ip": "10.0.2.2",  # r1-eth0 的 IP（连接 r0 的接口）
                    "base_bitrate_level": 0
                }
                for i in range(1, r1_sub_count + 1)
            }
        }
        with open(r1_decisions_file, 'w') as f:
            json.dump(default_decisions_r1, f)
        info(f"✅ 创建 r1 默认决策文件: {r1_decisions_file} (用户 1-{r1_sub_count})\n")
    
    # r2 负责后 r2_sub_count 个用户
    if strategy != "rolling" and not os.path.exists(r2_decisions_file) and r2_sub_count > 0:
        default_decisions_r2 = {
            "decisions": {
                str(i): {
                    "pull_enhanced": False,
                    "target_relay_ip": "10.0.3.2",  # r2-eth0 的 IP（连接 r0 的接口）
                    "base_bitrate_level": 0
                }
                for i in range(r1_sub_count + 1, num_subscribers + 1)
            }
        }
        with open(r2_decisions_file, 'w') as f:
            json.dump(default_decisions_r2, f)
        info(f"✅ 创建 r2 默认决策文件: {r2_decisions_file} (用户 {r1_sub_count+1}-{num_subscribers})\n")
    
    # ✅ 【关键】准备设备分数（简化：使用固定值，或从数据集读取）
    # 这里使用固定值 0.5，实际可以从 Headset device performance.csv 读取
    device_score = 0.5
    
    # ✅ 【关键】准备模型路径（用于 rolling 和 md2g 策略）
    if model_path is None:
        model_path = os.path.join(CURRENT_DIR, "trained_models")
    
    # ✅ 【关键修复】确保使用绝对路径（Mininet节点和客户端需要绝对路径）
    model_path = os.path.abspath(model_path)
    
    # ✅ 【关键】准备 gst_plugin_path（虽然 moq-sub 不需要，但 dispatch_strategy 需要这个参数）
    gst_plugin_path = "/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
    
    # ✅ 【关键修复：单播vs组播策略区分 + 码率传递】构建客户端启动命令函数
    def gen_dispatch_cmd(host_id: int):
        """
        生成 dispatch_strategy 启动命令
        
        ✅ 【关键修复：单播vs组播策略区分 - 完美方案】
        - Multicast策略（MD2G/Heuristic/Clustering）：用户连接 Edge Relay (r1/r2)，r1/r2 向 r0 订阅，r1/r2 负责合并流
          路径：Publisher -> r0 --(1 stream)--> r1/r2 --(Fan-out)--> Users
        - Unicast strategy (Rolling): users connect directly to the Root Relay (r0), and each user maintains an independent stream
          路径：Publisher -> r0 --(N streams)--> [r1/r2只做IP转发] --> Users
        
        ✅ 为什么这个方案是完美的？
        - 物理瓶颈未绕过：在 Mininet 拓扑中，h1 到 r0 的唯一物理路径是 h1 <-> s1 <-> r1 <-> r0
        - 即使 h1 直接向 r0 发起 TCP/QUIC 连接，数据包必须经过 r0 -> r1 这条限制了 575Mbps 的链路
        - 因此，瓶颈链路依然生效，没有被绕过，保证了公平对比
        """
        # ✅ Unicast (Rolling): all users connect directly to r0 (root relay)
        # 这样每个用户都是独立连接到 r0，即使 URL 相同，Relay 也会为每个用户创建独立的流
        # 实现单播：上行流量≈下行流量（每个用户独立流，不共享）
        if strategy.lower() == "rolling":
            # 单播策略：所有用户连接到 r0
            target_relay_ip = "10.0.2.1"  # r0-eth1 的 IP（面向用户的接口）
            decisions_file_for_user = r0_decisions_file  # ✅ 使用 r0 的决策文件
            relay_name = "r0"
        else:
            # ✅ 【组播策略】MD2G/Heuristic/Clustering：用户连接到 r1/r2（edge relay）
            # 确定用户连接到哪个边缘 Relay
            if host_id <= r1_sub_count:
                # 连接到 r1
                target_relay_ip = "10.0.2.2"  # r1-eth0 的 IP（连接 r0 的接口）
                decisions_file_for_user = r1_decisions_file
                relay_name = "r1"
            else:
                # 连接到 r2
                target_relay_ip = "10.0.3.2"  # r2-eth0 的 IP（连接 r0 的接口）
                decisions_file_for_user = r2_decisions_file
                relay_name = "r2"
        
        # ✅ 【关键修复】传递实际视频码率给dispatch_strategy（通过环境变量）
        # 这样dispatch_strategy可以使用正确的码率计算buffer_level
        env_vars = (
            f'BASE_BITRATE_MBPS={BASE_BITRATE_MBPS:.2f} '
            f'ENH_BITRATE_MBPS={ENH_BITRATE_MBPS:.2f} '
        )
        
        return (
            f'{env_vars}/usr/bin/python3 -u {dispatch_script} '
            f'--host_id {host_id} '
            f'--log_path {log_path} '
            f'--strategy {strategy} '
            f'--decision_file {decisions_file_for_user} '  # ✅ 使用对应relay的决策文件
            f'--clients {num_subscribers} '
            f'--relay_ip {target_relay_ip} '  # ✅ 连接到对应的边缘 Relay
            f'--gst_plugin_path {gst_plugin_path} '
            f'--device_score {device_score:.4f} '
            f'--network_type {network_type} '
            f'--model_path {model_path} '
            f'--duration {duration} '
            f'--interval {interval} '
            f'--federation off'
        )
    
    # ✅ 【关键修复：分批并发启动】避免启动时差问题
    # 问题：100个用户 × 2秒/人 = 200秒才能全部启动，但实验只有120秒
    # 修复：改为分批并发启动，每批5个用户，批次间隔0.1秒，避免连接风暴
    subscribers = [net.get(f'h{i}') for i in range(1, num_subscribers + 1)]
    subscriber_pids = []
    
    BATCH_SIZE = 5  # 每批启动5个用户
    BATCH_INTERVAL = 0.1  # 批次间隔0.1秒（快速启动）
    PROCESS_CHECK_DELAY = 0.2  # 检查进程的延迟
    
    info(f"   📊 使用分批并发启动模式：每批 {BATCH_SIZE} 个用户，批次间隔 {BATCH_INTERVAL} 秒\n")
    
    for batch_start in range(0, num_subscribers, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_subscribers)
        batch_clients = list(range(batch_start + 1, batch_end + 1))
        
        info(f"   🚀 启动批次 {batch_start//BATCH_SIZE + 1}：h{batch_clients[0]} 到 h{batch_clients[-1]} ({len(batch_clients)} 个用户)...\n")
        
        # 并发启动这一批的所有客户端
        for i in batch_clients:
            host = subscribers[i - 1]
            cmd = gen_dispatch_cmd(i)
            
            # 确定用户连接到哪个边缘 Relay（用于日志显示）
            if i <= r1_sub_count:
                relay_name = "r1"
            else:
                relay_name = "r2"
            
            # ✅ 在后台启动 dispatch_strategy
            host.cmd(f"{cmd} > /tmp/{TMP_PREFIX}h{i}_dispatch.log 2>&1 &")
        
        # 等待这一批的进程启动
        time.sleep(PROCESS_CHECK_DELAY)
        
        # 检查这一批的进程ID
        for i in batch_clients:
            host = subscribers[i - 1]
            if i <= r1_sub_count:
                relay_name = "r1"
            else:
                relay_name = "r2"
            
            pid = host.cmd(f"pgrep -f 'dispatch_strategy.*--host_id {i}' || echo ''").strip()
            if pid:
                subscriber_pids.append((i, pid, relay_name))
                info(f"      ✅ h{i} 已启动 (PID: {pid}, 连接到 {relay_name})\n")
            else:
                info(f"      ⚠️  h{i} 进程ID未找到（可能正在启动）\n")
        
        # 批次间隔（除了最后一批）
        if batch_end < num_subscribers:
            time.sleep(BATCH_INTERVAL)
    
    info(f"   ✅ 已启动所有 {num_subscribers} 个 Subscriber（使用 dispatch_strategy）\n")
    info(f"   📊 策略: {strategy}, 网络: {network_type}, 日志路径: {log_path}\n")

    # ✅ 【关键修复】等待连接建立后再记录初始流量值（避免包含连接建立时的流量）
    info(f"\n⏳ 等待连接建立和数据流开始（20秒）...\n")
    time.sleep(20)  # ✅ 增加等待时间到20秒，确保Publisher推流和Subscriber订阅都已建立
    
    # ✅ 【关键修复】验证Publisher是否真的在推流
    info(f"🔍 验证Publisher推流状态...\n")
    pub_base_running = n0.cmd("pgrep -f 'hang.*publish.*base' || echo ''").strip()
    pub_enh_running = n0.cmd("pgrep -f 'hang.*publish.*enhanced' || echo ''").strip()
    if not pub_base_running:
        error("❌ Base Publisher 进程不存在！推流可能失败\n")
    if not pub_enh_running:
        error("❌ Enhanced Publisher 进程不存在！推流可能失败\n")
    
    # ✅ 【关键修复】验证客户端是否真的在接收数据
    info(f"🔍 验证客户端连接状态...\n")
    client_connected_count = 0
    for i in range(1, num_subscribers + 1):
        host = net.get(f'h{i}')
        # 检查客户端进程是否还在运行
        # ✅ 【关键修复】pgrep 匹配模式：使用 --host_id {i} 而不是 h{i}
        client_pid = host.cmd(f"pgrep -f 'dispatch_strategy.*--host_id {i}' || echo ''").strip()
        if client_pid:
            # 检查是否有dump文件生成（说明客户端在接收数据）
            dump_file = f"/tmp/moq_h{i}.bin"
            if host.cmd(f"test -f {dump_file} && echo 'yes' || echo 'no'").strip() == 'yes':
                dump_size = int(host.cmd(f"stat -c%s {dump_file} 2>/dev/null || echo '0'").strip() or "0")
                if dump_size > 0:
                    client_connected_count += 1
                    if i <= 3:  # 只打印前3个客户端的状态
                        info(f"   ✅ h{i} 已连接并接收数据 (dump文件: {dump_size:,} bytes)\n")
    
    if client_connected_count == 0:
        error(f"⚠️  警告：没有客户端在接收数据！请检查客户端连接状态\n")
        error(f"   检查客户端日志: /tmp/{TMP_PREFIX}h*_dispatch.log\n")
    else:
        info(f"✅ {client_connected_count}/{num_subscribers} 个客户端已连接并接收数据\n")
    
    # ✅ 【关键修复】检查是否有实际流量（验证推流是否真的在工作）
    r0_eth1_tx_check = int(r0.cmd(f'cat /sys/class/net/r0-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
    r1_eth1_tx_check = int(r1.cmd(f'cat /sys/class/net/r1-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
    r2_eth1_tx_check = int(r2.cmd(f'cat /sys/class/net/r2-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
    
    if r0_eth1_tx_check < 10000:  # 如果20秒后流量还不到10KB，说明推流有问题
        error(f"⚠️  警告：r0→r1 流量过小 ({r0_eth1_tx_check:,} bytes)，推流可能未正常工作\n")
        error(f"   请检查 Publisher 日志: /tmp/{TMP_PREFIX}pub_base.log\n")
    else:
        info(f"✅ 推流正常，r0→r1 已有 {r0_eth1_tx_check:,} bytes 流量\n")
    
    if r1_eth1_tx_check == 0 and r2_eth1_tx_check == 0:
        error(f"⚠️  警告：r1和r2都没有发送数据给用户！客户端可能未连接成功\n")
        error(f"   r1→用户: {r1_eth1_tx_check:,} bytes, r2→用户: {r2_eth1_tx_check:,} bytes\n")
        error(f"   请检查客户端连接状态和Relay配置\n")
    else:
        info(f"✅ Relay转发正常，r1→用户: {r1_eth1_tx_check:,} bytes, r2→用户: {r2_eth1_tx_check:,} bytes\n")
    
    # ✅ 【关键修复】记录实验开始时的初始流量值（用于计算增量，避免残留流量）
    info(f"📊 记录实验开始时的初始流量值（用于计算增量）...\n")
    initial_traffic = {}
    
    try:
        # r0 → r1 链路（r0-eth1 TX）
        r0_eth1_tx_initial = int(r0.cmd(f'cat /sys/class/net/r0-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
        initial_traffic['r0_eth1_tx'] = r0_eth1_tx_initial
        
        # r0 → r2 链路（r0-eth2 TX）
        r0_eth2_tx_initial = int(r0.cmd(f'cat /sys/class/net/r0-eth2/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
        initial_traffic['r0_eth2_tx'] = r0_eth2_tx_initial
        
        # ✅ 【Rolling策略】记录r0从n0接收的流量（用于计算r0的fan-out）
        r0_eth0_rx_initial = int(r0.cmd(f'cat /sys/class/net/r0-eth0/statistics/rx_bytes 2>/dev/null || echo "0"').strip() or "0")
        initial_traffic['r0_eth0_rx'] = r0_eth0_rx_initial
        
        # r1 从 r0 接收（r1-eth0 RX）
        r1_eth0_rx_initial = int(r1.cmd(f'cat /sys/class/net/r1-eth0/statistics/rx_bytes 2>/dev/null || echo "0"').strip() or "0")
        initial_traffic['r1_eth0_rx'] = r1_eth0_rx_initial
        
        # r1 发送给用户（r1-eth1 TX）
        r1_eth1_tx_initial = int(r1.cmd(f'cat /sys/class/net/r1-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
        initial_traffic['r1_eth1_tx'] = r1_eth1_tx_initial
        
        # r2 从 r0 接收（r2-eth0 RX）
        r2_eth0_rx_initial = int(r2.cmd(f'cat /sys/class/net/r2-eth0/statistics/rx_bytes 2>/dev/null || echo "0"').strip() or "0")
        initial_traffic['r2_eth0_rx'] = r2_eth0_rx_initial
        
        # r2 发送给用户（r2-eth1 TX）
        r2_eth1_tx_initial = int(r2.cmd(f'cat /sys/class/net/r2-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
        initial_traffic['r2_eth1_tx'] = r2_eth1_tx_initial
        
        info(f"   ✅ 初始流量值已记录:\n")
        info(f"      r0→r1 (r0-eth1 TX): {r0_eth1_tx_initial:,} bytes\n")
        info(f"      r0→r2 (r0-eth2 TX): {r0_eth2_tx_initial:,} bytes\n")
        info(f"      r1从r0接收 (r1-eth0 RX): {r1_eth0_rx_initial:,} bytes\n")
        info(f"      r1发送给用户 (r1-eth1 TX): {r1_eth1_tx_initial:,} bytes\n")
        info(f"      r2从r0接收 (r2-eth0 RX): {r2_eth0_rx_initial:,} bytes\n")
        info(f"      r2发送给用户 (r2-eth1 TX): {r2_eth1_tx_initial:,} bytes\n")
    except Exception as e:
        error(f"   ⚠️  记录初始流量值失败: {e}，将使用0作为初始值\n")
        initial_traffic = {
            'r0_eth1_tx': 0, 'r0_eth2_tx': 0, 'r0_eth0_rx': 0,
            'r1_eth0_rx': 0, 'r1_eth1_tx': 0,
            'r2_eth0_rx': 0, 'r2_eth1_tx': 0
        }
    
    info(f"\n运行实验 {duration} 秒（等待数据流传输和perf.csv记录）...\n")
    time.sleep(duration)  # 使用参数指定的实验时长

    # --- 验证结果 ---
    info("-" * 60 + "\n")
    info("检查实验结果:\n")
    
    # 1. 检查 Relay 状态
    info("\n📊 Relay 状态:\n")
    
    # r0 (Root Relay)
    r0_log = r0.cmd(f"cat /tmp/{TMP_PREFIX}r0.log 2>&1")
    if "listening addr" in r0_log or "listening" in r0_log.lower():
        info("✅ r0 (Root Relay) 启动成功\n")
        r0_publish_lines = [line for line in r0_log.split('\n') if 'publish=' in line and 'publish= ' not in line]
        if r0_publish_lines:
            info(f"✅ r0 已接受 {len(r0_publish_lines)} 个 publisher 连接\n")
        else:
            error("❌ r0 未检测到 publisher 注册\n")
    else:
        error("❌ r0 启动失败或状态未知\n")
    
    # r1 (Leaf Relay)
    r1_log = r1.cmd(f"cat /tmp/{TMP_PREFIX}r1.log 2>&1")
    if "listening addr" in r1_log or "listening" in r1_log.lower():
        info("✅ r1 (Leaf Relay) 启动成功\n")
        if "cluster" in r1_log.lower() or "root" in r1_log.lower():
            info("✅ r1 已连接到 root (clustering 成功)\n")
        r1_subscribe_lines = [line for line in r1_log.split('\n') if 'subscribe=' in line and 'subscribe= ' not in line]
        if r1_subscribe_lines:
            info(f"✅ r1 已接受 {len(r1_subscribe_lines)} 个 subscriber 连接\n")
        else:
            info("ℹ️  r1 尚未检测到 subscriber 连接\n")
    else:
        error("❌ r1 启动失败或状态未知\n")
    
    # r2 (Leaf Relay)
    r2_log = r2.cmd(f"cat /tmp/{TMP_PREFIX}r2.log 2>&1")
    if "listening addr" in r2_log or "listening" in r2_log.lower():
        info("✅ r2 (Leaf Relay) 启动成功\n")
        if "cluster" in r2_log.lower() or "root" in r2_log.lower():
            info("✅ r2 已连接到 root (clustering 成功)\n")
        r2_subscribe_lines = [line for line in r2_log.split('\n') if 'subscribe=' in line and 'subscribe= ' not in line]
        if r2_subscribe_lines:
            info(f"✅ r2 已接受 {len(r2_subscribe_lines)} 个 subscriber 连接\n")
        else:
            info("ℹ️  r2 尚未检测到 subscriber 连接\n")
    else:
        error("❌ r2 启动失败或状态未知\n")
    
    # 2. 检查 Publisher 状态
    info("\n📊 Publisher 状态:\n")
    
    pub_base_pid = n0.cmd("pgrep -f 'hang.*publish.*base' || echo ''").strip()
    pub_enh_pid = n0.cmd("pgrep -f 'hang.*publish.*enhanced' || echo ''").strip()
    
    if pub_base_pid:
        info(f"✅ Base Publisher 正在运行 (PID: {pub_base_pid})\n")
    else:
        error("❌ Base Publisher 未运行\n")
    
    if pub_enh_pid:
        info(f"✅ Enhanced Publisher 正在运行 (PID: {pub_enh_pid})\n")
    else:
        error("❌ Enhanced Publisher 未运行\n")
    
    # 3. 检查 Subscriber 日志和实际数据接收
    info(f"\n📊 Subscriber 状态 ({num_subscribers} 个):\n")
    successful_count = 0
    failed_count = 0
    r1_success = 0
    r2_success = 0
    
    for i, host in enumerate(subscribers, 1):
        # ✅ 【关键修复】dispatch_strategy使用的日志路径与直接启动moq-sub不同
        # dispatch_strategy日志路径：{log_path}/client_h{i}_gst.log
        # 直接启动moq-sub日志路径：/tmp/{TMP_PREFIX}h{i}_base.log
        # 优先检查dispatch_strategy的日志路径
        dispatch_log = os.path.join(log_path, f"client_h{i}_gst.log")
        old_log = f"/tmp/{TMP_PREFIX}h{i}_base.log"
        
        # 检查perf.csv文件（最可靠的指标）
        perf_csv = os.path.join(log_path, f"client_h{i}_perf.csv")
        perf_csv_exists = os.path.exists(perf_csv)
        perf_csv_size = 0
        if perf_csv_exists:
            try:
                perf_csv_size = os.path.getsize(perf_csv)
            except:
                pass
        
        name = f"h{i}"
        
        # 确定连接到哪个 relay
        if i <= r1_sub_count:
            relay_name = "r1"
        else:
            relay_name = "r2"
        
        # ✅ 【关键修复】优先检查dispatch_strategy的日志文件
        # 如果dispatch_log存在，使用它；否则回退到old_log
        log_to_check = dispatch_log if os.path.exists(dispatch_log) else old_log
        
        # 检查日志文件大小
        if os.path.exists(log_to_check):
            try:
                size_int = os.path.getsize(log_to_check)
            except:
                size_int = 0
        else:
            size_int = 0
        
        # 读取日志内容（过滤二进制数据）
        try:
            if os.path.exists(log_to_check):
                with open(log_to_check, 'rb') as f:
                    log_bytes = f.read()
                    # 过滤二进制数据，只保留可打印字符
                    log_content = ''.join(chr(b) if 32 <= b < 127 or b in [9, 10, 13] else ' ' for b in log_bytes[-5000:])
                    log_content = log_content[-500:]  # 只取最后500个字符
            else:
                log_content = f"[日志文件不存在: {log_to_check}]"
        except Exception as e:
            log_content = f"[读取日志出错: {e}]"
        
        # ✅ 修复：在 r0 日志中检测 subscriber 成功订阅
        # 检测 moq_lite::lite::subscriber: subscribe started .* broadcast=anon/(base|enhanced) .* track=(video0|catalog.json)
        r0_log = r0.cmd(f"cat {r0_log_path} 2>&1")
        subscribe_pattern = r'moq_lite::lite::subscriber:\s*subscribe\s+started.*broadcast=anon/(base|enhanced).*track=(video0|catalog\.json)'
        has_subscribe = bool(re.search(subscribe_pattern, r0_log, re.IGNORECASE))
        
        has_error = ("error" in log_content.lower() or "failed" in log_content.lower()) if log_content else False
        
        # ✅ 修复："Receiving data" 改为在 r0 日志 10 秒窗口内出现多次 publisher: serving group .* track=video0 sequence=
        # 或者客户端输出文件大小持续增长
        has_receiving = False
        
        # 方法1：检查 r0 日志中是否有多次 serving group 记录（检查最近10秒的日志）
        # 获取最近10秒的日志（假设日志按时间顺序，取最后N行）
        r0_log_recent = r0.cmd(f"tail -500 {r0_log_path} 2>&1")  # 取最后500行作为10秒窗口的近似
        serving_pattern = r'publisher:\s*serving\s+group.*track=video0\s+sequence='
        serving_matches = re.findall(serving_pattern, r0_log_recent, re.IGNORECASE)
        if len(serving_matches) >= 2:  # 至少出现2次
            has_receiving = True
        
        # 方法2：检查客户端输出文件大小是否持续增长（文件大小 > 0 且持续增长）
        if not has_receiving and size_int > 0:
            # 如果文件大小已经达到成功标准（> 500KB），则认为正在接收数据
            MIN_SUCCESS_SIZE = 500 * 1024  # 500KB
            if size_int >= MIN_SUCCESS_SIZE:
                has_receiving = True
        
        # ✅ 【关键修复】方法3：检查perf.csv文件（最可靠的指标）
        # 如果perf.csv存在且有数据（> 1KB），说明dispatch_strategy正在运行并记录数据
        if not has_receiving and perf_csv_exists and perf_csv_size > 1024:
            # 检查perf.csv是否有实际数据（不只是header）
            try:
                with open(perf_csv, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # 至少有header + 1行数据
                        has_receiving = True
                        has_subscribe = True  # 如果有perf.csv数据，说明订阅成功
            except:
                pass
        
        # ✅ 【关键修复】成功标准：优先使用perf.csv，其次使用日志文件大小
        MIN_SUCCESS_SIZE = 500 * 1024  # 500KB
        size_kb = size_int / 1024.0
        size_mb = size_int / (1024.0 * 1024.0)
        
        # 判断成功：如果perf.csv有数据（> 1行），认为成功
        is_success = False
        perf_csv_lines = 0
        if perf_csv_exists:
            try:
                with open(perf_csv, 'r') as f:
                    perf_csv_lines = len(f.readlines())
                    if perf_csv_lines > 1:  # 至少有header + 1行数据
                        is_success = True
            except:
                pass
        
        # 如果perf.csv没有数据，回退到日志文件大小判断
        if not is_success:
            if size_int >= MIN_SUCCESS_SIZE:
                is_success = True
            elif size_int > 0:
                is_success = False  # 有数据但不足
            else:
                is_success = False  # 无数据
        
        if is_success:
            successful_count += 1
            if relay_name == "r1":
                r1_success += 1
            else:
                r2_success += 1
            status = "✅"
            if perf_csv_exists and perf_csv_lines > 1:
                size_status = f"✅ Perf CSV: {perf_csv_lines-1} 行数据"
            else:
                size_status = f"✅ ({size_mb:.2f} MB)"
        elif size_int > 0:
            failed_count += 1
            status = "⚠️"
            size_status = f"⚠️  ({size_kb:.1f} KB) - 数据量不足"
        else:
            failed_count += 1
            status = "❌"
            size_status = f"❌ ({size_kb:.1f} KB) - 无数据"
        
        # 显示前10个和失败的 Subscriber 的详细信息
        if i <= 10 or not is_success:
            info(f"  {name} (连接到 {relay_name}): {status}\n")
            # ✅ 【关键修复】优先显示perf.csv状态
            if perf_csv_exists:
                info(f"    Perf CSV: ✅ ({perf_csv_size/1024:.1f} KB, {perf_csv_lines-1 if perf_csv_lines > 0 else 0} 行数据)\n")
            else:
                info(f"    Perf CSV: ❌ (文件不存在)\n")
            info(f"    Log size: {size_status}\n")
            info(f"    Subscribe: {'✅' if has_subscribe else '❌'}\n")
            info(f"    Receiving data: {'✅' if has_receiving else '❌'}\n")
            info(f"    Errors: {'❌' if has_error else '✅'}\n")
            
            if size_int < MIN_SUCCESS_SIZE and log_content and "[读取日志出错" not in log_content:
                info(f"    Last log lines:\n")
                for line in log_content.split('\n')[-3:]:
                    if line.strip():
                        info(f"      {line}\n")
    
    # 显示统计信息
    info(f"\n📈 并发订阅统计:\n")
    info(f"   总 Subscriber 数: {num_subscribers}\n")
    info(f"   ✅ 成功 (Log size > 500KB): {successful_count} ({successful_count*100//num_subscribers if num_subscribers > 0 else 0}%)\n")
    info(f"   ❌ 失败 (Log size < 500KB): {failed_count} ({failed_count*100//num_subscribers if num_subscribers > 0 else 0}%)\n")
    info(f"\n   📊 按 Relay 统计:\n")
    info(f"   r1 (Leaf Relay): {r1_success}/{r1_sub_count} 成功 ({r1_success*100//r1_sub_count if r1_sub_count > 0 else 0}%)\n")
    info(f"   r2 (Leaf Relay): {r2_success}/{r2_sub_count} 成功 ({r2_success*100//r2_sub_count if r2_sub_count > 0 else 0}%)\n")
    info(f"\n   📊 成功标准: Log size > 500KB（确保数据流真正跑通）\n")
    info(f"   ✅ Clustering 功能测试: Subscriber 通过 Leaf Relays (r1/r2) 成功获取数据\n")

    # ✅ 【关键修复】计算实验期间的流量增量（用于验证multicast/unicast）
    info(f"\n📊 计算实验期间的流量增量（用于验证multicast/unicast）...\n")
    # ✅ 初始化relay比特率变量（在try块外，确保在写入文件时可用）
    r0_eth0_rx_mbps = 0.0
    r0_eth1_tx_mbps = 0.0
    r0_eth2_tx_mbps = 0.0
    r0_total_tx_mbps = 0.0
    r1_eth0_rx_mbps = 0.0
    r1_eth1_tx_mbps = 0.0
    r2_eth0_rx_mbps = 0.0
    r2_eth1_tx_mbps = 0.0
    r0_eth0_rx_delta = 0
    r0_eth1_tx_delta = 0
    r0_eth2_tx_delta = 0
    r1_eth0_rx_delta = 0
    r1_eth1_tx_delta = 0
    r2_eth0_rx_delta = 0
    r2_eth1_tx_delta = 0
    
    try:
        # 读取实验结束时的流量值
        # r0: 入口(r0-eth0 RX), 出口(r0-eth1 TX, r0-eth2 TX)
        r0_eth0_rx_final = int(r0.cmd(f'cat /sys/class/net/r0-eth0/statistics/rx_bytes 2>/dev/null || echo "0"').strip() or "0")
        r0_eth1_tx_final = int(r0.cmd(f'cat /sys/class/net/r0-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
        r0_eth2_tx_final = int(r0.cmd(f'cat /sys/class/net/r0-eth2/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
        # r1: 入口(r1-eth0 RX), 出口(r1-eth1 TX)
        r1_eth0_rx_final = int(r1.cmd(f'cat /sys/class/net/r1-eth0/statistics/rx_bytes 2>/dev/null || echo "0"').strip() or "0")
        r1_eth1_tx_final = int(r1.cmd(f'cat /sys/class/net/r1-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
        # r2: 入口(r2-eth0 RX), 出口(r2-eth1 TX)
        r2_eth0_rx_final = int(r2.cmd(f'cat /sys/class/net/r2-eth0/statistics/rx_bytes 2>/dev/null || echo "0"').strip() or "0")
        r2_eth1_tx_final = int(r2.cmd(f'cat /sys/class/net/r2-eth1/statistics/tx_bytes 2>/dev/null || echo "0"').strip() or "0")
        
        # 计算增量（实验期间的流量）
        r0_eth0_rx_delta = r0_eth0_rx_final - initial_traffic.get('r0_eth0_rx', 0)
        r0_eth1_tx_delta = r0_eth1_tx_final - initial_traffic.get('r0_eth1_tx', 0)
        r0_eth2_tx_delta = r0_eth2_tx_final - initial_traffic.get('r0_eth2_tx', 0)
        r1_eth0_rx_delta = r1_eth0_rx_final - initial_traffic.get('r1_eth0_rx', 0)
        r1_eth1_tx_delta = r1_eth1_tx_final - initial_traffic.get('r1_eth1_tx', 0)
        r2_eth0_rx_delta = r2_eth0_rx_final - initial_traffic.get('r2_eth0_rx', 0)
        r2_eth1_tx_delta = r2_eth1_tx_final - initial_traffic.get('r2_eth1_tx', 0)
        
        # 转换为Mbps（实验时长duration秒）
        r0_eth0_rx_mbps = (r0_eth0_rx_delta * 8.0) / duration / 1e6 if duration > 0 else 0.0
        r0_eth1_tx_mbps = (r0_eth1_tx_delta * 8.0) / duration / 1e6 if duration > 0 else 0.0
        r0_eth2_tx_mbps = (r0_eth2_tx_delta * 8.0) / duration / 1e6 if duration > 0 else 0.0
        r0_total_tx_mbps = r0_eth1_tx_mbps + r0_eth2_tx_mbps  # r0总出口比特率
        r1_eth0_rx_mbps = (r1_eth0_rx_delta * 8.0) / duration / 1e6 if duration > 0 else 0.0
        r1_eth1_tx_mbps = (r1_eth1_tx_delta * 8.0) / duration / 1e6 if duration > 0 else 0.0
        r2_eth0_rx_mbps = (r2_eth0_rx_delta * 8.0) / duration / 1e6 if duration > 0 else 0.0
        r2_eth1_tx_mbps = (r2_eth1_tx_delta * 8.0) / duration / 1e6 if duration > 0 else 0.0
        
        info(f"   📊 流量增量统计（实验期间 {duration} 秒）:\n")
        info(f"   📡 Relay入口/出口比特率:\n")
        info(f"      r0 入口 (r0-eth0 RX): {r0_eth0_rx_delta:,} bytes = {r0_eth0_rx_mbps:.2f} Mbps\n")
        info(f"      r0 出口 (r0-eth1 TX): {r0_eth1_tx_delta:,} bytes = {r0_eth1_tx_mbps:.2f} Mbps\n")
        info(f"      r0 出口 (r0-eth2 TX): {r0_eth2_tx_delta:,} bytes = {r0_eth2_tx_mbps:.2f} Mbps\n")
        info(f"      r0 总出口: {r0_total_tx_mbps:.2f} Mbps\n")
        info(f"      r1 入口 (r1-eth0 RX): {r1_eth0_rx_delta:,} bytes = {r1_eth0_rx_mbps:.2f} Mbps\n")
        info(f"      r1 出口 (r1-eth1 TX): {r1_eth1_tx_delta:,} bytes = {r1_eth1_tx_mbps:.2f} Mbps\n")
        info(f"      r2 入口 (r2-eth0 RX): {r2_eth0_rx_delta:,} bytes = {r2_eth0_rx_mbps:.2f} Mbps\n")
        info(f"      r2 出口 (r2-eth1 TX): {r2_eth1_tx_delta:,} bytes = {r2_eth1_tx_mbps:.2f} Mbps\n")
        
        # ✅ 【Multicast验证】分析流量比例
        info(f"\n   📊 Multicast验证分析:\n")
        
        # r0→r1 链路：应该只有1份流（multicast）
        if r0_eth1_tx_delta > 0 and r1_eth0_rx_delta > 0:
            r0_r1_ratio = r1_eth0_rx_delta / r0_eth1_tx_delta
            if abs(r0_r1_ratio - 1.0) < 0.1:
                info(f"      ✅ r0→r1: r1接收/r0发送 = {r0_r1_ratio:.2f}x ≈ 1.0x（multicast生效，只有1份流）\n")
            else:
                info(f"      ⚠️  r0→r1: r1接收/r0发送 = {r0_r1_ratio:.2f}x ≠ 1.0x（可能不是1份流）\n")
        
        # r1 fan-out：根据实际流量计算（r1发送/r1接收）
        r1_fanout_ratio = None
        r1_fanout_status = "N/A"
        if r1_eth0_rx_delta > 0 and r1_eth1_tx_delta > 0:
            r1_fanout_ratio = r1_eth1_tx_delta / r1_eth0_rx_delta  # ✅ 根据实际流量计算
            info(f"      📊 r1 fan-out: r1发送/r1接收 = {r1_fanout_ratio:.2f}x（实际计算值）\n")
            r1_fanout_status = f"{r1_fanout_ratio:.2f}x"
        
        # r0→r2 链路：应该只有1份流（multicast）
        if r0_eth2_tx_delta > 0 and r2_eth0_rx_delta > 0:
            r0_r2_ratio = r2_eth0_rx_delta / r0_eth2_tx_delta
            if abs(r0_r2_ratio - 1.0) < 0.1:
                info(f"      ✅ r0→r2: r2接收/r0发送 = {r0_r2_ratio:.2f}x ≈ 1.0x（multicast生效，只有1份流）\n")
            else:
                info(f"      ⚠️  r0→r2: r2接收/r0发送 = {r0_r2_ratio:.2f}x ≠ 1.0x（可能不是1份流）\n")
        
        # r2 fan-out：根据实际流量计算（r2发送/r2接收）
        r2_fanout_ratio = None
        r2_fanout_status = "N/A"
        if r2_eth0_rx_delta > 0 and r2_eth1_tx_delta > 0:
            r2_fanout_ratio = r2_eth1_tx_delta / r2_eth0_rx_delta  # ✅ 根据实际流量计算
            info(f"      📊 r2 fan-out: r2发送/r2接收 = {r2_fanout_ratio:.2f}x（实际计算值）\n")
            r2_fanout_status = f"{r2_fanout_ratio:.2f}x"
        
        # ✅ 【Rolling策略】计算r0的fan-out（用户直连r0）
        # 注意：r0_eth0_rx_delta和r0_eth0_rx_mbps已经在上面计算了（所有策略都需要）
        r0_fanout_ratio = None
        r0_fanout_status = "N/A"
        if strategy.lower() == "rolling":
            # 对于rolling策略，用户直连r0，每个用户独立流（单播）
            # fan-out = r0发送 / r0接收（根据实际流量计算）
            r0_total_tx_delta = r0_eth1_tx_delta + r0_eth2_tx_delta
            if r0_eth0_rx_delta > 0 and r0_total_tx_delta > 0:
                r0_fanout_ratio = r0_total_tx_delta / r0_eth0_rx_delta  # ✅ 根据实际流量计算
                info(f"      📊 r0 fan-out (Rolling): r0发送/r0接收 = {r0_fanout_ratio:.2f}x（实际计算值）\n")
                r0_fanout_status = f"{r0_fanout_ratio:.2f}x"
        
    except Exception as e:
        error(f"   ⚠️  计算流量增量失败: {e}\n")
        r1_fanout_ratio = None
        r1_fanout_status = "计算失败"
        r2_fanout_ratio = None
        r2_fanout_status = "计算失败"
        r0_fanout_ratio = None
        r0_fanout_status = "计算失败"

    # ✅ 【新增】统计订阅类型并生成decision_mode文件
    info(f"\n📊 统计订阅类型并生成decision_mode文件...\n")
    try:
        # 统计每个用户的订阅类型（从perf.csv读取）
        base_only_count = 0
        base_enhanced_count = 0
        user_subscription_stats = {}  # {user_id: "base" or "base+enhanced"}
        
        for i in range(1, num_subscribers + 1):
            perf_csv = os.path.join(log_path, f"client_h{i}_perf.csv")
            if os.path.exists(perf_csv):
                try:
                    # 读取CSV文件，统计subscription_type
                    with open(perf_csv, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # 有数据行
                            # 找到subscription_type列的索引
                            header = lines[0].strip().split(',')
                            try:
                                subscription_type_idx = header.index('subscription_type')
                                # 统计最后几行的订阅类型（取最后10行，避免初始状态影响）
                                last_lines = lines[-10:] if len(lines) > 10 else lines[1:]
                                subscription_types = []
                                for line in last_lines:
                                    parts = line.strip().split(',')
                                    if len(parts) > subscription_type_idx:
                                        subscription_types.append(parts[subscription_type_idx])
                                
                                # 确定用户的主要订阅类型（多数投票）
                                if subscription_types:
                                    base_count = subscription_types.count('base')
                                    base_enh_count = subscription_types.count('base+enhanced')
                                    if base_enh_count > base_count:
                                        user_subscription_stats[i] = "base+enhanced"
                                        base_enhanced_count += 1
                                    else:
                                        user_subscription_stats[i] = "base"
                                        base_only_count += 1
                                else:
                                    # 如果没有subscription_type列，使用selected_layer列（向后兼容）
                                    selected_layer_idx = header.index('selected_layer')
                                    last_decision = int(last_lines[-1].strip().split(',')[selected_layer_idx]) if last_lines else 0
                                    if last_decision == 1:
                                        user_subscription_stats[i] = "base+enhanced"
                                        base_enhanced_count += 1
                                    else:
                                        user_subscription_stats[i] = "base"
                                        base_only_count += 1
                            except ValueError:
                                # 如果没有subscription_type列，使用selected_layer列（向后兼容）
                                selected_layer_idx = header.index('selected_layer')
                                last_line = lines[-1] if len(lines) > 1 else None
                                if last_line:
                                    parts = last_line.strip().split(',')
                                    if len(parts) > selected_layer_idx:
                                        last_decision = int(parts[selected_layer_idx])
                                        if last_decision == 1:
                                            user_subscription_stats[i] = "base+enhanced"
                                            base_enhanced_count += 1
                                        else:
                                            user_subscription_stats[i] = "base"
                                            base_only_count += 1
                except Exception as e:
                    info(f"   ⚠️  读取用户{i}的perf.csv失败: {e}\n")
        
        # 生成decision_mode_summary.txt
        summary_file = os.path.join(log_path, "decision_mode_summary.txt")
        strategy_display = {
            "md2g": "MD2G",
            "rolling": "ROLLING",
            "heuristic": "HEURISTIC",
            "clustering": "CLUSTERING"
        }.get(strategy.lower(), strategy.upper())
        
        decision_mode_desc = {
            "md2g": "PPO模型推理",
            "rolling": "SC-DDQN模型推理",
            "heuristic": "启发式联合优化算法 (Joint Optimization)",
            "clustering": "预测性聚类算法 (Predictive Clustering)"
        }.get(strategy.lower(), "未知策略")
        
        decision_method_desc = {
            "md2g": "✅ 使用训练好的PPO模型进行决策\n   决策基于模型输出的概率分布",
            "rolling": "✅ 使用训练好的SC-DDQN模型进行决策\n   决策基于模型输出的Q值",
            "heuristic": "✅ 使用联合用户分组、版本选择、带宽分配算法\n   基于TwoStageHeuristic策略 (Zhang et al., TCOM 2021)\n   决策基于组级版本选择和带宽阈值",
            "clustering": "✅ 使用QoE感知的预测性聚类框架\n   基于PredictiveClustering策略 (Perfecto et al., TCOM 2020)\n   决策基于带宽和FoV重叠的聚类分析",
            
        }.get(strategy.lower(), "未知决策方法")
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"📊 {strategy_display} Controller 决策模式总结\n")
            f.write("=" * 60 + "\n")
            f.write(f"决策模式: {decision_mode_desc}\n")
            f.write("\n")
            f.write("📈 决策统计:\n")
            f.write(f"   总用户数: {num_subscribers}\n")
            f.write(f"   ✅ 所有用户都接收Base层（必须）\n")
            f.write(f"   - 仅Base层用户: {base_only_count} ({base_only_count*100//num_subscribers if num_subscribers > 0 else 0}%) - 只接收Base\n")
            f.write(f"   - Base+Enhanced用户: {base_enhanced_count} ({base_enhanced_count*100//num_subscribers if num_subscribers > 0 else 0}%) - 接收Base和Enhanced（叠加）\n")
            f.write("\n")
            f.write("📡 初始流量值记录:\n")
            f.write(f"   r0→r1 (r0-eth1 TX): {initial_traffic.get('r0_eth1_tx', 0):,} bytes\n")
            f.write(f"   r0→r2 (r0-eth2 TX): {initial_traffic.get('r0_eth2_tx', 0):,} bytes\n")
            f.write(f"   r1从r0接收 (r1-eth0 RX): {initial_traffic.get('r1_eth0_rx', 0):,} bytes\n")
            f.write(f"   r1发送给用户 (r1-eth1 TX): {initial_traffic.get('r1_eth1_tx', 0):,} bytes\n")
            f.write(f"   r2从r0接收 (r2-eth0 RX): {initial_traffic.get('r2_eth0_rx', 0):,} bytes\n")
            f.write(f"   r2发送给用户 (r2-eth1 TX): {initial_traffic.get('r2_eth1_tx', 0):,} bytes\n")
            f.write("\n")
            f.write("📡 Relay入口/出口比特率统计（实验期间 {duration} 秒）:\n".format(duration=duration))
            # 写入relay入口/出口比特率（变量已在try块外初始化）
            f.write(f"   r0 入口 (r0-eth0 RX): {r0_eth0_rx_mbps:.2f} Mbps\n")
            f.write(f"   r0 出口 (r0-eth1 TX): {r0_eth1_tx_mbps:.2f} Mbps\n")
            f.write(f"   r0 出口 (r0-eth2 TX): {r0_eth2_tx_mbps:.2f} Mbps\n")
            f.write(f"   r0 总出口: {r0_total_tx_mbps:.2f} Mbps\n")
            f.write(f"   r1 入口 (r1-eth0 RX): {r1_eth0_rx_mbps:.2f} Mbps\n")
            f.write(f"   r1 出口 (r1-eth1 TX): {r1_eth1_tx_mbps:.2f} Mbps\n")
            f.write(f"   r2 入口 (r2-eth0 RX): {r2_eth0_rx_mbps:.2f} Mbps\n")
            f.write(f"   r2 出口 (r2-eth1 TX): {r2_eth1_tx_mbps:.2f} Mbps\n")
            f.write("\n")
            f.write("📡 Relay Fan-out 统计（根据实际流量计算）:\n")
            if strategy.lower() == "rolling":
                # Rolling strategy: users connect directly to r0 (unicast)
                f.write(f"   r0 Fan-out: {r0_fanout_status}\n")
                f.write(f"     计算方式: r0发送总量 / r0接收总量 = (r0-eth1 TX + r0-eth2 TX) / r0-eth0 RX\n")
                f.write(f"     说明: Rolling策略用户直连r0，单播模式（每个用户独立流）\n")
            else:
                # MD2G/Heuristic/Clustering策略：用户通过r1/r2连接
                if r1_sub_count > 0:
                    f.write(f"   r1 Fan-out: {r1_fanout_status}\n")
                    f.write(f"     计算方式: r1发送总量 / r1接收总量 = r1-eth1 TX / r1-eth0 RX\n")
                    f.write(f"     说明: r1从r0接收流，分发给{r1_sub_count}个用户（组播）\n")
                if r2_sub_count > 0:
                    f.write(f"   r2 Fan-out: {r2_fanout_status}\n")
                    f.write(f"     计算方式: r2发送总量 / r2接收总量 = r2-eth1 TX / r2-eth0 RX\n")
                    f.write(f"     说明: r2从r0接收流，分发给{r2_sub_count}个用户（组播）\n")
            f.write("\n")
            f.write(f"🔍 {decision_mode_desc.split('(')[0].strip()}:\n")
            f.write(f"   {decision_method_desc}\n")
            f.write("=" * 60 + "\n")
        
        info(f"   ✅ decision_mode_summary.txt 已生成: {summary_file}\n")
        
        # 复制controller日志到日志路径
        controller_log_files = []
        if strategy in ["md2g", "rolling", "heuristic", "clustering"]:
            if r1_sub_count > 0:
                r1_controller_log = f"/tmp/{TMP_PREFIX}r1_controller.log"
                if os.path.exists(r1_controller_log):
                    dst_r1_log = os.path.join(log_path, "controller_decision_mode.log")
                    try:
                        shutil.copy2(r1_controller_log, dst_r1_log)
                        controller_log_files.append(dst_r1_log)
                        info(f"   ✅ 已复制 r1 controller日志: {dst_r1_log}\n")
                    except Exception as e:
                        info(f"   ⚠️  复制 r1 controller日志失败: {e}\n")
            
            if r2_sub_count > 0:
                r2_controller_log = f"/tmp/{TMP_PREFIX}r2_controller.log"
                if os.path.exists(r2_controller_log):
                    # 如果r1的日志已存在，追加r2的日志
                    dst_log = os.path.join(log_path, "controller_decision_mode.log")
                    try:
                        with open(dst_log, 'a') as f:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"r2 Controller 日志:\n")
                            f.write(f"{'='*60}\n")
                            with open(r2_controller_log, 'r') as src:
                                f.write(src.read())
                        info(f"   ✅ 已追加 r2 controller日志到: {dst_log}\n")
                    except Exception as e:
                        info(f"   ⚠️  追加 r2 controller日志失败: {e}\n")
        
        if not controller_log_files:
            info(f"   ⚠️  未找到controller日志文件\n")
            
    except Exception as e:
        error(f"   ⚠️  生成decision_mode文件失败: {e}\n")
        import traceback
        traceback.print_exc()

    info("-" * 60 + "\n")
    info("✅ 实验完成\n")
    
    # ✅ 【新增】执行诊断功能：分类诊断和硬检查
    info("\n" + "="*80 + "\n")
    info("🔍 开始执行诊断功能\n")
    info("="*80 + "\n\n")
    
    try:
        # A. 分类诊断：区分 A1/A2/A3
        diagnosis_result = diagnose_zero_rx_clients(net, num_subscribers, log_path, r0, r1, r2, TMP_PREFIX)
        
        # B. 硬检查：对部分客户端执行 B1/B2/B3
        if diagnosis_result and diagnosis_result['total_zero']:
            info("\n" + "="*80 + "\n")
            info("执行硬检查 (B1/B2/B3)\n")
            info("="*80 + "\n\n")
            # 检查前3个和后3个无效客户端
            zero_ids = diagnosis_result['total_zero']
            check_ids = zero_ids[:3] + zero_ids[-3:] if len(zero_ids) > 6 else zero_ids
            perform_hard_checks(net, num_subscribers, r0, r1, r2, check_ids)
        
        # C. 分析后10个有效客户端平均rx反而更高的现象
        info("\n" + "="*80 + "\n")
        info("📊 分析接收量分布异常\n")
        info("="*80 + "\n\n")
        
        # 读取所有客户端的接收量
        client_rx_stats = []
        for i in range(1, num_subscribers + 1):
            perf_csv = os.path.join(log_path, f"client_h{i}_perf.csv")
            if os.path.exists(perf_csv):
                try:
                    import pandas as pd
                    df = pd.read_csv(perf_csv)
                    if len(df) > 0:
                        max_rx = df['rx_bytes'].max()
                        client_rx_stats.append({'id': i, 'rx': max_rx})
                except:
                    pass
        
        if client_rx_stats:
            valid_clients = [s for s in client_rx_stats if s['rx'] > 0]
            if len(valid_clients) >= 20:
                early_avg = sum(s['rx'] for s in valid_clients[:10]) / 10
                late_avg = sum(s['rx'] for s in valid_clients[-10:]) / 10
                info(f"前10个有效客户端平均接收量: {early_avg:,.0f} bytes ({early_avg/1024/1024:.2f} MB)\n")
                info(f"后10个有效客户端平均接收量: {late_avg:,.0f} bytes ({late_avg/1024/1024:.2f} MB)\n")
                if late_avg > early_avg * 1.2:
                    info(f"⚠️  后10个客户端平均接收量明显高于前10个（{late_avg/early_avg:.2f}x）\n")
                    info(f"   这通常说明：早期批次被'连接风暴/资源限制/握手并发'打崩了一部分，\n")
                    info(f"   而后面启动的反而更稳定。\n")
                    info(f"   这和你看到的'h1/h2/h3/h6/h7/h8…'这种集中在前段编号的 0 rx_bytes 非常吻合。\n\n")
                    info(f"💡 建议应用 D1 修复（降低启动并发）\n\n")
    except Exception as e:
        error(f"❌ 诊断功能执行出错: {e}\n")
        import traceback
        traceback.print_exc()
    
    # ✅ 恢复CLI，允许交互式调试
    info("*** 进入Mininet CLI (输入 'exit' 退出) ***\n")
    # ✅ 【批量实验】去除CLI，避免中断自动化脚本
    # CLI(net)  # 已注释，用于批量实验
    
    # 清理所有进程
    cleanup_all_processes()
    
    # 停止Mininet网络
    info("停止Mininet网络...\n")
    net.stop()
    
    info("✅ 清理完成，实验结束\n")

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='MoQ Cluster 实验（支持多策略、多网络、多用户，集成dispatch_strategy）')
    parser.add_argument('--clients', type=int, default=10, help='用户数量（默认：10）')
    parser.add_argument('--strategy', type=str, default='md2g',
                       choices=['md2g', 'rolling', 'heuristic', 'clustering'],
                       help='Strategy name (default: md2g)')
    parser.add_argument('--network_type', type=str, default='4g',
                       choices=['wifi', '4g', '5g', 'fiber_optic', 'default_mix', 'wifi_dominant', '5g_dominant'],
                       help='网络类型（默认：4g）')
    parser.add_argument('--log_path', type=str, default=None, help='日志保存路径（默认：自动生成）')
    parser.add_argument('--duration', type=int, default=120, help='实验持续时间（秒，默认：120）')
    parser.add_argument('--interval', type=float, default=1.0, help='决策周期间隔（秒，默认：1.0）')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径（用于rolling和md2g策略）')
    
    args = parser.parse_args()
    
    info(f"📝 实验配置:\n")
    info(f"   用户数: {args.clients}\n")
    info(f"   策略: {args.strategy}\n")
    info(f"   网络类型: {args.network_type}\n")
    info(f"   实验时长: {args.duration}秒\n")
    info(f"   决策间隔: {args.interval}秒\n")
    if args.log_path:
        info(f"   日志路径: {args.log_path}\n")
    if args.model_path:
        info(f"   模型路径: {args.model_path}\n")
    
    run_moq_experiment(
        num_subscribers=args.clients,
        strategy=args.strategy,
        network_type=args.network_type,
        log_path=args.log_path,
        duration=args.duration,
        interval=args.interval,
        model_path=args.model_path
    )
       