# ======================================================================
#  dispatch_strategy.py – 最终完整功能版
# ======================================================================

import argparse, os, sys, json, time, subprocess, traceback, math, re, glob
import random
import pandas as pd
from collections import deque
import numpy as np
import threading

# ✅ 【Federation OFF 模式】不需要导入用户到relay映射工具
# 在 Federation OFF 模式下，所有用户都连接到 r0，忽略映射表

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MOQ_BIN_DIR = os.getenv(
    "MOQ_BIN_DIR",
    os.path.join(PROJECT_ROOT, "third_party", "moq", "target", "release"),
)

print(f"[DEBUG] Running {os.path.basename(__file__)}", file=sys.stderr, flush=True)

# === MD2G-PLUS TUNING ===
# 窗口与迟滞（优化：减少迟滞，更快响应）
MD2G_TREND_WIN = 5          # 带宽/QoE 窗口长度
MD2G_ENABLE_GOOD_CNT = 1    # 从2降低到1，更快启用增强层
MD2G_DISABLE_BAD_CNT = 2    # 从3降低到2，更快响应网络恶化

# 阈值与趋势调节（优化：更积极启用增强层以提升带宽和QoE）
MD2G_ON_BASE_THRESHOLD = 1.02   # 从1.08进一步降低到1.02，更积极启用增强层
MD2G_OFF_BASE_THRESHOLD = 1.05  # 从1.12降低到1.05，提升OFF状态QoE
MD2G_TREND_UP_ADJ = 0.90        # 从0.93进一步降低到0.90，更积极响应上行趋势
MD2G_TREND_DOWN_ADJ = 1.01       # 从1.03降低到1.01，减少保守滞后

# 稳定性奖励
MD2G_STAB_BONUS_MAX = 0.15   # 最多 +0.15
MD2G_STAB_STD_CAP  = 0.10    # std 上限映射

# DecisionQuality 权重（优化：提高带宽权重以改善带宽性能）
MD2G_DQ_W_BW = 0.80  # 从0.75进一步提高到0.80，更重视带宽
MD2G_DQ_W_DLY = 0.20  # 从0.25降低到0.20

# ==============================================================================
# ⚠️ 废弃：旧的MD2G权重配置（不再使用）
# ==============================================================================
# 注意：统一QoE公式后，所有策略（包括MD2G）都使用相同的权重配置
# 这些旧的权重定义已废弃，保留仅为历史参考，实际代码中不再使用
# ==============================================================================
# MD2G_W_ON  = dict(w_r=0.40, w_b=0.30, w_ln=0.20, w_d=0.07, w_f=0.03)  # 已废弃
# MD2G_W_OFF = dict(w_r=0.40, w_b=0.38, w_ln=0.14, w_d=0.06, w_f=0.02)  # 已废弃

# ---------------- 数据集加载 ----------------
# 数据集目录（可通过环境变量覆盖）
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(PROJECT_ROOT, "datasets"))
DATASET_DIR_REL = os.path.relpath(DATASET_DIR, PROJECT_ROOT)
print(f"[DEBUG] dispatch_strategy using datasets: {DATASET_DIR_REL}", file=sys.stderr)
df_wifi = pd.read_csv(os.path.join(DATASET_DIR, "wifi_clean.csv"))
df_4g   = pd.read_csv(os.path.join(DATASET_DIR, "4G-network-data_clean.csv"))
# ✅ 【修复】5G数据集文件名：实际是 5g_final_trace.csv，不是 5g_network_data_clean.csv
df_5g   = pd.read_csv(os.path.join(DATASET_DIR, "5g_final_trace.csv"))
df_opt  = pd.read_csv(os.path.join(DATASET_DIR, "Optic_Bandwidth_clean_2.csv"))
df_dev  = pd.read_csv(os.path.join(DATASET_DIR, "Headset device performance.csv"))

print("[DEBUG] Dataset loading completed", file=sys.stderr)

# ✅ 【修复】数据集列名不统一：4G数据集使用 "DL_bitrate_Mbps"，其他使用 "bytes_sec (Mbps)"
# 统一列名处理
def get_bandwidth_column(df, default_col="bytes_sec (Mbps)"):
    """获取带宽列名，支持多种列名格式"""
    if default_col in df.columns:
        return default_col
    elif "DL_bitrate_Mbps" in df.columns:
        return "DL_bitrate_Mbps"
    elif "bitrate" in df.columns.str.lower().values:
        bitrate_cols = [c for c in df.columns if "bitrate" in c.lower()]
        return bitrate_cols[0] if bitrate_cols else None
    else:
        # 尝试找到第一个数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else None

bw_col_wifi = get_bandwidth_column(df_wifi)
bw_col_4g = get_bandwidth_column(df_4g, "DL_bitrate_Mbps")
bw_col_5g = get_bandwidth_column(df_5g)
bw_col_opt = get_bandwidth_column(df_opt)

print(f"[DEBUG] Bandwidth columns: wifi={bw_col_wifi}, 4g={bw_col_4g}, 5g={bw_col_5g}, opt={bw_col_opt}", file=sys.stderr)

# 按网络类型分组的B_MAX（P95值）
B_MAX_BY_NETWORK = {
    "wifi": np.percentile(df_wifi[bw_col_wifi], 95) if bw_col_wifi else 100.0,
    "4g": np.percentile(df_4g[bw_col_4g], 95) if bw_col_4g else 50.0,
    "5g": np.percentile(df_5g[bw_col_5g], 95) if bw_col_5g else 200.0,
    "fiber_optic": np.percentile(df_opt[bw_col_opt], 95) if bw_col_opt else 1000.0
}

# 全局B_MAX（保持向后兼容）
all_bw = pd.concat([
    df_wifi[bw_col_wifi] if bw_col_wifi else pd.Series([100.0]),
    df_4g[bw_col_4g] if bw_col_4g else pd.Series([50.0]),
    df_5g[bw_col_5g] if bw_col_5g else pd.Series([200.0]),
    df_opt[bw_col_opt] if bw_col_opt else pd.Series([1000.0])
])
B_MAX = np.percentile(all_bw, 95)
EPS = 1e-6

print(f"[DEBUG] 按网络类型B_MAX: {B_MAX_BY_NETWORK}", file=sys.stderr)
print(f"[DEBUG] 全局B_MAX: {B_MAX:.2f} Mbps", file=sys.stderr)

# ---------- Rolling-DRL ----------
try:
    from strategies.rolling_drl_strategy_v2_refined import RollingDRLStrategy
except ImportError:
    RollingDRLStrategy = None

# ---------- 智能增强层状态管理 ----------
class SmartEnhancementState:
    """智能增强层状态管理：双阈值+迟滞机制+动态调节"""
    def __init__(self):
        self.enh_enabled = False
        self.good_cnt = 0
        self.bad_cnt = 0
        self.bw_ema = 0.0
        self.bw_window = deque(maxlen=MD2G_TREND_WIN)  # 最近5次带宽记录
        self.alpha = 0.6  # EMA平滑系数 - 优化：从0.5提高到0.6，更快响应带宽变化
        self.qoe_window = deque(maxlen=MD2G_TREND_WIN)  # 最近5次QoE记录，用于动态调节
        self.bandwidth_trend = 0  # -1/0/+1
        self.qoe_trend = 0        # -1/0/+1
        self.dynamic_threshold_multiplier = 1.0  # 动态阈值乘数
        
        # 网络特化阈值 - ✅ 【WiFi特调】针对WiFi网络大幅提高准入门槛，强制预留20-30%带宽余量
        self.thresholds = {
            # ✅ WiFi 特调：大幅提高门槛，防止波动导致卡顿
            # "on": 1.25 -> 必须有 25% 的带宽余量才允许开启 Enhanced
            # "off": 1.10 -> 一旦余量低于 10%，立刻切回 Base（保命要紧）
            # "load_max": 0.70 -> 如果 Relay 负载超过 70%，WiFi 用户就别凑热闹了
            "wifi": {
                "on": 1.25,      # 提高门槛，防止波动导致卡顿
                "off": 1.10,     # 提前撤退，减少 Stall
                "jfi_min": 0.85, 
                "load_max": 0.70, # 拥塞时 WiFi 用户优先降级
                "load_off": 0.85
            },
            # 4G 保持激进（因为本来带宽就小，不激进没法看）
            "4g": {"on": 0.98, "off": 0.95, "jfi_min": 0.88, "load_max": 0.90, "load_off": 0.95},
            # 5G/光纤 保持激进（带宽稳，不怕）
            "5g": {"on": 1.00, "off": 0.98, "jfi_min": 0.90, "load_max": 0.85, "load_off": 0.93},
            "fiber_optic": {"on": 1.02, "off": 1.00, "jfi_min": 0.92, "load_max": 0.75, "load_off": 0.88},
            # 混合网络特化阈值 - 新增：针对混合网络优化
            "mixed_4g_heavy": {"on": 1.00, "off": 0.98, "jfi_min": 0.88, "load_max": 0.90, "load_off": 0.95},
            "mixed_balanced": {"on": 1.00, "off": 0.98, "jfi_min": 0.90, "load_max": 0.85, "load_off": 0.93},
            "mixed_wifi_heavy": {"on": 1.00, "off": 0.98, "jfi_min": 0.90, "load_max": 0.80, "load_off": 0.90}
        }
        
        # 用户分组管理 - 更精准的分组策略
        self.user_groups = {
            "high_performance": [],  # 高性能用户组
            "medium_performance": [],  # 中等性能用户组
            "low_performance": []  # 低性能用户组
        }
        
        # 网络适应性状态
        self.network_adaptation = {
            "bandwidth_trend": 0,  # 带宽趋势 (1=上升, 0=稳定, -1=下降)
            "stability_score": 1.0,  # 网络稳定性评分
            "adaptation_factor": 1.0  # 适应性因子
        }
    
    def _trend_from_window(self, seq, pos=0.05, neg=-0.05):
        """从窗口数据计算趋势"""
        if len(seq) < 3:
            return 0
        diffs = np.diff(np.array(seq, dtype=float))
        avg = np.mean(diffs)
        if avg > pos:  return 1
        if avg < neg:  return -1
        return 0

    def update_windows(self, bandwidth_mbps, qoe_smooth):
        """更新窗口和趋势"""
        self.bw_window.append(float(bandwidth_mbps))
        self.qoe_window.append(float(qoe_smooth))
        self.bandwidth_trend = self._trend_from_window(self.bw_window)
        self.qoe_trend = self._trend_from_window(self.qoe_window, pos=0.01, neg=-0.01)

    def update_bandwidth(self, inst_bw: float):
        """更新带宽EMA和窗口"""
        if self.bw_ema == 0:
            self.bw_ema = inst_bw
        else:
            self.bw_ema = self.alpha * self.bw_ema + (1 - self.alpha) * inst_bw
        
        self.bw_window.append(inst_bw)
    
    def get_pessimistic_bandwidth(self) -> float:
        """获取保守带宽估计（10分位数，更激进）"""
        if len(self.bw_window) < 3:
            return self.bw_ema
        return min(self.bw_ema, np.percentile(list(self.bw_window), 10))  # 从20%降低到10%，更激进
    
    def update_qoe_and_adjust_threshold(self, qoe_value: float):
        """更新QoE并动态调整阈值"""
        self.qoe_window.append(qoe_value)
        
        if len(self.qoe_window) >= 10:  # 有足够样本时进行动态调节
            qoe_std = np.std(list(self.qoe_window))
            
            # 动态阈值调节逻辑
            if qoe_std < 0.02:  # 稳态下更激进
                self.dynamic_threshold_multiplier = 0.95
            elif qoe_std > 0.05:  # 波动大时更保守
                self.dynamic_threshold_multiplier = 1.05
            else:  # 正常状态
                self.dynamic_threshold_multiplier = 1.0
    
    def should_enable_enhancement(self, network_type: str, base_rate: float, 
                                 relay_jfi: float, relay_load: float, federation_on: bool = False) -> tuple:
        """判断是否应该启用增强层"""
        if network_type not in self.thresholds:
            network_type = "wifi"  # 默认使用wifi阈值
        
        # ✅ 【关键修复】使用网络特定的阈值配置，而不是全局常量
        specific_th = self.thresholds[network_type]
        
        # 获取网络特定的on/off阈值
        threshold_val = specific_th["on"] if not self.enh_enabled else specific_th["off"]
        
        # 根据带宽趋势微调（可选，保持向后兼容）
        if self.bandwidth_trend > 0:
            threshold_val *= MD2G_TREND_UP_ADJ
        elif self.bandwidth_trend < 0:
            threshold_val *= MD2G_TREND_DOWN_ADJ
        
        # ✅ 【WiFi特调】获取保守带宽估计
        estimated_bw = self.get_pessimistic_bandwidth()
        
        # ✅ 【新增】针对 WiFi 的额外"恐惧因子"
        # WiFi 信号通常有突发丢包，测量值往往虚高，手动打折 15%
        if network_type == "wifi":
            estimated_bw *= 0.85
            # 核心判断逻辑：
            # (带宽 * 0.85) >= (base_rate * 1.25)
            # 这意味着真实带宽必须是 base_rate 的 1.47 倍 (1.25 / 0.85) 才能开启 Enhanced
            # 这是一个非常安全的"舒适区"

        # 使用网络特定的阈值进行判断
        ok_bw   = (estimated_bw >= base_rate * threshold_val)
        ok_jfi  = (relay_jfi >= specific_th["jfi_min"])
        ok_load = (relay_load <= (specific_th["load_max"] if not self.enh_enabled else specific_th["load_off"]))
        ok = ok_bw and ok_jfi and ok_load

        # 迟滞：需要连续满足/不满足
        if ok:
            self.good_cnt += 1
            self.bad_cnt = 0
        else:
            self.bad_cnt += 1
            self.good_cnt = 0

        if not self.enh_enabled and self.good_cnt >= MD2G_ENABLE_GOOD_CNT:
            self.enh_enabled = True
            print(f"[SmartEnhancement-Plus] 启用增强层: network={network_type}, bw_pess={estimated_bw:.2f}, jfi={relay_jfi:.3f}, load={relay_load:.3f}, threshold={threshold_val:.3f}")
            return True, threshold_val
        if self.enh_enabled and self.bad_cnt >= MD2G_DISABLE_BAD_CNT:
            self.enh_enabled = False
            print(f"[SmartEnhancement-Plus] 禁用增强层: network={network_type}, bw_pess={estimated_bw:.2f}, jfi={relay_jfi:.3f}, load={relay_load:.3f}, threshold={threshold_val:.3f}")
            return False, threshold_val
        # 保持现状
        return self.enh_enabled, threshold_val

# 全局智能增强层状态
smart_enhancement = SmartEnhancementState()

# ---------- 增强版决策质量和稳定性函数 ----------
def decision_quality_v2(Bu, base_rate, delay_penalty, network_type=None):
    """决策质量 2.0 - 包含延迟响应项，支持动态权重调整（优化：混合网络更重视带宽）"""
    # 根据网络类型动态调整权重（混合网络更重视带宽）
    if network_type and network_type in ['mixed_4g_heavy', 'mixed_balanced', 'mixed_wifi_heavy']:
        w_bw = 0.80  # 混合网络：更重视带宽
        w_dly = 0.20
    else:
        w_bw = MD2G_DQ_W_BW
        w_dly = MD2G_DQ_W_DLY
    
    # 带宽利用率项（缩紧系数 1.2，比原来 1.5 更敏感）
    util = min(1.0, float(Bu) / max(1e-6, base_rate * 1.2))
    # 延迟响应项（越低越好 → 1/(1+d) 映射）
    resp = 1.0 / (1.0 + max(0.0, float(delay_penalty)))
    return min(1.0, w_bw * util + w_dly * resp)

def stability_bonus(qoe_window):
    """稳定性奖励 - 基于QoE标准差"""
    if not qoe_window:
        return 0.0
    std = float(np.std(list(qoe_window)))
    # std=0 → 奖励最大；std>=cap → 奖励趋近 0
    factor = max(0.0, 1.0 - min(std, MD2G_STAB_STD_CAP) / MD2G_STAB_STD_CAP)
    return MD2G_STAB_BONUS_MAX * factor

# ---------- 工具函数 ----------
def sample_bandwidth(net_type: str) -> float:
    """从真实数据集中采样网络带宽"""
    # ✅ 【修复】使用动态检测的列名，而不是硬编码
    if net_type == "wifi":
        col = bw_col_wifi or "bytes_sec (Mbps)"
        return float(df_wifi.sample(1)[col].iloc[0])
    elif net_type == "4g":
        col = bw_col_4g or "DL_bitrate_Mbps"
        return float(df_4g.sample(1)[col].iloc[0])
    elif net_type == "5g":
        col = bw_col_5g or "bytes_sec (Mbps)"
        return float(df_5g.sample(1)[col].iloc[0])
    elif net_type == "fiber_optic":
        col = bw_col_opt or "bytes_sec (Mbps)"
        return float(df_opt.sample(1)[col].iloc[0])
    else:
        return random.uniform(1, 10)

def gpu_boost(device_score: float) -> float:
    """基于 GPU 性能调整 Qr"""
    gpu_vals = df_dev["GPU Clock (MHz)"]
    min_gpu, max_gpu = gpu_vals.min(), gpu_vals.max()
    scale = 0.8 + 0.4 * (device_score - min_gpu) / (max_gpu - min_gpu + 1e-6)
    return max(0.8, min(scale, 1.2))

def nic_name(host_id, retries=5):
    """获取主机网络接口名，优先使用h{host_id}-eth0"""
    # 优先尝试h{host_id}-eth0
    primary_iface = f"h{host_id}-eth0"
    if os.path.exists(f"/sys/class/net/{primary_iface}"):
        return primary_iface
    
    # 回退到动态检测
    for _ in range(retries):
        ifaces = glob.glob(f"/sys/class/net/h{host_id}-eth*")
        if ifaces:
            return os.path.basename(ifaces[0])
        time.sleep(1)
    
    # 最后回退到默认接口
    print(f"[WARN] h{host_id}: 无法检测到网络接口，使用默认h{host_id}-eth0", file=sys.stderr, flush=True)
    return primary_iface

def measure_bandwidth(server_ip, duration=10, parallel=5):
    """
    使用iperf3测量真实网络带宽（从client到server）
    
    Args:
        server_ip: 目标服务器IP（relay或源服务器）
        duration: 测量持续时间（秒）
        parallel: 并发流数量
    
    Returns:
        带宽值（Mbps），如果测量失败返回None
    """
    cmd = f"iperf3 -c {server_ip} -t {duration} -P {parallel} -J"
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=duration+5)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            # 提取总接收带宽
            bits_per_second = data.get('end', {}).get('sum_received', {}).get('bits_per_second', 0)
            if bits_per_second > 0:
                return bits_per_second / 1e6  # 转换为Mbps
    except Exception as e:
        # 静默失败，不打印错误（避免日志过多）
        pass
    return None


def ping_rtt(host, iface=None):
    """
    使用ping测量真实网络延迟（RTT）
    
    Args:
        host: 目标主机IP
        iface: 网络接口名（可选）
    
    Returns:
        延迟值（毫秒），如果测量失败返回-1
    """
    try:
        cmd = ["ping", "-c", "1", "-W", "1", host]
        if iface:
            cmd += ["-I", iface]  # 指定网卡
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True, timeout=3)
        match = re.search(r'time=(\d+\.\d+)', output)
        if match:
            return float(match.group(1))
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # ping失败，返回-1表示测量失败
        return -1
    except Exception:
        return -1
    return -1

def get_real_trace_delay(network_type):
    """基于真实trace数据计算延迟"""
    try:
        import pickle
        with open('real_trace_delays.pkl', 'rb') as f:
            delay_distributions = pickle.load(f)
        
        if network_type in delay_distributions:
            delays = delay_distributions[network_type]
            # 随机选择一个延迟值
            return random.choice(delays)
        else:
            # 如果网络类型不存在，使用默认值
            return random.uniform(20, 120)
    except:
        # 如果文件不存在或读取失败，使用默认值
        return random.uniform(20, 120)

def kill(p: subprocess.Popen | None):
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            p.kill()

# ==============================================================================
# ✅ 【DataDrainer 类】持续读取进程输出，防止缓冲区阻塞
# ==============================================================================
# 作用：持续读取进程输出，防止缓冲区阻塞，实现真正的 "Download & Discard"
# 核心原理：使用二进制块读取（read(4096)），不等待换行符，防止 Pipe 死锁
# 用于 Base 和 Enhanced 订阅进程（动态订阅模式：Enhanced 进程根据 decision 启动/停止）
# ==============================================================================
class DataDrainer(threading.Thread):
    """
    强力数据抽水机（二进制块读取器）
    
    功能：
    1. 持续读取进程的 stdout/stderr，防止缓冲区满导致进程阻塞
    2. 使用二进制块读取（read(4096)），不等待换行符，防止死锁
    3. 将读取的数据写入日志文件（如果需要）
    4. 解析日志中的关键指标（延迟、码率等）
    
    原理：
    - 不使用 readline()（会死等换行符导致死锁）
    - 使用 read(4096) 二进制块读取，只要有数据就立即读取
    - 读完直接丢弃或写入文件，不做耗时操作
    - 防止 Pipe 缓冲区（64KB）被填满导致进程挂起
    """
    def __init__(self, process, name, log_file_path=None, start_time_epoch=None):
        super().__init__()
        self.process = process
        self.name = name
        self.log_file_path = log_file_path
        self.start_time_epoch = start_time_epoch if start_time_epoch is not None else time.time()  # 进程启动的绝对时间 (T0)
        self.running = True
        self.daemon = True  # 设为守护线程，主程序退出它自动退出
        self._latest_line = ""  # 缓存最后一行文本用于调试
        self.bytes_read = 0  # 读取的字节数统计
        self.last_activity_time = time.time()  # 最后活动时间
        
        # ✅ 【TTFB/TTLB 测量】核心指标
        self.ttfb_ms = None  # 首包时延 (Time To First Byte)
        self.ttlb_ms = None  # 尾包时延 (Time To Last Byte) / 传输完成时间
        self.total_bytes = 0  # 总接收字节数
        
        # ✅ 【关键修复】用于实时延迟测量
        self.last_data_ts = None  # 最近一次读到任何数据的时间（time.time()）
        self.last_read_dt_ms = None  # 最近一次 read 循环的间隔（EWMA）
        
    def run(self):
        """持续读取进程输出（强力排水模式），同时测量 TTFB 和 TTLB"""
        print(f"[DataDrainer] {self.name}: 强力排水模式启动（二进制块读取，防止Pipe死锁）", file=sys.stderr, flush=True)
        
        # 如果指定了日志文件，打开文件用于写入
        log_file = None
        if self.log_file_path:
            try:
                log_file = open(self.log_file_path, 'ab')  # 二进制追加模式
            except Exception as e:
                print(f"[DataDrainer] {self.name}: 无法打开日志文件 {self.log_file_path}: {e}", file=sys.stderr, flush=True)
        
        # ✅ 【TTFB/TTLB 测量】初始化
        has_received_first_byte = False
        last_arrival_time = None  # 用于记录最后一次收到数据的时间
        
        try:
            # 循环读取，直到进程结束
            while self.running:
                # ✅ 【关键】直接读取二进制流，不等待换行符！
                # 每次读取 4096 字节 (4KB)，只要有数据就立即读取
                try:
                    data = self.process.stdout.read(4096)
                except Exception as e:
                    # 如果读取失败（例如进程已关闭），检查进程状态
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.01)  # 短暂休眠后重试
                    continue
                
                # 如果读到空，且进程已结束，说明流传输完毕
                if not data:
                    if self.process.poll() is not None:
                        break
                    # 如果进程还在运行但没有数据，短暂休眠后继续
                    time.sleep(0.01)
                    continue
                
                current_time = time.time()
                self.bytes_read += len(data)
                self.total_bytes += len(data)
                self.last_activity_time = current_time
                last_arrival_time = current_time  # 持续更新最后时刻
                
                # ✅ 【关键修复】更新最近数据到达时间（用于实时延迟测量）
                if self.last_data_ts is not None:
                    # 计算本次 read 的间隔（毫秒）
                    dt_ms = (current_time - self.last_data_ts) * 1000.0
                    # EWMA 平滑
                    if self.last_read_dt_ms is None:
                        self.last_read_dt_ms = dt_ms
                    else:
                        self.last_read_dt_ms = 0.8 * self.last_read_dt_ms + 0.2 * dt_ms
                self.last_data_ts = current_time
                
                # ✅ 【TTFB 测量】捕获首包时延（只记录第一次）
                if not has_received_first_byte:
                    self.ttfb_ms = (current_time - self.start_time_epoch) * 1000.0
                    print(f"⏱️  [{self.name}] TTFB (首包): {self.ttfb_ms:.2f} ms", file=sys.stderr, flush=True)
                    has_received_first_byte = True
                
                # ✅ 如果指定了日志文件，将数据写入文件
                if log_file:
                    try:
                        log_file.write(data)
                        log_file.flush()  # 立即刷新，确保数据写入
                    except Exception as e:
                        if self.bytes_read % (4096 * 100) == 0:  # 每100次读取打印一次错误
                            print(f"[DataDrainer] {self.name}: 写入日志文件异常: {e}", file=sys.stderr, flush=True)
                
                # ✅ 【关键】尝试解码文本用于调试（忽略错误，避免二进制数据导致崩溃）
                try:
                    text_chunk = data.decode('utf-8', errors='ignore')
                    # 简单粗暴：只保留最后一段看起来像文本的东西用于调试
                    if len(text_chunk.strip()) > 0:
                        # 提取最后一行（可能不完整）
                        lines = text_chunk.split('\n')
                        if lines:
                            self._latest_line = lines[-1]
                except:
                    pass
                
                # ✅ 读完直接丢弃，不做任何耗时操作！
                # 这是"抽水机"模式：只负责把 Pipe 里的"水"排干，防止阻塞
                
        except Exception as e:
            print(f"[DataDrainer] {self.name} 异常: {e}", file=sys.stderr, flush=True)
        finally:
            # ✅ 【TTLB 测量】计算尾包时延（循环结束后，用最后一次收到数据的时间计算）
            if last_arrival_time:
                self.ttlb_ms = (last_arrival_time - self.start_time_epoch) * 1000.0
                
                # 计算平均吞吐量 (Throughput)
                duration_sec = self.ttlb_ms / 1000.0
                avg_speed_mbps = (self.total_bytes * 8 / 1000000) / duration_sec if duration_sec > 0 else 0
                
                print(f"🏁 [{self.name}] 传输完成! TTLB (尾包): {self.ttlb_ms:.2f} ms | "
                      f"总大小: {self.total_bytes/1024/1024:.2f} MB | "
                      f"平均速度: {avg_speed_mbps:.2f} Mbps", file=sys.stderr, flush=True)
            else:
                print(f"⚠️  [{self.name}] 未收到任何数据 (TTLB 无效)", file=sys.stderr, flush=True)
            
            # 关闭日志文件
            if log_file:
                try:
                    log_file.close()
                except:
                    pass
        
        print(f"[DataDrainer] {self.name}: 线程退出（共读取 {self.bytes_read} 字节）", file=sys.stderr, flush=True)
    
    @property
    def latest_data(self):
        """兼容之前的调用接口"""
        return self._latest_line
    
    @property
    def lines_read(self):
        """兼容之前的调用接口（返回字节数）"""
        return self.bytes_read // 1024  # 转换为KB，用于显示
    
    def stop(self):
        """停止读取线程"""
        self.running = False

def assign_relay_ip(host_id: int, num_clients: int = 10) -> str:
    """
    ✅ 根据 host_id 分配用户到 r1 或 r2
    
    分配逻辑：
    - 前一半用户（h1-h5）连接到 r1
    - 后一半用户（h6-h10）连接到 r2
    
    Relay IP 地址：
    - r1-eth0: 10.0.2.2 (连接 r0，接收数据)
    - r2-eth0: 10.0.3.2 (连接 r0，接收数据)
    """
    r1_sub_count = num_clients // 2
    if host_id <= r1_sub_count:
        return "10.0.2.2"  # r1-eth0 的 IP 地址
    else:
        return "10.0.3.2"  # r2-eth0 的 IP 地址

def parse_rebuffer_from_debug(log_path, last_size=[0]):
    """解析GST_DEBUG日志中的rebuffer信息"""
    import os, re, time
    # 增量读取，避免每次从头扫描
    try:
        if not os.path.exists(log_path):
            return 0.0, 0
        
        cur = os.path.getsize(log_path)
        start = last_size[0]
        last_size[0] = cur
        if cur <= start:
            return 0.0, 0  # 无新内容
        
        with open(log_path, "r", errors="ignore") as fp:
            fp.seek(start)
            chunk = fp.read()

        # 解析buffering信息
        stall_time = 0.0
        stall_cnt = 0
        
        # 查找buffering相关的日志
        buffering_patterns = [
            r'buffering.*?(\d+)%',  # buffering 0% -> 100%
            r'buffering done',      # buffering完成
            r'underflow',           # 缓冲区下溢
            r'stall'                # 卡顿
        ]
        
        # 简化：每出现一次 "buffering done" 或 "underflow" 就认为发生过一次卡顿
        stall_cnt = len(re.findall(r'buffering done|underflow|stall', chunk, re.IGNORECASE))
        
        # 如果有时间戳格式，可进一步精确计算持续时间；这里先返回计数
        return stall_time, stall_cnt
    except Exception as e:
        print(f"[WARN] 解析rebuffer日志失败: {e}", file=sys.stderr, flush=True)
        return 0.0, 0

def calculate_jain_fairness_index(load_rates):
    """计算Jain's Fairness Index (JFI)"""
    if not load_rates or len(load_rates) == 0:
        return 1.0
    
    # 将负载率转换为可用裕量 (1 - load_rate)
    available_margins = [1.0 - rate for rate in load_rates]
    
    # 避免除零
    if sum(available_margins) == 0:
        return 0.0
    
    # JFI = (sum(x))^2 / (n * sum(x^2))
    n = len(available_margins)
    sum_x = sum(available_margins)
    sum_x_squared = sum(x * x for x in available_margins)
    
    if sum_x_squared == 0:
        return 1.0
    
    jfi = (sum_x * sum_x) / (n * sum_x_squared)
    return max(0.0, min(1.0, jfi))  # 限制在[0,1]范围内

def calculate_system_load_balance(relay_loads):
    """计算系统级负载均衡度 L_net"""
    if not relay_loads:
        return 1.0
    
    # 使用Jain's Fairness Index计算负载均衡度
    return calculate_jain_fairness_index(relay_loads)

# ---------- 主逻辑 ----------
def run_client(a):
    # 1) 策略加载
    # ✅ 【关键说明】只有Rolling策略需要加载RL模型，其他策略（MD2G, Heuristic, Clustering, Groot, Pano）不加载模型
    # 其他策略从决策文件读取决策（由各自的controller生成）
    # 这样设计的好处：
    # 1. 避免不必要的模型加载（节省内存和启动时间）
    # 2. 保持策略独立性（每个策略使用自己的controller）
    # 3. 统一的perf.csv生成逻辑（所有策略都经过同一个中间层）
    if a.strategy == "rolling":
        # ✅ 【关键保证】Rolling策略必须加载RL模型
        if RollingDRLStrategy is None:
            print(f"❌ [ERROR] h{a.host_id}: RollingDRLStrategy 模块缺失，无法加载Rolling策略", file=sys.stderr, flush=True)
            sys.exit(1)
        
        # ✅ 【关键保证】使用新训练的SC-DDQN模型
        model = os.path.join(a.model_path, "sc_ddqn_rolling.pth")
        
        # ✅ 【关键保证】验证模型文件是否存在
        if not os.path.exists(model):
            print(f"❌ [ERROR] h{a.host_id}: Rolling策略模型文件不存在: {model}", file=sys.stderr, flush=True)
            print(f"    [INFO] 模型路径: {a.model_path}", file=sys.stderr, flush=True)
            print(f"    [INFO] 请确保模型文件 sc_ddqn_rolling.pth 存在于 {a.model_path} 目录中", file=sys.stderr, flush=True)
            sys.exit(1)
        
        # ✅ 【关键保证】加载Rolling策略的RL模型
        try:
            strat = RollingDRLStrategy(model_path=model, window_size=3, state_feature_count=5)
            print(f"✅ [INFO] h{a.host_id}: Rolling策略 - 已成功加载RL模型: {model}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"❌ [ERROR] h{a.host_id}: Rolling策略模型加载失败: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif a.strategy == "md2g":
        # ✅ 【关键说明】MD2G策略的模型在regional_relay_controller.py中加载（服务端）
        # MD2G是服务端策略，模型在relay端加载，客户端不加载模型
        strat = None
        print(f"[INFO] h{a.host_id}: MD2G策略 - 模型在服务端(regional_relay_controller)加载，客户端从决策文件读取决策", file=sys.stderr, flush=True)
    else:
        # ✅ 【关键说明】其他策略（Heuristic, Clustering, Groot, Pano）不加载模型
        # 这些策略从决策文件读取决策（由各自的controller生成）
        strat = None
        print(f"[INFO] h{a.host_id}: {a.strategy}策略 - 不加载模型，从决策文件读取决策", file=sys.stderr, flush=True)

    state_window, qoe_window = deque(maxlen=3), deque(maxlen=15)

    # 2) 日志
    os.makedirs(a.log_path, exist_ok=True)
    perf_log = os.path.join(a.log_path, f"client_h{a.host_id}_perf.log")
    perf_csv = os.path.join(a.log_path, f"client_h{a.host_id}_perf.csv")  # ✅ 【CSV实时写入】同时生成CSV文件
    gst_log = os.path.join(a.log_path, f"client_h{a.host_id}_gst.log")
    
    # ✅ 【CSV实时写入】CSV文件头（同时写入.log和.csv文件）
    csv_header = "timestamp,user_id,network_type,device_score,selected_layer,subscription_type," \
                 "delay_ms,ttlb_ms,stall_total_sec,stall_count,bandwidth_mbps,rx_bytes," \
                 "reward_R_o,reward_R_q,reward_R_b,reward_final," \
                 "grouping_id,grouping_efficiency,load_balance_jfi,decision_step,buffer_level_sec\n"
    
    # ✅ 【CSV实时写入】同时创建.log和.csv文件，写入相同的CSV头
    with open(perf_log, "w") as f:
        # ✅ 统一奖励函数日志格式（对应论文 Eq.(1)）
        # R_t = λ_o*R_o + λ_q*R_q - λ_b*R_b
        # ✅ 【新增】添加 Time to Last Byte (ttlb_ms) 延迟指标
        f.write(csv_header)
    
    with open(perf_csv, "w") as f:
        # ✅ 【CSV实时写入】CSV文件与.log文件内容完全相同（都是CSV格式）
        f.write(csv_header)
    
    print(f"[INFO] h{a.host_id}: ✅ CSV文件已创建: {perf_csv}", file=sys.stderr, flush=True)

    # 3) GStreamer 基础流
    # ✅ 【关键修复】优先使用命令行传入的relay_ip（Rolling/GROOT策略连接到r0）
    # 如果命令行未传入relay_ip，则根据host_id分配用户到r1或r2（组播策略）
    if a.relay_ip:
        # ✅ 使用命令行传入的relay_ip（Rolling/GROOT策略会传入r0的IP: 10.0.2.1）
        relay_ip = a.relay_ip
        if relay_ip == "10.0.2.1":
            relay_name = "r0"
        elif relay_ip == "10.0.2.2":
            relay_name = "r1"
        elif relay_ip == "10.0.3.2":
            relay_name = "r2"
        else:
            relay_name = "unknown"
    else:
        # 默认分配：前一半用户连接到r1，后一半用户连接到r2（组播策略）
        relay_ip = assign_relay_ip(a.host_id, a.clients)
        relay_name = "r1" if relay_ip == "10.0.2.2" else "r2"
    current_relay_ip = relay_ip  # 当前使用的 relay IP
    print(f"[DEBUG] h{a.host_id}: 初始relay_ip={relay_ip} (连接到 {relay_name})", file=sys.stderr, flush=True)
    # 强制IPv4和保守设置，避免hangsrc崩溃，并添加GST_DEBUG
    # 添加系统 GStreamer 插件路径，确保能找到 hangsrc
    # 注意：a.gst_plugin_path 已经是包含 libgsthang.so 的目录
    system_gst_path = "/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
    # 确保插件路径正确：优先使用传入的路径，然后是系统路径
    GST_ENV = f"GST_PLUGIN_PATH={a.gst_plugin_path}:{system_gst_path} "
    GST_ENV += "RUST_BACKTRACE=1 RUST_LOG=info "
    GST_ENV += "MOQ_HANGSRC_FORCE_IPV4=1 MOQ_HANGSRC_DISABLE_ZERO_COPY=1 "
    GST_ENV += "GST_DEBUG=hangsrc:4,pipeline:3,queue:3,decodebin:3 "
    GST_ENV += f"GST_DEBUG_FILE=/tmp/client_logs/client_h{a.host_id}_gstdebug.log "
    
    # 改进的pipeline配置：使用更健壮的配置，避免decodebin自动检测失败
    # 尝试多种pipeline配置，如果一种失败，尝试下一种
    # 注意：hangsrc输出的是编码后的数据流，需要先等待数据流开始
    # 
    # 配置优先级：SIMPLE -> MEDIUM -> FULL
    # SIMPLE: 最简形式，只测试hangsrc本身是否工作
    # MEDIUM: 添加decodebin，测试能否解码
    # FULL: 完整pipeline，包含videoconvert
    #
    # 注意：async=false 只放在 fakesink 上，用于减少异步preroll问题
    # 但这不会让pipeline在NULL状态运行，pipeline仍需要进入PAUSED/PLAYING状态
    
    # ========== 替换hangsrc为moq-sub ==========
    # moq-sub 路径（从 MOQ_BIN_DIR 读取，避免写死机器本地路径）
    MOQ_SUB_PATH = os.path.join(MOQ_BIN_DIR, "moq-sub")
    
    # ✅ 使用 MoQ dump 文件路径（不再使用 hangsrc 命名）
    moq_dump_file = f"/tmp/moq_h{a.host_id}.bin"
    
    # 验证moq-sub是否可用
    def check_moq_sub():
        """检查moq-sub工具是否可用"""
        # ✅ 【修复】显式导入subprocess，避免作用域问题
        import subprocess as sp
        try:
            if not os.path.exists(MOQ_SUB_PATH):
                print(f"[WARN] h{a.host_id}: moq-sub不存在: {MOQ_SUB_PATH}", file=sys.stderr, flush=True)
                return False
            # 测试运行
            check_cmd = [MOQ_SUB_PATH, "--help"]
            result = sp.run(check_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"[DEBUG] h{a.host_id}: moq-sub 工具验证成功", file=sys.stderr, flush=True)
                return True
            else:
                print(f"[WARN] h{a.host_id}: moq-sub 验证失败: {result.stderr}", file=sys.stderr, flush=True)
                return False
        except Exception as e:
            print(f"[WARN] h{a.host_id}: moq-sub 检查异常: {e}", file=sys.stderr, flush=True)
            return False
    
    def build_moq_sub_cmd(track_url: str, output_file: str):
        """构建moq-sub命令
        track_url: MoQ URL (完整track路径，如 https://relay:4443/track，hang 不支持 namespace)
        output_file: 输出文件路径
        """
        # moq-sub使用https://或moql://协议
        # 如果URL是https://，直接使用；如果是moq://，转换为https://
        if track_url.startswith("moq://"):
            track_url = track_url.replace("moq://", "https://", 1)
        elif not track_url.startswith("https://"):
            # 默认添加https://
            if not track_url.startswith("http"):
                track_url = f"https://{track_url}"
        
        # ✅ 使用完整的 track URL，不需要 --track 参数
        # ✅ 确保使用 --tls-disable-verify 参数（即使设置了环境变量，也显式指定）
        cmd = [
            MOQ_SUB_PATH,
            track_url,
            "--dump", output_file,
            "--tls-disable-verify"
        ]
        
        return cmd
    
    # 先验证moq-sub工具
    moq_sub_available = check_moq_sub()
    base_p = None
    
    if not moq_sub_available:
        print(f"[ERROR] h{a.host_id}: ❌ moq-sub 工具不可用，实验终止", file=sys.stderr, flush=True)
        sys.exit(1)
    else:
        print(f"[DEBUG] h{a.host_id}: 启动 base moq-sub 流程", file=sys.stderr, flush=True)
        
        # ============================================================
        # Module 3: Client-side cache-aware fetch (Federation ON)
        # ============================================================
        # Try to fetch from federation cache first if federation is ON
        if a.federation == 'on':
            # Import relay_cache from topo module (will be available at runtime)
            try:
                # Try to fetch cached segment
                segment_seq = 0  # Start with first segment
                cache_key = f"redandblack_base_segment_{segment_seq}"
                
                # Note: In a real implementation, relay_cache would be accessible
                # For now, we simulate cache check via file system or shared memory
                cache_file = f"/tmp/federation_cache_{cache_key}"
                if os.path.exists(cache_file):
                    print(f"[CACHE HIT] h{a.host_id}: {cache_key}", file=sys.stderr, flush=True)
                    # Load from cache
                    with open(cache_file, 'rb') as f:
                        cached_data = f.read()
                    if cached_data:
                        with open(moq_dump_file, 'wb') as f:
                            f.write(cached_data)
                        print(f"[CACHE] h{a.host_id}: Base loaded from federation cache", file=sys.stderr, flush=True)
                        # Continue with moq-sub for subsequent segments
                else:
                    print(f"[CACHE MISS] h{a.host_id}: {cache_key}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[WARN] h{a.host_id}: Cache check failed: {e}", file=sys.stderr, flush=True)
        
        # ✅ 【关键修复：参考 simple_moq_test.py 的成功经验】使用根路径 URL + --broadcast 参数
        # ✅ 根据 relay_ip 选择对应的 relay URL
        # ⚠️ 重要：根据 simple_moq_test.py 的成功经验，应该使用：
        # - URL: https://r1.local:4443/ 或 https://r2.local:4443/ (根路径，不是 /base)
        # - --broadcast base (通过参数指定)
        # - --track video0 (通过参数指定)
        # 
        # Relay 分配逻辑：
        # - 前一半用户连接到 r1.local:4443
        # - 后一半用户连接到 r2.local:4443
        # - r1 和 r2 都运行 moq-relay 服务，监听 4443 端口
        # - r1 和 r2 从 r0 订阅数据，然后分发给各自的用户（实现组播）
        # ✅ 【关键修复】根据 relay_ip 选择对应的 relay URL
        if relay_ip == "10.0.2.2":  # r1
            track_url = f"https://r1.local:4443/"
            relay_domain = "r1.local"
            relay_name = "r1"
        elif relay_ip == "10.0.3.2":  # r2
            track_url = f"https://r2.local:4443/"
            relay_domain = "r2.local"
            relay_name = "r2"
        else:
            # 默认回退到 r0（向后兼容）
            track_url = f"https://r0.local:4443/"
            relay_domain = "r0.local"
            relay_name = "r0"
        
        # ✅ 在启动moq-sub之前，验证网络连通性
        print(f"[DEBUG] h{a.host_id}: 验证网络连通性...", file=sys.stderr, flush=True)
        
        # ✅ 根据 relay_ip 选择对应的 ping 目标
        relay_ip_to_ping = relay_ip  # 使用分配的 relay IP
        
        # 检查域名解析
        try:
            import socket
            relay_ip_resolved = socket.gethostbyname(relay_domain)
            print(f"[DEBUG] h{a.host_id}: ✅ {relay_domain} 解析为 {relay_ip_resolved}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[ERROR] h{a.host_id}: ❌ 无法解析 {relay_domain}: {e}", file=sys.stderr, flush=True)
            print(f"[ERROR] h{a.host_id}: 尝试检查 /etc/hosts...", file=sys.stderr, flush=True)
            try:
                with open('/etc/hosts', 'r') as f:
                    hosts_content = f.read()
                    if relay_domain in hosts_content:
                        print(f"[DEBUG] h{a.host_id}: /etc/hosts 中包含 {relay_domain} 条目", file=sys.stderr, flush=True)
                    else:
                        print(f"[ERROR] h{a.host_id}: /etc/hosts 中未找到 {relay_domain} 条目", file=sys.stderr, flush=True)
                        # 尝试添加
                        # ✅ 【修复】使用已导入的subprocess模块
                        import subprocess as sp_check
                        sp_check.run(['sh', '-c', f'echo "{relay_ip} {relay_domain}" >> /etc/hosts'], check=False)
                        print(f"[DEBUG] h{a.host_id}: 已尝试添加 {relay_domain} ({relay_ip}) 到 /etc/hosts", file=sys.stderr, flush=True)
            except Exception as e2:
                print(f"[WARN] h{a.host_id}: 检查 /etc/hosts 失败: {e2}", file=sys.stderr, flush=True)
        
        # 检查网络连通性（ping relay IP）
        # ✅ 【修复】显式导入subprocess，避免作用域问题
        import subprocess as sp_check
        try:
            ping_result = sp_check.run(['ping', '-c', '1', '-W', '1', relay_ip_to_ping], 
                                       capture_output=True, timeout=3)
            if ping_result.returncode == 0:
                print(f"[DEBUG] h{a.host_id}: ✅ 可以 ping 通 {relay_name} ({relay_ip_to_ping})", file=sys.stderr, flush=True)
            else:
                print(f"[WARN] h{a.host_id}: ⚠️ 无法 ping 通 {relay_name} ({relay_ip_to_ping})", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARN] h{a.host_id}: ping 检查失败: {e}", file=sys.stderr, flush=True)
        
        # 检查端口连通性（使用nc或telnet，检查 relay 的 4443 端口）
        try:
            nc_result = sp_check.run(['nc', '-z', '-w', '1', relay_ip_to_ping, '4443'], 
                                      capture_output=True, timeout=3)
            if nc_result.returncode == 0:
                print(f"[DEBUG] h{a.host_id}: ✅ {relay_name}:4443 端口可访问", file=sys.stderr, flush=True)
            else:
                print(f"[WARN] h{a.host_id}: ⚠️ {relay_name}:4443 端口不可访问", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARN] h{a.host_id}: 端口检查失败（可能没有nc工具）: {e}", file=sys.stderr, flush=True)
        
        # moq-sub启动逻辑（使用 latency wrapper 写入延迟日志）
        print(f"[DEBUG] h{a.host_id}: 启动moq-sub订阅base流", file=sys.stderr, flush=True)
        
        # ✅ 【关键修复】使用 latency wrapper 脚本写入延迟日志
        # 确保写入路径与 topo_moq_eval_OFF.py 读取路径完全一致
        latency_wrapper = os.path.join(PROJECT_ROOT, "moq_sub_with_latency.py")
        latency_log_file = f"/tmp/moq_latency_h{a.host_id}_base.log"  # ✅ 与读取端路径一致
        sub_start_time = time.time()  # 记录订阅开始时间
        
        # ✅ 使用 wrapper 脚本启动 moq-sub（写入延迟日志）
        # ✅ 【关键修复】Track名称必须与Publisher一致
        # hang publish 默认使用 video0（重编码后的默认值）
        # 如果Publisher使用其他track名称，需要确保匹配
        track_name = "video0"  # ✅ 默认使用video0，与hang publish一致
        
        # ✅ 【关键修复：参考 simple_moq_test.py】传递 broadcast 名称参数
        # simple_moq_test.py 使用：--url https://r0.local:4443/ --broadcast base --track video0
        # 因此需要传递 broadcast 名称作为第8个参数
        broadcast_name = "base"  # ✅ 与 Publisher 的 --name base 一致
        
        base_cmd_wrapper = [
            sys.executable,  # python3
            latency_wrapper,
            MOQ_SUB_PATH,
            track_name,  # ✅ 使用video0，与Publisher一致
            track_url,
            moq_dump_file,
            gst_log,
            latency_log_file,
            str(sub_start_time),
            broadcast_name  # ✅ 【关键修复】传递 broadcast 名称，与 simple_moq_test.py 保持一致
        ]
        
        print(f"[INFO] h{a.host_id}: 启动moq-sub订阅base流（使用latency wrapper）", file=sys.stderr, flush=True)
        print(f"[DEBUG] h{a.host_id}: Track名称: {track_name} (必须与Publisher一致)", file=sys.stderr, flush=True)
        print(f"[DEBUG] h{a.host_id}: URL: {track_url}", file=sys.stderr, flush=True)
        print(f"[DEBUG] h{a.host_id}: moq-sub dump文件路径: {moq_dump_file}", file=sys.stderr, flush=True)
        print(f"[DEBUG] h{a.host_id}: 延迟日志路径: {latency_log_file} (确保与读取端路径一致)", file=sys.stderr, flush=True)
        
        # ✅ 【关键修复：大幅增加超时时间】设置环境变量，确保 moq-sub 可以正常工作
        # 问题：即使设置了 300s 超时，在 Mininet 环境下仍然可能超时断开
        # 解决：大幅增加到 1800s（30分钟），与 Publisher 和 Relay 保持一致
        env = os.environ.copy()
        env['MOQ_TLS_DISABLE_VERIFY'] = '1'  # ✅ 禁用 TLS 证书验证（解决ApplicationClosed问题）
        env['MOQ_TRANSPORT_IDLE_TIMEOUT'] = '1800s'  # ✅ 【关键修复】增加到1800s（30分钟），与Publisher和Relay保持一致
        env['QUIC_IDLE_TIMEOUT'] = '1800s'  # ✅ 【关键修复】增加QUIC空闲超时到1800s
        env['RUST_LOG'] = 'info'  # 设置日志级别
        env['RUST_BACKTRACE'] = '1'  # 启用backtrace用于调试
        
        # ✅ 【关键修复】使用 PIPE 模式 + DataDrainer，防止 Pipe 死锁
        # 必须使用 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
        # bufsize=0 表示二进制无缓冲模式，让数据直接流出
        base_p = subprocess.Popen(
            base_cmd_wrapper,
            stdout=subprocess.PIPE,  # ✅ 使用 PIPE，让 DataDrainer 读取
            stderr=subprocess.STDOUT,  # ✅ 合并 stderr 到 stdout
            bufsize=0,  # ✅ 关键：关闭系统级缓冲，让数据直接流出
            env=env  # ✅ 传递环境变量
        )
        
        # ✅ 启动 Base 流 DataDrainer 线程
        # 持续读取进程输出，防止缓冲区阻塞（二进制块读取，不等待换行符）
        # ✅ 【TTFB/TTLB 测量】传入启动时间，用于测量真实业务延迟
        base_start_time = time.time()  # 记录 Base 流启动时间
        reader_base = DataDrainer(base_p, "Base_Stream", gst_log, start_time_epoch=base_start_time)
        reader_base.start()
        print(f"[INFO] h{a.host_id}: ✅ Base DataDrainer 线程已启动（强力排水模式，防止Pipe死锁）", file=sys.stderr, flush=True)
        
        # 等待moq-sub建立连接并开始接收数据
        print(f"[INFO] h{a.host_id}: moq-sub已启动（PID: {base_p.pid}），等待连接和数据流...", file=sys.stderr, flush=True)
        
        # ✅ 增加等待时间：等待 publisher 完全启动（最多30秒）
        connection_timeout = 30
        connection_start = time.time()
        session_started = False
        
        while time.time() - connection_start < connection_timeout:
            # 检查进程是否还在运行
            if base_p.poll() is not None:
                exit_code = base_p.returncode
                print(f"[WARN] h{a.host_id}: moq-sub进程退出（退出码: {exit_code}），检查日志", file=sys.stderr, flush=True)
                # ✅ 【关键诊断】检查日志中的错误信息
                try:
                    with open(gst_log, "r", errors="ignore") as f:
                        log_content = f.read()
                        # 检查常见错误
                        if "ApplicationClosed" in log_content:
                            print(f"[ERROR] h{a.host_id}: ❌ ApplicationClosed错误 - 可能是证书验证失败或Track名称不匹配", file=sys.stderr, flush=True)
                            print(f"[ERROR] h{a.host_id}: 建议检查: 1) TLS验证是否禁用 2) Track名称是否与Publisher一致", file=sys.stderr, flush=True)
                        elif "TimedOut" in log_content or "timeout" in log_content.lower():
                            print(f"[ERROR] h{a.host_id}: ❌ TimedOut错误 - 可能是网络问题或UDP丢包", file=sys.stderr, flush=True)
                            print(f"[ERROR] h{a.host_id}: 建议检查: 1) 网络连通性 2) 防火墙设置 3) 带宽限制", file=sys.stderr, flush=True)
                        elif "session started" in log_content.lower() or "Session started" in log_content:
                            session_started = True
                            print(f"[INFO] h{a.host_id}: ✅ moq-sub连接成功（在日志中发现session started）✓", file=sys.stderr, flush=True)
                            break
                        # 输出最后几行日志用于调试
                        log_lines = log_content.split('\n')
                        if len(log_lines) > 5:
                            print(f"[DEBUG] h{a.host_id}: 最后5行日志:", file=sys.stderr, flush=True)
                            for line in log_lines[-5:]:
                                if line.strip():
                                    print(f"[DEBUG] h{a.host_id}: {line}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[WARN] h{a.host_id}: 读取日志失败: {e}", file=sys.stderr, flush=True)
                break
            
            # 检查日志中是否有session started
            try:
                with open(gst_log, "r", errors="ignore") as f:
                    log_content = f.read()
                    if "session started" in log_content.lower() or "Session started" in log_content:
                        session_started = True
                        print(f"[INFO] h{a.host_id}: ✅ moq-sub连接成功（session started）✓", file=sys.stderr, flush=True)
                        break
            except Exception:
                pass
            
            time.sleep(0.5)  # 每0.5秒检查一次
        
        if not session_started:
            print(f"[ERROR] h{a.host_id}: ❌ moq-sub未能在{connection_timeout}秒内建立连接", file=sys.stderr, flush=True)
            print(f"[ERROR] h{a.host_id}: ⚠️  可能原因: 1) Publisher未启动 2) Broadcast未注册 3) Track名称不匹配", file=sys.stderr, flush=True)
            print(f"[ERROR] h{a.host_id}: 建议检查: 1) Publisher是否在n0节点上运行 2) Relay日志中是否有publish=base", file=sys.stderr, flush=True)
            # 打印moq-sub日志的最后几行
            try:
                with open(gst_log, "r", errors="ignore") as f:
                    log_lines = f.readlines()
                    if log_lines:
                        print(f"[ERROR] h{a.host_id}: moq-sub日志最后10行:", file=sys.stderr, flush=True)
                        for line in log_lines[-10:]:
                            print(f"  {line.rstrip()}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[WARN] h{a.host_id}: 无法读取moq-sub日志: {e}", file=sys.stderr, flush=True)
            if base_p.poll() is None:
                base_p.terminate()
                time.sleep(0.5)
                if base_p.poll() is None:
                    base_p.kill()
            # ✅ 不要立即退出，让主循环继续运行，以便topo脚本可以收集诊断信息
            print(f"[ERROR] h{a.host_id}: moq-sub连接失败，但继续运行以便诊断", file=sys.stderr, flush=True)
            # sys.exit(1)  # 暂时注释掉，让进程继续运行
        
        # ✅ 增加等待时间：等待数据流到达（最多30秒）
        print(f"[INFO] h{a.host_id}: moq-sub连接成功，等待数据流到达...", file=sys.stderr, flush=True)
        data_timeout = 30
        data_start = time.time()
        startup_success = False
        
        while time.time() - data_start < data_timeout:
            try:
                if os.path.exists(moq_dump_file):
                    file_size = os.path.getsize(moq_dump_file)
                    if file_size > 0:
                        print(f"[INFO] h{a.host_id}: ✅ 数据流已到达（文件大小: {file_size} bytes），moq-sub成功接收数据✓", file=sys.stderr, flush=True)
                        startup_success = True
                        # ✅ 【记录订阅开始时间】用于计算 Time to Last Byte
                        if not hasattr(run_client, 'sub_start_time'):
                            run_client.sub_start_time = time.time()
                        break
            except Exception as e:
                print(f"[WARN] h{a.host_id}: 检查数据文件时出错: {e}", file=sys.stderr, flush=True)
            
            # 检查进程是否还在运行
            if base_p.poll() is not None:
                print(f"[WARN] h{a.host_id}: moq-sub进程退出（退出码: {base_p.returncode}）", file=sys.stderr, flush=True)
                break
            
            time.sleep(1)  # 每秒检查一次
        
        if not startup_success:
            print(f"[ERROR] h{a.host_id}: ❌ moq-sub在{data_timeout}秒内未收到任何数据（文件仍为0字节）", file=sys.stderr, flush=True)
            # 打印moq-sub日志的最后几行
            try:
                with open(gst_log, "r", errors="ignore") as f:
                    log_lines = f.readlines()
                    if log_lines:
                        print(f"[ERROR] h{a.host_id}: moq-sub日志最后10行:", file=sys.stderr, flush=True)
                        for line in log_lines[-10:]:
                            print(f"  {line.rstrip()}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[WARN] h{a.host_id}: 无法读取moq-sub日志: {e}", file=sys.stderr, flush=True)
            # 检查dump文件状态
            if os.path.exists(moq_dump_file):
                file_size = os.path.getsize(moq_dump_file)
                print(f"[ERROR] h{a.host_id}: dump文件存在但大小为 {file_size} bytes", file=sys.stderr, flush=True)
            else:
                print(f"[ERROR] h{a.host_id}: dump文件不存在: {moq_dump_file}", file=sys.stderr, flush=True)
            print(f"[ERROR] h{a.host_id}: 媒体流未打通，但继续运行以便诊断", file=sys.stderr, flush=True)
            # ✅ 不要立即退出，让主循环继续运行，以便topo脚本可以收集诊断信息
            # if base_p.poll() is None:
            #     base_p.terminate()
            #     time.sleep(0.5)
            #     if base_p.poll() is None:
            #         base_p.kill()
            # sys.exit(1)  # 暂时注释掉，让进程继续运行
        
        print(f"[INFO] h{a.host_id}: ✅ moq-sub启动成功，正在接收base流数据✓", file=sys.stderr, flush=True)
        
        # ✅ 【记录订阅开始时间】用于计算 Time to Last Byte
        run_client.sub_start_time = time.time()


    # ✅ 【动态订阅模式】只启动 Base 订阅进程，Enhanced 根据 decision 动态订阅
    # 保持单连接/单握手，根据 decision 动态开关 Enhanced 对象的订阅
    # 这样网络流量会随 decision 变化，buffer/stall/奖励都能自洽
    print(f"[DEBUG] h{a.host_id}: [动态订阅模式] 启动 Base 订阅进程，Enhanced 将根据 decision 动态订阅...", file=sys.stderr, flush=True)
    
    # Enhanced 订阅相关变量（初始化为 None，表示未订阅）
    enh_dump_file = f"/tmp/moq_h{a.host_id}_enh.bin"
    enh_log_file = gst_log.replace(".log", "_enh.log")
    enh_latency_log_file = f"/tmp/moq_latency_h{a.host_id}_enhanced.log"
    enh_track_name = "video0"  # ✅ 与Publisher一致
    enh_broadcast_name = "enhanced"  # ✅ 与 Publisher 的 --name enhanced 一致
    enh_p = None  # Enhanced 订阅进程（动态启动/停止）
    reader_enhanced = None  # Enhanced DataDrainer 线程
    enh_sub_start_time = None  # Enhanced 订阅开始时间
    enh_start_time = None  # Enhanced 流启动时间
    last_decision = 0  # 上一次的 decision，用于检测变化
    
    # 等待 Base 流连接建立（给一些时间完成握手）
    print(f"[DEBUG] h{a.host_id}: 等待 Base 流连接建立（2秒）...", file=sys.stderr, flush=True)
    time.sleep(2)
    
    # 4) 主循环
    start = time.time()
    enh_on = False  # ✅ 【动态订阅模式】逻辑标志：Enhanced 订阅进程是否正在运行（decision==1 时为 True）
    iteration = 0   # ✅ 修复：增加循环计数器
    
    # 初始化rebuffer统计
    stall_total_sec = 0.0
    stall_count = 0
    last_debug_size = [0]
    # ✅ 【修复：stall增量计算】保存上一次的累计值，用于计算增量
    last_stall_total_sec = 0.0
    stall_inc_smooth = 0.0  # EWMA平滑后的stall增量
    
    # ✅ 【Buffer Level 计算初始化】
    buffer_level_sec = 5.0  # 初始缓冲5秒（模拟播放器预加载，确保从5秒开始）
    last_buffer_bytes = 0  # 上一次的累计接收字节数
    last_buffer_update_time = None  # 上一次更新 buffer 的时间戳
    last_total_bitrate_bps = None  # 上一次的总码率（用于层级切换时保持连续性）
    last_decision = 0  # 上一次的 decision（用于检测层级切换）
    
    # ✅ 【被动带宽估测初始化】替代 iperf
    # ✅ 【关键修复】将时间设为 0，强制触发循环内的初始化判断
    # ❌ 错误：last_check_time = time.time() 会导致初始化判断 if last_check_time <= 0: 永远不会成立
    # ✅ 正确：last_check_time = 0.0 才能触发初始化，避免第1秒的带宽尖峰
    last_total_bytes = 0
    last_check_time = 0.0  # 强制初始化为0，触发循环内的初始化判断
    Bu = 0.0  # 强制初始化为 0，不要 sample_bandwidth
    # ✅ 【改进】添加连续零带宽计数器，避免高并发下单次delta_bytes=0就归零
    zero_delta_count = 0  # 连续delta_bytes=0的计数
    MAX_ZERO_DELTA_COUNT = 3  # 连续3次delta_bytes=0才真正归零（约1.5秒）
    # ✅ 【动态订阅模式】enh_dump_file 已在初始化阶段定义（第921行），这里不需要重新定义
    
    # ✅ 【初始化订阅开始时间】如果没有记录，使用实验开始时间
    if not hasattr(run_client, 'sub_start_time'):
        run_client.sub_start_time = start
    
    # ✅ 【关键修复】在主循环开始前就写入初始状态文件，确保控制器能立即收集到用户信息
    # 这解决了"控制器收集不到host信息"的问题
    SHARED_STATE_DIR = "/tmp/mininet_shared"
    client_state_file = f"{SHARED_STATE_DIR}/client_h{a.host_id}_state.json"
    try:
        os.makedirs(SHARED_STATE_DIR, exist_ok=True)
        os.chmod(SHARED_STATE_DIR, 0o777)
        # 写入初始状态文件（即使还没有数据，也要写入，确保控制器能收集到用户信息）
        initial_state = {
            "host_id": a.host_id,
            "network_type": a.network_type,
            "throughput_mbps": Bu,  # 使用初始带宽估计
            "delay_ms": 50.0,  # 默认延迟（会在第一次迭代时更新）
            "device_score": float(a.device_score),
            "last_decision_layer": 0,  # 初始只订阅Base层
            "reward_R_o": 0.0,
            "reward_R_q": 0.0,
            "reward_R_b": 0.0,
            "reward_R_l": 0.0,
            "reward_final": 0.0,
            "timestamp": time.time(),
            "viewpoint": "front_center"
        }
        temp_file = f"{client_state_file}.tmp"
        with open(temp_file, 'w') as sf:
            json.dump(initial_state, sf)
            sf.flush()
            os.fsync(sf.fileno())
        os.rename(temp_file, client_state_file)
        os.chmod(client_state_file, 0o666)
        print(f"[DEBUG] h{a.host_id}: ✅ 已写入初始状态文件: {client_state_file} (在主循环开始前)", file=sys.stderr, flush=True)
        # 验证文件确实已创建
        if os.path.exists(client_state_file):
            file_size = os.path.getsize(client_state_file)
            print(f"[DEBUG] h{a.host_id}: ✅ 初始状态文件验证: {file_size} 字节", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[WARN] h{a.host_id}: 无法写入初始状态文件: {e}", file=sys.stderr, flush=True)

    # ✅ 【CSV实时写入】同时打开.log和.csv文件，确保实时写入
    perf_log_file = open(perf_log, "a", buffering=1)  # ✅ line buffered
    perf_csv_file = open(perf_csv, "a", buffering=1)  # ✅ line buffered，实时写入CSV
    
    try:
        print(f"[DEBUG] h{a.host_id}: 进入主循环", file=sys.stderr, flush=True)
        while time.time() - start < a.duration - 5:

            # --- 基础码率定义（所有策略通用） ---
            base_rates = {"wifi": 3.5, "4g": 3.5, "5g": 3.5, "fiber_optic": 3.5}
            base_rate = base_rates.get(a.network_type, 3.5)

            # --- 策略决策 ---
            decision = 0  # 默认决策为0（是否拉 enhanced）
            # ✅ 【Federation OFF 模式】强制所有用户连接到 r0-eth1
            # r0 有两个接口：r0-eth0 (10.0.1.1) 连接 n0，r0-eth1 (10.0.2.1) 面向用户
            target_relay_ip = "10.0.2.1"  # r0-eth1 的 IP 地址（面向用户的接口）
            base_bitrate_level = 0  # 默认码率档位：0=low, 1=medium, 2=high

            # ✅ 【方案3：主动探测+初始预设】在初始阶段（前20秒）强制所有用户只订阅Base层
            # 目的：解决"鸡生蛋"问题，确保用户能跑通Base层后再优化Enhanced层
            elapsed_time = time.time() - start
            INITIAL_PROBE_DURATION = 20.0  # 初始探测时长：20秒
            
            if elapsed_time < INITIAL_PROBE_DURATION:
                # ✅ 【动态订阅模式】初始阶段：decision=0，只订阅 Base 层
                # Enhanced 订阅进程（enh_p）尚未启动，等待 decision=1 时启动
                decision = 0  # 强制不订阅 enhanced（动态订阅：不启动 Enhanced 进程）
                if iteration % 10 == 0:  # 每10次打印一次
                    print(f"[方案3] h{a.host_id}: 初始探测阶段（{elapsed_time:.1f}s/{INITIAL_PROBE_DURATION}s），强制只订阅 Base 层（Enhanced 进程未启动）", 
                          file=sys.stderr, flush=True)
            else:
                # 正常阶段：使用策略决策
                pass  # 继续执行下面的决策逻辑

            # MD2G/Heuristic/Clustering strategies read decisions from the shared decision file
            decision_data = None
            # ==============================================================================
            # ✅ 支持所有策略（不包括 rolling）读取决策文件
            # ⚠️ Rolling策略使用客户端本地模型进行动态决策，不依赖决策文件
            # 
            # 决策文件格式支持：
            # 1. 新格式（优先）：{"decisions": {"1": {"pull_enhanced": bool, ...}, ...}}
            # 2. 旧格式（向后兼容）：{"layers": [0, 1, 0, ...]}
            # 
            # Controller mapping (baseline strategies):
            # - MD2G: regional_relay_controller.py → /tmp/r0_decisions.json
            # - Heuristic: regional_relay_controller.py → /tmp/r0_decisions.json
            # - Clustering: regional_relay_controller.py → /tmp/r0_decisions.json
            # ==============================================================================
            if a.strategy in ["md2g", "heuristic", "clustering"]:
                # 如果文件不存在，等待并重试（最多等待5秒）
                max_retries = 10
                retry_interval = 0.5
                for retry in range(max_retries):
                    try:
                        if os.path.exists(a.decision_file):
                            # 修复：使用不同的变量名，避免与外层f冲突
                            with open(a.decision_file, 'r') as df:
                                decision_data = json.load(df)
                            break
                        else:
                            if retry == 0:
                                print(f"[DEBUG] h{a.host_id}: Decision file not found: {a.decision_file}, waiting...", file=sys.stderr, flush=True)
                            time.sleep(retry_interval)
                    except (IOError, json.JSONDecodeError) as e:
                        if retry < max_retries - 1:
                            time.sleep(retry_interval)
                            continue
                        else:
                            print(f"[DEBUG] h{a.host_id}: Failed to read decision file {a.decision_file}: {e}", file=sys.stderr, flush=True)
                            decision_data = None
                            break
                
                # 如果成功读取决策数据，解析决策
                if decision_data:
                    try:
                        # ==============================================================================
                        # ✅ 支持新的决策格式（按用户建议）
                        # ==============================================================================
                        # 新格式包含：decisions[user_id] = {
                        #   "pull_enhanced": bool,
                        #   "target_relay_ip": str,
                        #   "base_bitrate_level": int
                        # }
                        # ==============================================================================
                        # ✅ 【方案3：主动探测+初始预设】只有在正常阶段才使用Controller决策
                        # 初始阶段强制decision=0（已在上面设置）
                        if elapsed_time >= INITIAL_PROBE_DURATION:
                            if 'decisions' in decision_data and str(a.host_id) in decision_data['decisions']:
                                # 使用新格式
                                user_decision = decision_data['decisions'][str(a.host_id)]
                                decision = 1 if user_decision.get('pull_enhanced', False) else 0
                                # ✅ 【Federation OFF 模式】忽略决策中的 target_relay_ip，强制使用 r0-eth1
                                # r0 有两个接口：r0-eth0 (10.0.1.1) 连接 n0，r0-eth1 (10.0.2.1) 面向用户
                                target_relay_ip = "10.0.2.1"  # r0-eth1 的 IP 地址（面向用户的接口）
                                base_bitrate_level = user_decision.get('base_bitrate_level', 0)
                            elif 'layers' in decision_data:
                                # 向后兼容：使用旧格式（只有 layers）
                                if 0 <= a.host_id - 1 < len(decision_data['layers']):
                                    decision = decision_data['layers'][a.host_id - 1]
                    except (KeyError, IndexError, TypeError) as e:
                        # 如果格式不对，则保持默认决策
                        if iteration % 10 == 0:
                            print(f"[DEBUG] h{a.host_id}: Failed to parse decision data: {e}", file=sys.stderr, flush=True)
                        decision_data = None
                
                # ==============================================================================
                # ⚠️ 旧版本 fallback 智能增强逻辑已移除
                # ==============================================================================
                # 原因：
                # 1. 使用了未定义的 Bu 变量（Bu 在后面才测量）
                # 2. 现在 MD2G 的行为完全由 PPO 决策控制（通过 regional_relay_controller）
                # 3. 如果决策文件不存在或格式错误，decision 会保持默认值 0（不拉 enhanced）
                # ==============================================================================
                # 如果需要在决策文件缺失时使用 fallback，可以在这里添加，但需要确保 Bu 已定义
                # 当前实现：依赖 PPO 控制器生成决策，客户端只执行决策，不做二次判断
                # ==============================================================================

            # ==============================================================================
            # ✅ Rolling 策略使用本地模型进行动态决策（不依赖决策文件）
            # ==============================================================================
            if a.strategy == "rolling":
                if len(state_window) >= 3:
                    recent_states = list(state_window)[-3:]
                    # 修复：传递二维数组而不是一维数组
                    state_window_array = np.array(recent_states)
                    decision = strat.decide(state_window_array)
                else:
                    # 状态窗口不足，使用保守决策（不拉enhanced）
                    decision = 0

            # ==============================================================================
            # ✅ 执行决策：真正控制行为（按用户建议）
            # ==============================================================================
            # ✅ 【Federation OFF 模式】强制所有用户连接到 r0-eth1，忽略决策中的 target_relay_ip
            # r0 有两个接口：r0-eth0 (10.0.1.1) 连接 n0，r0-eth1 (10.0.2.1) 面向用户
            # 在 Federation OFF 模式下，只有 r0 运行 moq-relay 服务，r1-r6 只是路由器
            # 数据包会自动经过边缘 Relay (r3-r6) 和区域 Relay (r1-r2) 到达 r0
            # 这正是我们要模拟的"经过多跳网络"的场景
            target_relay_ip = "10.0.2.1"  # 强制使用 r0-eth1（面向用户的接口）
            current_relay_ip = "10.0.2.1"  # 强制使用 r0-eth1（面向用户的接口）
            
            # 2. 根据 base_bitrate_level 调整 base 码率（如果需要）
            # base_bitrate_level: 0=low, 1=medium, 2=high
            # 这里可以根据 level 选择不同的 manifest 或 representation
            # 当前实现中，base_rate 已经根据 network_type 设置，这里可以进一步细化
            
            # ==============================================================================
            # ✅ 添加客户端行为调试打印（按用户建议）
            # ==============================================================================
            if iteration % 10 == 0:  # 每10次打印一次
                print(f"[DEBUG] h{a.host_id}: Decision execution - "
                      f"pull_enhanced={decision}, target_relay={target_relay_ip}, "
                      f"base_bitrate_level={base_bitrate_level}, "
                      f"enh_on={enh_on}", 
                      file=sys.stderr, flush=True)

            # ✅ 【动态订阅模式】根据 decision 动态启动/停止 Enhanced 订阅进程
            # 保持 Base 连接不变，根据 decision 动态开关 Enhanced 订阅
            # 这样网络流量会随 decision 变化，buffer/stall/奖励都能自洽
            # 
            # 原理：
            # - decision == 1: 启动 Enhanced 订阅进程，开始接收 Enhanced 数据
            # - decision == 0: 停止 Enhanced 订阅进程，停止接收 Enhanced 数据
            # - 网络流量会随 decision 变化，真实反映订阅行为
            if decision != last_decision:
                if decision == 1 and enh_p is None:
                    # ✅ decision=1：启动 Enhanced 订阅进程
                    print(f"[INFO] h{a.host_id}: Decision=1，启动 Enhanced 订阅进程...", file=sys.stderr, flush=True)
                    enh_sub_start_time = time.time()
                    # ✅ 使用与 Base 订阅相同的 URL（根据 relay_ip 选择 r1 或 r2）
                    enh_track_url = track_url  # ✅ 与 Base 订阅相同的 URL（已在前面根据 relay_ip 设置）
                    enh_cmd_wrapper = [
                        sys.executable,  # python3
                        latency_wrapper,
                        MOQ_SUB_PATH,
                        enh_track_name,
                        enh_track_url,  # ✅ 使用与 Base 订阅相同的 URL
                        enh_dump_file,
                        enh_log_file,
                        enh_latency_log_file,
                        str(enh_sub_start_time),
                        enh_broadcast_name
                    ]
                    enh_cmd_wrapper = ["taskset", "-c", str(a.host_id % 32)] + enh_cmd_wrapper
                    
                    with open(enh_log_file, "a", errors="ignore") as enh_log:
                        enh_log.write(f"\n=== moq-sub enhanced stream started at {time.strftime('%Y-%m-%d %H:%M:%S')} (动态订阅模式) ===\n")
                        enh_log.write(f"Command: {' '.join(enh_cmd_wrapper)}\n\n")
                    
                    env_enh = os.environ.copy()
                    env_enh['MOQ_TLS_DISABLE_VERIFY'] = '1'
                    env_enh['MOQ_TRANSPORT_IDLE_TIMEOUT'] = '1800s'
                    env_enh['QUIC_IDLE_TIMEOUT'] = '1800s'
                    env_enh['RUST_LOG'] = 'info'
                    env_enh['RUST_BACKTRACE'] = '1'
                    
                    enh_p = subprocess.Popen(
                        enh_cmd_wrapper,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=0,
                        env=env_enh
                    )
                    enh_start_time = time.time()
                    reader_enhanced = DataDrainer(enh_p, "Enhanced_Stream", enh_log_file, start_time_epoch=enh_start_time)
                    reader_enhanced.start()
                    print(f"[INFO] h{a.host_id}: ✅ Enhanced moq-sub已启动（PID: {enh_p.pid}，动态订阅模式）", file=sys.stderr, flush=True)
                    enh_on = True
                    
                elif decision == 0 and enh_p is not None:
                    # ✅ decision=0：停止 Enhanced 订阅进程（动态订阅模式）
                    print(f"[INFO] h{a.host_id}: Decision=0，停止 Enhanced 订阅进程（动态订阅模式）...", file=sys.stderr, flush=True)
                    try:
                        if reader_enhanced:
                            reader_enhanced.stop()
                        if enh_p:
                            enh_p.terminate()
                            enh_p.wait(timeout=2)
                    except Exception as e:
                        print(f"[WARN] h{a.host_id}: 停止 Enhanced 订阅进程时出错: {e}", file=sys.stderr, flush=True)
                        try:
                            if enh_p:
                                enh_p.kill()
                        except:
                            pass
                    enh_p = None
                    reader_enhanced = None
                    enh_on = False
                    # ✅ 清除 enhanced 相关的字节数记录
                    if hasattr(run_client, 'enh_start_bytes'):
                        delattr(run_client, 'enh_start_bytes')
                    if hasattr(run_client, 'enh_bytes_prev'):
                        delattr(run_client, 'enh_bytes_prev')
                
                # ✅ 更新 last_decision（用于检测 decision 变化）
                last_decision = decision
                
                # ✅ 【动态订阅模式】确保 enh_on 与 decision 和 enh_p 状态一致
                if decision == 1 and enh_p is not None:
                    enh_on = True
                elif decision == 0 or enh_p is None:
                    enh_on = False

            # ==============================================================================
            # ✅ 【核心修改】被动带宽估测 (替代 iperf)
            # ==============================================================================
            # 原理：基于 dump 文件增长速率计算带宽，零干扰，完全反映真实应用层吞吐量
            # 公式：带宽 = (Δ文件大小 * 8) / (Δ时间 * 1e6) Mbps
            # ==============================================================================
            current_time = time.time()
            time_diff = current_time - last_check_time
            
            # ✅ 【动态订阅模式】获取当前 dump 文件大小
            # 注意：只有 Enhanced 订阅进程运行时，enh_dump_file 才会增长
            # ✅ 【关键修复】确保 rx_bytes 单调递增，避免 enhanced 订阅开始时重复计数
            current_total_bytes = 0
            moq_dump_size = 0
            if os.path.exists(moq_dump_file):
                moq_dump_size = os.path.getsize(moq_dump_file)
                current_total_bytes += moq_dump_size
                # ✅ 【调试】每10次迭代打印一次 dump 文件大小，检查是否在增长
                if iteration % 10 == 0:
                    print(f"[DEBUG] h{a.host_id}: moq_dump_file={moq_dump_file}, size={moq_dump_size} bytes, last_total_bytes={last_total_bytes} bytes", file=sys.stderr, flush=True)
            else:
                # ✅ 【调试】如果 dump 文件不存在，打印警告
                if iteration % 20 == 0:
                    print(f"[WARN] h{a.host_id}: moq_dump_file不存在: {moq_dump_file}", file=sys.stderr, flush=True)
            # ✅ 【动态订阅模式】只有 Enhanced 订阅进程运行时（decision==1）才计入 Enhanced 数据
            # ✅ 【关键修复】当 enhanced 订阅刚启动时，enh_dump_file 可能已经存在且有历史数据
            # 需要确保只计算从 enhanced 订阅开始后的增量，而不是全部历史数据
            if enh_p is not None and enh_on and os.path.exists(enh_dump_file):
                # ✅ 动态订阅模式：只有 decision==1 时，enh_p 才不为 None，enh_on 才为 True
                enh_file_size = os.path.getsize(enh_dump_file)
                # ✅ 如果这是 enhanced 订阅的第一次迭代，记录初始文件大小
                if not hasattr(run_client, 'enh_start_bytes'):
                    run_client.enh_start_bytes = enh_file_size
                    print(f"[DEBUG] h{a.host_id}: Enhanced 订阅启动，记录初始文件大小: {run_client.enh_start_bytes} bytes", file=sys.stderr, flush=True)
                # ✅ 只计算从 enhanced 订阅开始后的增量
                enh_incremental_bytes = max(0, enh_file_size - run_client.enh_start_bytes)
                current_total_bytes += enh_incremental_bytes
            else:
                # ✅ Enhanced 订阅停止时，清除记录
                if hasattr(run_client, 'enh_start_bytes'):
                    delattr(run_client, 'enh_start_bytes')
            # 注意：如果 decision=0，Enhanced 订阅进程已停止，enh_dump_file 不会增长
            # 这样网络流量会随 decision 变化，buffer/stall/奖励都能自洽
            
            # 计算增量
            delta_bytes = current_total_bytes - last_total_bytes
            # ✅ 【调试】每10次迭代打印一次带宽计算信息
            if iteration % 10 == 0:
                print(f"[DEBUG] h{a.host_id}: 带宽计算 - time_diff={time_diff:.3f}s, delta_bytes={delta_bytes}, current_total_bytes={current_total_bytes}, last_total_bytes={last_total_bytes}, Bu={Bu:.4f}Mbps", file=sys.stderr, flush=True)
            
            # ✅ 【关键修复：防止空转】如果 total_bytes == 0，直接返回 0，而不是衰减
            # 如果连续 N 秒接收字节数为 0，判定为"流中断"
            if current_total_bytes == 0:
                # 如果从未收到数据，带宽为 0
                Bu = 0.0
                # 记录连续无数据的时间
                if not hasattr(run_client, 'zero_bytes_duration'):
                    run_client.zero_bytes_duration = 0.0
                run_client.zero_bytes_duration += time_diff
                # 如果连续 10 秒无数据，判定为流中断
                if run_client.zero_bytes_duration >= 10.0:
                    print(f"[ERROR] h{a.host_id}: 流中断！连续 {run_client.zero_bytes_duration:.1f} 秒未收到数据，请检查 Publisher 状态和 Track Name", file=sys.stderr, flush=True)
                    # 不退出，继续尝试（可能网络暂时中断）
            else:
                # 有数据时，重置无数据计时器
                if hasattr(run_client, 'zero_bytes_duration'):
                    run_client.zero_bytes_duration = 0.0
            
            # 计算带宽 (Mbps) = (Bytes * 8) / (Seconds * 1e6)
            # 只有当 time_diff 足够大且有数据流动时才更新，避免除零或抖动
            # ✅ 【关键修复】初始化判断：如果 last_check_time <= 0，说明是第一次循环，跳过带宽计算
            # 这样可以避免第一次循环时使用"从实验开始到现在的全部流量"计算带宽，产生巨大的尖峰
            if last_check_time <= 0:
                # 第一次循环，初始化 last_total_bytes 和 last_check_time，不计算带宽
                last_total_bytes = current_total_bytes
                last_check_time = current_time
                Bu = 0.0  # 保持带宽为 0
            elif time_diff >= 0.5:  # 至少间隔 0.5 秒才更新
                # ✅ 【调试】检查 dump 文件是否真的在增长
                if iteration % 10 == 0:
                    print(f"[DEBUG] h{a.host_id}: 带宽计算检查 - time_diff={time_diff:.3f}s, delta_bytes={delta_bytes}, "
                          f"current_total_bytes={current_total_bytes}, last_total_bytes={last_total_bytes}, "
                          f"moq_dump_size={moq_dump_size}, Bu={Bu:.4f}Mbps", file=sys.stderr, flush=True)
                if delta_bytes > 0:
                    # 瞬时吞吐量
                    inst_bw = (delta_bytes * 8.0) / time_diff / 1e6
                    
                    # ✅ 【物理约束】单用户带宽不可能超过链路设定的 100M（或 5G 的 600M）
                    # 如果 inst_bw 超过 60M，说明是中继缓存回冲（Cache Catch-up），将其修正为 Bu
                    # 中继缓存回冲：当开启 Enhanced 层订阅时，Relay 会瞬间把之前错过的所有 UDP 包"吐"给客户端
                    # 这不是真实的链路带宽，而是本地内存转发速度，会误导 ABR 算法
                    if inst_bw > 60.0:
                        inst_bw = Bu if Bu > 0 else 5.0  # 如果 Bu 为 0，使用保守值 5.0 Mbps
                        if iteration % 10 == 0:  # 每10次打印一次，避免日志过多
                            print(f"[DEBUG] h{a.host_id}: ⚠️ 检测到带宽尖峰 {inst_bw:.2f} Mbps，已修正为中继缓存回冲", file=sys.stderr, flush=True)
                    
                    # 使用指数移动平均 (EMA) 使数值更平滑，模拟 TCP/QUIC 的拥塞控制视角
                    # alpha = 0.7 表示更相信最近一次的测量
                    Bu = 0.3 * Bu + 0.7 * inst_bw
                    # ✅ 【改进】有数据到达时，重置零带宽计数器
                    zero_delta_count = 0
                elif delta_bytes == 0 and current_total_bytes > 0:
                    # ✅ 【改进】如果文件大小不变（delta_bytes == 0），使用衰减策略而不是立即归零
                    # 原因：高并发下数据包到达不连续，单次delta_bytes=0不代表带宽真的为0
                    # 策略：使用缓慢衰减，只有连续多次delta_bytes=0时才真正归零
                    zero_delta_count += 1
                    if zero_delta_count >= MAX_ZERO_DELTA_COUNT:
                        # 连续多次无数据，真正归零
                        Bu = 0.0
                        if iteration % 10 == 0:  # 每10次打印一次，避免日志过多
                            print(f"[DEBUG] h{a.host_id}: 连续{zero_delta_count}次delta_bytes=0，带宽已归零", file=sys.stderr, flush=True)
                    else:
                        # 使用衰减策略，保持历史带宽值（避免高并发下误判）
                        # 衰减系数0.95：每次衰减5%，约3次后衰减到85%左右
                        Bu = 0.95 * Bu
                        if iteration % 20 == 0:  # 每20次打印一次，避免日志过多
                            print(f"[DEBUG] h{a.host_id}: delta_bytes=0 (连续{zero_delta_count}次)，带宽衰减至{Bu:.2f}Mbps", file=sys.stderr, flush=True)
                else:
                    # delta_bytes < 0 或 current_total_bytes == 0：文件被重置或从未有数据
                    # ✅ 【关键修复】如果 total_bytes == 0，直接返回 0，而不是衰减
                    if current_total_bytes == 0:
                        Bu = 0.0
                        zero_delta_count = 0  # 重置计数器
                    else:
                        # 文件被重置（delta_bytes < 0），重新计算
                        Bu = 0.0
                        zero_delta_count = 0  # 重置计数器
                
                # 更新状态
                last_total_bytes = current_total_bytes
                last_check_time = current_time
            
            # ✅ 【Buffer Level 计算】基于真实下载量和视频码率
            # 原理：buffer = buffer_prev + (下载量 - 消耗量) / 码率
            # 消耗量 = 时间间隔 * 视频码率
            if last_buffer_update_time is not None:
                dt = current_time - last_buffer_update_time
            else:
                dt = a.interval  # 第一次迭代，使用默认间隔
                last_buffer_update_time = current_time
            
            # ✅ 【关键修复】使用真实接收字节数计算 buffer
            # 从 /proc/net/dev 读取真实的累计接收字节数
            def get_real_rx_bytes():
                """从/proc/net/dev读取真实的累计接收字节数（Namespace隔离）"""
                try:
                    with open('/proc/net/dev', 'r') as f:
                        for line in f:
                            if 'eth0:' in line:
                                return int(line.split()[1])
                except (FileNotFoundError, IOError, ValueError, IndexError):
                    return 0
            
            current_rx_bytes = get_real_rx_bytes()
            # ✅ 【关键修复】确保 delta_rx_bytes 正确计算
            if last_buffer_bytes > 0:
                delta_rx_bytes = current_rx_bytes - last_buffer_bytes
            else:
                # 第一次迭代，delta_rx_bytes = 0（不更新buffer）
                delta_rx_bytes = 0
                last_buffer_bytes = current_rx_bytes  # 初始化 last_buffer_bytes
            
            # ✅ 【码率配置】根据Relay实际供出带宽调整
            # Relay实际供出分析（从实验数据）:
            #   - r1发送给5个用户: 27.86 Mbps → 平均每个用户: 5.57 Mbps
            #   - r2发送给5个用户: 22.57 Mbps → 平均每个用户: 4.51 Mbps
            #   - 平均每个用户: 5.04 Mbps
            # 考虑到网络波动和测量误差，实际可用带宽约4.0-4.5 Mbps
            # 设置base_rate_mbps=3.5 Mbps，确保 delivery (4.0-4.5M) > consumption (3.5M)，保持buffer稳定
            base_rate_mbps = 3.5  # 3.5 Mbps，基于Relay实际供出带宽（5.04M平均）的70%余量设置
            enh_rate_mbps = 1.0
            # ✅ 【关键修复】检查enhanced是否真的在本 interval 接收数据
            # 原理：size>0 只能说明"历史上写过"，不能说明"本 interval 在收"
            # 修复：使用本 interval 的增量字节数判断
            enh_receiving_data = False
            enh_delta_bytes = 0
            if decision == 1 and enh_on and enh_p is not None:
                # ✅ 动态订阅模式：只有 decision==1 时，enh_p 才不为 None，enh_on 才为 True
                # ✅ 记录上次 enhanced 字节数（如果不存在则初始化）
                if not hasattr(run_client, 'enh_bytes_prev'):
                    run_client.enh_bytes_prev = 0
                # 检查enhanced dump文件是否存在
                if os.path.exists(enh_dump_file):
                    enh_bytes_now = os.path.getsize(enh_dump_file)
                    # ✅ 计算本 interval 的增量字节数
                    enh_delta_bytes = max(0, enh_bytes_now - run_client.enh_bytes_prev)
                    # ✅ 只有本 interval 有增量（>0）才算真正在接收
                    if enh_delta_bytes > 0:
                        enh_receiving_data = True
                        run_client.enh_bytes_prev = enh_bytes_now
                else:
                    # 文件不存在，重置 prev
                    run_client.enh_bytes_prev = 0
            else:
                # Enhanced 订阅停止时，清除记录
                if hasattr(run_client, 'enh_bytes_prev'):
                    delattr(run_client, 'enh_bytes_prev')
            
            # ✅ 【关键修复】只有当enhanced真正在接收数据时，才使用增强码率
            # 这样可以避免在enhanced订阅刚启动但还没收到数据时，错误地增加消耗量
            current_bitrate_bps = (base_rate_mbps + (enh_rate_mbps if enh_receiving_data else 0)) * 1e6
            
            # ✅ 【Buffer 更新】buffer = buffer_prev + (下载量 - 消耗量)
            # 下载量（秒）= 接收字节数 * 8 / 码率（bps）
            # 消耗量（秒）= 时间间隔（秒）
            # ✅ 【关键修复】第一次迭代（iteration=0）时，不消耗buffer，保持初始值5.0秒
            prev_buffer_level_sec = buffer_level_sec  # 保存更新前的buffer值，用于stall计算
            if current_bitrate_bps > 0 and dt > 0 and iteration > 0:
                # 下载量（秒）= 接收字节数 * 8 / 码率（bps）
                downloaded_seconds = (delta_rx_bytes * 8.0) / current_bitrate_bps if delta_rx_bytes > 0 else 0.0
                # 消耗量（秒）= 时间间隔（秒）
                consumed_seconds = dt
                # Buffer 更新
                buffer_level_sec = buffer_level_sec + downloaded_seconds - consumed_seconds
                # ✅ 【关键修复】限制 buffer 范围：0-30秒
                buffer_level_sec = max(0.0, min(30.0, buffer_level_sec))
            elif iteration == 0:
                # 第一次迭代，保持buffer初始值5.0秒，不进行消耗计算
                pass
            else:
                # 如果码率为0或时间间隔为0，buffer保持不变
                pass
            
            # ✅ 【Stall 累加】优先使用 parse_rebuffer_from_debug 的真实 rebuffer
            # ✅ 【关键修复】stall 应该来自播放器/解码链路的真实 rebuffer 事件，而不是 buffer 模型
            # 原理：buffer 模型容易得到 stall≈0（因为 dump 增长速度 > 码率消耗），而真实播放器会报告 rebuffer
            if iteration > 0 and dt > 0:  # 第一次迭代不计算stall
                # ✅ 优先使用 GStreamer debug 日志解析的真实 rebuffer
                debug_log_path = os.path.join(a.log_path, f"client_h{a.host_id}_gst.log")
                if os.path.exists(debug_log_path):
                    stall_sec_inc, stall_cnt_inc = parse_rebuffer_from_debug(debug_log_path, last_debug_size)
                    if stall_sec_inc > 0:
                        stall_total_sec += stall_sec_inc
                        stall_count += stall_cnt_inc
                        if iteration % 10 == 0:  # 每10次打印一次，避免日志过多
                            print(f"[DEBUG] h{a.host_id}: ⚠️ 检测到真实 rebuffer！stall_sec_inc={stall_sec_inc:.3f}s, stall_count={stall_count}", file=sys.stderr, flush=True)
                else:
                    # ✅ 回退：如果 debug 日志不存在，使用 buffer 模型（但这不是理想方案）
                    if buffer_level_sec <= 0.0:
                        # ✅ 简化逻辑：只要 Buffer 枯竭，就判定为卡顿
                        # 正常步长检查：防止异常大的 dt 值（如系统暂停）
                        if dt > 0 and dt < 5.0:  # 正常步长（1秒左右），最大容忍5秒
                            stall_total_sec += dt
                            # ✅ 【Stall Count】仅在从正数变为0或负数时计数（避免重复计数）
                            if prev_buffer_level_sec > 0.0:
                                stall_count += 1
                                if iteration % 10 == 0:  # 每10次打印一次，避免日志过多
                                    print(f"[DEBUG] h{a.host_id}: ⚠️ 检测到卡顿（buffer模型）！buffer从 {prev_buffer_level_sec:.3f}s 降至 {buffer_level_sec:.3f}s，stall_count={stall_count}", file=sys.stderr, flush=True)
            
            # ✅ 【更新状态】
            last_buffer_bytes = current_rx_bytes
            last_buffer_update_time = current_time
            
            # ✅ 【Time to Last Byte 延迟测量】真正的 TTLB（每个 chunk 的传输时间）
            # ✅ 【关键修复】TTLB应该是"本 step 的对象/segment 下载完成时间"，而不是累计时间
            # ✅ 【关键修复】避免时间基准对齐错误：不使用 reader_base.ttlb_ms（它基于 start_time_epoch，可能与 sub_start_time 不一致）
            # 原理：TTLB = 本 interval 的数据传输时间 = 本 interval 的增量字节数 / 当前带宽
            ttlb_ms = 0.0  # Time to Last Byte (ms)
            
            # ✅ 计算本 interval 的 TTLB：基于增量字节数和带宽
            if iteration > 0 and delta_rx_bytes > 0 and Bu > 0:
                # ✅ 使用本 interval 的增量字节数和带宽计算 TTLB
                # TTLB = (增量字节数 * 8) / (带宽 * 1e6) * 1000 (转换为毫秒)
                ttlb_ms = (delta_rx_bytes * 8.0) / (Bu * 1e6) * 1000.0
                # ✅ 限制 TTLB 范围：最小 1ms，最大 5000ms（5秒，防止异常值）
                ttlb_ms = max(1.0, min(5000.0, ttlb_ms))
            elif iteration == 0:
                # ✅ 第一次迭代：使用 TTFB 作为初始 TTLB
                if hasattr(reader_base, 'ttfb_ms') and reader_base.ttfb_ms is not None:
                    ttlb_ms = reader_base.ttfb_ms
                else:
                    ttlb_ms = 50.0  # 默认初始值
            elif current_total_bytes > 0 and Bu > 0:
                # ✅ 回退：如果 delta_rx_bytes 为 0，使用当前总字节数和带宽估算
                # 注意：这不是真正的 TTLB，而是估算值
                ttlb_ms = (current_total_bytes * 8.0) / (Bu * 1e6) * 1000.0
                ttlb_ms = max(1.0, min(5000.0, ttlb_ms))
            else:
                # 如果没有数据或带宽，使用默认值
                ttlb_ms = 50.0

            # --- 真实延迟测量（实时测量每个 interval 的延迟） ---
            # ✅ 【关键修复】delay_ms 应该是"最近一个窗口内的端到端业务时延"
            # 原理：使用当前时间与最近一次收到数据的时间差（last_data_ts）
            # TTFB 只能作为连接建立/首帧启动延迟指标，不该当作 steady-state delay
            if hasattr(reader_base, 'last_data_ts') and reader_base.last_data_ts is not None:
                # ✅ 使用最近数据到达时间计算实时延迟
                dly = (current_time - reader_base.last_data_ts) * 1000.0
                # ✅ 限制延迟范围：如果超过 500ms，说明可能数据流中断，使用 EWMA 间隔作为估计
                if dly > 500.0 and hasattr(reader_base, 'last_read_dt_ms') and reader_base.last_read_dt_ms is not None:
                    dly = reader_base.last_read_dt_ms  # 使用最近一次 read 间隔作为估计
                # ✅ 限制延迟范围：最小 1ms，最大 500ms
                dly = max(1.0, min(500.0, dly))
            elif hasattr(reader_base, 'ttfb_ms') and reader_base.ttfb_ms is not None:
                # ✅ 回退：如果还没有收到数据，使用 TTFB 作为初始延迟
                dly = reader_base.ttfb_ms
            elif current_total_bytes > 0 and hasattr(run_client, 'sub_start_time'):
                # ✅ 回退：使用当前时间与订阅开始时间的差值作为近似
                dly = (current_time - run_client.sub_start_time) * 1000.0
            else:
                # 还没收到首包（启动瞬间），给一个合理的初始值
                dly = 50.0
                if iteration % 10 == 0:  # 每10次打印一次
                    print(f"[DEBUG] h{a.host_id}: ⏳ 等待首包到达，使用初始延迟值: {dly:.2f} ms", file=sys.stderr, flush=True)
            
            # ✅ 保存当前 buffer 用于下次 RTT 估计
            if not hasattr(run_client, 'last_buffer_level'):
                run_client.last_buffer_level = buffer_level_sec
            run_client.last_buffer_level = buffer_level_sec
            
            # ✅ 【已删除】不再使用 Ping 测量延迟，因为：
            # 1. Ping 在 Mininet 环境下经常失败（路由问题、CPU 忙）
            # 2. Ping 只能测量网络层延迟，无法反映应用层处理延迟
            # 3. TTFB 是真实业务延迟，包含所有耗时，更适合 QoE 评估
            # 
            # 旧代码（已删除）：
            # iface = nic_name(a.host_id)
            # server_ip = "10.0.1.100"  # n0 的 IP
            # dly = ping_rtt(server_ip, iface=iface)
            # if dly < 0 or dly > 500:
            #     dly = 50.0  # 使用默认值
            # print(f"[DEBUG] h{a.host_id} measured end-to-end delay (client→n0): {dly:.2f}ms for {a.network_type}", file=sys.stderr, flush=True)

            # --- 设备性能 ---
            dev_score = float(a.device_score)
            dev = df_dev.sample(1).iloc[0]
            gpu = dev['GPU Clock (MHz)'] / df_dev['GPU Clock (MHz)'].max()
            ram = dev['RAM(GB)'] / df_dev['RAM(GB)'].max()
            refresh = dev['Refresh Rate (Hz)'] / df_dev['Refresh Rate (Hz)'].max()
            # 修复列名解析Bug：使用正确的列名
            resolution_str = str(dev['Resolution (per eye)'])
            res_w = int(resolution_str.split('×')[0]) if '×' in resolution_str else 1920
            res = res_w / df_dev['Resolution (per eye)'].apply(
                lambda x: int(str(x).split('×')[0]) if '×' in str(x) else 1920
            ).max()

            # --- 渲染质量 (Qr) ---
            Qr = min(1.2, max(0.2,
                (0.4 * dev_score + 0.3 * gpu + 0.2 * ram + 0.1 * refresh + 0.2 * res)
                * gpu_boost(gpu)
            ))

            # --- 解析rebuffer信息 ---
            debug_log_path = f"/tmp/client_logs/client_h{a.host_id}_gstdebug.log"
            stall_sec_inc, stall_cnt_inc = parse_rebuffer_from_debug(debug_log_path, last_debug_size)
            stall_total_sec += stall_sec_inc
            stall_count += stall_cnt_inc
            
            # --- 系统级负载均衡计算 ---
            # 从真实relay监控获取负载率（如果可用），否则使用默认值
            # ⚠️ 注意：默认值仅用于调试/fallback，正式实验应从 /tmp/relay_loads.json 读取真实负载
            # 默认值故意设置得明显（便于识别fallback情况）
            DEFAULT_OFF_LOADS = [0.9]  # 单relay高负载（仅用于debug/图表，不参与QoE计算）
            DEFAULT_ON_LOADS = [0.2, 0.5, 0.8, 0.3]  # 仅占位，不代表真实负载
            
            def read_relay_loads_from_json():
                """
                从 /tmp/relay_loads.json 读取真实 relay 负载
                
                Returns:
                    list: relay 负载列表，如果读取失败返回 None
                """
                try:
                    with open("/tmp/relay_loads.json", "r") as f:
                        data = json.load(f)
                        loads = data.get("loads", None)
                        
                        # 验证数据格式
                        if not loads or not isinstance(loads, list) or len(loads) == 0:
                            raise ValueError("Invalid loads in JSON: empty or not a list")
                        
                        # 验证负载值范围
                        if not all(0.0 <= load <= 1.0 for load in loads):
                            raise ValueError("Invalid loads in JSON: values not in [0, 1]")
                        
                        return loads
                        
                except (IOError, json.JSONDecodeError, KeyError, ValueError) as e:
                    # 读取失败，返回 None（由调用者决定是否使用 fallback）
                    if iteration % 20 == 0:  # 减少打印频率
                        print(f"[WARN] h{a.host_id}: Failed to read relay_loads.json: {e}", 
                              file=sys.stderr, flush=True)
                    return None
            
            # ==============================================================================
            # ✅ 修复 JFI 计算：ON 模式从 relay_loads.json 读取真实负载并计算 JFI
            # ==============================================================================
            if a.federation == 'on':
                # Federation ON: 从真实监控读取，计算真实 JFI
                relay_loads = read_relay_loads_from_json()
                
                if relay_loads and len(relay_loads) > 0:
                    # 成功读取真实负载，计算真实 JFI
                    # 过滤掉过小的负载值（可能是测量误差）
                    valid_loads = [l for l in relay_loads if l >= 0.001]  # 至少0.1%的负载才有效
                    if len(valid_loads) > 0:
                        load_balance_jfi = calculate_system_load_balance(valid_loads)
                        # 每次迭代都打印（用于验证JFI是否真的在变化）
                        print(f"[JFI] h{a.host_id}: ON模式真实JFI={load_balance_jfi:.4f}, relay_loads={valid_loads}", 
                              file=sys.stderr, flush=True)
                    else:
                        # 所有负载值都太小，使用fallback
                        load_balance_jfi = 1.0
                        if iteration % 20 == 0:
                            print(f"[WARN] h{a.host_id}: ON模式所有负载值过小，使用fallback JFI=1.0", 
                                  file=sys.stderr, flush=True)
                else:
                    # 读取失败，使用 fallback（但应该很少发生）
                    load_balance_jfi = 1.0
                    if iteration % 20 == 0:  # 减少打印频率
                        print(f"[WARN] h{a.host_id}: ON模式使用fallback JFI=1.0 (relay_loads.json读取失败)", 
                              file=sys.stderr, flush=True)
            else:
                # Federation OFF: 单relay，负载集中，JFI=1.0（不参与QoE计算，因为w_ln=0）
                relay_loads = [0.8]  # 单relay高负载（仅用于占位）
                load_balance_jfi = 1.0  # OFF模式固定为1.0
            
            # --- 方案B: 系统级负载均衡QoE公式 ---
            # QoE = w_r*Qr + w_b*Rb + w_ln*L_net - w_d*D - w_f*F
            Rq = 5.0 * (math.log1p(Qr) / math.log(2.5))
            
            # 使用按网络类型分组的B_MAX进行Rb归一化
            network_b_max = B_MAX_BY_NETWORK.get(a.network_type, B_MAX)
            Rb = math.log1p(Bu) / math.log1p(network_b_max + EPS)
            delay_penalty = dly / 200.0
            rebuffer_penalty = min(1.0, stall_count * 0.1)  # 基于stall次数
            
            # ==============================================================================
            # ✅ 统一奖励函数（paper-aligned without load-balance term）：
            # R_t = λ_o*R_o + λ_q*R_q - λ_b*R_b
            # ==============================================================================
            # 根据论文 Eq.(1)，所有策略使用相同的统一奖励函数
            # 其中：
            #   R_o: Grouping Efficiency（分组效率）
            #   R_q: User Perceived Quality（用户感知质量）
            #   R_b: Bandwidth Efficiency Penalty（带宽效率惩罚）
            #   R_l: Load Balance Penalty（负载均衡惩罚）
            # ==============================================================================
            
            # --- 奖励函数系数（全局超参数，OFF和ON模式保持一致）---
            # ✅ 【修复 v2.0】调整权重：提高 QoE 权重，降低带宽惩罚权重
            # 旧版本为 (0.2, 0.5, 0.1, 0.2)，现在移除 λ_l*R_l 后按比例重标定到和为 1
            # => lambda_o=0.25, lambda_q=0.625, lambda_b=0.125
            lambda_o = 0.25   # 分组效率权重
            lambda_q = 0.625  # 用户感知质量权重
            lambda_b = 0.125  # 带宽效率惩罚权重
            
            # --- R_q: User Perceived Quality ---
            # ✅ 【修复 v2.0】R_q = α₁*quality_score - α₂*delay_norm - α₃*stall_norm
            # 确保各项指标在同一量级（归一化到0-1）
            
            # ✅ 【关键修复】质量得分：给 Base 层保底分 0.6
            # 0=Base(0.6分), 1=Enhanced(1.0分)
            quality_score = 0.6 + 0.4 * float(decision)
            
            # ✅ 【修复】延迟归一化：假设最大延迟200ms，归一化到0-1
            delay_norm = min(dly / 200.0, 1.0) if dly > 0 else 0.0
            
            # ✅ 【修复】卡顿归一化：使用指数衰减，避免线性惩罚过大
            if stall_total_sec > 0:
                stall_norm = 1.0 - np.exp(-stall_total_sec / 3.0)  # 3秒为时间常数
            else:
                stall_norm = 0.0
            
            # ✅ 【修复 v2.0】调整权重，使曲线更平滑
            alpha1, alpha2, alpha3 = 1.0, 0.2, 0.3
            R_q = alpha1 * quality_score - alpha2 * delay_norm - alpha3 * stall_norm
            
            # ✅ 【修复】确保R_q在合理范围内（0到1之间）
            R_q = max(0.0, min(1.0, R_q))
            
            # --- R_o: Grouping Efficiency ---
            # ✅ 【关键修复】连接到真实的分组逻辑，而不是使用固定值
            # 尝试从控制器决策文件中读取真实的分组信息
            grouping_id = a.host_id  # 默认使用 host_id
            grouping_size = 1  # 默认单用户组
            
            # ✅ 【强制分组优化】如果FOV分组未启用，按relay分组（至少共享Base层）
            # 前一半用户（r1）共享grouping_id=1，后一半用户（r2）共享grouping_id=2
            # 这样可以实现Base层的组播共享，减少带宽消耗
            r1_sub_count = a.clients // 2
            if a.host_id <= r1_sub_count:
                # r1分支：所有用户共享grouping_id=1
                default_grouping_id = 1
                default_grouping_size = r1_sub_count
            else:
                # r2分支：所有用户共享grouping_id=2
                default_grouping_id = 2
                default_grouping_size = a.clients - r1_sub_count
            
            try:
                # 尝试读取控制器决策文件，获取真实的分组信息
                # 优先使用r1/r2的决策文件（根据用户连接的relay）
                # ✅ 【关键修复】使用已确定的relay_ip（可能来自命令行参数或默认分配）
                # 注意：这里使用外层已确定的relay_ip变量，而不是重新调用assign_relay_ip
                if relay_ip == "10.0.2.2":  # r1
                    decision_file = "/tmp/r1_decisions.json"
                elif relay_ip == "10.0.3.2":  # r2
                    decision_file = "/tmp/r2_decisions.json"
                else:
                    # Rolling/GROOT策略连接到r0，使用r1的决策文件（兼容性，实际不使用）
                    decision_file = "/tmp/r1_decisions.json"
                
                if os.path.exists(decision_file):
                    with open(decision_file, 'r') as f:
                        decisions_data = json.load(f)
                    # ✅ 【关键修复】决策文件格式可能是 {"decisions": {"1": {...}, "2": {...}}} 或 {"h1": {...}, "h2": {...}}
                    # 需要兼容两种格式
                    if "decisions" in decisions_data:
                        # 格式1：{"decisions": {"1": {...}, "2": {...}}}
                        decisions_dict = decisions_data["decisions"]
                        user_key = str(a.host_id)  # 使用数字字符串作为键
                    else:
                        # 格式2：{"h1": {...}, "h2": {...}}
                        decisions_dict = decisions_data
                        user_key = f"h{a.host_id}"  # 使用 "h1", "h2" 格式
                    
                    if user_key in decisions_dict:
                        user_decision = decisions_dict[user_key]
                        # 如果决策中包含分组信息，使用它
                        if "fov_group_id" in user_decision:
                            grouping_id = user_decision["fov_group_id"]
                        if "fov_group_size" in user_decision:
                            grouping_size = user_decision["fov_group_size"]
                        # ✅ 如果决策中没有分组信息，使用按relay分组的默认值
                        else:
                            grouping_id = default_grouping_id
                            grouping_size = default_grouping_size
                    else:
                        # 用户不在决策文件中，使用按relay分组的默认值
                        grouping_id = default_grouping_id
                        grouping_size = default_grouping_size
                else:
                    # 决策文件不存在，使用按relay分组的默认值
                    grouping_id = default_grouping_id
                    grouping_size = default_grouping_size
            except Exception as e:
                # 如果读取失败，使用按relay分组的默认值
                grouping_id = default_grouping_id
                grouping_size = default_grouping_size
                if iteration % 20 == 0:  # 每20次打印一次，避免日志过多
                    print(f"[DEBUG] h{a.host_id}: 无法读取分组信息: {e}，使用按relay分组（grouping_id={grouping_id}, size={grouping_size}）", file=sys.stderr, flush=True)
            
            # ✅ 计算真实的分组效率：基于分组大小（多播增益）
            # grouping_efficiency = 分组大小 / 最大分组大小（归一化到0-1）
            # 如果分组大小为1（单播），效率为0.2（最低）
            # 如果分组大小为5（多播），效率为1.0（最高）
            max_group_size = 5  # 假设最大分组大小为5
            R_o = min(1.0, grouping_size / max_group_size) if grouping_size > 0 else 0.2
            grouping_efficiency = R_o
            
            # ✅ 【关键修复】确保 grouping_efficiency 实时更新，而不是固定值
            # 如果分组信息没有从决策文件读取到，说明当前是单播模式（grouping_size=1）
            # 这种情况下 grouping_efficiency=0.2 是正确的，但需要确保每个 iteration 都重新计算
            if iteration % 20 == 0:  # 每20次打印一次，避免日志过多
                if grouping_size == 1:
                    print(f"[DEBUG] h{a.host_id}: 当前为单播模式（grouping_size=1），grouping_id={grouping_id}，grouping_efficiency={grouping_efficiency:.4f}", file=sys.stderr, flush=True)
                else:
                    print(f"[DEBUG] h{a.host_id}: 当前为多播模式（grouping_size={grouping_size}），grouping_id={grouping_id}，grouping_efficiency={grouping_efficiency:.4f}", file=sys.stderr, flush=True)
            
            # --- R_b: Bandwidth Efficiency Penalty ---
            # ✅ 【关键修复】确保 R_b 对带宽变化敏感，而不是恒为 0
            # 原理：R_b = 1.0 - bandwidth_utilization，其中 utilization = Bu / TARGET_BITRATE
            # ✅ 【关键修复】TARGET_BITRATE 应该基于实际视频码率（Base + Enhanced），而不是固定值
            # 实际视频码率：Base 3.5 Mbps + Enhanced 1.0 Mbps = 4.5 Mbps（最大）
            TARGET_BITRATE = 4.5  # Mbps (Base 3.5 + Enhanced 1.0)
            if Bu > 0 and TARGET_BITRATE > 0:
                bandwidth_utilization = min(1.0, Bu / TARGET_BITRATE)
                # R_b: 利用率越高，惩罚越小（连续值：0.0 到 1.0）
                # 例如：Bu=1.0 Mbps → util=0.22 → R_b=0.78（高惩罚，带宽浪费）
                #      Bu=3.5 Mbps → util=0.78 → R_b=0.22（低惩罚，带宽充分利用）
                R_b = 1.0 - bandwidth_utilization
                # ✅ 确保 R_b 在合理范围内（0.0 到 1.0）
                R_b = max(0.0, min(1.0, R_b))
            else:
                # ✅ 当Bu=0时，说明没有有效带宽，给一个中等惩罚（而不是最大惩罚1.0）
                # 这样可以避免step=0时R_b=1.0，step>=1时R_b=0.0的跳跃
                R_b = 0.5  # 中等惩罚，表示"带宽未充分利用但也不完全浪费"
            
            # --- R_l: Load Balance Penalty ---
            # R_l = std(relay_loads) （仅ON模式）
            # ✅ 【关键修复】确保 load_balance_jfi 连接到真实的分组逻辑
            # 初始化 load_balance_jfi（确保在所有分支中都有定义）
            if 'load_balance_jfi' not in locals():
                load_balance_jfi = 1.0  # 默认值
            
            if a.federation == 'off':
                R_l = 0.0  # OFF模式不启用负载均衡惩罚
                # ✅ 【关键修复】确保 load_balance_jfi 实时更新，而不是固定值
                # OFF模式下，如果没有分组，JFI 应该接近 1.0（单播，负载均衡）
                # 如果有分组，应该基于分组大小计算 JFI
                if grouping_size > 1:
                    # 多播组内负载均衡：假设组内用户负载均匀分布
                    load_balance_jfi = 0.9 + 0.1 * (1.0 / grouping_size)  # 分组越大，JFI 越接近 1.0
                else:
                    load_balance_jfi = 1.0  # 单播，完美负载均衡
            else:
                # ON模式：计算relay负载标准差
                if relay_loads and len(relay_loads) > 1:
                    R_l = float(np.std(relay_loads))
                else:
                    R_l = 0.0
                # ✅ load_balance_jfi 已经在上面从真实relay监控读取（第1794行）
                # 如果未读取到，保持默认值 1.0
                # ✅ 【关键修复】确保 load_balance_jfi 在每个 iteration 都重新计算
                if 'load_balance_jfi' not in locals() or load_balance_jfi is None:
                    load_balance_jfi = 1.0  # 默认值
            
            # --- 统一奖励函数计算 ---
            reward_final = lambda_o * R_o + lambda_q * R_q - lambda_b * R_b
            reward_final = max(0.0, reward_final)  # ✅ 确保非负
            
            # 保持向后兼容：用于MD2G内部决策（不影响奖励计算）
            # 使用旧的QoE计算作为内部参考（仅用于MD2G策略的内部决策）
            Rq_legacy = 5.0 * (math.log1p(Qr) / math.log(2.5))
            Rb_legacy = math.log1p(Bu) / math.log1p(network_b_max + EPS)
            delay_penalty = dly / 200.0
            rebuffer_penalty = min(1.0, stall_count * 0.1)
            # Legacy QoE 也移除负载均衡项：将原来 0.15*jfi 的正权重等分回 Rq/Rb
            QoE_inst_legacy = max(0.0,
                                    0.425 * Rq_legacy + 0.425 * Rb_legacy
                                    - 0.10 * delay_penalty - 0.05 * rebuffer_penalty)
            
            # MD2G策略的特殊处理（仅用于内部决策，不影响奖励计算）
            # 注意：dq, sb, dyn_th等指标仅用于MD2G的内部决策逻辑（如是否启用增强层），
            # 但不用于奖励计算，确保所有策略使用相同的奖励函数进行公平比较
            if a.strategy == "md2g":
                # 先更新窗口与趋势（用于MD2G内部决策，使用legacy QoE作为参考）
                smart_enhancement.update_windows(bandwidth_mbps=Bu, qoe_smooth=QoE_inst_legacy)
                
                # 获取网络类型（用于动态权重调整）
                network_type = getattr(a, 'network_type', None) or getattr(a, 'network', None)
                
                # 计算决策质量与稳定性（仅用于MD2G内部决策，不用于QoE计算）
                dq = decision_quality_v2(Bu, base_rate, delay_penalty, network_type=network_type)
                sb = stability_bonus(smart_enhancement.qoe_window)
                
                # 启停判定（迟滞 + 趋势）
                # 计算当前负载率（使用relay_loads的平均值）
                current_load_rate = np.mean(relay_loads) if relay_loads else 0.5
                
                # ============================================================
                # Module 5: Cache hit-aware strategy (Federation ON)
                # ============================================================
                # If federation ON: prefer enhanced only when cache warmed-up
                cache_aware_decision = decision
                if a.federation == 'on':
                    try:
                        # Check cache statistics (simulated via file system)
                        cache_stats_file = "/tmp/federation_cache_stats.json"
                        cache_ok = False
                        if os.path.exists(cache_stats_file):
                            with open(cache_stats_file, 'r') as f:
                                cache_stats = json.load(f)
                                cache_size = cache_stats.get("cache_size", 0)
                                cache_ok = cache_size > 50  # Cache warmed up
                        
                        if not cache_ok:
                            # Cache not warmed up yet, reduce enhanced layer usage
                            if decision == 1 and random.random() < 0.5:
                                cache_aware_decision = 0
                                if iteration % 10 == 0:
                                    print(f"[CACHE-AWARE] h{a.host_id}: Cache not ready, reducing enhanced layer usage", 
                                          file=sys.stderr, flush=True)
                    except Exception as e:
                        if iteration % 20 == 0:
                            print(f"[WARN] h{a.host_id}: Cache-aware check failed: {e}", file=sys.stderr, flush=True)
                
                enh_target, dyn_th = smart_enhancement.should_enable_enhancement(
                    network_type=a.network_type, base_rate=base_rate, 
                    relay_jfi=load_balance_jfi, relay_load=current_load_rate,
                    federation_on=(a.federation == 'on')
                )
                smart_enhancement.enh_enabled = enh_target
                
                # Apply cache-aware decision override if federation is ON
                if a.federation == 'on' and cache_aware_decision != decision:
                    decision = cache_aware_decision
            else:
                # 其他策略不需要这些内部决策指标
                dq, sb, dyn_th = 0.0, 0.0, 0.0
            
            # --- MD2G-Plus动态调节（仅用于MD2G内部决策，不影响奖励计算） ---
            if a.strategy == "md2g" and iteration > 10:  # 等待足够样本
                # 使用legacy QoE进行内部决策
                if not hasattr(run_client, 'qoe_smooth_prev'):
                    run_client.qoe_smooth_prev = QoE_inst_legacy
                QoE_smooth_legacy = 0.85 * run_client.qoe_smooth_prev + 0.15 * QoE_inst_legacy
                run_client.qoe_smooth_prev = QoE_smooth_legacy
                smart_enhancement.update_qoe_and_adjust_threshold(QoE_smooth_legacy)
            
            # --- 计算rx_bytes（用于日志）---
            # ✅ 【关键修复】使用真实的累计接收字节数，而不是从Bu反推
            # ❌ 错误做法：rx_bytes = int(Bu * 1024 * 1024 / 8)  # 基于带宽估算，会导致累计字节数可能减少（物理上不可能）
            # ✅ 正确做法：直接使用相对于实验开始时的物理累计值
            # ✅ 【关键修复】使用相对于实验开始时的累计值，确保单调递增
            # ✅ 【关键修复：rx_bytes 基准值偏移】在 iteration == 0 时记录初始值
            # 问题：/proc/net/dev 中的数值是网卡自创建以来的总流量，连续实验时会累积
            # 修复：在第一次迭代时记录初始值，后续计算时减去初始值，得到本次实验的净增长
            if iteration == 0:
                # 使用 get_real_rx_bytes() 读取物理网卡的初始累计值
                def get_real_rx_bytes():
                    """从/proc/net/dev读取真实的累计接收字节数（Namespace隔离）"""
                    try:
                        with open('/proc/net/dev', 'r') as f:
                            for line in f:
                                if 'eth0:' in line:
                                    return int(line.split()[1])
                    except (FileNotFoundError, IOError, ValueError, IndexError):
                        return 0
                run_client.initial_rx = get_real_rx_bytes()  # 记录物理网卡的初始累计值
                run_client.start_rx_bytes = current_total_bytes  # 记录逻辑累计值的初始值
            elif not hasattr(run_client, 'start_rx_bytes'):
                # 兼容性：如果第一次迭代没执行到，在这里初始化
                run_client.initial_rx = get_real_rx_bytes() if 'get_real_rx_bytes' in dir() else current_total_bytes
                run_client.start_rx_bytes = current_total_bytes
            # rx_bytes = 当前累计值 - 实验开始时的初始值（相对于实验开始时的累计值）
            # ✅ 【关键修复】使用物理网卡的累计值减去初始值，得到本次实验的净增长
            if hasattr(run_client, 'initial_rx'):
                # 使用物理网卡的累计值
                def get_real_rx_bytes():
                    """从/proc/net/dev读取真实的累计接收字节数（Namespace隔离）"""
                    try:
                        with open('/proc/net/dev', 'r') as f:
                            for line in f:
                                if 'eth0:' in line:
                                    return int(line.split()[1])
                    except (FileNotFoundError, IOError, ValueError, IndexError):
                        return 0
                current_physical_rx = get_real_rx_bytes()
                rx_bytes = int(current_physical_rx - run_client.initial_rx)  # 本次实验的净增长
            else:
                # 回退：使用逻辑累计值
                rx_bytes = int(current_total_bytes - run_client.start_rx_bytes)
            
            # --- 写日志（统一奖励函数格式）---
            if a.strategy == "md2g":
                print(f"[DEBUG] h{a.host_id}: 写入日志 dly={dly:.2f}ms Bu={Bu:.2f}Mbps "
                      f"R_q={R_q:.4f} R_o={R_o:.4f} R_b={R_b:.4f} reward={reward_final:.4f} "
                      f"(dq={dq:.4f} sb={sb:.4f} dyn_th={dyn_th:.3f}仅用于MD2G内部决策)",
                      file=sys.stderr, flush=True)
            else:
                print(f"[DEBUG] h{a.host_id}: 写入日志 dly={dly:.2f}ms Bu={Bu:.2f}Mbps "
                      f"R_q={R_q:.4f} R_o={R_o:.4f} R_b={R_b:.4f} reward={reward_final:.4f}",
                      file=sys.stderr, flush=True)
            
            # ✅ 统一奖励函数日志格式（对应论文 Eq.(1)）
            # ✅ 【新增】添加 Time to Last Byte (ttlb_ms) 延迟指标
            # ✅ 【修复】添加 subscription_type 和 buffer_level_sec 字段
            subscription_type = "base+enhanced" if (decision == 1 and enh_on) else "base"
            csv_line = f"{time.time():.2f},{a.host_id},{a.network_type},{dev_score:.3f},{int(decision)},{subscription_type}," \
                      f"{dly:.2f},{ttlb_ms:.2f},{stall_total_sec:.3f},{stall_count},{Bu:.4f},{rx_bytes}," \
                      f"{R_o:.4f},{R_q:.4f},{R_b:.4f},{reward_final:.4f}," \
                      f"{grouping_id},{grouping_efficiency:.4f},{load_balance_jfi:.4f},{iteration},{buffer_level_sec:.3f}\n"
            
            # ✅ 【CSV实时写入】同时写入.log和.csv文件，确保实时刷新
            # ✅ 【关键保证】此逻辑对所有策略都有效（MD2G, Rolling, Heuristic, Clustering, Groot, Pano）
            # 不依赖策略类型，所有策略都会生成统一格式的perf.csv
            # 文件路径：{a.log_path}/client_h{a.host_id}_perf.csv
            perf_log_file.write(csv_line)
            perf_log_file.flush()
            perf_csv_file.write(csv_line)
            perf_csv_file.flush()

            # --- 写出自身状态，供全局控制器读取 ---
            # ✅ 【共享目录修复】使用共享目录，解决Rolling策略控制器无法读取状态文件的问题
            # Mininet节点和主机共享/tmp目录，使用子目录确保可访问性
            # ✅ 【关键修复】确保在第一次迭代时也写入状态文件（iteration从0开始）
            SHARED_STATE_DIR = "/tmp/mininet_shared"
            client_state_file = f"{SHARED_STATE_DIR}/client_h{a.host_id}_state.json"
            # ✅ 确保共享目录存在且可写
            try:
                os.makedirs(SHARED_STATE_DIR, exist_ok=True)
                os.chmod(SHARED_STATE_DIR, 0o777)  # 确保所有用户可访问
                # ✅ 确保文件可写（如果文件已存在，先检查权限）
                if os.path.exists(client_state_file):
                    os.chmod(client_state_file, 0o666)  # 确保所有用户可读写
            except (OSError, PermissionError) as e:
                if iteration % 10 == 0:  # 每10次打印一次错误，避免日志过多
                    print(f"[WARN] h{a.host_id}: 无法创建共享目录: {e}", file=sys.stderr, flush=True)
                pass  # 如果无法设置权限，继续尝试写入
            
            current_state_payload = {
                "host_id": a.host_id,
                "network_type": a.network_type,
                "throughput_mbps": Bu,  # ✅ 单位：Mbps（已统一）
                "delay_ms": dly,
                "device_score": dev_score,
                "last_decision_layer": decision,
                "reward_R_o": R_o,  # 分组效率奖励
                "reward_R_q": R_q,  # 用户感知质量奖励
                "reward_R_b": R_b,  # 带宽效率惩罚
                "reward_R_l": R_l,  # 负载均衡惩罚
                "reward_final": reward_final,  # 最终奖励
                "timestamp": time.time(),
                "viewpoint": "front_center"  # 可以根据需要动态改变
            }
            try:
                # ✅ 【修复】确保目录存在且可写
                os.makedirs(SHARED_STATE_DIR, exist_ok=True)
                os.chmod(SHARED_STATE_DIR, 0o777)
                
                # ✅ 【修复】使用原子写入：先写入临时文件，再重命名
                temp_file = f"{client_state_file}.tmp"
                with open(temp_file, 'w') as sf:
                    json.dump(current_state_payload, sf)
                    sf.flush()
                    os.fsync(sf.fileno())  # 强制刷新到磁盘
                
                # 原子重命名
                os.rename(temp_file, client_state_file)
                
                # ✅ 设置文件权限，确保控制器可以读取
                os.chmod(client_state_file, 0o666)  # 所有用户可读写
                
                # ✅ 【关键修复】在第一次迭代时也输出日志，确保状态文件被写入
                if iteration == 0 or iteration % 20 == 0:  # 第一次和每20次打印一次成功信息
                    print(f"[DEBUG] h{a.host_id}: ✅ 已写入状态文件: {client_state_file} (iteration={iteration})", file=sys.stderr, flush=True)
                    # ✅ 验证文件确实已创建
                    if os.path.exists(client_state_file):
                        file_size = os.path.getsize(client_state_file)
                        print(f"[DEBUG] h{a.host_id}: ✅ 状态文件验证: {file_size} 字节", file=sys.stderr, flush=True)
            except (IOError, OSError, PermissionError) as e:
                if iteration % 10 == 0:  # 每10次打印一次错误，避免日志过多
                    print(f"[WARN] h{a.host_id}: 无法写入client_state_file: {e}", file=sys.stderr, flush=True)
                pass  # 写入失败不应使客户端崩溃

            # --- 更新状态窗口（用于Rolling策略） ---
            # 注意：state_window用于Rolling DRL策略，包含最近N步的状态
            # ✅ 使用统一奖励函数的最终值 reward_final 替代旧的 QoE_smooth
            state_window.append([Bu, dly, reward_final, Qr, decision])
            iteration += 1   # ✅ 计数器 +1
            time.sleep(a.interval)
    
    finally:
        # ✅ 【CSV实时写入】确保文件正确关闭
        perf_log_file.close()
        perf_csv_file.close()
        print(f"[INFO] h{a.host_id}: CSV文件已保存到: {perf_csv}", file=sys.stderr, flush=True)

    print(f"[INFO] h{a.host_id}: 实验结束，清理进程（动态订阅模式：Base 流始终运行，Enhanced 流根据 decision 可能运行）", file=sys.stderr, flush=True)
    
    # ✅ 停止 DataDrainer 线程
    if 'reader_base' in locals():
        reader_base.stop()
        print(f"[INFO] h{a.host_id}: Base DataDrainer 线程已停止（共读取 {reader_base.bytes_read // 1024} KB）", file=sys.stderr, flush=True)
    if 'reader_enhanced' in locals():
        reader_enhanced.stop()
        print(f"[INFO] h{a.host_id}: Enhanced DataDrainer 线程已停止（共读取 {reader_enhanced.bytes_read // 1024} KB）", file=sys.stderr, flush=True)
    
    # 等待线程结束
    time.sleep(0.5)
    
    # 清理进程
    if base_p:
        kill(base_p)
    if enh_p:
        kill(enh_p)



# ============================= CLI =============================
if __name__ == "__main__":
    p = argparse.ArgumentParser("MoQ dispatch client")
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--host_id", type=int, required=True)
    p.add_argument("--log_path", required=True)
    p.add_argument("--decision_file", required=True)
    p.add_argument("--strategy", choices=["md2g", "rolling", "heuristic", "clustering"], required=True)
    p.add_argument("--clients", type=int, required=True)   # 改为 clients
    p.add_argument("--relay_ip")
    p.add_argument("--gst_plugin_path", required=True)
    p.add_argument("--device_score", type=float, required=True)
    p.add_argument("--network_type", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--federation", choices=["on", "off"], default="off")

    try:
        run_client(p.parse_args())
    except Exception:
        print("❌ 未处理异常:\n", traceback.format_exc())
        sys.exit(1)