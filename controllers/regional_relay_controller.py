import sys
import json
import time
import argparse
import random
import os
import numpy as np

# 确保 controller 作为独立脚本被 Mininet 节点调用时也能找到项目根目录模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ✅ 【PPO分组学习】导入Canonicalization函数
try:
    from ppo_agent import canonicalize_groups
except ImportError:
    # 如果ppo_agent不可用，定义本地版本
    def canonicalize_groups(group_assignments, device_scores, num_groups):
        """Canonicalization: 按组平均device_score重排组号"""
        if isinstance(group_assignments, torch.Tensor):
            group_assignments = group_assignments.cpu().numpy()
        if isinstance(device_scores, torch.Tensor):
            device_scores = device_scores.cpu().numpy()
        
        num_users = len(group_assignments)
        group_avg_scores = {}
        for g in range(num_groups):
            group_members = [u for u in range(num_users) if group_assignments[u] == g]
            if group_members:
                group_avg_scores[g] = np.mean([device_scores[u] for u in group_members])
            else:
                group_avg_scores[g] = -1.0
        
        sorted_groups = sorted(group_avg_scores.items(), key=lambda x: x[1])
        remap_dict = {old_g: new_g for new_g, (old_g, _) in enumerate(sorted_groups)}
        canonical_groups = np.array([remap_dict[g] for g in group_assignments], dtype=np.int32)
        return canonical_groups, remap_dict

# ✅ 【改进分组逻辑】基于device_score、带宽和FOV相似性的分组函数
def group_by_similarity(device_scores, bandwidths, fov_groups, num_groups=3):
    """
    基于device_score、带宽和FOV相似性进行分组
    
    Args:
        device_scores: 设备分数数组 [num_users]
        bandwidths: 带宽数组 [num_users] (Mbps)
        fov_groups: FOV分组列表 [[user_ids], ...] 或 None
        num_groups: 分组数量（默认3）
    
    Returns:
        group_assignments: 分组分配数组 [num_users]，值范围[0..num_groups-1]
    """
    num_users = len(device_scores)
    if num_users == 0:
        return np.array([], dtype=np.int32)
    
    # 归一化特征（使不同量纲的特征可以比较）
    # device_score已经在[0,1]范围内，不需要归一化
    # 带宽需要归一化（假设范围0-100 Mbps）
    max_bandwidth = max(bandwidths) if len(bandwidths) > 0 and max(bandwidths) > 0 else 100.0
    normalized_bandwidths = bandwidths / max_bandwidth
    
    # 构建用户特征矩阵 [num_users, 2]：device_score + normalized_bandwidth
    user_features = np.column_stack([device_scores, normalized_bandwidths])
    
    # ✅ 优先考虑FOV分组：如果用户在同一个FOV组，尽量分到同一个MD2G组
    fov_user_to_group = {}  # {user_id: fov_group_idx}
    if fov_groups:
        for fov_group_idx, fov_group in enumerate(fov_groups):
            for user_id_str in fov_group:
                try:
                    user_id = int(user_id_str)
                    # user_id是1-based，转换为0-based索引
                    if 1 <= user_id <= num_users:
                        fov_user_to_group[user_id - 1] = fov_group_idx
                except (ValueError, TypeError):
                    continue
    
    # ✅ 使用K-means聚类（如果可用）或基于距离的贪心分组
    try:
        from sklearn.cluster import KMeans
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
        group_assignments = kmeans.fit_predict(user_features)
        print(f"[Controller] ✅ 使用K-means聚类进行分组（基于device_score和带宽）", flush=True)
    except ImportError:
        # Fallback：基于距离的贪心分组
        print(f"[Controller] ⚠️ sklearn不可用，使用基于距离的贪心分组", flush=True)
        group_assignments = np.zeros(num_users, dtype=np.int32)
        group_centers = []  # 每个组的中心点
        
        # 初始化：将用户分配到最近的组
        for user_idx in range(num_users):
            if len(group_centers) < num_groups:
                # 初始化组中心
                group_centers.append(user_features[user_idx].copy())
                group_assignments[user_idx] = len(group_centers) - 1
            else:
                # 找到最近的组中心
                distances = [np.linalg.norm(user_features[user_idx] - center) 
                            for center in group_centers]
                nearest_group = np.argmin(distances)
                group_assignments[user_idx] = nearest_group
        
        # 迭代优化：更新组中心并重新分配
        for _ in range(10):  # 最多迭代10次
            # 更新组中心
            for g in range(num_groups):
                group_members = user_features[group_assignments == g]
                if len(group_members) > 0:
                    group_centers[g] = np.mean(group_members, axis=0)
            
            # 重新分配
            changed = False
            for user_idx in range(num_users):
                distances = [np.linalg.norm(user_features[user_idx] - center) 
                            for center in group_centers]
                nearest_group = np.argmin(distances)
                if group_assignments[user_idx] != nearest_group:
                    group_assignments[user_idx] = nearest_group
                    changed = True
            if not changed:
                break
    
    # ✅ 后处理：优先考虑FOV分组约束
    # 如果用户在同一个FOV组，尽量分到同一个MD2G组
    if fov_user_to_group:
        # 统计每个FOV组在哪些MD2G组中
        fov_to_md2g = {}  # {fov_group_idx: {md2g_group: count}}
        for user_idx in range(num_users):
            if user_idx in fov_user_to_group:
                fov_group_idx = fov_user_to_group[user_idx]
                md2g_group = group_assignments[user_idx]
                if fov_group_idx not in fov_to_md2g:
                    fov_to_md2g[fov_group_idx] = {}
                fov_to_md2g[fov_group_idx][md2g_group] = fov_to_md2g[fov_group_idx].get(md2g_group, 0) + 1
        
        # 对于每个FOV组，将其用户分配到该FOV组中占主导地位的MD2G组
        for fov_group_idx, md2g_distribution in fov_to_md2g.items():
            if len(md2g_distribution) > 1:  # 如果FOV组中的用户分散在多个MD2G组中
                # 找到占主导地位的MD2G组
                dominant_md2g = max(md2g_distribution.items(), key=lambda x: x[1])[0]
                # 将该FOV组的所有用户重新分配到主导MD2G组
                for user_idx in range(num_users):
                    if user_idx in fov_user_to_group and fov_user_to_group[user_idx] == fov_group_idx:
                        group_assignments[user_idx] = dominant_md2g
                print(f"[Controller] ✅ FOV组{fov_group_idx}的用户统一分配到MD2G组{dominant_md2g}", flush=True)
    
    return group_assignments.astype(np.int32)

# 尝试导入 PyTorch，如果环境没有，则使用 Fallback
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ✅ 【新功能】导入FOV分组管理器
try:
    from fov_grouping import FOVGroupManager, load_user_tiles_from_trace, load_fov_from_head_movement
    HAS_FOV_GROUPING = True
except ImportError:
    HAS_FOV_GROUPING = False
    print("[Controller] ⚠️ FOV grouping module not available, will use default grouping")
    # 定义fallback函数
    def load_fov_from_head_movement(*args, **kwargs):
        return {}

# ================= PPO 模型定义 (需与训练代码一致) =================
# ✅ 【关键修复】使用与 ppo_agent.py 中 ActorNet 完全一致的结构
# Student 模型：hidden_size=128, 使用 ReLU 激活函数
class ActorNet(nn.Module):
    """
    MD2G Actor Network with dual heads (for Controller loading compatibility)
    """
    def __init__(self, state_dim, num_users, num_groups=3, hidden_size=128, freeze_enhanced=False):
        super(ActorNet, self).__init__()
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
        
        # 分组head：每用户输出K个logits
        self.group_head = nn.Linear(hidden_size, num_users * num_groups)
        
        # Enhanced head：每用户输出1个logit
        self.enhanced_head = nn.Linear(hidden_size, num_users)
    
    def forward(self, state):
        """
        Returns:
            group_logits: [B, num_users, num_groups]
            enhanced_logits: [B, num_users]
        """
        shared = self.shared_net(state)
        B = shared.size(0)
        
        group_logits = self.group_head(shared).view(B, self.num_users, self.num_groups)
        enhanced_logits = self.enhanced_head(shared).view(B, self.num_users)
        
        if self.freeze_enhanced:
            enhanced_logits = torch.zeros_like(enhanced_logits)
        
        return group_logits, enhanced_logits

def load_model_from_file(model_file, MAX_USERS=100, state_dim=301, num_groups=3):  # ✅ 新格式：MAX_USERS * 3 + 1 = 301（device_score, bandwidth, fov_score）
    """加载单个模型文件（支持分组学习）"""
    try:
        hidden_size = 128  # Student 模型默认值
        
        # 检查模型结构是否兼容
        state_dict = torch.load(model_file, map_location='cpu')
        first_key = list(state_dict.keys())[0]
        
        # ✅ 【关键修复】检查输入维度是否匹配（支持新的ActorNet结构）
        if first_key.startswith("net."):
            # 旧Teacher模型格式：net.0.weight
            input_dim = state_dict['net.0.weight'].shape[1]
            output_dim = state_dict['net.4.weight'].shape[0]
            hidden_size = state_dict['net.0.weight'].shape[0]
        elif first_key.startswith("shared_net."):
            # ✅ 新ActorNet结构格式：shared_net.0.weight（双head输出）
            input_dim = state_dict['shared_net.0.weight'].shape[1]
            # 从group_head或enhanced_head推断hidden_size
            if 'group_head.weight' in state_dict:
                hidden_size = state_dict['group_head.weight'].shape[1]  # group_head: [300, 128] -> 128
            elif 'shared_net.2.weight' in state_dict:
                hidden_size = state_dict['shared_net.2.weight'].shape[0]  # 第二层输出维度
            else:
                hidden_size = state_dict['shared_net.0.weight'].shape[0]  # 第一层输出维度
            # 新结构有双head，不需要检查单一output_dim
            output_dim = None  # 新结构：group_head + enhanced_head
        elif '0.weight' in state_dict:
            # 旧Student模型格式：0.weight
            input_dim = state_dict['0.weight'].shape[1]
            output_dim = state_dict['4.weight'].shape[0] if '4.weight' in state_dict else None
            hidden_size = state_dict['0.weight'].shape[0]
        else:
            # 未知格式，尝试从第一个权重层推断
            first_weight_key = [k for k in state_dict.keys() if 'weight' in k][0]
            input_dim = state_dict[first_weight_key].shape[1]
            hidden_size = state_dict[first_weight_key].shape[0]
            output_dim = None
            print(f"[Controller] ⚠️ Unknown model format, using first weight layer: {first_weight_key}", flush=True)
        
        # 检查模型结构是否兼容
        if input_dim != state_dim:
            raise ValueError(f"Model input dimension mismatch: expected {state_dim} (for {MAX_USERS} users), but model has {input_dim}.")
        
        # ✅ 【新结构】新ActorNet有双head，不需要检查单一output_dim
        if output_dim is not None and output_dim != MAX_USERS and output_dim != 1:
            print(f"[Controller] ⚠️ Warning: Model output dimension is {output_dim}, expected {MAX_USERS} or 1")
        
        # 根据模型文件判断是 Student (128) 还是 Teacher (512)
        if first_key.startswith("net."):
            if hidden_size == 512:
                print(f"[Controller] Detected Teacher model (hidden_size=512)")
            else:
                print(f"[Controller] Detected model (hidden_size={hidden_size})")
        elif first_key.startswith("shared_net."):
            # ✅ 新ActorNet结构（双head）
            if hidden_size == 128:
                print(f"[Controller] Detected Student model (hidden_size=128, new ActorNet structure with dual heads)")
            elif hidden_size == 512:
                print(f"[Controller] Detected Teacher model (hidden_size=512, new ActorNet structure with dual heads)")
            else:
                print(f"[Controller] Detected model (hidden_size={hidden_size}, new ActorNet structure with dual heads)")
        else:
            if hidden_size == 128:
                print(f"[Controller] Detected Student model (hidden_size=128)")
            elif hidden_size == 512:
                print(f"[Controller] Detected Teacher model (hidden_size=512)")
            else:
                print(f"[Controller] Detected model (hidden_size={hidden_size})")
        
        # ✅ 【PPO分组学习】检查模型是否有双head结构
        # 尝试加载新模型（双head），如果失败则使用旧模型（单head，向后兼容）
        try:
            model = ActorNet(state_dim, MAX_USERS, num_groups, hidden_size)
            # 尝试加载state_dict，如果键名不匹配则使用旧模型
            test_state_dict = {k: v for k, v in list(state_dict.items())[:5]}  # 只检查前5个键
            model_keys = set(model.state_dict().keys())
            test_keys = set(test_state_dict.keys())
            # 检查是否有group_head或enhanced_head
            has_dual_head = any('group_head' in k or 'enhanced_head' in k for k in test_keys)
            if not has_dual_head:
                # 旧模型格式，使用单head（向后兼容）
                print(f"[Controller] ⚠️ Detected old model format (single head), using compatibility mode", flush=True)
                # 这里需要创建一个兼容的旧模型，但为了简化，我们假设新模型可以处理
                model = ActorNet(state_dim, MAX_USERS, num_groups, hidden_size)
        except Exception as e:
            print(f"[Controller] ⚠️ Failed to create dual-head model, using single-head: {e}", flush=True)
            # 向后兼容：如果创建失败，使用旧模型
            from ppo_agent import ActorNet as OldActorNet
            model = OldActorNet(state_dim, MAX_USERS, hidden_size)
        
        # 处理模型键名不匹配问题
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        # ✅ 【关键修复】处理旧模型格式（单head）到新模型格式（双head）的转换
        # 旧格式：0.weight, 0.bias, 2.weight, 2.bias, 4.weight, 4.bias
        # 新格式：shared_net.0.weight, shared_net.0.bias, shared_net.2.weight, shared_net.2.bias, group_head.weight, enhanced_head.weight
        if model_keys != state_dict_keys:
            # 检查是否是旧格式（只有数字键，如"0.weight", "2.weight", "4.weight"）
            is_old_format = all(k.split('.')[0].isdigit() for k in state_dict_keys if '.' in k and not k.startswith('net.'))
            
            if is_old_format and any('shared_net' in k for k in model_keys):
                # ✅ 【关键修复】旧格式转新格式：将数字键映射到shared_net和heads
                print(f"[Controller] ⚠️ Detected old model format (single head), converting to new format...")
                fixed_state_dict = {}
                
                # 映射规则：
                # 0.weight, 0.bias -> shared_net.0.weight, shared_net.0.bias
                # 2.weight, 2.bias -> shared_net.2.weight, shared_net.2.bias
                # 4.weight, 4.bias -> 需要判断：如果是group_head维度，映射到group_head；否则映射到enhanced_head
                
                for k, v in state_dict.items():
                    if k == '0.weight' or k == '0.bias':
                        fixed_state_dict[f'shared_net.{k}'] = v
                    elif k == '2.weight' or k == '2.bias':
                        fixed_state_dict[f'shared_net.{k}'] = v
                    elif k == '4.weight' or k == '4.bias':
                        # 判断维度：group_head的维度是 MAX_USERS * num_groups，enhanced_head的维度是 MAX_USERS
                        # 如果v的shape[0] == MAX_USERS * num_groups，则是group_head；否则是enhanced_head
                        if len(v.shape) == 2:
                            if v.shape[0] == MAX_USERS * num_groups:
                                fixed_state_dict[f'group_head.{k[2:]}'] = v  # 移除"4."，保留"weight"或"bias"
                            else:
                                fixed_state_dict[f'enhanced_head.{k[2:]}'] = v
                        else:
                            # 1D tensor，通常是bias，根据MAX_USERS判断
                            if v.shape[0] == MAX_USERS * num_groups:
                                fixed_state_dict[f'group_head.{k[2:]}'] = v
                            else:
                                fixed_state_dict[f'enhanced_head.{k[2:]}'] = v
                    else:
                        fixed_state_dict[k] = v
                
                state_dict = fixed_state_dict
                print(f"[Controller] ✅ Converted old model format to new format")
            
            # 检查是否需要添加"net."前缀
            elif any(not k.startswith('net.') for k in state_dict_keys) and not is_old_format:
                # 模型文件中的键没有"net."前缀，需要添加
                fixed_state_dict = {}
                for k, v in state_dict.items():
                    if not k.startswith('net.'):
                        fixed_state_dict[f'net.{k}'] = v
                    else:
                        fixed_state_dict[k] = v
                state_dict = fixed_state_dict
                print(f"[Controller] ✅ Fixed state_dict keys (added 'net.' prefix)")
            # 或者需要移除"net."前缀
            elif any(k.startswith('net.') for k in state_dict_keys) and not any(k.startswith('net.') for k in model_keys):
                # 模型文件中的键有"net."前缀，但模型定义没有，需要移除
                fixed_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('net.'):
                        fixed_state_dict[k[4:]] = v  # 移除"net."前缀
                    else:
                        fixed_state_dict[k] = v
                state_dict = fixed_state_dict
                print(f"[Controller] ✅ Fixed state_dict keys (removed 'net.' prefix)")
        
        # ✅ 【关键修复】使用strict=False允许部分匹配，如果仍有不匹配的键
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[Controller] ⚠️ Warning: Some keys could not be loaded: {e}")
            # 尝试只加载匹配的键
            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            if len(matched_dict) > 0:
                model.load_state_dict(matched_dict, strict=False)
                print(f"[Controller] ✅ Loaded {len(matched_dict)}/{len(state_dict)} keys successfully")
            else:
                raise e
        model.eval()
        print(f"[Controller] ✅ [MD2G] Successfully loaded PPO model from {model_file}")
        print(f"[Controller] ✅ [MD2G] Model type: Student (hidden_size={hidden_size})" if hidden_size == 128 else f"[Controller] ✅ [MD2G] Model type: Teacher (hidden_size={hidden_size})")
        return model
    except Exception as e:
        print(f"[Controller] ⚠️ Failed to load model from {model_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_controller(args):
    """
    策略路由控制器：根据策略类型调用对应的controller
    """
    decision_file = args.decision_file
    num_users = args.max_users
    strategy = getattr(args, 'strategy', 'md2g').lower()
    
    # ✅ Strategy routing (baseline strategies)
    if strategy == 'heuristic':
        # Heuristic策略：调用heuristic_controller_v2_refined.py
        from strategies.heuristic_controller_v2_refined import run_enhanced_heuristic_controller
        print(f"[Controller] Routing to Heuristic Controller for {num_users} users...", flush=True)
        run_enhanced_heuristic_controller(args)
        return
    elif strategy == 'clustering':
        # Clustering策略：调用predictive_controller_v2_refined.py
        from strategies.predictive_controller_v2_refined import run_enhanced_predictive_controller
        print(f"[Controller] Routing to Clustering (Predictive) Controller for {num_users} users...", flush=True)
        run_enhanced_predictive_controller(args)
        return
    elif strategy == 'rolling':
        # ✅ 【关键修改】Rolling策略：在服务端（r1/r2）使用SC-DDQN模型做决策
        # 与MD2G策略保持一致，统一服务端决策架构
        print(f"[Controller] Starting SC-DDQN logic for {num_users} users...", flush=True)
        print(f"[Controller] Note: Rolling策略在服务端部署SC-DDQN模型（与MD2G策略架构一致）", flush=True)
    elif strategy == 'md2g':
        # MD2G策略：使用PPO模型
        print(f"[Controller] Starting PPO logic for {num_users} users...", flush=True)
    else:
        # 未知策略：默认使用PPO逻辑
        print(f"[Controller] ⚠️ Unknown strategy '{strategy}', using PPO logic as fallback...", flush=True)
    
    print(f"[Controller] Decision file: {decision_file}", flush=True)
    print(f"[Controller] Model path: {args.model_path}", flush=True)
    print(f"[Controller] Strategy: {strategy}", flush=True)
    
    # ✅ 【新功能】边缘计算架构：每个relay只处理自己负责的用户
    # 从环境变量或参数获取当前relay名称
    relay_name = os.environ.get('RELAY_NAME', getattr(args, 'relay_name', 'r0'))
    print(f"[Controller] Running on relay: {relay_name}", flush=True)
    
    # ✅ 加载用户到relay的映射
    try:
        from user_relay_mapping import load_mapping_from_file, get_users_for_relay
        user_relay_mapping = load_mapping_from_file("/tmp/user_relay_mapping.json")
        
        # 如果映射文件不存在，使用默认分配（基于relay名称）
        if not user_relay_mapping:
            print(f"[Controller] ⚠️ Mapping file not found, using default assignment", flush=True)
            # 默认分配：根据relay索引分配用户
            relay_idx = int(relay_name[1]) if relay_name[1].isdigit() else 0
            relay_list = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6']
            users_per_relay = num_users // len(relay_list)
            remainder = num_users % len(relay_list)
            start_user = relay_idx * users_per_relay + min(relay_idx, remainder) + 1
            end_user = start_user + users_per_relay + (1 if relay_idx < remainder else 0) - 1
            assigned_users = list(range(start_user, end_user + 1))
        else:
            # 从映射中提取当前relay负责的用户
            assigned_users = [uid for uid, r in user_relay_mapping.items() if r == relay_name]
            assigned_users.sort()
        
        print(f"[Controller] ✅ Assigned users for {relay_name}: {assigned_users} ({len(assigned_users)} users)", flush=True)
    except ImportError:
        print(f"[Controller] ⚠️ user_relay_mapping module not found, processing all users", flush=True)
        assigned_users = list(range(1, num_users + 1))  # 处理所有用户（fallback）
    except Exception as e:
        print(f"[Controller] ⚠️ Failed to load user-relay mapping: {e}, processing all users", flush=True)
        assigned_users = list(range(1, num_users + 1))  # 处理所有用户（fallback）
    
    # ✅ 【新功能】读取用户网络类型映射（用于混合网络场景）
    user_network_types = None
    log_path = getattr(args, 'log_path', None)
    if log_path:
        mapping_file = os.path.join(log_path, "host_network_mapping.csv")
        if os.path.exists(mapping_file):
            try:
                import pandas as pd
                df = pd.read_csv(mapping_file)
                # 创建用户ID到网络类型的映射（host_id从1开始，但索引从0开始）
                user_network_types = {}
                for _, row in df.iterrows():
                    host_id = int(row['host_id'])
                    net_type = str(row['network_type']).strip()
                    user_network_types[host_id] = net_type
                print(f"[Controller] ✅ Loaded user network mapping from {mapping_file}")
                print(f"[Controller] Network types distribution: {pd.Series(list(user_network_types.values())).value_counts().to_dict()}")
            except Exception as e:
                print(f"[Controller] ⚠️ Failed to load network mapping: {e}")
                user_network_types = None
    
    # ✅ 【新功能】初始化FOV分组管理器（如果启用）
    fov_group_manager = None
    trace_csv_path = getattr(args, 'trace_csv', None)
    current_frame_id = getattr(args, 'frame_id', 0)
    use_fov_grouping = getattr(args, 'use_fov_grouping', False)
    
    if use_fov_grouping and HAS_FOV_GROUPING:
        overlap_threshold = getattr(args, 'fov_overlap_threshold', 0.5)
        fov_group_manager = FOVGroupManager(overlap_threshold=overlap_threshold)
        print(f"[Controller] ✅ FOV grouping enabled (threshold={overlap_threshold})", flush=True)
    elif use_fov_grouping and not HAS_FOV_GROUPING:
        print(f"[Controller] ⚠️ FOV grouping requested but module not available", flush=True)
    
    # ✅ 【MD2G分组】构建用户分组（基于host_id % 3，与客户端一致）
    # ✅ 【修复】group_id范围统一为[0..K-1]，不是[1..K]
    # 分组逻辑：user_group = (host_id - 1) % 3  # 0, 1, 或 2
    user_groups = {0: [], 1: [], 2: []}  # {group_id: [user_ids]}，范围[0..K-1]
    for user_id in range(1, num_users + 1):
        group_id = (user_id - 1) % 3  # 0, 1, 或 2
        user_groups[group_id].append(user_id)
    print(f"[Controller] ✅ MD2G分组: Group0={len(user_groups[0])}用户, Group1={len(user_groups[1])}用户, Group2={len(user_groups[2])}用户", flush=True)
    
    # 1. 尝试加载模型
    # ✅ 【关键修复】MD2G 使用学生模型 (Student Model)
    # ✅ 【关键修改】新的状态维度：MAX_USERS * 3 + 1 = 100 * 3 + 1 = 301
    # 模型结构：ActorNet(state_dim=301, num_users=100, hidden_size=128)
    # 状态格式：每个用户3个特征（device_score, bandwidth, fov_score）+ 全局capacity
    # 注意：虽然维度是301，但内容完全不同（之前是priority, last_layer, device_score，现在是device_score, bandwidth, fov_score）
    # 输出维度：MAX_USERS = 100 (每个用户的动作概率)
    
    # ✅ 【新功能】支持多模型加载（混合网络场景）
    # 如果有用户网络类型映射，为每个网络类型加载对应的模型
    models_dict = {}  # {network_type: model}
    model_files_dict = {}  # {network_type: model_file}
    model = None
    model_file = None  # ✅ 确保在函数作用域内可用（用于单模型场景）
    
    # ✅ 【关键修复】根据策略类型查找模型文件
    # MD2G: 查找 PPO 学生模型 (ppo_actor_student_{network_type}.pth)
    # Rolling: 查找 SC-DDQN 模型 (sc_ddqn_rolling.pth)
    if HAS_TORCH and args.model_path:
        # ✅ 【关键修复】确保model_path是绝对路径（Mininet节点需要绝对路径）
        if not os.path.isabs(args.model_path):
            # 如果是相对路径，尝试转换为绝对路径
            # 首先尝试基于当前工作目录
            abs_model_path = os.path.abspath(args.model_path)
            if os.path.exists(abs_model_path):
                args.model_path = abs_model_path
                print(f"[Controller] ✅ Converted relative model_path to absolute: {abs_model_path}", flush=True)
            else:
                # 如果基于当前目录不存在，尝试基于脚本所在目录
                script_dir = os.path.dirname(os.path.abspath(__file__))
                abs_model_path = os.path.join(script_dir, args.model_path)
                if os.path.exists(abs_model_path):
                    args.model_path = abs_model_path
                    print(f"[Controller] ✅ Converted relative model_path to absolute (based on script dir): {abs_model_path}", flush=True)
                else:
                    print(f"[Controller] ⚠️ Model path '{args.model_path}' not found, will try to use as-is", flush=True)
        else:
            # 已经是绝对路径，验证是否存在
            if not os.path.exists(args.model_path):
                print(f"[Controller] ⚠️ Model path '{args.model_path}' does not exist, will try to use as-is", flush=True)
        # ✅ 【Rolling策略】查找SC-DDQN模型
        if strategy == 'rolling':
            # Rolling策略使用SC-DDQN模型
            if os.path.isfile(args.model_path):
                model_file = args.model_path
            elif os.path.isdir(args.model_path):
                # 查找 sc_ddqn_rolling.pth
                sc_ddqn_model = os.path.join(args.model_path, "sc_ddqn_rolling.pth")
                if os.path.exists(sc_ddqn_model):
                    model_file = sc_ddqn_model
                    print(f"[Controller] ✅ [Rolling] Found SC-DDQN model: {sc_ddqn_model}", flush=True)
                else:
                    # 回退到旧格式
                    rolling_drl_model = os.path.join(args.model_path, "rolling_drl_actor.pth")
                    if os.path.exists(rolling_drl_model):
                        model_file = rolling_drl_model
                        print(f"[Controller] ⚠️ [Rolling] SC-DDQN model not found, using fallback: {rolling_drl_model}", flush=True)
                    else:
                        print(f"[Controller] ❌ [Rolling] No SC-DDQN model found in {args.model_path}", flush=True)
                        model_file = None
            else:
                model_file = None
        # ✅ 【MD2G/其他策略】查找PPO学生模型
        elif strategy == 'md2g' or strategy not in ['rolling']:
            # 检查是否是文件路径
            if os.path.isfile(args.model_path):
                model_file = args.model_path
            # 检查是否是目录路径
            elif os.path.isdir(args.model_path):
                # ✅ 【新功能】如果有用户网络类型映射，为每个网络类型加载对应的模型
                # ✅ 【关键修复】只有当 user_network_types 不为空且不是 default_mix 时才进入混合网络场景
                # default_mix 虽然是混合网络，但如果没有用户映射文件，应该使用单网络场景的fallback逻辑
                if user_network_types and len(user_network_types) > 0:
                    # 混合网络场景：为每个网络类型加载对应的模型
                    unique_net_types = set(user_network_types.values())
                    print(f"[Controller] Mixed network scenario detected. Loading models for: {unique_net_types}", flush=True)
                    
                    # ✅ 【关键修复】创建标准化映射，避免重复加载相同模型
                    # 例如：fiber_optic 和 optic 应该映射到同一个模型
                    normalized_to_original = {}  # {normalized_type: [original_types]}
                    for net_type in unique_net_types:
                        normalized_net_type = net_type.lower().strip()
                        if normalized_net_type in ['fiber_optic', 'fiber', 'optic']:
                            normalized_net_type = 'optic'
                        if normalized_net_type not in normalized_to_original:
                            normalized_to_original[normalized_net_type] = []
                        normalized_to_original[normalized_net_type].append(net_type)
                    
                    # 为每个标准化后的网络类型加载模型（避免重复）
                    for normalized_net_type, original_types in normalized_to_original.items():
                        # 使用第一个原始类型作为代表（用于日志显示）
                        representative_type = original_types[0]
                        
                        # ✅ 【新结构】优先查找新目录结构
                        deploy_student_dir = os.path.join(args.model_path, "deploy_student_128")
                        new_model_path = os.path.join(deploy_student_dir, f"ppo_actor_student_{normalized_net_type}.pth")
                        
                        model_file_found = None
                        # 尝试新结构
                        if os.path.exists(new_model_path):
                            model_file_found = new_model_path
                            print(f"[Controller] ✅ Found model (new structure) for {representative_type} (normalized: {normalized_net_type}): {new_model_path}", flush=True)
                        else:
                            # 回退到旧结构（向后兼容）
                            old_model_path = os.path.join(args.model_path, f"ppo_actor_{normalized_net_type}_general.pth")
                            if os.path.exists(old_model_path):
                                model_file_found = old_model_path
                                print(f"[Controller] ✅ Found model (old structure) for {representative_type} (normalized: {normalized_net_type}): {old_model_path}", flush=True)
                            else:
                                # 回退到通用模型（按优先级）
                                fallback_models = ['wifi', '4g', '5g', 'optic']
                                found_fallback = False
                                for fallback_type in fallback_models:
                                    # 先尝试新结构
                                    fallback_new = os.path.join(deploy_student_dir, f"ppo_actor_student_{fallback_type}.pth")
                                    if os.path.exists(fallback_new):
                                        model_file_found = fallback_new
                                        print(f"[Controller] ⚠️ Model for '{representative_type}' (normalized: {normalized_net_type}) not found, using fallback (new): {fallback_new}", flush=True)
                                        found_fallback = True
                                        break
                                    # 再尝试旧结构
                                    fallback_old = os.path.join(args.model_path, f"ppo_actor_{fallback_type}_general.pth")
                                    if os.path.exists(fallback_old):
                                        model_file_found = fallback_old
                                        print(f"[Controller] ⚠️ Model for '{representative_type}' (normalized: {normalized_net_type}) not found, using fallback (old): {fallback_old}", flush=True)
                                        found_fallback = True
                                        break
                                if not found_fallback:
                                    print(f"[Controller] ⚠️ No suitable model found for {representative_type} (normalized: {normalized_net_type}), will use heuristic", flush=True)
                        
                        # ✅ 【关键修复】为所有原始网络类型映射到同一个模型文件
                        # 这样 fiber_optic 和 optic 都会使用同一个模型
                        if model_file_found:
                            for original_type in original_types:
                                model_files_dict[original_type] = model_file_found
                                print(f"[Controller]   → Mapped {original_type} to model: {os.path.basename(model_file_found)}", flush=True)
                
                # 旧代码（已替换，保留作为参考）
                # for net_type in unique_net_types:
                    # ✅ 【关键修复】标准化网络类型名称，确保与模型文件名匹配
                    normalized_net_type = net_type.lower().strip()
                    # 处理 fiber_optic 的各种变体
                    if normalized_net_type in ['fiber_optic', 'fiber', 'optic']:
                        normalized_net_type = 'optic'
                    # 确保网络类型是标准格式（wifi, 4g, 5g, optic）
                    elif normalized_net_type not in ['wifi', '4g', '5g']:
                        # 如果是不认识的网络类型，尝试直接使用（可能是其他变体）
                        pass
                    
                    # ✅ 【新结构】优先查找新目录结构
                    deploy_student_dir = os.path.join(args.model_path, "deploy_student_128")
                    new_model_path = os.path.join(deploy_student_dir, f"ppo_actor_student_{normalized_net_type}.pth")
                    
                    # 尝试新结构
                    if os.path.exists(new_model_path):
                        # ✅ 【关键修复】使用原始 net_type 作为键，但使用标准化后的路径
                        # 这样在推理时，无论用户映射中是 fiber_optic 还是 optic，都能找到对应的模型
                        model_files_dict[net_type] = new_model_path
                        print(f"[Controller] ✅ Found model (new structure) for {net_type} (normalized: {normalized_net_type}): {new_model_path}", flush=True)
                    else:
                        # 回退到旧结构（向后兼容）
                        old_model_path = os.path.join(args.model_path, f"ppo_actor_{normalized_net_type}_general.pth")
                        if os.path.exists(old_model_path):
                            model_files_dict[net_type] = old_model_path
                            print(f"[Controller] ✅ Found model (old structure) for {net_type} (normalized: {normalized_net_type}): {old_model_path}", flush=True)
                        else:
                            # 回退到通用模型（按优先级）
                            fallback_models = ['wifi', '4g', '5g', 'optic']
                            found_fallback = False
                            for fallback_type in fallback_models:
                                # 先尝试新结构
                                fallback_new = os.path.join(deploy_student_dir, f"ppo_actor_student_{fallback_type}.pth")
                                if os.path.exists(fallback_new):
                                    model_files_dict[net_type] = fallback_new
                                    print(f"[Controller] ⚠️ Model for '{net_type}' (normalized: {normalized_net_type}) not found, using fallback (new): {fallback_new}", flush=True)
                                    found_fallback = True
                                    break
                                # 再尝试旧结构
                                fallback_old = os.path.join(args.model_path, f"ppo_actor_{fallback_type}_general.pth")
                                if os.path.exists(fallback_old):
                                    model_files_dict[net_type] = fallback_old
                                    print(f"[Controller] ⚠️ Model for '{net_type}' (normalized: {normalized_net_type}) not found, using fallback (old): {fallback_old}", flush=True)
                                    found_fallback = True
                                    break
                            if not found_fallback:
                                print(f"[Controller] ⚠️ No suitable model found for {net_type} (normalized: {normalized_net_type}), will use heuristic", flush=True)
                
                # ✅ 【关键修复】如果没有 user_network_types 或为空，进入单网络场景
                if not user_network_types or len(user_network_types) == 0:
                    # 单网络场景：查找对应网络类型的学生模型
                    network_type = getattr(args, 'network_type', 'wifi')
                normalized_net_type = network_type.lower().strip()
                # ✅ 【关键修复】处理混合网络类型：wifi_dominant -> wifi, 5g_dominant -> 5g
                if normalized_net_type == 'wifi_dominant':
                    normalized_net_type = 'wifi'
                elif normalized_net_type == '5g_dominant':
                    normalized_net_type = '5g'
                elif normalized_net_type in ['fiber_optic', 'fiber', 'optic']:
                    normalized_net_type = 'optic'
                
                print(f"[Controller] Looking for student model for network_type: {network_type} (normalized: {normalized_net_type})", flush=True)
                
                # ✅ 【新结构】优先查找新目录结构
                deploy_student_dir = os.path.join(args.model_path, "deploy_student_128")
                new_model_path = os.path.join(deploy_student_dir, f"ppo_actor_student_{normalized_net_type}.pth")
                
                if os.path.exists(new_model_path):
                    model_file = new_model_path
                    print(f"[Controller] ✅ Found student model (new structure): {new_model_path}", flush=True)
                else:
                    # 回退到旧结构（向后兼容）
                    old_model_path = os.path.join(args.model_path, f"ppo_actor_{normalized_net_type}_general.pth")
                    if os.path.exists(old_model_path):
                        model_file = old_model_path
                        print(f"[Controller] ✅ Found student model (old structure): {old_model_path}", flush=True)
                    else:
                        # ✅ 【关键修复】对于 default_mix 或其他不存在的网络类型，使用通用模型
                        # 优先级：wifi > 4g > 5g > optic（因为 wifi 是最常见的混合网络场景）
                        fallback_models = ['wifi', '4g', '5g', 'optic']
                        model_file = None
                        for fallback_type in fallback_models:
                            # 先尝试新结构
                            fallback_new = os.path.join(deploy_student_dir, f"ppo_actor_student_{fallback_type}.pth")
                            if os.path.exists(fallback_new):
                                model_file = fallback_new
                                print(f"[Controller] ⚠️ Student model for '{network_type}' not found, using fallback (new): {fallback_new}", flush=True)
                                break
                            # 再尝试旧结构
                            fallback_old = os.path.join(args.model_path, f"ppo_actor_{fallback_type}_general.pth")
                            if os.path.exists(fallback_old):
                                model_file = fallback_old
                                print(f"[Controller] ⚠️ Student model for '{network_type}' not found, using fallback (old): {fallback_old}", flush=True)
                                break
                    
                    if model_file is None:
                        # 如果所有学生模型都不存在，才使用 rolling_drl_actor.pth（但会检查结构）
                        rolling_model = os.path.join(args.model_path, "rolling_drl_actor.pth")
                        if os.path.exists(rolling_model):
                            model_file = rolling_model
                            print(f"[Controller] ⚠️ No student models found, using rolling model: {rolling_model} (will check compatibility)", flush=True)
                        else:
                            print(f"[Controller] ❌ No suitable model files found in {args.model_path}", flush=True)
        else:
            # 尝试默认路径
            default_model = os.path.join(args.model_path, "rolling_drl_actor.pth")
            if os.path.exists(default_model):
                model_file = default_model
        
        # ✅ 【新功能】加载模型（单模型或多模型）
        MAX_USERS = 100  # 训练时的最大用户数
        # ✅ 【关键修改】新的状态维度：MAX_USERS * 3 + 1 = 301
        # 每个用户3个特征：device_score, bandwidth, fov_score
        # 注意：虽然维度是301，但内容完全不同（之前是priority, last_layer, device_score，现在是device_score, bandwidth, fov_score）
        state_dim = MAX_USERS * 3 + 1  # 301
        
        if model_files_dict:
            # ✅ 【关键保证】混合网络场景：为每个网络类型加载MD2G PPO模型
            print(f"[Controller] ✅ [MD2G] Loading {len(model_files_dict)} PPO models for mixed network scenario...", flush=True)
            for net_type, m_file in model_files_dict.items():
                print(f"[Controller] ✅ [MD2G] Loading PPO model for {net_type} from {m_file}...", flush=True)
                loaded_model = load_model_from_file(m_file, MAX_USERS, state_dim)
                if loaded_model:
                    models_dict[net_type] = loaded_model
                    print(f"[Controller] ✅ [MD2G] Successfully loaded PPO model for {net_type}: {m_file}", flush=True)
                else:
                    print(f"[Controller] ❌ [ERROR] [MD2G] Failed to load PPO model for {net_type} from {m_file}, will use heuristic for these users", flush=True)
            if models_dict:
                print(f"[Controller] ✅ [MD2G] Successfully loaded {len(models_dict)} PPO models", flush=True)
            else:
                print(f"[Controller] ❌ [ERROR] [MD2G] No PPO models loaded, will use heuristic", flush=True)
        elif model_file and os.path.exists(model_file):
            # ✅ 【Rolling策略】加载SC-DDQN模型（服务端部署）
            if strategy == 'rolling':
                # ✅ 【关键修改】Rolling策略现在在服务端（r1/r2）加载和使用SC-DDQN模型
                # 与MD2G策略保持一致，统一服务端决策架构
                print(f"[Controller] ✅ [Rolling] Loading SC-DDQN model from: {model_file}", flush=True)
                try:
                    # 导入Rolling策略类
                    import sys
                    # ✅ os已在文件顶部导入，不需要重复导入
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    strategies_dir = os.path.join(current_dir, "strategies")
                    if strategies_dir not in sys.path:
                        sys.path.insert(0, strategies_dir)
                    
                    from rolling_drl_strategy_v2_refined import RollingDRLStrategy
                    
                    # 加载SC-DDQN模型（与客户端使用相同的接口）
                    rolling_strategy = RollingDRLStrategy(
                        model_path=model_file,
                        window_size=3,
                        state_feature_count=5
                    )
                    model = rolling_strategy  # 使用Rolling策略对象作为model
                    print(f"[Controller] ✅ [Rolling] Successfully loaded SC-DDQN model: {model_file}", flush=True)
                    print(f"[Controller] ✅ [Rolling] SC-DDQN model deployed on relay (service-side decision)", flush=True)
                except Exception as e:
                    print(f"[Controller] ❌ [ERROR] [Rolling] Failed to load SC-DDQN model: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    model = None  # 回退到占位模式
            else:
                # ✅ 【关键保证】单网络场景：加载单个MD2G PPO模型
                print(f"[Controller] ✅ [MD2G] Loading PPO model from: {model_file}", flush=True)
                model = load_model_from_file(model_file, MAX_USERS, state_dim)
                if model:
                    print(f"[Controller] ✅ [MD2G] Successfully loaded PPO model: {model_file}", flush=True)
                    print(f"[Controller] ✅ [MD2G] Model structure: state_dim={state_dim}, num_users={MAX_USERS}", flush=True)
                else:
                    print(f"[Controller] ❌ [ERROR] [MD2G] Failed to load model from {model_file}. Switching to Heuristic Mode.", flush=True)
        else:
            print(f"[Controller] ℹ️ Model file not found at {args.model_path}. Using Heuristic Mode.")
    else:
        if not HAS_TORCH:
            print("[Controller] ℹ️ PyTorch not found. Using Heuristic Mode.")
        else:
            print("[Controller] ℹ️ Model path not provided. Using Heuristic Mode.")

    # 2. 生成决策
    # ✅ 【新功能】支持动态仿真：如果启用FOV分组，在循环中运行并更新frame_id
    # 否则只运行一次（静态模式）
    
    # 检查是否需要动态模式（启用FOV分组且duration > 0）
    dynamic_mode = use_fov_grouping and trace_csv_path and os.path.exists(trace_csv_path)
    update_interval = getattr(args, 'interval', 0.5)  # 默认0.5秒更新一次
    frame_rate = 30.0  # 假设30fps，每0.5秒更新一次对应15帧
    
    if dynamic_mode:
        print(f"[Controller] ✅ Dynamic mode enabled: frame_id will auto-increment", flush=True)
        print(f"[Controller] Update interval: {update_interval}s, Frame rate: {frame_rate} fps", flush=True)
    
    decisions = {}
    
    # ✅ 【修复】从设备分数文件或默认值获取设备分数
    device_scores = []
    device_csv_path = getattr(args, 'device_csv', None)
    if device_csv_path and os.path.exists(device_csv_path):
        try:
            import pandas as pd
            df_dev = pd.read_csv(device_csv_path)
            # 使用设备性能分数（如果有）
            for i in range(num_users):
                if i < len(df_dev):
                    # 假设有device_score列，或者使用其他列计算
                    if 'device_score' in df_dev.columns:
                        device_scores.append(float(df_dev.iloc[i]['device_score']))
                    else:
                        # 使用默认计算
                        device_scores.append(0.4 + (0.6 * (i / num_users)))
                else:
                    device_scores.append(0.4 + (0.6 * (i / num_users)))
        except Exception as e:
            print(f"[Controller] ⚠️ Failed to load device scores: {e}, using defaults", flush=True)
            device_scores = [0.4 + (0.6 * (i / num_users)) for i in range(num_users)]
    else:
        # 默认设备分数分布
        for i in range(num_users):
            score = 0.4 + (0.6 * (i / num_users))
            device_scores.append(score)

    # ✅ 【新功能】从Head_movement数据加载真实FOV信息
    fov_overlap_dict = {}
    head_movement_csv_path = getattr(args, 'head_movement_csv', None)
    if not head_movement_csv_path:
        # 尝试默认路径
        default_path = os.path.join(PROJECT_ROOT, "datasets", "Head_movement_clean_1.csv")
        if os.path.exists(default_path):
            head_movement_csv_path = default_path
    
    if head_movement_csv_path and os.path.exists(head_movement_csv_path) and HAS_FOV_GROUPING:
        try:
            print(f"[Controller] 📊 Loading FOV data from Head_movement CSV: {head_movement_csv_path}", flush=True)
            fov_overlap_dict = load_fov_from_head_movement(head_movement_csv_path, num_users, timestamp=None)
            print(f"[Controller] ✅ Loaded FOV overlap for {len(fov_overlap_dict)} users", flush=True)
            # 打印前几个用户的FOV重叠度
            sample_users = list(fov_overlap_dict.items())[:5]
            for uid, fov_score in sample_users:
                print(f"[Controller]   User {uid}: FOV overlap = {fov_score:.3f}", flush=True)
        except Exception as e:
            print(f"[Controller] ⚠️ Failed to load FOV from Head_movement: {e}", flush=True)
            import traceback
            traceback.print_exc()
            fov_overlap_dict = {}
    else:
        if head_movement_csv_path:
            print(f"[Controller] ⚠️ Head_movement CSV not found: {head_movement_csv_path}", flush=True)
        else:
            print(f"[Controller] ℹ️ Head_movement CSV path not provided, using default FOV values", flush=True)
    
    # 如果没有FOV数据，使用默认值
    if not fov_overlap_dict:
        fov_overlap_dict = {i: 0.5 for i in range(1, num_users + 1)}

    # ✅ 【新功能】FOV分组（如果启用tiles-based分组）
    fov_groups = None
    user_tiles_map = None
    
    if fov_group_manager and trace_csv_path and os.path.exists(trace_csv_path):
        try:
            # 加载当前帧的用户Tile数据
            user_tiles_map = load_user_tiles_from_trace(trace_csv_path, current_frame_id)
            
            if user_tiles_map:
                # 执行FOV分组
                fov_groups = fov_group_manager.group_users(user_tiles_map)
                stats = fov_group_manager.get_group_stats(fov_groups, user_tiles_map)
                print(f"[Controller] ✅ FOV grouping completed: {stats['total_groups']} groups, "
                      f"avg_size={stats['avg_group_size']:.2f}, "
                      f"multicast_gain={stats['multicast_gain_estimate']:.2%}", flush=True)
            else:
                print(f"[Controller] ⚠️ No user tiles found for frame {current_frame_id}", flush=True)
        except Exception as e:
            print(f"[Controller] ⚠️ FOV grouping failed: {e}", flush=True)
            fov_groups = None
    
    if models_dict:
        # === 混合网络场景：为每个用户根据其网络类型选择对应的模型 ===
        print(f"[Controller] Using multi-model inference for mixed network scenario", flush=True)
        MAX_USERS = 100  # 训练时的最大用户数
        state_dim = MAX_USERS * 3 + 1  # 301
        
        # ✅ 【修复】读取真实客户端状态数据
        import glob
        user_metrics_dict = {}  # {user_id: user_metric}
        
        # 尝试从两个位置读取状态文件
        state_file_patterns = [
            "/tmp/mininet_shared/client_h*_state.json",
            "/tmp/client_h*_state.json"
        ]
        
        state_files = []
        for pattern in state_file_patterns:
            found_files = glob.glob(pattern)
            state_files.extend(found_files)
            if found_files:
                print(f"[Controller] Found {len(found_files)} files matching pattern: {pattern}", flush=True)
        
        print(f"[Controller] Found {len(state_files)} total client state files", flush=True)
        
        # ✅ 【调试】如果找不到文件，输出调试信息
        if len(state_files) == 0:
            print(f"[Controller] ⚠️  No client state files found!", flush=True)
            print(f"[Controller]   检查路径: {state_file_patterns}", flush=True)
            # ✅ 检查共享目录是否存在
            shared_dir = "/tmp/mininet_shared"
            if os.path.exists(shared_dir):
                try:
                    dir_contents = os.listdir(shared_dir)
                    print(f"[Controller]   共享目录存在，内容: {dir_contents[:10]}", flush=True)
                except:
                    print(f"[Controller]   共享目录存在但无法读取", flush=True)
            else:
                print(f"[Controller]   ⚠️  共享目录不存在: {shared_dir}", flush=True)
                print(f"[Controller]   建议：检查客户端是否已启动并写入状态文件", flush=True)
        
        # ✅ 【边缘计算架构】只读取当前relay负责的用户状态数据
        for file_path in sorted(state_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                host_id = data.get("host_id")
                if host_id is None or host_id < 1 or host_id > num_users:
                    continue
                
                # ✅ 【关键】只处理分配给当前relay的用户
                if host_id not in assigned_users:
                    continue  # 跳过不属于当前relay的用户
                
                # 提取用户指标
                user_metric = {
                    "host_id": host_id,
                    "bandwidth": float(data.get("throughput_mbps", 0.0)),
                    "device_score": float(data.get("device_score", 0.5)),
                    "network_type": data.get("network_type", user_network_types.get(host_id, 'wifi')),
                    "delay_ms": float(data.get("delay_ms", 0.0)),
                }
                user_metrics_dict[host_id] = user_metric
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"[Controller] ⚠️ Failed to read {file_path}: {e}", flush=True)
                continue
        
        print(f"[Controller] ✅ Collected {len(user_metrics_dict)} user metrics for {relay_name} (assigned: {len(assigned_users)})", flush=True)
        
        # 按网络类型分组用户
        users_by_net_type = {}
        for user_id in range(1, num_users + 1):
            # 优先使用状态文件中的网络类型
            if user_id in user_metrics_dict:
                net_type = user_metrics_dict[user_id]["network_type"]
            else:
                net_type = user_network_types.get(user_id, 'wifi')  # 默认使用wifi
            if net_type not in users_by_net_type:
                users_by_net_type[net_type] = []
            users_by_net_type[net_type].append(user_id)
        
        # 为每个网络类型的用户组使用对应的模型进行推理
        for net_type, user_ids in users_by_net_type.items():
            if net_type in models_dict:
                model_for_type = models_dict[net_type]
                print(f"[Controller] Using {net_type} model for {len(user_ids)} users", flush=True)
                
                # ✅ 【关键修改】构建真实状态向量（新格式：MAX_USERS * 3 + 1 = 301）
                # 注意：状态向量使用全局用户索引（0-99），不是按网络类型分组的索引
                # 格式：[user1_device_score, user1_bandwidth, user1_fov_score, user2_device_score, ..., global_capacity]
                user_features = np.zeros(MAX_USERS * 3, dtype=np.float32)
                all_throughputs = []
                
                for user_id in user_ids:
                    user_idx = user_id - 1  # 转换为0-based全局索引
                    if user_idx >= MAX_USERS:
                        continue
                    base_idx = user_idx * 3  # ✅ 每个用户3个特征
                    
                    # 从状态文件读取真实数据，如果没有则使用默认值
                    if user_id in user_metrics_dict:
                        user_metric = user_metrics_dict[user_id]
                        device_score = float(user_metric.get("device_score", 0.5))
                        bandwidth = float(user_metric.get("bandwidth", 0.0))
                    else:
                        device_score = device_scores[user_id - 1] if user_id - 1 < len(device_scores) else 0.5
                        bandwidth = 0.0
                    
                    # ✅ 【新功能】获取FOV分数
                    fov_score = float(fov_overlap_dict.get(user_id, 0.5))
                    # ✅ 归一化带宽（假设范围0-2000 Mbps）
                    normalized_bandwidth = bandwidth / 2000.0
                    
                    user_features[base_idx] = device_score
                    user_features[base_idx + 1] = normalized_bandwidth  # ✅ 每个用户的独立带宽（归一化）
                    user_features[base_idx + 2] = fov_score  # ✅ 每个用户的FOV分数
                    all_throughputs.append(bandwidth)
                
                # 全局特征：平均带宽（capacity），归一化（保持向后兼容）
                capacity = np.mean(all_throughputs) if all_throughputs else 0.0
                normalized_capacity = capacity / 2000.0  # 归一化到 [0, 1]
                
                # 组合状态向量：user_features (300) + normalized_capacity (1) = 301
                real_state = np.concatenate([user_features, [normalized_capacity]]).astype(np.float32)
                state_tensor = torch.FloatTensor(real_state).unsqueeze(0)  # shape: [1, 301]
                
                print(f"[Controller] ✅ Built real state for {net_type}: avg_bandwidth={capacity:.2f}Mbps, users={len(user_ids)}", flush=True)
                
                with torch.no_grad():
                    # ✅ 【PPO分组学习】检查模型输出格式（双head或单head）
                    model_output = model_for_type(state_tensor)
                    
                    if isinstance(model_output, tuple):
                        # 新模型：双head输出
                        group_logits, enhanced_logits = model_output
                        # group_logits: [1, num_users, num_groups]
                        # enhanced_logits: [1, num_users]
                        
                        gl = group_logits.squeeze(0)  # [num_users, num_groups]
                        el = enhanced_logits.squeeze(0)  # [num_users]
                        
                        # 分组：使用argmax（deterministic模式）
                        group_assignments = gl.argmax(dim=-1).cpu().numpy()  # [num_users], 范围[0..K-1]
                        
                        # Enhanced：使用sigmoid > 0.5（deterministic模式）
                        enhanced_decisions = (torch.sigmoid(el) > 0.5).float().cpu().numpy()  # [num_users], 范围[0,1]
                        
                        use_ppo_grouping = True
                    else:
                        # 旧模型：单head输出（向后兼容）
                        logits = model_output  # [1, MAX_USERS]
                        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [MAX_USERS]
                        
                        # 使用硬编码分组（向后兼容）
                        group_assignments = np.array([(uid - 1) % 3 for uid in user_ids], dtype=np.int32)
                        enhanced_decisions = (probs[:len(user_ids)] > 0.5).astype(float)
                        
                        use_ppo_grouping = False
                        print(f"[Controller] ⚠️ Using old model format (single head), falling back to hardcoded grouping", flush=True)
                
                # ✅ 【关键修改】PPO模型现在可以学习基于device_score、每个用户的带宽和FOV的分组
                device_scores_array = np.array([user_metrics_dict.get(uid, {}).get("device_score", 0.5) 
                                               for uid in user_ids])
                bandwidths_array = np.array([user_metrics_dict.get(uid, {}).get("bandwidth", 0.0) 
                                             for uid in user_ids])
                
                # ✅ 【策略】完全信任PPO模型的分组（如果有效），只在fallback时使用基于相似度的分组
                if not use_ppo_grouping or len(set(group_assignments)) < 2:
                    # PPO模型没有输出有效分组，使用基于相似度的分组作为fallback
                    print(f"[Controller] 🔄 PPO模型未输出有效分组，使用基于相似度的分组作为fallback（device_score + 带宽 + FOV）", flush=True)
                    # 获取FOV分组信息（如果可用）
                    fov_groups_for_users = None
                    if fov_group_manager and hasattr(fov_group_manager, 'get_groups'):
                        try:
                            fov_groups_for_users = fov_group_manager.get_groups()
                        except:
                            pass
                    group_assignments = group_by_similarity(
                        device_scores_array, bandwidths_array, fov_groups_for_users, num_groups=3
                    )
                else:
                    # ✅ PPO模型输出了有效分组，完全信任模型的学习结果
                    print(f"[Controller] ✅ 使用PPO模型的分组（基于device_score、每个用户的带宽和FOV，训练时学习）", flush=True)
                
                # ✅ 【Canonicalization】按组平均device_score重排组号（保持组内相似性）
                canonical_groups, remap_dict = canonicalize_groups(
                    group_assignments, device_scores_array, num_groups=3
                )
                if use_ppo_grouping:
                    print(f"[Controller] ✅ PPO grouping + Canonicalized: remap={remap_dict}", flush=True)
                else:
                    print(f"[Controller] ✅ Hardcoded grouping + Canonicalized: remap={remap_dict}", flush=True)
                
                # ✅ 【MD2G分组策略】使用PPO学习的分组（不再硬编码）
                # 策略：
                # 1. Base版本：组内统一（由客户端根据组内最低水平选择）
                # 2. Enhanced层：PPO学习是否订阅enhanced（0/1）
                # 3. Enhanced level：规则决定（enhanced=1时，根据bandwidth_headroom决定level 1/2）
                
                # 为这个网络类型的用户生成决策
                for idx, user_id in enumerate(user_ids):
                    user_idx = user_id - 1  # 转换为0-based索引
                    
                    # ✅ 【PPO分组学习】使用PPO输出的分组（不再是硬编码）
                    if user_idx < len(canonical_groups):
                        group_id = int(canonical_groups[user_idx])  # 范围[0..K-1]
                    else:
                        # 如果用户索引超出范围，使用fallback（按user_id取模）
                        group_id = (user_id - 1) % 3
                    
                    # ✅ 【PPO分组学习】使用PPO输出的enhanced决策
                    if user_idx < len(enhanced_decisions):
                        enhanced_decision = int(enhanced_decisions[user_idx])  # 0或1
                    else:
                        # 如果用户索引超出范围，使用启发式
                        enhanced_decision = 1 if (user_idx >= int(num_users * 0.3)) else 0
                    
                    # ✅ 【语义统一】规则决定enh_level（0/1/2），从pull_enhanced（0/1）映射
                    # pull_enhanced[u]: PPO的0/1输出（只要不要增强）
                    # enh_level[u]: 运行时0/1/2（由规则把pull_enhanced映射出来）
                    enh_level = 0
                    if enhanced_decision == 1:  # pull_enhanced=1
                        # 读取用户状态，计算bandwidth_headroom
                        user_throughput = 0.0
                        buffer_level = 0.0
                        state_file = f"/tmp/mininet_shared/client_h{user_id}_state.json"
                        try:
                            if os.path.exists(state_file):
                                with open(state_file, 'r') as f:
                                    state = json.load(f)
                                    user_throughput = state.get('throughput_mbps', 0.0)
                                    buffer_level = state.get('buffer_level_sec', 0.0)
                        except:
                            pass
                        
                        # 规则：根据bandwidth_headroom和buffer决定level
                        base_rate = 2.4  # 假设base3（实际应该从组内最低水平确定）
                        bandwidth_headroom = max(0, user_throughput - base_rate)
                        THRESHOLD_LEVEL2 = 2.0  # Mbps（base+enh1+enh2需要约1.8Mbps额外带宽）
                        MIN_BUFFER_LEVEL2 = 3.0  # 秒
                        
                        if bandwidth_headroom >= THRESHOLD_LEVEL2 and buffer_level >= MIN_BUFFER_LEVEL2:
                            enh_level = 2  # base+enh1+enh2
                        else:
                            enh_level = 1  # base+enh1
                    # pull_enhanced=0时，enh_level=0（只base）
                    
                    # ✅ 【新功能】添加FOV分组信息
                    decision_data = {
                        "pull_enhanced": bool(enhanced_decision),  # PPO输出：0/1
                        "enhanced_level": enh_level,  # 规则决定：0/1/2
                        "md2g_group_id": group_id,  # 范围[0..K-1]
                        "md2g_group_members": [uid for uid in range(1, num_users + 1) 
                                              if user_idx < len(canonical_groups) and 
                                              canonical_groups[user_idx] == group_id],
                        "md2g_group_size": len([uid for uid in range(1, num_users + 1) 
                                              if user_idx < len(canonical_groups) and 
                                              canonical_groups[user_idx] == group_id])
                    }
                    
                    # ✅ 【验证日志】记录每个用户的决策信息
                    if idx == 0 or (idx + 1) % 5 == 0:  # 每5个用户打印一次，避免日志过多
                        print(f"[Controller] User {user_id}: group_id={group_id}, pull_enhanced={enhanced_decision}, enh_level={enh_level}", flush=True)
                    
                    # 如果用户属于某个FOV组，添加分组信息
                    if fov_groups:
                        for group_idx, group in enumerate(fov_groups):
                            if str(user_id) in group:
                                decision_data["fov_group_id"] = group_idx
                                decision_data["fov_group_members"] = group
                                decision_data["fov_group_size"] = len(group)
                                break
                    
                    decisions[str(user_id)] = decision_data
            else:
                # 如果没有对应网络类型的模型，使用启发式（每个用户独立决策enhanced）
                print(f"[Controller] ⚠️ No model for {net_type}, using heuristic for {len(user_ids)} users", flush=True)
                
                # ✅ 【MD2G分组策略】Enhanced层：每个用户独立决策（不再要求组内统一）
                for user_id in user_ids:
                    user_idx = user_id - 1
                    group_id = (user_id - 1) % 3 + 1
                    
                    # 启发式：如果用户ID >= 30%，则允许enhanced
                    pull_enhanced = (user_idx >= int(num_users * 0.3))
                    
                    # ✅ 【语义统一】规则决定enh_level（0/1/2），从pull_enhanced（0/1）映射
                    enh_level = 0
                    if pull_enhanced:
                        # 读取用户状态，计算bandwidth_headroom
                        user_throughput = 0.0
                        state_file = f"/tmp/mininet_shared/client_h{user_id}_state.json"
                        try:
                            if os.path.exists(state_file):
                                with open(state_file, 'r') as f:
                                    state = json.load(f)
                                    user_throughput = state.get('throughput_mbps', 0.0)
                        except:
                            pass
                        
                        base_rate = 2.4  # 假设base3
                        bandwidth_headroom = max(0, user_throughput - base_rate)
                        if bandwidth_headroom >= 2.0:  # THRESHOLD_LEVEL2
                            enh_level = 2
                        else:
                            enh_level = 1
                    # pull_enhanced=0时，enh_level=0
                    
                    # ✅ 【修复】group_id范围应该是[0..K-1]，不是[1..K]
                    group_id_0based = (user_id - 1) % 3  # 直接计算0-based
                    group_members = [uid for uid in range(1, num_users + 1) if (uid - 1) % 3 == group_id_0based]
                    
                    decisions[str(user_id)] = {
                        "pull_enhanced": bool(pull_enhanced),  # PPO输出：0/1
                        "enhanced_level": enh_level,  # 规则决定：0/1/2
                        "md2g_group_id": group_id_0based,  # 范围[0..K-1]
                        "md2g_group_members": group_members,
                        "md2g_group_size": len(group_members)
                    }
    elif model and strategy != 'rolling':
        # === 单网络场景：使用单个模型（MD2G策略） ===
        # ✅ 【关键修复】从真实客户端状态文件读取数据，而不是使用dummy_state
        MAX_USERS = 100  # 训练时的最大用户数
        state_dim = MAX_USERS * 3 + 1  # 301
        
        # ✅ 【修复】读取真实客户端状态数据
        import glob
        user_metrics = []
        
        # 尝试从两个位置读取状态文件
        state_file_patterns = [
            "/tmp/mininet_shared/client_h*_state.json",
            "/tmp/client_h*_state.json"
        ]
        
        state_files = []
        for pattern in state_file_patterns:
            state_files.extend(glob.glob(pattern))
        
        print(f"[Controller] Found {len(state_files)} client state files", flush=True)
        
        # ✅ 【边缘计算架构】只读取当前relay负责的用户状态数据
        for file_path in sorted(state_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                host_id = data.get("host_id")
                if host_id is None or host_id < 1 or host_id > num_users:
                    continue
                
                # ✅ 【关键】只处理分配给当前relay的用户
                if host_id not in assigned_users:
                    continue  # 跳过不属于当前relay的用户
                
                # 提取用户指标
                user_metric = {
                    "host_id": host_id,
                    "bandwidth": float(data.get("throughput_mbps", 0.0)),
                    "device_score": float(data.get("device_score", 0.5)),
                    "network_type": data.get("network_type", network_type if 'network_type' in locals() else "4g"),
                    "delay_ms": float(data.get("delay_ms", 0.0)),
                }
                user_metrics.append(user_metric)
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"[Controller] ⚠️ Failed to read {file_path}: {e}", flush=True)
                continue
        
        print(f"[Controller] ✅ Collected {len(user_metrics)} user metrics for {relay_name} (assigned: {len(assigned_users)})", flush=True)
        
        # 按 host_id 排序
        user_metrics.sort(key=lambda x: x["host_id"])
        print(f"[Controller] ✅ Collected {len(user_metrics)} user metrics", flush=True)
        
        # ✅ 【验证日志】统计分组信息（用于验证）
        if len(user_metrics) > 0:
            # 统计每组人数和group_base（需要先运行决策逻辑，这里先打印基本信息）
            print(f"[Controller] 📊 User metrics summary: {len(user_metrics)} users", flush=True)
        
        # ✅ 【关键修改】构建真实状态向量（新格式：MAX_USERS * 3 + 1 = 301）
        # 格式：[user1_device_score, user1_bandwidth, user1_fov_score, user2_device_score, ..., global_capacity]
        # 每个用户3个特征：device_score, bandwidth, fov_score（删除无用的priority和last_layer）
        user_features = np.zeros(MAX_USERS * 3, dtype=np.float32)
        all_throughputs = []
        
        for i, user in enumerate(user_metrics):
            if i >= MAX_USERS:
                break
            base_idx = i * 3  # ✅ 每个用户3个特征
            host_id = user.get("host_id", i + 1)
            device_score = float(user.get("device_score", 0.5))
            bandwidth = float(user.get("bandwidth", 0.0))
            # ✅ 【新功能】获取FOV分数（从fov_overlap_dict中获取）
            fov_score = float(fov_overlap_dict.get(host_id, 0.5))
            
            # ✅ 归一化带宽（假设范围0-2000 Mbps）
            normalized_bandwidth = bandwidth / 2000.0
            
            user_features[base_idx] = device_score
            user_features[base_idx + 1] = normalized_bandwidth  # ✅ 每个用户的独立带宽（归一化）
            user_features[base_idx + 2] = fov_score  # ✅ 每个用户的FOV分数
            all_throughputs.append(bandwidth)
        
        # 全局特征：平均带宽（capacity），归一化（保持向后兼容）
        capacity = np.mean(all_throughputs) if all_throughputs else 0.0
        normalized_capacity = capacity / 2000.0  # 归一化到 [0, 1]
        
        # 组合状态向量：user_features (300) + normalized_capacity (1) = 301
        real_state = np.concatenate([user_features, [normalized_capacity]]).astype(np.float32)
        
        # 如果用户数不足，使用默认值填充
        if len(user_metrics) == 0:
            print(f"[Controller] ⚠️ No user metrics found, using default values", flush=True)
            # 使用设备分数、带宽和FOV作为fallback
            for i in range(min(num_users, MAX_USERS)):
                base_idx = i * 3  # ✅ 每个用户3个特征
                user_id = i + 1
                device_s = device_scores[i] if i < len(device_scores) else 0.5
                bandwidth_default = 10.0  # 默认带宽10 Mbps
                fov_score_default = float(fov_overlap_dict.get(user_id, 0.5))
                user_features[base_idx] = device_s  # device_score
                user_features[base_idx + 1] = bandwidth_default / 2000.0  # ✅ 归一化带宽
                user_features[base_idx + 2] = fov_score_default  # ✅ FOV分数
            real_state = np.concatenate([user_features, [0.5]]).astype(np.float32)
        
        # 转换为torch tensor
        state_tensor = torch.FloatTensor(real_state).unsqueeze(0)  # shape: [1, 301]
        
        print(f"[Controller] ✅ Built real state vector: shape={state_tensor.shape}, "
              f"avg_bandwidth={capacity:.2f}Mbps, users={len(user_metrics)}", flush=True)
        print(f"[Controller] ✅ 状态向量包含：每个用户的device_score、独立带宽和FOV分数（已删除无用的priority和last_layer）", flush=True)
        
        with torch.no_grad():
            # ✅ 【PPO分组学习】检查模型输出格式
            model_output = model(state_tensor)
            
            if isinstance(model_output, tuple):
                # 新模型：双head输出
                group_logits, enhanced_logits = model_output
                gl = group_logits.squeeze(0)  # [num_users, num_groups]
                el = enhanced_logits.squeeze(0)  # [num_users]
                
                group_assignments = gl.argmax(dim=-1).cpu().numpy()  # [num_users]
                enhanced_decisions = (torch.sigmoid(el) > 0.5).float().cpu().numpy()  # [num_users]
                
                use_ppo_grouping = True
            else:
                # 旧模型：单head输出（向后兼容）
                logits = model_output
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                group_assignments = np.array([(i) % 3 for i in range(num_users)], dtype=np.int32)
                enhanced_decisions = (probs[:num_users] > 0.5).astype(float)
                use_ppo_grouping = False
        
        # ✅ 【关键修复】只使用当前relay的用户数，而不是MAX_USERS
        device_scores_array = np.array([user_metrics[i].get("device_score", 0.5) 
                                       for i in range(len(user_metrics))])
        bandwidths_array = np.array([user_metrics[i].get("bandwidth", 0.0) 
                                     for i in range(len(user_metrics))])
        
        # ✅ 【关键修复】只使用前len(user_metrics)个group_assignments，避免索引越界
        valid_group_assignments = group_assignments[:len(user_metrics)]
        
        # ✅ 【关键修改】PPO模型训练时的状态向量现在包含：
        #   - 每个用户的device_score ✅
        #   - 每个用户的独立带宽 ✅
        #   - 每个用户的FOV分数 ✅
        #   因此PPO模型可以学习基于device_score、带宽和FOV的分组
        #   
        # ✅ 【策略】完全信任PPO模型的分组（基于device_score、带宽和FOV），
        #   只在PPO模型没有输出有效分组时才使用基于相似度的分组作为fallback
        if not use_ppo_grouping or len(set(valid_group_assignments)) < 2:
            # PPO模型没有输出有效分组，使用基于相似度的分组（device_score + 带宽 + FOV）作为fallback
            print(f"[Controller] 🔄 PPO模型未输出有效分组，使用基于相似度的分组作为fallback（device_score + 带宽 + FOV）", flush=True)
            valid_group_assignments = group_by_similarity(
                device_scores_array, bandwidths_array, fov_groups, num_groups=3
            )
        else:
            # ✅ PPO模型输出了有效分组，完全信任模型的学习结果
            #   模型在训练时已经学习了基于device_score、每个用户的带宽和FOV的分组策略
            print(f"[Controller] ✅ 使用PPO模型的分组（基于device_score、每个用户的带宽和FOV，训练时学习）", flush=True)
            # 保持PPO模型的分组输出，不替换
        
        # ✅ 【Canonicalization】按组平均device_score重排组号（保持组内相似性）
        canonical_groups, remap_dict = canonicalize_groups(
            valid_group_assignments, device_scores_array, num_groups=3
        )
        
        # ✅ 【PPO分组学习】为每个用户生成决策
        for i in range(num_users):
            user_id = i + 1
            
            # ✅ 使用PPO输出的分组（不再是硬编码）
            if i < len(canonical_groups):
                group_id = int(canonical_groups[i])  # 范围[0..K-1]
            else:
                group_id = i % 3  # fallback
            
            # ✅ 【语义统一】使用PPO输出的pull_enhanced（0/1）
            if i < len(enhanced_decisions):
                pull_enhanced = int(enhanced_decisions[i])  # PPO输出：0/1
            else:
                pull_enhanced = 1 if (i >= int(num_users * 0.3)) else 0
            
            # ✅ 【规则决定enh_level】从pull_enhanced映射到enh_level（0/1/2）
            enh_level = 0
            if pull_enhanced == 1:
                user_throughput = user_metrics[i].get("bandwidth", 0.0) if i < len(user_metrics) else 0.0
                base_rate = 2.4  # 假设base3
                bandwidth_headroom = max(0, user_throughput - base_rate)
                if bandwidth_headroom >= 2.0:  # THRESHOLD_LEVEL2
                    enh_level = 2
                else:
                    enh_level = 1
            # pull_enhanced=0时，enh_level=0
            
            decisions[str(user_id)] = {
                "pull_enhanced": bool(pull_enhanced),  # PPO输出：0/1
                "enhanced_level": enh_level,  # 规则决定：0/1/2
                "md2g_group_id": group_id,
                "md2g_group_members": [uid for uid in range(1, num_users + 1) 
                                      if (uid - 1) < len(canonical_groups) and 
                                      canonical_groups[uid - 1] == group_id],
                "md2g_group_size": len([uid for uid in range(1, num_users + 1) 
                                       if (uid - 1) < len(canonical_groups) and 
                                       canonical_groups[uid - 1] == group_id])
            }
            
    elif strategy == 'rolling':
        # === Rolling策略：在服务端（r1/r2）使用SC-DDQN模型做决策 ===
        # ✅ 【关键修改】Rolling策略现在在服务端部署SC-DDQN模型，与MD2G策略保持一致
        print(f"[Controller] ✅ [Rolling] Using SC-DDQN model for service-side decision making", flush=True)
        
        # ✅ 【关键修复】读取用户状态数据（与MD2G策略保持一致）
        import glob
        user_metrics = []
        
        # 尝试从两个位置读取状态文件
        state_file_patterns = [
            "/tmp/mininet_shared/client_h*_state.json",
            "/tmp/client_h*_state.json"
        ]
        
        state_files = []
        for pattern in state_file_patterns:
            state_files.extend(glob.glob(pattern))
        
        print(f"[Controller] [Rolling] Found {len(state_files)} client state files", flush=True)
        
        # ✅ 【边缘计算架构】只读取当前relay负责的用户状态数据
        for file_path in sorted(state_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                host_id = data.get("host_id")
                if host_id is None or host_id < 1 or host_id > num_users:
                    continue
                
                # ✅ 【关键】只处理分配给当前relay的用户
                if host_id not in assigned_users:
                    continue  # 跳过不属于当前relay的用户
                
                # 提取用户指标
                user_metric = {
                    "host_id": host_id,
                    "bandwidth": float(data.get("throughput_mbps", 0.0)),
                    "device_score": float(data.get("device_score", 0.5)),
                    "network_type": data.get("network_type", network_type if 'network_type' in locals() else "4g"),
                    "delay_ms": float(data.get("delay_ms", 0.0)),
                }
                user_metrics.append(user_metric)
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"[Controller] ⚠️ [Rolling] Failed to read {file_path}: {e}", flush=True)
                continue
        
        print(f"[Controller] ✅ [Rolling] Collected {len(user_metrics)} user metrics for {relay_name} (assigned: {len(assigned_users)})", flush=True)
        
        if model is None or not hasattr(model, 'decide'):
            print(f"[Controller] ⚠️  [Rolling] SC-DDQN model not loaded, using fallback decisions", flush=True)
            # 回退：生成默认决策
            for i in range(num_users):
                user_id = i + 1
                decisions[str(user_id)] = {"pull_enhanced": False}
        else:
            # ✅ 使用SC-DDQN模型为每个用户做决策
            print(f"[Controller] ✅ [Rolling] Making decisions using SC-DDQN model for {num_users} users", flush=True)
            
            for i in range(num_users):
                user_id = i + 1
                
                # ✅ 只处理分配给当前relay的用户
                if user_id not in assigned_users:
                    # 不属于当前relay的用户，生成默认决策
                    decisions[str(user_id)] = {"pull_enhanced": False}
                    continue
                
                # 获取用户状态信息
                user_metric = next((u for u in user_metrics if u.get("host_id") == user_id), None)
                if user_metric is None:
                    # 没有状态信息，使用默认决策
                    decisions[str(user_id)] = {"pull_enhanced": False}
                    continue
                
                # ✅ 构造状态窗口（Rolling策略需要状态窗口）
                # 状态格式：[timestamp, bandwidth, qoe, device_score, ...]
                # 简化：使用当前状态构造一个状态窗口
                current_state = np.array([
                    time.time(),  # timestamp
                    user_metric.get("bandwidth", 10.0),  # bandwidth
                    0.5,  # qoe (简化)
                    user_metric.get("device_score", 0.5),  # device_score
                    0.0,  # buffer (简化)
                ])
                
                # 构造状态窗口（Rolling策略需要window_size=3的历史状态）
                state_window = np.array([current_state] * 3)  # 简化：使用当前状态重复3次
                
                try:
                    # ✅ 使用SC-DDQN模型做决策
                    decision_result = model.decide(state_window)
                    # decision_result: 0=Base, 1=Enhanced
                    pull_enhanced = bool(decision_result == 1)
                    
                    # ✅ 【规则决定enh_level】从decision_result映射到enh_level（0/1/2）
                    enh_level = 0
                    if pull_enhanced:
                        user_throughput = user_metric.get("bandwidth", 0.0)
                        base_rate = 2.4  # 假设base3
                        bandwidth_headroom = max(0, user_throughput - base_rate)
                        if bandwidth_headroom >= 2.0:  # THRESHOLD_LEVEL2
                            enh_level = 2
                        else:
                            enh_level = 1
                    
                    decisions[str(user_id)] = {
                        "pull_enhanced": pull_enhanced,
                        "enhanced_level": enh_level,
                    }
                except Exception as e:
                    print(f"[Controller] ⚠️  [Rolling] Failed to make decision for user {user_id}: {e}", flush=True)
                    # 回退：默认Base
                    decisions[str(user_id)] = {"pull_enhanced": False}
            
            print(f"[Controller] ✅ [Rolling] Generated decisions for {len([d for d in decisions.values() if d.get('pull_enhanced')])} users with Enhanced", flush=True)
    else:
        # === 启发式/Fallback 模式 (模仿 PPO 行为) ===
        # 策略：前 30% 低分设备 -> Base，后 70% 高分设备 -> Enhanced
        # 这与你之前跑出来的数据趋势一致
        print("[Controller] Running Heuristic Logic (High Score -> Enhanced)")
        
        # ✅ 【MD2G分组策略】Enhanced层：每个用户独立决策（不再要求组内统一）
        threshold_idx = int(num_users * 0.3)  # 30% Base
        
        # ✅ 【Heuristic策略分组】构建组内成员映射（用于一致性）
        heuristic_group_members = {str(g): [] for g in range(3)}  # 0, 1, 2
        
        for i in range(num_users):
            user_id = i + 1
            # ✅ 【修复】group_id范围应该是[0..K-1]，不是[1..K]
            group_id = (user_id - 1) % 3  # 0, 1, 或 2（范围[0..K-1]）
            # ✅ 记录到组内成员映射
            heuristic_group_members[str(group_id)].append(user_id)
            
            # ✅ 【MD2G分组】Enhanced层：每个用户根据启发式独立决策
            # 如果用户ID >= 30%，则允许enhanced
            pull_enhanced = (i >= threshold_idx)
            
            # ✅ 【语义统一】规则决定enh_level（0/1/2），从pull_enhanced（0/1）映射
            enh_level = 0
            if pull_enhanced:
                # 读取用户状态，计算bandwidth_headroom
                # ✅ 【修复】Heuristic模式下，user_metrics可能不存在，从状态文件读取
                user_throughput = 0.0
                state_file = f"/tmp/mininet_shared/client_h{user_id}_state.json"
                try:
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                            user_throughput = state.get('throughput_mbps', 0.0)
                except:
                    pass
                
                base_rate = 2.4  # 假设base3
                bandwidth_headroom = max(0, user_throughput - base_rate)
                if bandwidth_headroom >= 2.0:  # THRESHOLD_LEVEL2
                    enh_level = 2
                else:
                    enh_level = 1
            # pull_enhanced=0时，enh_level=0
            
            # ✅ 【Heuristic策略分组】构建组内成员列表（基于硬编码逻辑）
            # 注意：Heuristic策略不使用PPO分组，所以使用硬编码的 (id % 3) 逻辑
            group_members = []
            for member_id in range(1, num_users + 1):
                if (member_id - 1) % 3 == group_id:  # group_id是0-based
                    group_members.append(member_id)
            
            # ✅ 【新功能】添加FOV分组信息
            decision_data = {
                "pull_enhanced": bool(pull_enhanced),  # PPO输出：0/1
                "enhanced_level": enh_level,  # 规则决定：0/1/2
                "md2g_group_id": group_id,  # 范围[0..K-1]
                "md2g_group_members": group_members,
                "md2g_group_size": len(group_members)
            }
            
            # 如果用户属于某个FOV组，添加分组信息
            if fov_groups:
                for group_idx, group in enumerate(fov_groups):
                    if str(user_id) in group:
                        decision_data["fov_group_id"] = group_idx
                        decision_data["fov_group_members"] = group
                        decision_data["fov_group_size"] = len(group)
                        break
            
            decisions[str(user_id)] = decision_data

    # 3. 写入文件
    # ✅ 【边缘计算架构】添加relay信息到决策文件
    # ✅ 【单一事实来源】构建全局的 group_members 映射表，方便客户端查找组内成员
    group_members_map = {}  # {group_id: [user_id1, user_id2, ...]}
    for user_id_str, user_decision in decisions.items():
        group_id = user_decision.get('md2g_group_id')
        if group_id is not None:
            group_id_str = str(group_id)
            if group_id_str not in group_members_map:
                group_members_map[group_id_str] = []
            group_members_map[group_id_str].append(int(user_id_str))
    
    output_data = {
        "relay_name": relay_name,
        "assigned_users": assigned_users,
        "decisions": decisions,
        "group_members": group_members_map,  # ✅ 【单一事实来源】全局组内成员映射表
        "timestamp": time.time()
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(decision_file) if os.path.dirname(decision_file) else ".", exist_ok=True)
    
    # ✅ 【关键修复】确保目录存在
    decision_dir = os.path.dirname(decision_file)
    if decision_dir and not os.path.exists(decision_dir):
        os.makedirs(decision_dir, exist_ok=True)
        print(f"[Controller] Created directory: {decision_dir}", flush=True)
    
    # ✅ 【验证日志】统计分组和enhanced决策信息（在写入文件前）
    group_stats = {}  # {group_id: {'count': 0, 'pull_enhanced_1': 0, 'enh_level_1': 0, 'enh_level_2': 0}}
    for user_id_str, user_decision in decisions.items():
        group_id = user_decision.get('md2g_group_id', 0)
        pull_enhanced = user_decision.get('pull_enhanced', False)
        enh_level = user_decision.get('enhanced_level', 0)
        
        if group_id not in group_stats:
            group_stats[group_id] = {'count': 0, 'pull_enhanced_1': 0, 'enh_level_1': 0, 'enh_level_2': 0}
        
        group_stats[group_id]['count'] += 1
        if pull_enhanced:
            group_stats[group_id]['pull_enhanced_1'] += 1
        if enh_level == 1:
            group_stats[group_id]['enh_level_1'] += 1
        elif enh_level == 2:
            group_stats[group_id]['enh_level_2'] += 1
    
    # 打印统计信息
    print(f"[Controller] 📊 分组统计 ({relay_name}):", flush=True)
    for group_id in sorted(group_stats.keys()):
        stats = group_stats[group_id]
        print(f"  Group {group_id}: {stats['count']} users, "
              f"pull_enhanced=1: {stats['pull_enhanced_1']}, "
              f"enh_level=1: {stats['enh_level_1']}, "
              f"enh_level=2: {stats['enh_level_2']}", flush=True)
    # 注意：group_base由客户端根据组内最低水平选择，Controller无法直接获取
    
    with open(decision_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"[Controller] ✅ Written decisions to {decision_file} (relay={relay_name}, users={len(assigned_users)})", flush=True)
    # 打印前几个决策供调试
    if decisions:
        sample_keys = list(decisions.keys())[:3]
        sample_str = ", ".join([f"{k}->{decisions[k]}" for k in sample_keys])
        print(f"[Controller] Sample decisions: {sample_str}", flush=True)
    
    # ✅ 【关键修复】验证文件确实已创建
    if os.path.exists(decision_file):
        file_size = os.path.getsize(decision_file)
        print(f"[Controller] ✅ Decision file verified: {file_size} bytes", flush=True)
    else:
        print(f"[Controller] ❌ ERROR: Decision file was not created!", flush=True)
        sys.exit(1)
    
    # ✅ 【关键修复】在日志最后输出决策模式信息
    print("=" * 60, flush=True)
    print("📊 MD2G Controller 决策模式总结", flush=True)
    print("=" * 60, flush=True)
    
    # 统计决策
    base_count = sum(1 for d in decisions.values() if not d['pull_enhanced'])
    enhanced_count = sum(1 for d in decisions.values() if d['pull_enhanced'])
    total = len(decisions)
    
    # 判断是否使用回退机制（30%阈值）
    threshold_idx = int(total * 0.3)
    is_fallback = (base_count == threshold_idx)
    
    # ✅ 【关键修复】检查单模型场景或多模型场景
    if strategy == 'rolling':
        # ✅ 【关键修改】Rolling策略：在服务端部署SC-DDQN模型做决策
        print(f"✅ 决策模式: SC-DDQN模型推理（服务端部署）", flush=True)
        print(f"   模型文件: {model_file if model_file else 'N/A'}", flush=True)
        print(f"   模型类型: SC-DDQN (Serial Cyclic Dueling DQN)", flush=True)
        print(f"   部署位置: r1/r2 Controller（与MD2G策略架构一致）", flush=True)
    elif models_dict:
        # 混合网络场景：使用多个模型
        print(f"✅ 决策模式: PPO模型推理 (多模型混合网络场景)", flush=True)
        print(f"   已加载模型数: {len(models_dict)}", flush=True)
        for net_type, m_file in model_files_dict.items():
            if net_type in models_dict:
                # ✅ 【修复】正确判断模型类型：根据文件名和实际加载的hidden_size
                m_file_str = str(m_file)
                # 新结构：ppo_actor_student_xxx.pth -> Student (128)
                # 旧结构：ppo_actor_xxx_general.pth -> Student (128)
                # Teacher模型：ppo_actor_teacher_xxx.pth 或 archive_teacher_512/ 目录
                if 'student' in m_file_str.lower() or 'general' in m_file_str.lower() or 'deploy_student_128' in m_file_str:
                    model_type = "Student (hidden_size=128)"
                elif 'teacher' in m_file_str.lower() or 'archive_teacher_512' in m_file_str:
                    model_type = "Teacher (hidden_size=512)"
                else:
                    # 回退：根据实际加载的模型hidden_size判断
                    # 从models_dict中获取实际模型，检查其hidden_size
                    actual_model = models_dict[net_type]
                    # 通过检查第一层权重维度来判断
                    first_layer_weight = list(actual_model.parameters())[0]
                    if first_layer_weight.shape[0] == 128:
                        model_type = "Student (hidden_size=128)"
                    elif first_layer_weight.shape[0] == 512:
                        model_type = "Teacher (hidden_size=512)"
                    else:
                        model_type = f"Model (hidden_size={first_layer_weight.shape[0]})"
                print(f"   - {net_type}: {os.path.basename(m_file)} ({model_type})", flush=True)
    elif model and model_file:
        # 单网络场景：使用单个模型
        print(f"✅ 决策模式: PPO模型推理", flush=True)
        print(f"   模型文件: {model_file}", flush=True)
        # ✅ 【修复】正确判断模型类型
        model_file_str = str(model_file)
        if 'student' in model_file_str.lower() or 'general' in model_file_str.lower() or 'deploy_student_128' in model_file_str:
            model_type = "Student (hidden_size=128)"
        elif 'teacher' in model_file_str.lower() or 'archive_teacher_512' in model_file_str:
            model_type = "Teacher (hidden_size=512)"
        else:
            # 回退：根据实际模型判断
            first_layer_weight = list(model.parameters())[0]
            if first_layer_weight.shape[0] == 128:
                model_type = "Student (hidden_size=128)"
            elif first_layer_weight.shape[0] == 512:
                model_type = "Teacher (hidden_size=512)"
            else:
                model_type = f"Model (hidden_size={first_layer_weight.shape[0]})"
        print(f"   模型类型: {model_type}", flush=True)
    else:
        print(f"⚠️  决策模式: 启发式回退机制 (Heuristic Fallback)", flush=True)
        if not HAS_TORCH:
            print(f"   原因: PyTorch未安装", flush=True)
        elif not model_file and not models_dict:
            print(f"   原因: 模型文件未找到", flush=True)
        elif model_file and not os.path.exists(model_file):
            print(f"   原因: 模型文件不存在: {model_file}", flush=True)
        else:
            print(f"   原因: 模型加载失败", flush=True)
    
    print(f"", flush=True)
    print(f"📈 决策统计:", flush=True)
    print(f"   总用户数: {total}", flush=True)
    print(f"   ✅ 所有用户都接收Base层（必须）", flush=True)
    print(f"   - 仅Base层用户: {base_count} ({base_count/total*100:.1f}%) - 只接收Base", flush=True)
    print(f"   - Base+Enhanced用户: {enhanced_count} ({enhanced_count/total*100:.1f}%) - 接收Base和Enhanced（叠加）", flush=True)
    print(f"", flush=True)
    
    if strategy == 'rolling':
        print(f"🔍 SC-DDQN模型推理:", flush=True)
        print(f"   ✅ 使用训练好的SC-DDQN模型进行决策（服务端部署）", flush=True)
        print(f"   决策基于模型输出的Q值（Dueling DQN）", flush=True)
        print(f"   模型部署在r1/r2上，与MD2G策略保持一致的服务端决策架构", flush=True)
        if model is None or not hasattr(model, 'decide'):
            print(f"   ⚠️  模型未加载，使用fallback决策", flush=True)
        else:
            print(f"   ✅ 模型已成功加载并运行", flush=True)
    elif is_fallback and not model and not models_dict:
        print(f"🔍 回退机制检测:", flush=True)
        print(f"   ✅ 确认使用30%阈值启发式策略", flush=True)
        print(f"   阈值位置: 前{threshold_idx}个用户 -> Base, 后{total-threshold_idx}个用户 -> Enhanced", flush=True)
    elif models_dict or model:
        print(f"🔍 PPO模型推理:", flush=True)
        if models_dict:
            print(f"   ✅ 使用训练好的多模型PPO进行决策 (混合网络场景)", flush=True)
            print(f"   每个网络类型使用对应的模型进行推理", flush=True)
        else:
            print(f"   ✅ 使用训练好的PPO模型进行决策", flush=True)
        print(f"   决策基于模型输出的概率分布", flush=True)
    else:
        print(f"⚠️  注意: 决策模式异常，请检查日志", flush=True)
    
    print("=" * 60, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision_file", type=str, required=True)
    parser.add_argument("--max_users", type=int, default=20)
    parser.add_argument("--model_path", type=str, default="./trained_models")
    parser.add_argument("--strategy", type=str, default="md2g")
    # 兼容其他参数
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--relay_name", type=str, default="r0", help="Relay名称（r0, r1, r2等）")
    parser.add_argument("--relay_ip", type=str, default=None, help="Relay IP地址")
    parser.add_argument("--federation", type=str, default="off")
    parser.add_argument("--network_type", type=str, default="wifi")  # ✅ 用于查找对应的学生模型
    parser.add_argument("--log_path", type=str, default=None)  # ✅ 用于读取用户网络类型映射文件
    
    # ✅ 【新功能】FOV分组相关参数
    parser.add_argument("--use_fov_grouping", action="store_true",
                       help="启用基于FOV的用户分组（Multicast优化）")
    parser.add_argument("--trace_csv", type=str, default=None,
                       help="Trace CSV文件路径（train_tiles.csv或test_tiles.csv）")
    parser.add_argument("--head_movement_csv", type=str, default=None,
                       help="Head_movement CSV文件路径（Head_movement_clean_1.csv）")
    parser.add_argument("--frame_id", type=int, default=0,
                       help="当前帧ID（用于从trace数据中加载用户Tile）")
    parser.add_argument("--fov_overlap_threshold", type=float, default=0.5,
                       help="FOV重叠阈值（0.0-1.0），值越高分组越严格")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="动态模式下的更新间隔（秒），默认0.5秒")
    
    args, unknown = parser.parse_known_args()
    run_controller(args)

