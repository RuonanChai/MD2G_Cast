#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试训练设置
验证所有组件是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sc_ddqn_model import SCDDQN_Agent, SCDDQN_Network
from train_rolling import RealDatasetLoader, RollingSimEnv
import torch
import numpy as np

def test_model():
    """测试模型结构"""
    print("=" * 60)
    print("1. 测试 SC-DDQN 模型结构")
    print("=" * 60)
    
    state_dim = 93  # ✅ 修复：从64改为93，与train_rolling.py一致
    action_dim = 2
    
    # 测试网络
    net = SCDDQN_Network(state_dim, action_dim)
    test_state = torch.randn(1, state_dim)
    q_values = net(test_state)
    
    print(f"✅ 网络结构测试通过")
    print(f"   输入维度: {state_dim}")
    print(f"   输出维度: {q_values.shape[1]} (动作数: {action_dim})")
    print(f"   Q值范围: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    
    # 测试Agent
    agent = SCDDQN_Agent(state_dim, action_dim)
    test_state_np = np.random.randn(state_dim)
    action = agent.select_action(test_state_np, training=True)
    
    print(f"✅ Agent测试通过")
    print(f"   选择动作: {action} (0=Base, 1=Enhanced)")
    print(f"   当前Epsilon: {agent.epsilon:.4f}")
    
    return True

def test_dataset():
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("2. 测试数据集加载")
    print("=" * 60)
    
    loader = RealDatasetLoader()
    
    print(f"✅ 数据集加载成功")
    for net_type in ['wifi', '4g', '5g', 'fiber_optic']:
        if net_type in loader.bandwidth_data:
            data = loader.bandwidth_data[net_type]
            print(f"   {net_type:12s}: {len(data):6d} 条, "
                  f"范围: [{data.min():6.2f}, {data.max():6.2f}] Mbps")
        else:
            print(f"   {net_type:12s}: ⚠️  未加载")
    
    # 测试采样
    bw = loader.sample_bandwidth("wifi")
    device = loader.sample_device_score()
    print(f"\n✅ 数据采样测试")
    print(f"   WiFi带宽采样: {bw:.2f} Mbps")
    print(f"   设备分数采样: {device:.4f}")
    
    return True

def test_environment():
    """测试环境"""
    print("\n" + "=" * 60)
    print("3. 测试模拟环境")
    print("=" * 60)
    
    loader = RealDatasetLoader()
    env = RollingSimEnv(loader)
    
    # 重置环境
    y = env.reset()
    print(f"✅ 环境重置成功")
    print(f"   Y向量维度: {len(y)}")
    print(f"   当前网络: {env.current_network}")
    print(f"   带宽trace (前3步): {env.bw_trace[:3]}")
    print(f"   设备分数: {env.device_score:.4f}")
    
    # 测试几步
    total_reward = 0
    for step in range(5):
        action = np.random.randint(0, 2)  # 随机选择Base或Enhanced
        reward = env.step(action, step)
        total_reward += reward
        print(f"   Step {step}: action={action}, reward={reward:.4f}, buffer={env.buffer:.2f}s")
    
    print(f"\n✅ 环境step测试通过")
    print(f"   5步总奖励: {total_reward:.4f}")
    
    return True

def test_training_loop():
    """测试训练循环（单次迭代）"""
    print("\n" + "=" * 60)
    print("4. 测试训练循环（单次迭代）")
    print("=" * 60)
    
    loader = RealDatasetLoader()
    env = RollingSimEnv(loader)
    agent = SCDDQN_Agent(state_dim=93, action_dim=2)  # ✅ 修复：从64改为93
    
    # 模拟一个episode
    env_features_Y = env.reset()
    x_decisions = np.zeros(5 * 8)  # N_GOF * C_TILES
    
    total_reward = 0
    for step in range(5):  # 只测试5步
        state = np.concatenate((env_features_Y, x_decisions))
        action = agent.select_action(state, training=True)
        x_decisions[step] = (action + 1) / 2
        reward = env.step(action, step)
        total_reward += reward
        
        next_state = np.concatenate((env_features_Y, x_decisions))
        done = (step == 4)
        agent.store_transition(state, action, reward, next_state, done)
    
    # 尝试更新（如果经验池足够）
    if len(agent.memory) >= agent.batch_size:
        agent.update()
        print(f"✅ 网络更新成功")
    else:
        print(f"⚠️  经验池不足 ({len(agent.memory)}/{agent.batch_size})，跳过更新")
    
    print(f"✅ 训练循环测试通过")
    print(f"   总奖励: {total_reward:.4f}")
    print(f"   经验池大小: {len(agent.memory)}")
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 Rolling SC-DDQN 训练设置测试")
    print("=" * 60 + "\n")
    
    try:
        test_model()
        test_dataset()
        test_environment()
        test_training_loop()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！可以开始训练了。")
        print("=" * 60)
        print("\n运行以下命令开始训练：")
        print("  cd ./strategies")
        print("  python3 train_rolling.py")
        print()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

