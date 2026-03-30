# 🚀 Rolling SC-DDQN 训练指南

## 📋 概述

本训练脚本严格复现论文《Toward Optimal Real-Time Volumetric Video Streaming》中的 **SC-DDQN (Serial Cyclic Dueling DQN)** 算法，使用真实数据集进行训练。

## 📁 文件说明

1. **`sc_ddqn_model.py`**: SC-DDQN 模型核心文件
   - `LSTMPredictor`: LSTM预测模型（带宽和FoV预测）
   - `SCDDQN_Network`: Dueling DQN网络结构（761神经元）
   - `SCDDQN_Agent`: SC-DDQN智能体（经验回放、目标网络）

2. **`train_rolling.py`**: 训练脚本
   - 使用真实数据集（带宽、设备、FoV）
   - 实现串行循环决策（N*C步）
   - 使用真实QoE计算公式

## 🎯 核心特性

### 1. 严格遵循论文设计

- ✅ **SC-DDQN网络结构**: Dueling架构，761个隐藏神经元
- ✅ **串行循环决策**: N*C步串行决策（论文Algorithm 3）
- ✅ **LSTM预测**: 预测未来N步带宽和FoV
- ✅ **状态构造**: 遵循论文Eq.26 (S_m = [Y, x_m])

### 2. 使用真实数据集

- ✅ **带宽数据**: WiFi, 4G, 5G, Fiber Optic
- ✅ **设备数据**: Headset device performance
- ✅ **FoV数据**: 模拟FoV命中模式

### 3. 真实QoE计算公式

使用与 `dispatch_strategy_enhanced_unified.py` 相同的奖励函数：

```
R_t = λ_o*R_o + λ_q*R_q - λ_b*R_b - λ_l*R_l

其中：
- R_o: Grouping Efficiency (分组效率)
- R_q: User Perceived Quality (用户感知质量)
- R_b: Bandwidth Efficiency Penalty (带宽效率惩罚)
- R_l: Load Balance Penalty (负载均衡惩罚)
```

## 🚀 开始训练

### 步骤1: 检查数据集

确保以下数据集文件存在：

```bash
ls -lh ./datasets/
```

必需文件：
- `wifi_clean.csv`
- `4G-network-data_clean.csv` (或类似)
- `5g_network_data_clean.csv` (或 `5g_final_trace.csv`)
- `Optic_Bandwidth_clean_2.csv`
- `Headset device performance.csv`

### 步骤2: 运行训练

```bash
cd ./strategies
python3 train_rolling.py
```

### 步骤3: 监控训练进度

训练过程中会每100个episode打印一次进度：

```
Episode  100/20000 | Reward:  12.34 (Avg:  11.23) | Epsilon: 0.9950 | Speed: 45.2 iter/s
Episode  200/20000 | Reward:  13.45 (Avg:  12.34) | Epsilon: 0.9900 | Speed: 45.1 iter/s
...
```

每1000个episode自动保存模型。

## ⚙️ 配置参数

可以在 `train_rolling.py` 中调整以下参数：

```python
N_GOF = 5          # 滚动窗口大小 (论文中的N)
C_TILES = 8        # 每个GoF的切片数 (论文中的C)
L_LEVELS = 2       # 质量等级数 (Base=0, Enhanced=1)
STATE_DIM = 93     # 状态维度 (Y向量53维 + x_decisions 40维 = 93维)
MAX_EPISODES = 20000  # 训练轮数
TARGET_UPDATE_FREQ = 10  # 目标网络更新频率
```

## 📊 训练输出

训练完成后，模型会保存到：

```
./trained_models/sc_ddqn_rolling.pth
```

模型文件包含：
- `policy_net_state_dict`: 策略网络权重
- `target_net_state_dict`: 目标网络权重
- `episode`: 训练轮数
- `epsilon`: 最终探索率
- `avg_reward`: 平均奖励

## 🔄 使用训练好的模型

训练完成后，更新 `rolling_drl_strategy_v2_refined.py` 中的模型路径：

```python
model_path = "./trained_models/sc_ddqn_rolling.pth"
```

## 📈 训练建议

1. **快速测试**: 设置 `MAX_EPISODES = 2000`，验证训练流程
2. **正式训练**: 设置 `MAX_EPISODES = 20000` 或更多
3. **GPU加速**: 如果有GPU，训练会自动使用
4. **监控指标**: 关注平均奖励和epsilon衰减

## ⚠️ 注意事项

1. **数据集格式**: 确保CSV文件包含带宽列（自动检测多种列名）
2. **内存使用**: 经验回放池大小为200000，需要足够内存
3. **训练时间**: 20000个episode大约需要数小时（取决于硬件）

## 🔍 验证训练结果

训练完成后，可以：

1. 检查模型文件大小（应该约几MB）
2. 查看最终平均奖励（应该逐渐上升）
3. 检查epsilon值（应该接近0.05）

## 📝 论文对应关系

| 组件 | 论文章节 | 实现位置 |
|------|---------|---------|
| SC-DDQN网络 | V.B, Fig.3 | `sc_ddqn_model.py: SCDDQN_Network` |
| 训练算法 | V.A, Algo.2 | `sc_ddqn_model.py: SCDDQN_Agent` |
| 串行循环决策 | Algo.3 | `train_rolling.py: train()` |
| LSTM预测 | VI.A | `sc_ddqn_model.py: LSTMPredictor` |
| 状态构造 | V.A.1, Eq.26 | `train_rolling.py: RollingSimEnv._get_env_features()` |

---

*训练脚本基于论文严格实现，确保实验的可重复性和科学性。*

