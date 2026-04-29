# QAS 模块说明

`nexq.algorithms.qas` 是量子构架搜索（Quantum Architecture Search, QAS）模块。

该模块用于在给定目标量子态时，自动搜索量子线路结构。当前已定义基于强化学习的量子态制备实现：`state_qas.py`。

## 已提供能力

- 目标：输入 `State`，输出 `Circuit`
- 方法：基于强化学习逐步向空线路追加量子门
- 奖励：当保真度超过阈值时给正奖励，否则每步惩罚
- 约束：禁止 `unitary` 门，动作门需要是 `nexq/core/gates.py` 支持的门

## 主要接口

- `StateQASConfig`：强化学习配置
- `state_to_circuit(state, config=None)`：根据目标 `State` 搜索并返回 `Circuit`

可通过以下方式导入：

```python
from nexq.algorithms.qas import StateQASConfig, state_to_circuit
```

## 依赖

运行强化学习搜索需要安装：

```bash
pip install gymnasium stable-baselines3
```

## 使用示例

```python
import numpy as np

from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.state import State
from nexq.algorithms.qas import StateQASConfig, state_to_circuit

# 1) 构造目标态（示例：2 比特 Bell 态）
backend = NumpyBackend()
target = np.zeros(4, dtype=np.complex64)
target[0] = 1 / np.sqrt(2)
target[3] = 1 / np.sqrt(2)
state = State.from_array(target, n_qubits=2, backend=backend)

# 2) 配置强化学习超参数
config = StateQASConfig(
    algo="ppo",              # ppo | a2c | dqn
    total_timesteps=10000,
    max_timesteps=20,
    fidelity_threshold=0.95,
    reward_penalty=0.01,
    seed=42,
    eval_episodes=10,
)

# 3) 搜索线路（输入 State，输出 Circuit）
circuit = state_to_circuit(state, config=config)

print(circuit)
print(circuit.show())
```

## 可选：自定义动作门集合

`StateQASConfig.action_gates` 支持自定义动作门集合。每个动作是一个门字典，格式与 `Circuit` 门定义一致。

注意：

- 不允许包含 `{"type": "unitary", ...}`
- 建议只使用 `gate_to_matrix` 可解析的门

示例：

```python
custom_actions = [
    {"type": "hadamard", "target_qubit": 0},
    {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
]

config = StateQASConfig(action_gates=custom_actions)
circuit = state_to_circuit(state, config=config)
```
