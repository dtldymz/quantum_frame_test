"""
nexq/algorithms/qas/state_qas_demo.py

演示程序：使用 state_qas 模块从 GHZ 态构建量子线路

GHZ态（Greenberger–Horne–Zeilinger state）是一种最大纠缠态：
    |GHZ⟩ = (1/√2)(|000⟩ + |111⟩)

流程：
1. 生成 3 量子比特 GHZ 态系数
2. 使用 State.from_array() 创建量子态（自动归一化）
3. 使用 state_to_circuit() 生成量子线路
4. 将量子线路导出为 OpenQASM 3.0 格式文件
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from nexq.core.state import State
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.io.qasm import circuit_to_qasm
from nexq.algorithms.qas import state_to_circuit, StateQASConfig
from nexq.algorithms.qas.state_qas import _default_action_gates


def main():
    print("=" * 60)
    print("State QAS Demo: GHZ State to Circuit")
    print("=" * 60)

    # Step 1: 生成 3 量子比特 GHZ 态系数
    print("\n[Step 1] 生成 3 量子比特 GHZ 态系数...")
    n_qubits = 3
    # GHZ态: (1/√2)(|000⟩ + |111⟩)
    ghz_state = np.zeros(2**n_qubits, dtype=np.complex64)
    ghz_state[0] = 1.0  # |000⟩ 系数
    ghz_state[7] = 1.0  # |111⟩ 系数
    # 注：from_array() 会自动归一化，所以这里不用除以√2

    print(f"GHZ 态系数数组（长度 {len(ghz_state)}）:")
    for i, c in enumerate(ghz_state):
        if c != 0:
            print(f"  [{i}] |{''.join(format(i, '03b'))}⟩ = {c:.4f}")

    # Step 2: 使用 State.from_array() 创建量子态（自动归一化）
    print(f"\n[Step 2] 创建 GHZ 量子态（自动归一化）...")
    backend = NumpyBackend()
    target_state = State.from_array(ghz_state, n_qubits=n_qubits, backend=backend)
    print(f"目标量子态已创建:")
    print(f"  形状: {target_state.data.shape}")
    print(f"  范数: {float(np.linalg.norm(target_state.data)):.6f}")
    print(f"  量子比特数: {target_state.n_qubits}")
    print(f"  态向量（归一化后）:")
    state_vec = target_state.to_numpy().flatten()
    for i, c in enumerate(state_vec):
        if abs(c) > 1e-6:
            print(f"    [{i}] |{''.join(format(i, '03b'))}⟩ = {c:.6f}")

    # Step 3: 使用 state_to_circuit() 生成量子线路
    print(f"\n[Step 3] 使用 state_to_circuit() 生成量子线路...")
    config = StateQASConfig(
        algo="ppo",  # 使用 PPO 算法
        total_timesteps=5000,  # RL 总训练步数
        max_timesteps=15,  # 最大电路深度（门数）
        fidelity_threshold=0.95,  # 保真度阈值
        reward_penalty=0.3,  # 线路深度惩罚
        seed=46,
    )
    print(f"QAS 配置:")
    print(f"  算法: {config.algo}")
    print(f"  总训练步数: {config.total_timesteps}")
    print(f"  最大电路深度: {config.max_timesteps}")
    print(f"  保真度阈值: {config.fidelity_threshold}")
    print(f"  深度惩罚: {config.reward_penalty}")
    action_gates = _default_action_gates(n_qubits)
    cnot_gates = [gate for gate in action_gates if gate["type"] in {"cx", "cnot"}]
    print(f"  动作空间门数: {len(action_gates)}")
    print(f"  CNOT 有序对数量: {len(cnot_gates)}")
    print("  CNOT 有序对列表:")
    for gate in cnot_gates:
        print(f"    q{gate['control_qubits'][0]} -> q{gate['target_qubit']}")

    circuit = state_to_circuit(target_state, config)
    print(f"量子线路已生成:")
    print(f"  门数: {len(circuit.gates)}")
    print(f"  量子比特数: {circuit.n_qubits}")
    print(f"  门列表:")
    for i, gate in enumerate(circuit.gates[:10]):  # 显示前 10 个门
        print(
            f"    [{i}] {gate['type']} on qubit(s) {gate.get('target_qubit', gate.get('control_qubit'))}"
        )
    if len(circuit.gates) > 10:
        print(f"    ... 以及 {len(circuit.gates) - 10} 个其他门")

    # Step 4: 将量子线路导出为 OpenQASM 3.0 文件
    print(f"\n[Step 4] 导出到 OpenQASM 3.0 格式...")
    output_file = Path(__file__).parent / "state_qas_demo_circuit.qasm"
    qasm_str = circuit_to_qasm(circuit, version="3.0")
    with open(output_file, "w") as f:
        f.write(qasm_str)
    print(f"✓ 已保存到: {output_file}")
    print(f"QASM 文件内容（前 1000 字符）:")
    print(qasm_str[:1000])
    if len(qasm_str) > 1000:
        print(f"... (共 {len(qasm_str)} 字符)")

    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
