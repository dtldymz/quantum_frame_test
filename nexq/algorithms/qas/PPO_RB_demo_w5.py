"""nexq/algorithms/qas/PPO_RB_demo_w5.py

演示：使用 Trust Region-based PPO with Rollback 制备 5 量子比特 W 态。

W 态：|W5⟩ = (|00001⟩ + |00010⟩ + |00100⟩ + |01000⟩ + |10000⟩) / √5
（各仅含一个比特为 1 的等权叠加，qubit 0 = LSB）
目标输入为密度矩阵 rho_target = |W5⟩⟨W5|。

W 态线路构造关键门集：
  - Ry(2·arcsin(1/√k))  k=2,3,4,5  (递归幅度分配角度)
  - CX(ctrl=q|0⟩, tgt=q') (零控制 CNOT，W 态递归结构必需)
  - CX(ctrl=q|1⟩, tgt=q') (标准 CNOT)
  - X 翻转门
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from nexq.algorithms.qas.PPO_RB import PPORollbackConfig, ppo_rb_qas
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.gates import gate_to_matrix
from nexq.core.io.qasm import circuit_to_qasm
from nexq.core.state import State


def build_w5_density() -> np.ndarray:
    """构造 5 比特 W 态目标密度矩阵。

    |W5⟩ = (|00001⟩+|00010⟩+|00100⟩+|01000⟩+|10000⟩) / √5
    各分量索引 = 2^q（qubit q 单独为 1）：1, 2, 4, 8, 16。
    """
    n = 5
    w = np.zeros((1 << n, 1), dtype=np.complex64)
    amp = 1.0 / math.sqrt(n)
    for q in range(n):
        w[1 << q, 0] = amp   # indices: 1, 2, 4, 8, 16
    return w @ w.conj().T


def circuit_density(circuit) -> np.ndarray:
    """将线路作用在 |00000⟩ 后得到输出密度矩阵。"""
    backend = NumpyBackend()
    state = State.zero_state(circuit.n_qubits, backend=backend)
    for gate in circuit.gates:
        gm = gate_to_matrix(gate, cir_qubits=circuit.n_qubits, backend=backend)
        state = state.evolve(gm)
    psi = state.to_numpy().reshape(-1, 1).astype(np.complex64)
    return psi @ psi.conj().T


def fidelity_pure_density(rho_target: np.ndarray, rho_pred: np.ndarray) -> float:
    """纯态目标下的重叠保真度 Tr(rho_target · rho_pred)。"""
    return float(np.real(np.trace(rho_target @ rho_pred)))


def build_w5_action_gates(n: int = 5):
    """构建 W 态专用动作集合。

    动作类型：
      - pauli_x (X 门，5 个)
      - ry，4 种关键角度 × 5 比特 = 20 个
      - cx，ctrl=1（标准 CNOT），n×(n-1) = 20 个
      - cx，ctrl=0（零控制 CNOT，W 态递归必需），n×(n-1) = 20 个
    合计：5 + 20 + 20 + 20 = 65 个动作。
    """
    # W 态递归构造的关键 Ry 角度：2·arcsin(1/√k)，k=2..n
    w_angles = [2.0 * math.asin(math.sqrt(1.0 / k)) for k in range(2, n + 1)]
    # 近似值：π/2 ≈ 1.5708, ≈ 1.2310, ≈ 1.0472, ≈ 0.9273

    gates = []

    # X 门（5 个）
    for q in range(n):
        gates.append({"type": "pauli_x", "target_qubit": q})

    # Ry 门（4 角度 × 5 比特 = 20 个）
    for q in range(n):
        for theta in w_angles:
            gates.append({"type": "ry", "parameter": theta, "target_qubit": q})

    # 标准 CNOT ctrl=|1⟩（20 个）
    for ctrl in range(n):
        for tgt in range(n):
            if ctrl != tgt:
                gates.append({
                    "type": "cx",
                    "target_qubit": tgt,
                    "control_qubits": [ctrl],
                    "control_states": [1],
                })

    # 零控制 CNOT ctrl=|0⟩（20 个，W 态递归结构必需）
    for ctrl in range(n):
        for tgt in range(n):
            if ctrl != tgt:
                gates.append({
                    "type": "cx",
                    "target_qubit": tgt,
                    "control_qubits": [ctrl],
                    "control_states": [0],
                })

    return gates


def main() -> None:
    print("=" * 68)
    print("PPO-RB QAS Demo: Prepare 5-Qubit W State")
    print("=" * 68)

    n = 5
    rho_target = build_w5_density()
    epsilon = 0.95

    print(f"目标态: |W{n}⟩ = (", end="")
    parts = [f"|{'0'*q + '1' + '0'*(n-1-q)}⟩" for q in range(n - 1, -1, -1)]
    print(" + ".join(parts) + f") / √{n}")
    print(f"目标密度矩阵: {rho_target.shape}, 非零元素: {np.count_nonzero(np.abs(rho_target) > 1e-6)}")

    w5_action_gates = build_w5_action_gates(n)
    print(f"动作集合: {len(w5_action_gates)} 个门")

    # 5 比特观测维度 = 2 × 32² = 2048，适当加大网络容量
    config = PPORollbackConfig(
        # 与伪代码一致的核心超参数
        learning_rate=0.002,
        gamma=0.99,
        epsilon_clip=0.2,
        epoch_num=4,
        rollback_alpha=-0.3,
        kl_threshold=0.03,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        # W 态高成功率配置
        action_gates=w5_action_gates,
        terminal_bonus=5.0,       # W 态难度更高，加大终止奖励
        gate_penalty=0.001,       # 较小惩罚，允许探索更长线路
        episode_num=4000,         # 5 比特搜索空间更大
        max_steps_per_episode=15, # W 态最优线路约 9 步，留余量
        update_timestep=256,
        hidden_dim=256,           # 观测维度 2048，加大隐层
        seed=42,
        log_interval=400,
    )

    print("[1/4] 开始训练 PPO-RB...")
    theta, circuit = ppo_rb_qas(rho_target, epsilon=epsilon, config=config)

    print("[2/4] 训练完成，输出策略参数摘要...")
    print(f"参数张量数量: {len(theta)}")
    total_params = sum(int(v.numel()) for v in theta.values())
    print(f"参数总数: {total_params}")

    print("[3/4] 评估最终线路与 W 态目标保真度...")
    rho_pred = circuit_density(circuit)
    fidelity = fidelity_pure_density(rho_target, rho_pred)

    print(f"线路量子比特数: {circuit.n_qubits}")
    print(f"线路门数: {len(circuit.gates)}")
    print(f"最终保真度: {fidelity:.6f}")

    print("门序列（前 20 个）:")
    for idx, gate in enumerate(circuit.gates[:20]):
        print(f"  [{idx:02d}] {gate}")
    if len(circuit.gates) > 20:
        print(f"  ... 其余 {len(circuit.gates) - 20} 个门")

    print("[4/4] 导出 QASM 3.0...")
    out_path = Path(__file__).parent / "ppo_rb_w5_circuit.qasm"
    qasm_str = circuit_to_qasm(circuit)
    out_path.write_text(qasm_str, encoding="utf-8")
    print(f"QASM 已保存: {out_path}")

    print()
    print("Demo 完成。")


if __name__ == "__main__":
    main()
