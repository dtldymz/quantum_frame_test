"""Reinforcement-learning based state-to-circuit synthesis for nexq.

This module follows the same environment design used in quantum-arch-search:
- Paper: arXiv:2104.07715v1
- Start from an empty circuit (no gate).
- Action = append one gate from a predefined gate set.
- Observation = expectation values of X/Y/Z on each qubit.
- Reward = -penalty until fidelity reaches threshold, then fidelity - penalty.

Main entry:
    state_to_circuit(state: State, config: Optional[StateQASConfig]) -> Circuit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.gates import gate_to_matrix
from ...core.circuit import Circuit
from ...core.state import State

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


def _normalize_state_vector(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.complex64).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm <= 0:
        raise ValueError("目标态范数必须大于 0")
    return flat / norm


def _default_action_gates(n_qubits: int) -> List[Dict[str, Any]]:
    """Mirror the default action set from quantum-arch-search.

    For each qubit i:
    - rz(pi/4) on i
    - X/Y/Z/H on i
    - CNOT(i -> (i+1) mod n)
    """
    gates: List[Dict[str, Any]] = []
    for idx in range(n_qubits):
        nxt = (idx + 1) % n_qubits
        gates.append({"type": "rz", "target_qubit": idx, "parameter": np.pi / 4.0})
        gates.append({"type": "pauli_x", "target_qubit": idx})
        gates.append({"type": "pauli_y", "target_qubit": idx})
        gates.append({"type": "pauli_z", "target_qubit": idx})
        gates.append({"type": "hadamard", "target_qubit": idx})
        if n_qubits > 1:
            gates.append(
                {
                    "type": "cx",
                    "target_qubit": nxt,
                    "control_qubits": [idx],
                    "control_states": [1],
                }
            )
    return gates


def _optimize_circuit_gates(gates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cancel adjacent pairs on same qubit: XX, YY, ZZ, HH."""
    cancellable_types = {"pauli_x", "pauli_y", "pauli_z", "hadamard"}
    optimized: List[Dict[str, Any]] = []

    for gate in gates:
        if not optimized:
            optimized.append(dict(gate))
            continue

        prev = optimized[-1]
        same_type = prev.get("type") == gate.get("type")
        same_qubit = prev.get("target_qubit") == gate.get("target_qubit")
        no_controls = (not prev.get("control_qubits")) and (not gate.get("control_qubits"))
        no_params = (prev.get("parameter") is None) and (gate.get("parameter") is None)

        if same_type and same_qubit and gate.get("type") in cancellable_types and no_controls and no_params:
            optimized.pop()
        else:
            optimized.append(dict(gate))

    return optimized


def _default_observables(n_qubits: int) -> List[Dict[str, Any]]:
    observables: List[Dict[str, Any]] = []
    for idx in range(n_qubits):
        observables.append({"type": "pauli_x", "target_qubit": idx})
        observables.append({"type": "pauli_y", "target_qubit": idx})
        observables.append({"type": "pauli_z", "target_qubit": idx})
    return observables


def _fidelity(pred: np.ndarray, target: np.ndarray) -> float:
    inner = np.vdot(pred, target)
    return float(np.real(np.conj(inner) * inner))


def _is_gate_supported(gate: Dict[str, Any]) -> bool:
    gtype = gate.get("type")
    if gtype == "unitary":
        return False

    # Validate by materializing matrix with gate_to_matrix.
    # This enforces "must be defined in nexq/core/gates.py".
    if gtype in {"swap", "rzz"}:
        n_qubits = max(int(gate["qubit_1"]), int(gate["qubit_2"])) + 1
    elif gtype in {"cx", "cnot", "cy", "cz", "crx", "cry", "crz", "toffoli", "ccnot"}:
        controls = [int(q) for q in gate["control_qubits"]]
        n_qubits = max([int(gate["target_qubit"])] + controls) + 1
    elif gtype in {"identity", "I"}:
        n_qubits = int(gate["n_qubits"])
    else:
        n_qubits = int(gate.get("target_qubit", 0)) + 1

    try:
        _ = gate_to_matrix(gate, cir_qubits=n_qubits, backend=None)
        return True
    except Exception:
        return False


@dataclass
class StateQASConfig:
    algo: str = "ppo"  # ppo | a2c | dqn
    total_timesteps: int = 10000
    max_timesteps: int = 20
    fidelity_threshold: float = 0.95
    reward_penalty: float = 0.01
    seed: int = 42
    action_gates: Optional[List[Dict[str, Any]]] = None


class QuantumStateSearchEnvCore:
    """Framework-agnostic environment core for state-to-circuit RL."""

    def __init__(
        self,
        target_state: State,
        action_gates: Sequence[Dict[str, Any]],
        observables: Sequence[Dict[str, Any]],
        fidelity_threshold: float,
        reward_penalty: float,
        max_timesteps: int,
    ):
        self.backend = target_state.backend
        self.n_qubits = target_state.n_qubits
        self.target = _normalize_state_vector(target_state.to_numpy())

        if len(self.target) != (1 << self.n_qubits):
            raise ValueError("目标态维度与 n_qubits 不一致")
        if fidelity_threshold <= 0 or fidelity_threshold > 1:
            raise ValueError("fidelity_threshold 必须在 (0, 1] 区间")
        if reward_penalty < 0:
            raise ValueError("reward_penalty 不能为负数")
        if max_timesteps <= 0:
            raise ValueError("max_timesteps 必须是正整数")

        checked_gates: List[Dict[str, Any]] = []
        for gate in action_gates:
            if gate.get("type") == "unitary":
                raise ValueError("action_gates 不能包含 unitary 门")
            if not _is_gate_supported(gate):
                raise ValueError(f"action_gates 包含不受支持门: {gate}")
            checked_gates.append(dict(gate))
        if not checked_gates:
            raise ValueError("action_gates 不能为空")

        self.action_gates = checked_gates
        self.observables = [dict(obs) for obs in observables]
        self.fidelity_threshold = float(fidelity_threshold)
        self.reward_penalty = float(reward_penalty)
        self.max_timesteps = int(max_timesteps)

        self._observable_mats = [
            gate_to_matrix(obs, cir_qubits=self.n_qubits, backend=self.backend)
            for obs in self.observables
        ]

        self.circuit_gates: List[Dict[str, Any]] = []
        self.current_timestep = 0

    @property
    def observation_shape(self) -> Tuple[int]:
        return (len(self.observables),)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.circuit_gates = []
        self.current_timestep = 0
        return self._get_obs()

    def _build_state(self) -> State:
        state = State.zero_state(self.n_qubits, backend=self.backend)
        for gate in self.circuit_gates:
            gm = gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend)
            state = state.evolve(gm)
        return state

    def _get_obs(self) -> np.ndarray:
        state = self._build_state()
        values = [state.expectation(op) for op in self._observable_mats]
        return np.asarray(values, dtype=np.float32)

    def _get_fidelity(self) -> float:
        state = self._build_state()
        pred = _normalize_state_vector(state.to_numpy())
        return _fidelity(pred, self.target)

    def step(self, action: int):
        if action < 0 or action >= len(self.action_gates):
            raise ValueError(f"非法 action 索引: {action}")

        self.circuit_gates.append(dict(self.action_gates[action]))
        self.current_timestep += 1

        observation = self._get_obs()
        fidelity = self._get_fidelity()

        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty

        done = (reward > 0.0) or (self.current_timestep >= self.max_timesteps)
        info = {
            "fidelity": fidelity,
            "gate_count": len(self.circuit_gates),
            "circuit": self.build_circuit(),
        }
        return observation, float(reward), bool(done), info

    def build_circuit(self) -> Circuit:
        return Circuit(*self.circuit_gates, n_qubits=self.n_qubits, backend=self.backend)


if gym is not None:

    class QuantumStateSearchGymEnv(gym.Env):
        """Gymnasium wrapper compatible with stable-baselines3."""

        metadata = {"render_modes": ["human"]}

        def __init__(self, core: QuantumStateSearchEnvCore):
            super().__init__()
            self.core = core
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.core.observation_shape,
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(len(self.core.action_gates))

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            del options
            obs = self.core.reset(seed=seed)
            return obs, {}

        def step(self, action: int):
            obs, reward, done, info = self.core.step(int(action))
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info


else:
    QuantumStateSearchGymEnv = None


def _get_sb3_algo(algo: str):
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:
        raise ImportError(
            "需要安装 stable-baselines3 才能使用强化学习搜索: pip install stable-baselines3"
        ) from exc

    name = algo.lower()
    if name == "ppo":
        return PPO
    if name == "a2c":
        return A2C
    if name == "dqn":
        return DQN
    raise ValueError("algo 仅支持 'ppo'、'a2c'、'dqn'")


def _make_core(target_state: State, config: StateQASConfig) -> QuantumStateSearchEnvCore:
    n_qubits = target_state.n_qubits
    action_gates = config.action_gates if config.action_gates is not None else _default_action_gates(n_qubits)
    observables = _default_observables(n_qubits)

    return QuantumStateSearchEnvCore(
        target_state=target_state,
        action_gates=action_gates,
        observables=observables,
        fidelity_threshold=config.fidelity_threshold,
        reward_penalty=config.reward_penalty,
        max_timesteps=config.max_timesteps,
    )


def _rollout_once(model, env) -> Dict[str, Any]:
    """Run one post-training episode, mirroring the original repo usage pattern."""
    obs, _ = env.reset()
    done = False
    last_info: Dict[str, Any] = {}

    while not done:
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        last_info = info

    return {
        "fidelity": float(last_info.get("fidelity", -1.0)),
        "circuit": last_info.get("circuit"),
    }


def state_to_circuit(
    state: State,
    config: Optional[StateQASConfig] = None,
) -> Circuit:
    """Use RL-based QAS to synthesize a circuit for the input target State.

    Args:
        state: 目标量子态 State 实例。
        config: RL 超参数配置。若为 None，使用默认配置。

    Returns:
        Circuit: 训练后评估得到的最佳线路。
    """
    if not isinstance(state, State):
        raise TypeError("state 必须是 nexq.circuit.state.State 实例")

    cfg = StateQASConfig() if config is None else config

    if gym is None or QuantumStateSearchGymEnv is None:
        raise ImportError("需要安装 gymnasium 才能运行 state_qas: pip install gymnasium")

    algo_cls = _get_sb3_algo(cfg.algo)

    train_env = QuantumStateSearchGymEnv(_make_core(state, cfg))
    model = algo_cls("MlpPolicy", train_env, verbose=0, seed=cfg.seed)
    model.learn(total_timesteps=cfg.total_timesteps)

    eval_env = QuantumStateSearchGymEnv(_make_core(state, cfg))
    eval_result = _rollout_once(model, eval_env)
    circuit = eval_result["circuit"]
    if circuit is None:
        circuit = _make_core(state, cfg).build_circuit()

    optimized_gates = _optimize_circuit_gates(circuit.gates)
    return Circuit(*optimized_gates, n_qubits=circuit.n_qubits, backend=circuit.backend)


__all__ = [
    "StateQASConfig",
    "QuantumStateSearchEnvCore",
    "QuantumStateSearchGymEnv",
    "state_to_circuit",
]
