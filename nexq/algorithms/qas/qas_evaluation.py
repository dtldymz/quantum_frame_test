"""
nexq/algorithms/qas/qas_evaluation.py

Unified evaluation interface for quantum architecture search.

This module provides a comprehensive evaluation framework that combines:
- Expressibility metrics (ideal and noisy)
- Trainability assessment
- Noise robustness analysis
- Hardware efficiency evaluation
- Task performance metrics

Reference: Combined from expressibility.py and multi_objective_reward.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Sequence

import numpy as np

from ...core.circuit import Circuit
from ...core.state import State
from ...channel.backends.base import Backend
from ...channel.noise.model import NoiseModel

from .expressibility import (
    KL_Haar_divergence,
    MMD_relative,
)
from .noise_robustness import (
    KL_Haar_noisy,
    MMD_noisy,
    noise_sensitivity,
    NoiseSensitivityResult,
    comparative_expressibility,
    expressibility_score,
)
from .multi_objective_reward import (
    RewardWeights,
    ExpressibilityScore,
    TrainabilityScore,
    NoiseRobustnessScore,
    HardwareEfficiencyScore,
    MultiObjectiveReward,
)


# =============================================================================
# Data Classes
# =============================================================================

_TWO_QUBIT_GATE_TYPES = {
    "cx",
    "cnot",
    "cy",
    "cz",
    "crx",
    "cry",
    "crz",
    "swap",
    "toffoli",
    "ccnot",
    "rzz",
}


@dataclass
class MetricDefinition:
    """Metadata for one metric option inside a metric group."""

    name: str
    purpose: str
    status: str = "todo"
    active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "status": self.status,
            "active": self.active,
        }


@dataclass
class MetricGroupScore:
    """Score for one high-level objective group.

    Each group lists all known metric options, but only the active metric is
    executed in the first implementation pass.
    """

    name: str
    active_metric: str
    metrics: List[MetricDefinition]
    score: float
    raw_values: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active_metric": self.active_metric,
            "score": self.score,
            "raw_values": dict(self.raw_values),
            "notes": list(self.notes),
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


@dataclass
class ArchitectureSpec:
    """A candidate quantum architecture/ansatz to be scored by QAS."""

    name: str
    circuit: Circuit
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_gates(
        cls,
        name: str,
        gates: Sequence[Dict[str, Any]],
        n_qubits: int,
        backend: Optional[Backend] = None,
        **kwargs,
    ) -> "ArchitectureSpec":
        return cls(
            name=name,
            circuit=Circuit(*[dict(gate) for gate in gates], n_qubits=n_qubits, backend=backend),
            **kwargs,
        )

    @property
    def n_qubits(self) -> int:
        return int(self.circuit.n_qubits)

    @property
    def n_gates(self) -> int:
        return len(self.circuit.gates)

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(1 for gate in self.circuit.gates if gate.get("type") in _TWO_QUBIT_GATE_TYPES)

    @property
    def parameter_count(self) -> int:
        total = 0
        for gate in self.circuit.gates:
            if "parameter" not in gate:
                continue
            parameter = gate.get("parameter")
            if parameter is None:
                continue
            array = np.asarray(parameter)
            total += int(array.size) if array.shape else 1
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "n_qubits": self.n_qubits,
            "n_gates": self.n_gates,
            "n_parameters": self.parameter_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass
class ArchitectureScore:
    """Unified score report for one candidate architecture."""

    architecture: ArchitectureSpec
    expressibility: MetricGroupScore
    trainability: MetricGroupScore
    noise_robustness: MetricGroupScore
    hardware_efficiency: MetricGroupScore
    weights: Dict[str, float]
    weighted_score: float
    rank: Optional[int] = None

    def groups(self) -> Dict[str, MetricGroupScore]:
        return {
            "expressibility": self.expressibility,
            "trainability": self.trainability,
            "noise_robustness": self.noise_robustness,
            "hardware_efficiency": self.hardware_efficiency,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "architecture": self.architecture.to_dict(),
            "expressibility": self.expressibility.to_dict(),
            "trainability": self.trainability.to_dict(),
            "noise_robustness": self.noise_robustness.to_dict(),
            "hardware_efficiency": self.hardware_efficiency.to_dict(),
            "weights": dict(self.weights),
            "weighted_score": self.weighted_score,
        }

    def to_row(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.architecture.name,
            "n_qubits": self.architecture.n_qubits,
            "n_gates": self.architecture.n_gates,
            "n_parameters": self.architecture.parameter_count,
            "two_qubit_gate_count": self.architecture.two_qubit_gate_count,
            "expressibility": self.expressibility.score,
            "trainability": self.trainability.score,
            "noise_robustness": self.noise_robustness.score,
            "hardware_efficiency": self.hardware_efficiency.score,
            "weighted_score": self.weighted_score,
        }


def _metric_catalog(active_metrics: Optional[Dict[str, str]] = None) -> Dict[str, List[MetricDefinition]]:
    active_metrics = active_metrics or {}
    catalog = {
        "expressibility": [
            MetricDefinition("kl_haar", "KL divergence from the Haar fidelity distribution", "implemented"),
            MetricDefinition("mmd_relative", "MMD distance to Haar-like samples", "todo"),
            MetricDefinition("frame_potential", "Distance from unitary/state design behavior", "todo"),
            MetricDefinition("entangling_capability", "Average entanglement generated by sampled parameters", "todo"),
            MetricDefinition("transformer_predictor", "Learned surrogate for expressibility", "todo"),
        ],
        "trainability": [
            MetricDefinition("structure_proxy", "Depth, two-qubit density, and parameter-count heuristic", "implemented"),
            MetricDefinition("gradient_variance", "Sampled parameter-gradient variance", "todo"),
            MetricDefinition("gradient_norm", "Sampled gradient norm statistics", "todo"),
            MetricDefinition("barren_plateau_risk", "Risk estimate from expressibility, depth, and noise", "todo"),
        ],
        "noise_robustness": [
            MetricDefinition("ion_trap_error_budget_proxy", "Ion-trap error budget from the default noise configuration", "implemented"),
            MetricDefinition("ideal_noisy_score_gap", "Direct ideal-vs-noisy score gap", "todo"),
            MetricDefinition("noise_sensitivity", "Gate-type sensitivity under a NoiseModel", "todo"),
            MetricDefinition("per_source_ablation", "Noise-source ablation such as idle/crosstalk/twoq", "todo"),
        ],
        "hardware_efficiency": [
            MetricDefinition("native_depth_twoq_efficiency", "Native-gate, depth, and two-qubit-count heuristic", "implemented"),
            MetricDefinition("connectivity_penalty", "Penalty from non-native connectivity or routing cost", "todo"),
            MetricDefinition("calibrated_error_cost", "Hardware-calibrated error/resource cost", "todo"),
            MetricDefinition("latency_cost", "Gate-time or schedule-length cost", "todo"),
        ],
    }
    defaults = {
        "expressibility": "kl_haar",
        "trainability": "structure_proxy",
        "noise_robustness": "ion_trap_error_budget_proxy",
        "hardware_efficiency": "native_depth_twoq_efficiency",
    }
    for group_name, metrics in catalog.items():
        active_name = active_metrics.get(group_name, defaults[group_name])
        for metric in metrics:
            metric.active = metric.name == active_name
    return catalog


class ArchitectureEvaluator:
    """Evaluate architecture candidates with one active metric per objective group."""

    def __init__(
        self,
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
        weights: Optional[RewardWeights] = None,
        n_samples: int = 200,
        active_metrics: Optional[Dict[str, str]] = None,
    ):
        self.backend = backend
        self.noise_model = noise_model
        self.weights = weights or RewardWeights()
        self.n_samples = int(n_samples)
        self.active_metrics = active_metrics or {}

        self.expressibility_scorer = ExpressibilityScore(n_samples=self.n_samples)
        self.trainability_scorer = TrainabilityScore(n_samples=min(50, self.n_samples))
        self.noise_robustness_scorer = NoiseRobustnessScore(noise_model=noise_model, n_samples=self.n_samples)
        self.hardware_efficiency_scorer = HardwareEfficiencyScore()

    def _get_backend(self) -> Backend:
        if self.backend is None:
            from ...channel.backends.numpy_backend import NumpyBackend

            self.backend = NumpyBackend()
        return self.backend

    def _active_metric_name(self, group_name: str) -> str:
        catalog = _metric_catalog(self.active_metrics)
        return next(metric.name for metric in catalog[group_name] if metric.active)

    def _group_score(
        self,
        group_name: str,
        score: float,
        raw_values: Optional[Dict[str, Any]] = None,
        notes: Optional[List[str]] = None,
    ) -> MetricGroupScore:
        catalog = _metric_catalog(self.active_metrics)
        active = next(metric.name for metric in catalog[group_name] if metric.active)
        return MetricGroupScore(
            name=group_name,
            active_metric=active,
            metrics=catalog[group_name],
            score=float(np.clip(score, 0.0, 1.0)),
            raw_values=raw_values or {},
            notes=notes or [],
        )

    def _ion_trap_error_budget_proxy(self, circuit: Circuit) -> tuple[float, Dict[str, Any]]:
        from .ion_trap_noise_config import ONEQ_GATE_TYPES, TWOQ_GATE_TYPES, load_default_ion_trap_noise_config

        config = load_default_ion_trap_noise_config()
        resolved = config.resolved_parameters()

        oneq_gate_count = 0
        twoq_gate_count = 0
        measure_count = 0
        reset_count = 0
        for gate in circuit.gates:
            gate_type = str(gate.get("type", ""))
            if gate_type == "measure":
                measure_count += 1
            elif gate_type == "reset":
                reset_count += 1
            elif gate_type in TWOQ_GATE_TYPES or gate.get("control_qubits") or gate_type in {"swap", "rzz"}:
                twoq_gate_count += 1
            elif gate_type in ONEQ_GATE_TYPES or "target_qubit" in gate:
                oneq_gate_count += 1

        n_qubits = int(circuit.n_qubits)
        oneq_p = float(resolved.get("oneq_depol", 0.0) or 0.0) if resolved.get("enable_oneq_gate_noise", True) else 0.0
        twoq_p = float(resolved.get("twoq_depol", 0.0) or 0.0) if resolved.get("enable_twoq_gate_noise", True) else 0.0
        crosstalk_p = float(resolved.get("cross_talk", 0.0) or 0.0) if resolved.get("enable_crosstalk_noise", True) else 0.0
        measure_p = float(resolved.get("meas_bitflip", 0.0) or 0.0) if resolved.get("enable_measurement_noise", True) else 0.0
        reset_p = float(resolved.get("reset_bitflip", 0.0) or 0.0) if resolved.get("enable_initialization_noise", True) else 0.0
        if resolved.get("enable_idle_dephasing_noise", True):
            idle_oneq_p = config.idle_dephasing_probability(gate_family="oneq")
            idle_twoq_p = config.idle_dephasing_probability(gate_family="twoq")
        else:
            idle_oneq_p = 0.0
            idle_twoq_p = 0.0

        gate_error_budget = oneq_gate_count * oneq_p + twoq_gate_count * twoq_p
        idle_error_budget = (
            oneq_gate_count * max(n_qubits - 1, 0) * idle_oneq_p
            + twoq_gate_count * max(n_qubits - 2, 0) * idle_twoq_p
        )
        crosstalk_error_budget = (oneq_gate_count + twoq_gate_count) * n_qubits * crosstalk_p
        readout_reset_error_budget = measure_count * measure_p + reset_count * reset_p
        total_error_budget = gate_error_budget + idle_error_budget + crosstalk_error_budget + readout_reset_error_budget
        score = float(np.exp(-max(0.0, total_error_budget)))

        raw_values = {
            "oneq_gate_count": oneq_gate_count,
            "twoq_gate_count": twoq_gate_count,
            "measure_count": measure_count,
            "reset_count": reset_count,
            "oneq_depol": oneq_p,
            "twoq_depol": twoq_p,
            "cross_talk": crosstalk_p,
            "idle_oneq": idle_oneq_p,
            "idle_twoq": idle_twoq_p,
            "gate_error_budget": gate_error_budget,
            "idle_error_budget": idle_error_budget,
            "crosstalk_error_budget": crosstalk_error_budget,
            "readout_reset_error_budget": readout_reset_error_budget,
            "total_error_budget": total_error_budget,
        }
        return score, raw_values

    def evaluate(self, architecture: ArchitectureSpec) -> ArchitectureScore:
        backend = self._get_backend()
        circuit = architecture.circuit

        if self._active_metric_name("expressibility") != "kl_haar":
            raise NotImplementedError("当前仅实现 expressibility.kl_haar，其余表达能力指标已列为 todo")
        expr_score = self.expressibility_scorer.compute(circuit, backend)
        expressibility = self._group_score(
            "expressibility",
            expr_score,
            {"kl_haar_score": expr_score},
            ["Only the active expressibility metric is executed; other listed metrics are future work."],
        )

        if self._active_metric_name("trainability") != "structure_proxy":
            raise NotImplementedError("当前仅实现 trainability.structure_proxy，其余可训练性指标已列为 todo")
        train_score = self.trainability_scorer.compute(circuit, backend)
        trainability = self._group_score(
            "trainability",
            train_score,
            {"structure_proxy_score": train_score},
            ["Gradient-based trainability metrics are listed but not executed yet."],
        )

        if self._active_metric_name("noise_robustness") != "ion_trap_error_budget_proxy":
            raise NotImplementedError("当前仅实现 noise_robustness.ion_trap_error_budget_proxy，其余噪声鲁棒性指标已列为 todo")
        noise_score, noise_raw_values = self._ion_trap_error_budget_proxy(circuit)
        noise_robustness = self._group_score(
            "noise_robustness",
            noise_score,
            {"ion_trap_error_budget_proxy_score": noise_score, **noise_raw_values},
            ["Uses the default ion-trap noise configuration; direct noisy simulation and per-source ablations are future work."],
        )

        if self._active_metric_name("hardware_efficiency") != "native_depth_twoq_efficiency":
            raise NotImplementedError("当前仅实现 hardware_efficiency.native_depth_twoq_efficiency，其余硬件效率指标已列为 todo")
        hardware_score = self.hardware_efficiency_scorer.compute(circuit, backend)
        hardware_efficiency = self._group_score(
            "hardware_efficiency",
            hardware_score,
            {"native_depth_twoq_efficiency_score": hardware_score},
            ["Connectivity and calibrated hardware costs are listed but not executed yet."],
        )

        weight_dict = self.weights.to_dict()
        weighted_score = (
            weight_dict["expressibility"] * expressibility.score
            + weight_dict["trainability"] * trainability.score
            + weight_dict["noise_robustness"] * noise_robustness.score
            + weight_dict["hardware_efficiency"] * hardware_efficiency.score
        )
        return ArchitectureScore(
            architecture=architecture,
            expressibility=expressibility,
            trainability=trainability,
            noise_robustness=noise_robustness,
            hardware_efficiency=hardware_efficiency,
            weights=weight_dict,
            weighted_score=float(weighted_score),
        )

    def evaluate_many(self, architectures: Sequence[ArchitectureSpec]) -> List[ArchitectureScore]:
        scores = [self.evaluate(architecture) for architecture in architectures]
        scores.sort(key=lambda item: item.weighted_score, reverse=True)
        for rank, score in enumerate(scores, start=1):
            score.rank = rank
        return scores


def evaluate_architectures(
    architectures: Sequence[ArchitectureSpec],
    backend: Optional[Backend] = None,
    noise_model: Optional[NoiseModel] = None,
    weights: Optional[RewardWeights] = None,
    n_samples: int = 200,
    active_metrics: Optional[Dict[str, str]] = None,
) -> List[ArchitectureScore]:
    """Score and rank candidate architectures for noise-adaptive QAS."""
    evaluator = ArchitectureEvaluator(
        backend=backend,
        noise_model=noise_model,
        weights=weights,
        n_samples=n_samples,
        active_metrics=active_metrics,
    )
    return evaluator.evaluate_many(architectures)


@dataclass
class EvaluationResult:
    """
    Comprehensive evaluation result for a quantum circuit.

    Attributes:
        circuit: The evaluated circuit
        n_qubits: Number of qubits
        n_gates: Total gate count

        # Expressibility metrics
        kl_divergence: KL divergence from Haar (ideal)
        mmd_divergence: MMD from Haar (ideal)
        kl_noisy: KL divergence under noise
        mmd_noisy: MMD under noise

        # Individual objective scores
        expressibility_score: [0, 1] expressibility score
        trainability_score: [0, 1] trainability score
        noise_robustness_score: [0, 1] noise robustness score
        hardware_efficiency_score: [0, 1] hardware efficiency score

        # Combined metrics
        multi_objective_reward: Combined reward score

        # Noise sensitivity (if analyzed)
        noise_sensitivity_result: Detailed noise sensitivity analysis

        # Circuit properties
        depth_estimate: Estimated circuit depth
        two_qubit_gate_count: Number of 2-qubit gates
        single_qubit_gate_count: Number of single-qubit gates
    """
    circuit: Circuit
    n_qubits: int
    n_gates: int

    # Expressibility (ideal)
    kl_divergence: Optional[float] = None
    mmd_divergence: Optional[float] = None

    # Expressibility (noisy)
    kl_noisy: Optional[float] = None
    mmd_noisy: Optional[float] = None

    # Objective scores
    expressibility_score: Optional[float] = None
    trainability_score: Optional[float] = None
    noise_robustness_score: Optional[float] = None
    hardware_efficiency_score: Optional[float] = None

    # Combined reward
    multi_objective_reward: Optional[float] = None

    # Noise sensitivity
    noise_sensitivity_result: Optional[NoiseSensitivityResult] = None

    # Circuit properties
    depth_estimate: int = 0
    two_qubit_gate_count: int = 0
    single_qubit_gate_count: int = 0

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            "QAS Evaluation Result",
            "=" * 50,
            f"Circuit: {self.n_qubits} qubits, {self.n_gates} gates",
            f"Depth estimate: {self.depth_estimate}",
            "",
            "--- Expressibility (Ideal) ---",
            f"  KL divergence: {self.kl_divergence:.4f}" if self.kl_divergence else "  KL divergence: N/A",
            f"  MMD divergence: {self.mmd_divergence:.4f}" if self.mmd_divergence else "  MMD divergence: N/A",
            "",
            "--- Expressibility (Noisy) ---",
            f"  KL divergence (noisy): {self.kl_noisy:.4f}" if self.kl_noisy else "  KL divergence (noisy): N/A",
            f"  MMD divergence (noisy): {self.mmd_noisy:.4f}" if self.mmd_noisy else "  MMD divergence (noisy): N/A",
            "",
            "--- Objective Scores ---",
            f"  Expressibility: {self.expressibility_score:.4f}" if self.expressibility_score else "  Expressibility: N/A",
            f"  Trainability: {self.trainability_score:.4f}" if self.trainability_score else "  Trainability: N/A",
            f"  Noise robustness: {self.noise_robustness_score:.4f}" if self.noise_robustness_score else "  Noise robustness: N/A",
            f"  Hardware efficiency: {self.hardware_efficiency_score:.4f}" if self.hardware_efficiency_score else "  Hardware efficiency: N/A",
            "",
            "--- Combined ---",
            f"  Multi-objective reward: {self.multi_objective_reward:.4f}" if self.multi_objective_reward else "  Multi-objective reward: N/A",
            "=" * 50,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "n_qubits": self.n_qubits,
            "n_gates": self.n_gates,
            "depth_estimate": self.depth_estimate,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "single_qubit_gate_count": self.single_qubit_gate_count,
        }

        # Add optional fields if available
        if self.kl_divergence is not None:
            result["kl_divergence"] = self.kl_divergence
        if self.mmd_divergence is not None:
            result["mmd_divergence"] = self.mmd_divergence
        if self.kl_noisy is not None:
            result["kl_noisy"] = self.kl_noisy
        if self.mmd_noisy is not None:
            result["mmd_noisy"] = self.mmd_noisy

        if self.expressibility_score is not None:
            result["expressibility_score"] = self.expressibility_score
        if self.trainability_score is not None:
            result["trainability_score"] = self.trainability_score
        if self.noise_robustness_score is not None:
            result["noise_robustness_score"] = self.noise_robustness_score
        if self.hardware_efficiency_score is not None:
            result["hardware_efficiency_score"] = self.hardware_efficiency_score
        if self.multi_objective_reward is not None:
            result["multi_objective_reward"] = self.multi_objective_reward

        return result


# =============================================================================
# Evaluator Class
# =============================================================================

class QASEvaluator:
    """
    Unified evaluator for quantum architecture search.

    Provides comprehensive circuit evaluation with support for:
    - Expressibility metrics (KL, MMD)
    - Noise-aware evaluation
    - Multi-objective reward computation
    - Hardware efficiency assessment

    Example:
        evaluator = QASEvaluator(
            backend=backend,
            noise_model=noise_model,
            compute_expressibility=True,
            compute_trainability=True,
        )
        result = evaluator.evaluate(circuit)
        print(result.summary())
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
        n_samples: int = 300,
        compute_expressibility: bool = True,
        compute_trainability: bool = True,
        compute_noise_robustness: bool = True,
        compute_hardware_efficiency: bool = True,
        reward_weights: Optional[RewardWeights] = None,
    ):
        """
        Initialize evaluator.

        Args:
            backend: Backend for computation (default: NumpyBackend)
            noise_model: Noise model for noisy evaluation
            n_samples: Number of samples for Monte Carlo estimation
            compute_expressibility: Whether to compute expressibility metrics
            compute_trainability: Whether to compute trainability score
            compute_noise_robustness: Whether to compute noise robustness
            compute_hardware_efficiency: Whether to compute hardware efficiency
            reward_weights: Weights for multi-objective reward
        """
        self.backend = backend
        self.noise_model = noise_model
        self.n_samples = n_samples

        self.compute_expressibility = compute_expressibility
        self.compute_trainability = compute_trainability
        self.compute_noise_robustness = compute_noise_robustness
        self.compute_hardware_efficiency = compute_hardware_efficiency

        # Initialize scoring components
        if compute_expressibility:
            self.expressibility_scorer = ExpressibilityScore(n_samples=n_samples)

        if compute_trainability:
            self.trainability_scorer = TrainabilityScore(n_samples=min(50, n_samples))

        if compute_noise_robustness:
            self.noise_robustness_scorer = NoiseRobustnessScore(
                noise_model=noise_model,
                n_samples=n_samples,
            )

        if compute_hardware_efficiency:
            self.hardware_efficiency_scorer = HardwareEfficiencyScore()

        # Initialize multi-objective reward
        if reward_weights is None:
            reward_weights = RewardWeights(
                expressibility=0.25,
                trainability=0.25,
                noise_robustness=0.25,
                hardware_efficiency=0.25,
            )

        self.reward_weights = reward_weights
        self.multi_objective_reward = MultiObjectiveReward(
            weights=reward_weights,
            expressibility_score=self.expressibility_scorer if compute_expressibility else None,
            trainability_score=self.trainability_scorer if compute_trainability else None,
            noise_robustness_score=self.noise_robustness_scorer if compute_noise_robustness else None,
            hardware_efficiency_score=self.hardware_efficiency_scorer if compute_hardware_efficiency else None,
        )

    def _get_backend(self) -> Backend:
        """Get or create backend."""
        if self.backend is None:
            from ...channel.backends.numpy_backend import NumpyBackend
            self.backend = NumpyBackend()
        return self.backend

    def _count_gates(self, circuit: Circuit) -> tuple:
        """Count gates by type."""
        single_qubit = 0
        two_qubit = 0

        for gate in circuit.gates:
            gate_type = gate.get("type", "")
            if gate_type in {"cx", "cnot", "cy", "cz", "crx", "cry", "crz",
                            "swap", "toffoli", "ccnot"}:
                two_qubit += 1
            else:
                single_qubit += 1

        return single_qubit, two_qubit

    def evaluate(
        self,
        circuit: Circuit,
        fidelity: Optional[float] = None,
        include_noise_sensitivity: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a circuit comprehensively.

        Args:
            circuit: Circuit to evaluate
            fidelity: Task fidelity (if applicable)
            include_noise_sensitivity: Whether to include detailed noise sensitivity

        Returns:
            EvaluationResult with all computed metrics
        """
        backend = self._get_backend()
        n_qubits = circuit.n_qubits
        n_gates = len(circuit.gates)

        # Count gate types
        single_qubit, two_qubit = self._count_gates(circuit)

        # Initialize result
        result = EvaluationResult(
            circuit=circuit,
            n_qubits=n_qubits,
            n_gates=n_gates,
            depth_estimate=n_gates // max(1, n_qubits),
            two_qubit_gate_count=two_qubit,
            single_qubit_gate_count=single_qubit,
        )

        # Compute expressibility (ideal)
        if self.compute_expressibility:
            try:
                result.kl_divergence = KL_Haar_divergence(
                    circuit, samples=self.n_samples, backend=backend
                )
            except Exception:
                result.kl_divergence = None

            try:
                result.mmd_divergence = 1.0 - MMD_relative(
                    circuit, samples=self.n_samples, backend=backend
                )
            except Exception:
                result.mmd_divergence = None

            # Expressibility score
            result.expressibility_score = self.expressibility_scorer.compute(
                circuit, backend
            )

        # Compute expressibility (noisy)
        if self.compute_noise_robustness and self.noise_model is not None:
            try:
                result.kl_noisy = KL_Haar_noisy(
                    circuit, backend, self.n_samples,
                    noise_model=self.noise_model,
                    use_density_matrix=True,
                )
            except Exception:
                result.kl_noisy = None

            try:
                result.mmd_noisy = MMD_noisy(
                    circuit, backend, self.n_samples,
                    noise_model=self.noise_model,
                    use_density_matrix=True,
                )
            except Exception:
                result.mmd_noisy = None

        # Compute trainability
        if self.compute_trainability:
            result.trainability_score = self.trainability_scorer.compute(
                circuit, backend
            )

        # Compute noise robustness
        if self.compute_noise_robustness:
            result.noise_robustness_score = self.noise_robustness_scorer.compute(
                circuit, backend, self.noise_model
            )

            # Detailed noise sensitivity if requested
            if include_noise_sensitivity and self.noise_model is not None:
                try:
                    result.noise_sensitivity_result = noise_sensitivity(
                        circuit, backend, self.noise_model,
                        n_samples=min(100, self.n_samples)
                    )
                except Exception:
                    result.noise_sensitivity_result = None

        # Compute hardware efficiency
        if self.compute_hardware_efficiency:
            result.hardware_efficiency_score = self.hardware_efficiency_scorer.compute(
                circuit, backend
            )

        # Compute multi-objective reward
        result.multi_objective_reward = self.multi_objective_reward(
            circuit=circuit,
            backend=backend,
            fidelity=fidelity,
            noise_model=self.noise_model,
        )

        return result

    def evaluate_multi_objective(
        self,
        circuit: Circuit,
        fidelity: Optional[float] = None,
    ) -> tuple:
        """
        Evaluate circuit using multi-objective reward.

        Returns:
            Tuple of (reward, scores_dict)
        """
        backend = self._get_backend()

        reward = self.multi_objective_reward(
            circuit=circuit,
            backend=backend,
            fidelity=fidelity,
            noise_model=self.noise_model,
        )

        scores = self.multi_objective_reward.get_scores_detail(
            circuit, backend, self.noise_model
        )

        return reward, scores

    def compare_circuits(
        self,
        circuits: List[Circuit],
        fidelity: Optional[float] = None,
    ) -> List[EvaluationResult]:
        """
        Compare multiple circuits.

        Args:
            circuits: List of circuits to compare
            fidelity: Task fidelity for each circuit

        Returns:
            List of EvaluationResults sorted by multi_objective_reward
        """
        results = []
        for i, circuit in enumerate(circuits):
            fid = fidelity[i] if fidelity and i < len(fidelity) else None
            result = self.evaluate(circuit, fidelity=fid)
            results.append(result)

        # Sort by multi-objective reward (descending)
        results.sort(key=lambda r: r.multi_objective_reward or 0, reverse=True)

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_evaluate(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    noise_model: Optional[NoiseModel] = None,
    n_samples: int = 200,
) -> EvaluationResult:
    """
    Quick evaluation of a circuit with default settings.

    This is a convenience function for fast assessment.

    Args:
        circuit: Circuit to evaluate
        backend: Backend to use
        noise_model: Noise model (optional)
        n_samples: Number of samples

    Returns:
        EvaluationResult
    """
    evaluator = QASEvaluator(
        backend=backend,
        noise_model=noise_model,
        n_samples=n_samples,
        compute_expressibility=True,
        compute_trainability=True,
        compute_noise_robustness=noise_model is not None,
        compute_hardware_efficiency=True,
    )

    return evaluator.evaluate(circuit)


def noise_aware_reward(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    noise_model: Optional[NoiseModel] = None,
    weights: Optional[RewardWeights] = None,
    fidelity: Optional[float] = None,
) -> float:
    """
    Compute noise-aware reward for a circuit.

    Convenience function for quick reward computation.

    Args:
        circuit: Circuit to evaluate
        backend: Backend to use
        noise_model: Noise model
        weights: Reward weights
        fidelity: Task fidelity

    Returns:
        float: Reward value
    """
    if backend is None:
        from ...channel.backends.numpy_backend import NumpyBackend
        backend = NumpyBackend()

    if weights is None:
        weights = RewardWeights(
            expressibility=0.25,
            trainability=0.25,
            noise_robustness=0.25,
            hardware_efficiency=0.25,
        )

    reward_func = MultiObjectiveReward(
        weights=weights,
        expressibility_score=ExpressibilityScore(),
        trainability_score=TrainabilityScore(),
        noise_robustness_score=NoiseRobustnessScore(noise_model=noise_model),
        hardware_efficiency_score=HardwareEfficiencyScore(),
    )

    return reward_func(
        circuit=circuit,
        backend=backend,
        fidelity=fidelity,
        noise_model=noise_model,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MetricDefinition",
    "MetricGroupScore",
    "ArchitectureSpec",
    "ArchitectureScore",
    "ArchitectureEvaluator",
    "evaluate_architectures",
    "EvaluationResult",
    "QASEvaluator",
    "quick_evaluate",
    "noise_aware_reward",
]
