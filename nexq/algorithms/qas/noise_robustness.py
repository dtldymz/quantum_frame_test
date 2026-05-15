"""Noise-robustness metrics built from expressibility metrics and NoiseModel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ...channel.backends.base import Backend
from ...channel.noise.model import NoiseModel
from ...core.circuit import Circuit
from ...core.density import DensityMatrix
from ...core.gates import gate_to_matrix
from ...core.state import State
from .expressibility import (
    KL_Haar_divergence,
    MMD_relative,
    _count_total_parameters,
    _gaussian_kernel,
    _get_parametrized_gate_indices,
    _haar_population_distribution,
    _replace_circuit_parameters,
)


@dataclass
class NoiseSensitivityResult:
    """Result of noise sensitivity analysis for a quantum circuit."""

    circuit: Circuit
    n_qubits: int
    noise_model: Optional[NoiseModel]
    ideal_avg_fidelity: float
    noisy_avg_fidelity: float
    avg_fidelity_loss: float
    gate_type_sensitivity: Dict[str, float]
    n_gates_total: int
    n_gates_by_type: Dict[str, int]
    noise_strength: float

    def summary(self) -> str:
        lines = [
            "=== Noise Sensitivity Analysis ===",
            f"Circuit gates: {self.n_gates_total}",
            f"Ideal avg fidelity: {self.ideal_avg_fidelity:.4f}",
            f"Noisy avg fidelity: {self.noisy_avg_fidelity:.4f}",
            f"Fidelity loss: {self.avg_fidelity_loss:.4f}",
            f"Noise strength: {self.noise_strength:.4f}",
            "Gate-type sensitivity:",
        ]
        for gate_type, sensitivity in sorted(self.gate_type_sensitivity.items()):
            lines.append(f"  {gate_type}: {sensitivity:.4f}")
        return "\n".join(lines)


def _default_plus_state(n_qubits: int, backend: Backend) -> State:
    dim = 1 << n_qubits
    plus_data = np.ones(dim, dtype=np.complex64) / np.sqrt(dim)
    return State.from_array(plus_data, n_qubits=n_qubits, backend=backend)


def _evolve_density_gatewise(
    circuit: Circuit,
    backend: Backend,
    initial_state: State,
    noise_model: Optional[NoiseModel] = None,
) -> DensityMatrix:
    rho = initial_state.to_density_matrix()
    for gate in circuit.gates:
        gate_unitary = gate_to_matrix(gate, cir_qubits=circuit.n_qubits, backend=backend)
        rho = rho.evolve(gate_unitary)
        if noise_model is not None:
            rho = DensityMatrix(
                noise_model.apply(
                    rho.data,
                    n_qubits=circuit.n_qubits,
                    backend=backend,
                    gate_type=gate.get("type"),
                    gate=gate,
                ),
                circuit.n_qubits,
                backend,
            )
    return rho


def _sample_circuit_probability_vectors(
    circuit: Circuit,
    backend: Backend,
    noise_model: Optional[NoiseModel],
    n_samples: int = 500,
    initial_state: Optional[State] = None,
) -> np.ndarray:
    dim = 1 << circuit.n_qubits
    if initial_state is None:
        initial_state = _default_plus_state(circuit.n_qubits, backend)

    param_indices = _get_parametrized_gate_indices(circuit)
    total_params = _count_total_parameters(circuit, param_indices)
    samples = np.zeros((n_samples, dim), dtype=np.float64)

    for sample_index in range(n_samples):
        sampled_circuit = circuit
        if total_params > 0:
            params = np.random.uniform(0, 2 * np.pi, total_params)
            sampled_circuit = _replace_circuit_parameters(circuit, params)
        rho = _evolve_density_gatewise(sampled_circuit, backend, initial_state, noise_model=noise_model)
        samples[sample_index] = rho.probabilities()

    return samples


def _estimate_noise_strength(noise_model: NoiseModel) -> float:
    total_strength = 0.0
    n_channels = 0
    for rule in noise_model.rules:
        channel = rule.channel
        p = getattr(channel, "p", None)
        if p is not None:
            total_strength += p
            n_channels += 1
        gamma = getattr(channel, "gamma", None)
        if gamma is not None:
            total_strength += gamma
            n_channels += 1
    return total_strength / n_channels if n_channels > 0 else 0.01


def _apply_noise_to_fidelities(fidelities: np.ndarray, noise_strength: float = 0.01) -> np.ndarray:
    uniform_noise = np.random.uniform(size=fidelities.shape)
    noisy_fidelities = fidelities * (1 - noise_strength) + uniform_noise * noise_strength
    return np.clip(noisy_fidelities, 0, 1)


def KL_Haar_noisy(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    n_samples: int = 500,
    n_bins: int = 1000,
    noise_model: Optional[NoiseModel] = None,
    initial_state: Optional[State] = None,
    use_density_matrix: bool = True,
) -> float:
    """Compute KL divergence between noisy PQC output distribution and Haar distribution."""
    if backend is None:
        from ...channel.backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    if circuit.n_qubits > 10:
        raise ValueError(
            f"n_qubits={circuit.n_qubits} too large for expressibility evaluation. "
            "Max recommended: 10 qubits."
        )

    dim = 1 << circuit.n_qubits
    p_haar = _haar_population_distribution(dim, n_bins)
    noisy_probability_vectors = _sample_circuit_probability_vectors(
        circuit,
        backend,
        noise_model if use_density_matrix else None,
        n_samples=n_samples,
        initial_state=initial_state,
    )
    noisy_probs = np.max(noisy_probability_vectors, axis=1)
    if noise_model is not None and not use_density_matrix:
        noisy_probs = _apply_noise_to_fidelities(noisy_probs, _estimate_noise_strength(noise_model))

    hist_counts, _ = np.histogram(noisy_probs, bins=np.linspace(0, 1, n_bins + 1))
    p_pqc = hist_counts / n_samples
    epsilon = 1e-10
    p_pqc_safe = np.clip(p_pqc, epsilon, 1 - epsilon)
    p_haar_safe = np.clip(p_haar, epsilon, 1 - epsilon)
    p_pqc_safe = p_pqc_safe / np.sum(p_pqc_safe)
    p_haar_safe = p_haar_safe / np.sum(p_haar_safe)
    return float(np.sum(p_pqc_safe * np.log(p_pqc_safe / p_haar_safe)))


def MMD_noisy(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    n_samples: int = 500,
    sigma: float = 0.1,
    noise_model: Optional[NoiseModel] = None,
    initial_state: Optional[State] = None,
    use_density_matrix: bool = True,
) -> float:
    """Compute MMD between noisy PQC probability vectors and Haar-like samples."""
    if backend is None:
        from ...channel.backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    dim = 1 << circuit.n_qubits
    noisy_probs = _sample_circuit_probability_vectors(
        circuit,
        backend,
        noise_model if use_density_matrix else None,
        n_samples=n_samples,
        initial_state=initial_state,
    )
    expo = np.random.exponential(scale=1.0, size=(n_samples, dim))
    haar_samples = expo / np.sum(expo, axis=1, keepdims=True)
    k_xx = _gaussian_kernel(noisy_probs, noisy_probs, sigma)
    k_xz = _gaussian_kernel(noisy_probs, haar_samples.astype(np.float64), sigma)
    k_zz = _gaussian_kernel(haar_samples.astype(np.float64), haar_samples.astype(np.float64), sigma)
    mmd_sq = np.mean(k_xx) - 2 * np.mean(k_xz) + np.mean(k_zz)
    return float(max(0.0, mmd_sq))


def noise_sensitivity(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    noise_model: Optional[NoiseModel] = None,
    n_samples: int = 200,
    initial_state: Optional[State] = None,
) -> NoiseSensitivityResult:
    """Compare ideal and noisy execution and summarize gate-type sensitivity."""
    if backend is None:
        from ...channel.backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    n_qubits = circuit.n_qubits
    if initial_state is None:
        initial_state = _default_plus_state(n_qubits, backend)

    rho_ideal = _evolve_density_gatewise(circuit, backend, initial_state, noise_model=None)
    rho_noisy = _evolve_density_gatewise(circuit, backend, initial_state, noise_model=noise_model)
    probs_ideal = rho_ideal.probabilities()
    probs_noisy = rho_noisy.probabilities()
    fidelity_noisy = float(np.square(np.sum(np.sqrt(probs_ideal) * np.sqrt(probs_noisy))))
    avg_fidelity_loss = 1.0 - fidelity_noisy

    n_gates_by_type: Dict[str, int] = {}
    for gate in circuit.gates:
        gate_type = gate.get("type", "unknown")
        n_gates_by_type[gate_type] = n_gates_by_type.get(gate_type, 0) + 1

    return NoiseSensitivityResult(
        circuit=circuit,
        n_qubits=n_qubits,
        noise_model=noise_model,
        ideal_avg_fidelity=1.0,
        noisy_avg_fidelity=fidelity_noisy,
        avg_fidelity_loss=avg_fidelity_loss,
        gate_type_sensitivity=_analyze_gate_type_sensitivity(circuit, noise_model),
        n_gates_total=len(circuit.gates),
        n_gates_by_type=n_gates_by_type,
        noise_strength=_estimate_noise_strength(noise_model) if noise_model else 0.0,
    )


def _analyze_gate_type_sensitivity(circuit: Circuit, noise_model: Optional[NoiseModel]) -> Dict[str, float]:
    if noise_model is None:
        return {}
    sensitivity: Dict[str, float] = {}
    for gate_type in {gate.get("type", "unknown") for gate in circuit.gates}:
        count = sum(1 for gate in circuit.gates if gate.get("type", "unknown") == gate_type)
        if gate_type in {"cx", "cnot", "cy", "cz", "rzz", "swap", "crx", "cry", "crz"}:
            sensitivity[gate_type] = 0.02 * count
        elif gate_type in {"rx", "ry", "rz", "u2", "u3"}:
            sensitivity[gate_type] = 0.01 * count
        elif gate_type in {"hadamard", "s_gate", "t_gate"}:
            sensitivity[gate_type] = 0.005 * count
        else:
            sensitivity[gate_type] = 0.01 * count
    return sensitivity


def comparative_expressibility(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    n_samples: int = 500,
    noise_model: Optional[NoiseModel] = None,
    initial_state: Optional[State] = None,
) -> Dict[str, float]:
    """Compute ideal and noisy expressibility metrics for comparison."""
    if backend is None:
        from ...channel.backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    kl_ideal = KL_Haar_divergence(circuit, samples=n_samples, backend=backend)
    mmd_ideal = 1.0 - MMD_relative(circuit, samples=n_samples, backend=backend)
    result = {"kl_ideal": kl_ideal, "mmd_ideal": mmd_ideal}
    if noise_model is not None:
        kl_noisy = KL_Haar_noisy(circuit, backend=backend, n_samples=n_samples, noise_model=noise_model, initial_state=initial_state)
        mmd_noisy = MMD_noisy(circuit, backend=backend, n_samples=n_samples, noise_model=noise_model, initial_state=initial_state)
        result["kl_noisy"] = kl_noisy
        result["mmd_noisy"] = mmd_noisy
        result["noise_degradation_kl"] = (kl_noisy - kl_ideal) / kl_ideal if kl_ideal > 0 else 0.0
        result["noise_degradation_mmd"] = (mmd_noisy - mmd_ideal) / mmd_ideal if mmd_ideal > 0 else 0.0
    return result


def expressibility_score(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    n_samples: int = 500,
    noise_model: Optional[NoiseModel] = None,
    method: str = "auto",
    initial_state: Optional[State] = None,
) -> float:
    """Unified ideal/noisy expressibility distance interface for robustness analysis."""
    if backend is None:
        from ...channel.backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    if method == "auto":
        method = "kl" if (1 << circuit.n_qubits) <= 256 else "mmd"
    if noise_model is not None:
        if method == "kl":
            return KL_Haar_noisy(circuit, backend=backend, n_samples=n_samples, noise_model=noise_model, initial_state=initial_state)
        return MMD_noisy(circuit, backend=backend, n_samples=n_samples, noise_model=noise_model, initial_state=initial_state)
    if method == "kl":
        return KL_Haar_divergence(circuit, samples=n_samples, backend=backend)
    return 1.0 - MMD_relative(circuit, samples=n_samples, backend=backend)


__all__ = [
    "NoiseSensitivityResult",
    "KL_Haar_noisy",
    "MMD_noisy",
    "noise_sensitivity",
    "comparative_expressibility",
    "expressibility_score",
]