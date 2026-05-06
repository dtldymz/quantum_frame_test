"""Demo: encode (1, 2, 3, 4) with three basic encoders."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Allow running this file directly from the repository root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nexq.encoder.basic import AmplitudeEncoder, AngleEncoder, BasisEncoder


def _angle_equivalent(a, b, atol=1e-6):
    """Check angular equivalence under 2*pi periodicity."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.allclose(diff, 0.0, atol=atol), diff


def demo_amplitude(data):
    encoder = AmplitudeEncoder()
    circuit, state = encoder.encode(data, cir="dict")
    decoded = encoder.decode(state)

    print("=== AmplitudeEncoder ===")
    print("n_qubits:", state.n_qubits)
    print("circuit gates:", circuit.gates)
    print("state:", state.format())
    print("decoded amplitudes(real):", np.round(decoded, 6))
    print()


def demo_angle(data):
    encoder = AngleEncoder()
    circuit, state = encoder.encode(data, cir="dict")
    decoded = encoder.decode(state)
    ok, diff = _angle_equivalent(np.asarray(data, dtype=float), decoded)

    print("=== AngleEncoder ===")
    print("n_qubits:", state.n_qubits)
    print("circuit gates:", circuit.gates)
    print("state:", state.format())
    print("decoded angles:", np.round(decoded, 6))
    print("2*pi periodic equivalence:", ok)
    print("wrapped angle diff:", np.round(diff, 6))
    print(
        "note: current decode uses marginal |1> probability inversion, "
        "which is not one-to-one for all angles."
    )
    print()


def demo_basis(data, repeat=False):
    encoder = BasisEncoder(repeat=repeat)
    circuit, state = encoder.encode(data, cir="dict")
    decoded = encoder.decode(state)
    probs = state.probabilities()

    print(f"=== BasisEncoder (repeat={repeat}) ===")
    print("input array:", data)
    print("n_qubits:", state.n_qubits)
    print("circuit gates:", circuit.gates)
    print("state:", state.format())
    print("probabilities:", np.round(probs, 6))
    print("decoded bits:", decoded)
    print()


def main():
    data = (1, 1, 3, 4)
    print("Input data:", data)
    print()

    demo_amplitude(data)
    demo_angle(data)
    demo_basis(data, repeat=False)
    demo_basis(data, repeat=True)


if __name__ == "__main__":
    main()
