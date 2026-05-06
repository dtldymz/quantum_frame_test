"""Quantum data encoder implementations."""

from __future__ import annotations

import math

import numpy as np

from .abstract import BaseEncoder
from ..core.circuit import Circuit
from ..core.state import State
from ..core.io.qasm import circuit_to_qasm
from ..core.io.dag import circuit_to_dag
from ..channel.backends.numpy_backend import NumpyBackend


def _default_backend(backend):
    return NumpyBackend() if backend is None else backend


def _build_state_prep_unitary(psi):
    """Gram-Schmidt: build unitary whose first column is psi."""
    dim = len(psi)
    Q = np.zeros((dim, dim), dtype=np.complex64)
    Q[:, 0] = psi.astype(np.complex64)
    col = 1
    for i in range(dim):
        e = np.zeros(dim, dtype=np.complex64)
        e[i] = 1.0
        for k in range(col):
            e = e - np.vdot(Q[:, k], e) * Q[:, k]
        norm = np.linalg.norm(e)
        if norm > 1e-10:
            Q[:, col] = e / norm
            col += 1
            if col == dim:
                break
    return Q


def _emit_circuit(circuit, cir):
    if cir == "dict":
        return circuit
    if cir == "qasm":
        return circuit_to_qasm(circuit)
    if cir == "dag":
        gate_types = list(dict.fromkeys(g["type"] for g in circuit.gates))
        return circuit_to_dag(circuit, gate_types)
    raise ValueError(f"cir must be 'dict', 'qasm' or 'dag', got {cir!r}")


# ---------------------------------------------------------------------------
# AmplitudeEncoder
# ---------------------------------------------------------------------------

class AmplitudeEncoder(BaseEncoder):
    """
    Amplitude Encoding.

    Normalises the input vector and stores it as state amplitudes.
    n_qubits is inferred from data length when not supplied.
    """

    def __init__(self, n_qubits=None):
        self.n_qubits = n_qubits

    def encode(self, data, *, cir="dict", backend=None):
        """
        Args:
            data:    real/complex array of length 2^n (auto-normalised).
            cir:     "dict" | "qasm" | "dag"
            backend: compute backend (default: NumpyBackend)
        Returns:
            (circuit_repr, State)
        """
        bk = _default_backend(backend)
        arr = np.asarray(data, dtype=np.complex64).ravel()
        data_len = len(arr)

        if self.n_qubits is not None:
            n = self.n_qubits
            expected = 1 << n
            if data_len > expected:
                raise ValueError(f"data length {data_len} > 2^n_qubits={expected}")
            if data_len < expected:
                arr = np.concatenate([arr, np.zeros(expected - data_len, dtype=arr.dtype)])
        else:
            n = max(1, math.ceil(math.log2(data_len))) if data_len > 1 else 1
            expected = 1 << n
            if data_len < expected:
                arr = np.concatenate([arr, np.zeros(expected - data_len, dtype=arr.dtype)])

        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            raise ValueError("input norm is zero")
        psi = arr / norm

        U = _build_state_prep_unitary(psi)
        gate = {"type": "unitary", "parameter": U.tolist(), "n_qubits": n}
        circuit = Circuit(gate, n_qubits=n)

        state = State.from_array(psi, n_qubits=n, backend=bk)
        return _emit_circuit(circuit, cir), state

    def decode(self, quantum_state):
        """Return real part of state amplitudes as classical vector."""
        return np.real(quantum_state.to_numpy().ravel())


# ---------------------------------------------------------------------------
# AngleEncoder
# ---------------------------------------------------------------------------

class AngleEncoder(BaseEncoder):
    """
    Angle Encoding: x_i -> Ry(x_i) on qubit i.

    n_qubits is inferred from len(data) when not supplied.
    rotation can be "rx", "ry" (default), or "rz".
    """

    def __init__(self, n_qubits=None, rotation="ry"):
        self.n_qubits = n_qubits
        self.rotation = rotation.lower()
        if self.rotation not in {"rx", "ry", "rz"}:
            raise ValueError("rotation must be rx/ry/rz")

    def encode(self, data, *, cir="dict", backend=None):
        """
        Args:
            data:    real array of length n (rotation angles in radians).
            cir:     "dict" | "qasm" | "dag"
            backend: compute backend (default: NumpyBackend)
        Returns:
            (circuit_repr, State)
        """
        bk = _default_backend(backend)
        arr = np.asarray(data, dtype=np.float64).ravel()
        data_len = len(arr)

        if self.n_qubits is not None:
            n = self.n_qubits
            if data_len > n:
                raise ValueError(f"data length {data_len} > n_qubits={n}")
            if data_len < n:
                arr = np.concatenate([arr, np.zeros(n - data_len)])
        else:
            n = data_len

        gates = [
            {"type": self.rotation, "target_qubit": i, "parameter": float(angle)}
            for i, angle in enumerate(arr)
        ]
        circuit = Circuit(*gates, n_qubits=n)

        zero = State.zero_state(n, bk)
        U = circuit.unitary(backend=bk)
        state = zero.evolve(U)

        return _emit_circuit(circuit, cir), state

    def decode(self, quantum_state):
        """Recover rotation angles from marginal |1> probabilities (MSB qubit order)."""
        amplitudes = quantum_state.to_numpy()
        n = quantum_state.n_qubits
        angles = []
        for i in range(n):
            # gate_to_matrix uses MSB: qubit i is bit (n-1-i) of the index
            p1 = sum(
                abs(amplitudes[idx]) ** 2
                for idx in range(1 << n)
                if (idx >> (n - 1 - i)) & 1
            )
            angles.append(float(2 * np.arcsin(np.sqrt(np.clip(p1, 0.0, 1.0)))))
        return np.array(angles)


# ---------------------------------------------------------------------------
# BasisEncoder
# ---------------------------------------------------------------------------

class BasisEncoder(BaseEncoder):
    """
    Basis Encoding using Divide-and-Conquer Decision Tree Algorithm.

    Algorithm Overview:
    1. Input: integer dataset S, qubit count n
    2. Convert each element to n-bit binary string (build Trie)
    3. Count frequencies: compute subtree frequency at each Trie node
    4. Recursive traversal:
       - Calc P(0) = c0 / (c0 + c1)
       - Calc rotation angle: θ = 2 * arccos(sqrt(P(0)))
       - Generate controlled Ry(θ) gate at current qubit
    5. Circuit maps path to control configuration
    6. Final state is superposition of target basis states

    Repeat modes:
    - repeat=False: deduplicate inputs first, each unique value has weight 1
    - repeat=True: preserve input multiplicity, repeated values contribute by frequency

    Reference: Divide-and-Conquer method for efficient basis encoding
    """

    def __init__(self, n_qubits=None, repeat=False):
        self.n_qubits = n_qubits
        self.repeat = repeat

    def encode(self, data, *, cir="dict", backend=None):
        """
        Args:
            data:    integer array (e.g., [1, 2, 3, 4])
            cir:     "dict" | "qasm" | "dag"
            backend: compute backend (default: NumpyBackend)
        Returns:
            (circuit_repr, State)
        """
        bk = _default_backend(backend)
        
        # Convert to integer array
        arr = np.asarray(data, dtype=np.int64).ravel()
        if arr.size == 0:
            raise ValueError("data must be non-empty")
        
        # Validate non-negative integers
        if np.any(arr < 0):
            raise ValueError("data must contain non-negative integers")
        
        # Determine n_qubits
        max_val = int(np.max(arr))
        min_n = max(1, max_val.bit_length()) if max_val > 0 else 1
        if self.n_qubits is None:
            n = min_n
        else:
            n = max(min_n, int(self.n_qubits))
        
        # Convert to binary strings (Step 1: Format Conversion)
        binary_strings = [format(int(v), f"0{n}b") for v in arr]
        
        # Build Trie (Step 1: Trie Construction)
        trie = self._build_trie(binary_strings)
        
        # Recursively generate gates (Step 2-3: Traversal & Circuit Mapping)
        gates = []
        self._traverse_and_generate_gates(trie, "", n, gates)
        
        # Create circuit
        circuit = Circuit(*gates, n_qubits=n) if gates else Circuit(n_qubits=n)
        
        # Compute quantum state by evolving |0...0> through circuit
        zero_state = State.zero_state(n, bk)
        U = circuit.unitary(backend=bk)
        state = zero_state.evolve(U)
        
        return _emit_circuit(circuit, cir), state
    
    def _build_trie(self, binary_strings):
        """
        Build Trie structure and compute subtree frequencies.
        Returns root node of Trie.
        """
        trie = {}
        frequencies = {}
        for s in binary_strings:
            if self.repeat:
                frequencies[s] = frequencies.get(s, 0) + 1
            elif s not in frequencies:
                frequencies[s] = 1

        for s, freq in frequencies.items():
            node = trie
            for bit in s:
                if bit not in node:
                    node[bit] = {}
                node = node[bit]
            node["_terminal_count"] = node.get("_terminal_count", 0) + freq
        
        # Compute _count for each node (subtree total frequency)
        self._compute_counts(trie)
        return trie
    
    def _compute_counts(self, node):
        """
        Recursively compute subtree total frequency at each node.
        Returns the total frequency for this subtree.
        """
        if not isinstance(node, dict):
            return 0
        
        count = int(node.get("_terminal_count", 0))
        
        for bit in ["0", "1"]:
            if bit in node:
                count += self._compute_counts(node[bit])
        
        node["_count"] = count
        return count
    
    def _traverse_and_generate_gates(self, node, path, n, gates):
        """
        Recursively traverse Trie and generate controlled Ry gates.
        
        Args:
            node: current Trie node
            path: binary string path to current node (e.g., "10")
            n: total number of qubits
            gates: accumulator for generated gates
        """
        if len(path) >= n:
            return
        
        j = len(path)  # Current qubit index
        
        # Step A: Determine branch frequencies
        c0 = node.get("0", {}).get("_count", 0)
        c1 = node.get("1", {}).get("_count", 0)
        C = c0 + c1
        
        if C == 0:
            return
        
        # Step B: Calculate conditional probability and rotation angle
        if c0 == 0:
            # All frequency mass goes to 1 (right), so rotate to |1>
            theta = np.pi
        elif c1 == 0:
            # All frequency mass goes to 0 (left), no rotation needed
            theta = 0.0
        else:
            # Mixed case: use subtree frequency ratio
            P0 = c0 / C
            # Quantum amplitude relation: cos²(θ/2) = P(0)
            # So: θ = 2 * arccos(sqrt(P(0)))
            theta = 2.0 * np.arccos(np.sqrt(np.clip(P0, 0.0, 1.0)))
        
        # Step 3: Circuit Mapping - generate gate if angle is non-trivial
        if abs(theta) > 1e-10:
            # Build control configuration from path
            control_qubits = list(range(len(path)))
            control_states = [int(bit) for bit in path]
            
            # Generate controlled Ry gate
            if len(path) == 0:
                # No controls, just Ry
                gate = {
                    "type": "ry",
                    "target_qubit": j,
                    "parameter": float(theta)
                }
            else:
                # Single or multi-control Ry - use cry with control_states
                gate = {
                    "type": "cry",
                    "target_qubit": j,
                    "control_qubits": control_qubits,
                    "control_states": control_states,
                    "parameter": float(theta)
                }
            
            gates.append(gate)
        
        # Step 4: Termination & Recursion
        # Recursively handle left branch (bit 0)
        if "0" in node:
            self._traverse_and_generate_gates(node["0"], path + "0", n, gates)
        
        # Recursively handle right branch (bit 1)
        if "1" in node:
            self._traverse_and_generate_gates(node["1"], path + "1", n, gates)

    def decode(self, quantum_state):
        """
        Decode: return binary representation of most probable basis state (MSB).
        """
        amplitudes = quantum_state.to_numpy()
        idx = int(np.argmax(np.abs(amplitudes) ** 2))
        n = quantum_state.n_qubits
        return [(idx >> (n - 1 - i)) & 1 for i in range(n)]
