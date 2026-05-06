import numpy as np

from nexq.core.circuit import Circuit
from nexq.core.io.dag import circuit_to_dag
from nexq.optimizer import optimize_basic


def test_optimize_basic_dict_cancellations():
    circuit = Circuit(
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "pauli_x", "target_qubit": 0},
        {"type": "hadamard", "target_qubit": 1},
        {"type": "hadamard", "target_qubit": 1},
        {"type": "cx", "control_qubits": [0], "control_states": [1], "target_qubit": 1},
        {"type": "cx", "control_qubits": [0], "control_states": [1], "target_qubit": 1},
        {"type": "s_gate", "target_qubit": 0},
        {"type": "sdg", "target_qubit": 0},
        {"type": "pauli_z", "target_qubit": 0},
        n_qubits=2,
    )

    out = optimize_basic(circuit)
    assert isinstance(out, Circuit)
    assert len(out.gates) == 1
    assert out.gates[0]["type"] == "pauli_z"
    assert out.gates[0]["target_qubit"] == 0


def test_optimize_basic_qasm_keeps_type_and_applies_rules():
    qasm = """OPENQASM 3.0;
include \"stdgates.inc\";
qubit[2] q;
x q[0];
x q[0];
h q[1];
h q[1];
cx q[0], q[1];
cx q[0], q[1];
s q[0];
sdg q[0];
z q[0];
"""
    out = optimize_basic(qasm)
    assert isinstance(out, str)
    assert "x q[0];\nx q[0];" not in out
    assert "h q[1];\nh q[1];" not in out
    assert "cx q[0], q[1];\ncx q[0], q[1];" not in out
    assert "s q[0];\nsdg q[0];" not in out
    assert "z q[0];" in out


def test_optimize_basic_dag_tuple_output_type_same():
    circuit = Circuit(
        {"type": "pauli_y", "target_qubit": 0},
        {"type": "pauli_y", "target_qubit": 0},
        {"type": "pauli_z", "target_qubit": 1},
        n_qubits=2,
    )
    gate_types = ["pauli_x", "pauli_y", "pauli_z", "hadamard", "cx", "s_gate", "sdg"]
    dag = circuit_to_dag(circuit, gate_types)

    out = optimize_basic(dag, input_type="dag", dag_gate_types=gate_types)
    assert isinstance(out, tuple)
    X, A, T = out
    assert isinstance(X, np.ndarray)
    assert isinstance(A, np.ndarray)
    assert isinstance(T, np.ndarray)
    # Only one effective gate (pauli_z on q1) remains -> total nodes = gate + START + END = 3
    assert X.shape[0] == 3
    assert A.shape == (3, 3)
    assert T.shape[0] == 3


def test_optimize_basic_dict_merges_adjacent_rotations():
    circuit = Circuit(
        {"type": "rx", "target_qubit": 0, "parameter": 0.1},
        {"type": "rx", "target_qubit": 0, "parameter": 0.2},
        {"type": "ry", "target_qubit": 1, "parameter": 0.3},
        {"type": "ry", "target_qubit": 1, "parameter": -0.3},
        {"type": "rz", "target_qubit": 0, "parameter": 0.4},
        {"type": "rz", "target_qubit": 0, "parameter": 0.5},
        n_qubits=2,
    )

    out = optimize_basic(circuit)
    assert isinstance(out, Circuit)
    # rx(0.1)+rx(0.2)->rx(0.3), ry(0.3)+ry(-0.3)->I, rz(0.4)+rz(0.5)->rz(0.9)
    assert len(out.gates) == 2
    assert out.gates[0]["type"] == "rx"
    assert out.gates[0]["target_qubit"] == 0
    assert np.isclose(float(out.gates[0]["parameter"]), 0.3)
    assert out.gates[1]["type"] == "rz"
    assert out.gates[1]["target_qubit"] == 0
    assert np.isclose(float(out.gates[1]["parameter"]), 0.9)


def test_optimize_basic_qasm_merges_adjacent_rotations():
    qasm = """OPENQASM 3.0;
include \"stdgates.inc\";
qubit[2] q;
rx(0.1) q[0];
rx(0.2) q[0];
ry(pi/2) q[1];
ry(-pi/2) q[1];
rz(0.4) q[0];
rz(0.5) q[0];
"""
    out = optimize_basic(qasm)
    assert isinstance(out, str)
    assert "rx(0.3) q[0];" in out
    assert "ry(pi/2) q[1];" not in out
    assert "ry(-pi/2) q[1];" not in out
    assert "rz(0.9) q[0];" in out
