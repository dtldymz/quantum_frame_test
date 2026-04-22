"""Standalone test runner for basic/define torch and mindspore scripts."""


def run_basic_torch_tests():
    import torch
    import basic_torch as bt

    print("=== basic_torch tests ===")
    theta = torch.tensor(1.5, requires_grad=True, dtype=bt.MY_DTYPE, device=bt.DEVICE)
    psi = bt.KET_0.clone()

    rz_gate = bt._rz(theta)
    evolved_state = torch.matmul(rz_gate, psi)
    bra_0 = bt.dagger(psi)
    overlap = torch.matmul(bra_0, evolved_state)
    prob_0 = torch.abs(overlap) ** 2

    print(f"Input theta: {theta.item()}")
    print(f"Probability of |0> after rz({theta.item()}): {prob_0.item()}")

    prob_0.backward()
    print(f"Gradient of probability w.r.t. theta: {theta.grad.item()}")

    print("\n--- Matrix Product Test ---")
    i2 = bt.identity(1)
    x = bt._pauli_x()
    ix = bt.tensor_product(i2, x)
    xx = bt._pauli_x(1)
    print(f"I kron X:\n{ix}")
    print(f"X on qubit 1:\n{xx}")
    print(f"Are they equal? {torch.allclose(ix, xx)}")

    print("\n--- Gate Matrix Conversion Test ---")
    gate_info = {
        "type": "rx",
        "target_qubit": 0,
        "parameter": torch.tensor(0.5, requires_grad=True),
    }
    rx_mat = bt.gate_to_matrix(gate_info, cir_qubits=2)
    print(f"rx(0.5) gate matrix (2 qubits):\n{rx_mat}")
    print(f"Shape: {rx_mat.shape}")
    print(f"Requires grad: {rx_mat.requires_grad}")

    print("\n--- _rz with stack Test ---")
    rz_gate = bt._rz(torch.tensor(0.7))
    print(f"rz(0.7) gate matrix:\n{rz_gate}")

    print("\n--- _s_gate with stack Test ---")
    s_gate = bt._s_gate()
    print(f"S gate matrix:\n{s_gate}")

    print("\n--- _t_gate with stack Test ---")
    t_gate = bt._t_gate()
    print(f"T gate matrix:\n{t_gate}")

    print("\n--- _u3 with stack Test ---")
    u3_gate = bt._u3(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
    print(f"u3(0.1, 0.2, 0.3) gate matrix:\n{u3_gate}")

    print("\n--- RZZ with stack Test ---")
    rzz_gate = bt._rzz(torch.tensor(0.8))
    print(f"RZZ(0.8) gate matrix:\n{rzz_gate}")


def run_define_torch_tests():
    import torch
    import define_torch as dt

    print("\n=== define_torch tests ===")
    psi0 = dt.phi_0(2)
    bell_circuit = dt.circuit(dt.hadamard(0), dt.cnot(1, [0]), num_qubits=2)
    bell_state = torch.matmul(bell_circuit, psi0)
    exp_identity = dt.expectation(bell_state, dt.identity(2))

    print(f"bell_state shape: {bell_state.shape}")
    print(f"expectation on I: {exp_identity.item()}")


def run_basic_mind_tests():
    import mindspore as ms
    from mindspore import Tensor, ops
    import basic_mind as bm

    print("\n=== basic_mind tests ===")
    theta = Tensor(1.5, dtype=ms.complex64)
    psi = bm.KET_0

    def compute_prob(theta_val):
        rx_gate = bm._rx(theta_val)
        evolved_state = ops.matmul(rx_gate, psi)
        bra_0 = bm.dagger(psi)
        overlap = ops.matmul(bra_0, evolved_state)
        return ops.real(ops.abs(overlap) ** 2)

    prob_and_grad_func = ms.value_and_grad(compute_prob, grad_position=0)
    prob_0_val, grad_theta = prob_and_grad_func(theta)

    print(f"Input theta: {theta.asnumpy()}")
    print(f"Probability of |0> after rx({theta.asnumpy()}): {prob_0_val.asnumpy()}")
    print(f"Gradient of probability w.r.t. theta: {grad_theta.asnumpy()}")

    print("\n--- Matrix Product Test ---")
    i2 = bm.identity(1)
    x = bm._pauli_x()
    ix = bm.tensor_product(i2, x)
    xx = bm._pauli_x(1)
    print(f"I kron X:\n{ix}")
    print(f"X on qubit 1:\n{xx}")
    print(f"Are they equal? {ops.allclose(ix, xx).asnumpy()}")

    print("\n--- Gate Matrix Conversion Test ---")
    gate_info = {
        "type": "rx",
        "target_qubit": 0,
        "parameter": Tensor(0.5, dtype=ms.complex64),
    }
    rx_mat = bm.gate_to_matrix(gate_info, cir_qubits=2)
    print(f"rx(0.5) gate matrix (2 qubits):\n{rx_mat}")
    print(f"Shape: {rx_mat.shape}")

    print("\n--- _rz with stack Test ---")
    rz_gate = bm._rz(Tensor(0.7))
    print(f"rz(0.7) gate matrix:\n{rz_gate}")

    print("\n--- _s_gate with stack Test ---")
    s_gate = bm._s_gate()
    print(f"S gate matrix:\n{s_gate}")

    print("\n--- _t_gate with stack Test ---")
    t_gate = bm._t_gate()
    print(f"T gate matrix:\n{t_gate}")

    print("\n--- _u3 with stack Test ---")
    u3_gate = bm._u3(Tensor(0.1), Tensor(0.2), Tensor(0.3))
    print(f"u3(0.1, 0.2, 0.3) gate matrix:\n{u3_gate}")

    print("\n--- RZZ with stack Test ---")
    rzz_gate = bm._rzz(Tensor(0.8))
    print(f"RZZ(0.8) gate matrix:\n{rzz_gate}")


def run_define_mind_tests():
    import mindspore as ms
    from mindspore import Tensor, ops
    import define_mind as dm

    print("\n=== define_mind tests ===")
    theta = Tensor(1.5, dtype=ms.float64)
    psi = dm.KET_0

    def compute_prob(theta_val):
        rz_gate = dm._rz(theta_val)
        evolved_state = ops.matmul(rz_gate, psi)
        bra_0 = dm.dagger(psi)
        overlap = ops.matmul(bra_0, evolved_state)
        return ops.real(ops.abs(overlap) ** 2)

    prob_and_grad_func = ms.value_and_grad(compute_prob, grad_position=0)
    prob_0_val, grad_theta = prob_and_grad_func(theta)

    print(f"Input theta: {theta.asnumpy()}")
    print(f"Probability of |0> after rz({theta.asnumpy()}): {prob_0_val.asnumpy()}")
    print(f"Gradient of probability w.r.t. theta: {grad_theta.asnumpy()}")

    print("\n--- Matrix Product Test ---")
    i2 = dm.identity(1)
    x = dm._pauli_x()
    ix = dm.tensor_product(i2, x)
    xx = dm._pauli_x(1)
    print(f"I kron X:\n{ix}")
    print(f"X on qubit 1:\n{xx}")
    print(f"Are they equal? {ops.allclose(ix, xx).asnumpy()}")

    print("\n--- Gate Matrix Conversion Test ---")
    gate_info = {
        "type": "rx",
        "target_qubit": 0,
        "parameter": Tensor(0.5, dtype=ms.float64),
    }
    rx_mat = dm.gate_to_matrix(gate_info, cir_qubits=2)
    print(f"rx(0.5) gate matrix (2 qubits):\n{rx_mat}")
    print(f"Shape: {rx_mat.shape}")

    print("\n--- _rz with stack Test ---")
    rz_gate = dm._rz(Tensor(0.7))
    print(f"rz(0.7) gate matrix:\n{rz_gate}")

    print("\n--- _s_gate with stack Test ---")
    s_gate = dm._s_gate()
    print(f"S gate matrix:\n{s_gate}")

    print("\n--- _t_gate with stack Test ---")
    t_gate = dm._t_gate()
    print(f"T gate matrix:\n{t_gate}")

    print("\n--- _u3 with stack Test ---")
    u3_gate = dm._u3(Tensor(0.1), Tensor(0.2), Tensor(0.3))
    print(f"u3(0.1, 0.2, 0.3) gate matrix:\n{u3_gate}")

    print("\n--- RZZ with stack Test ---")
    rzz_gate = dm._rzz(Tensor(0.8))
    print(f"RZZ(0.8) gate matrix:\n{rzz_gate}")


def main():
    run_basic_torch_tests()
    run_define_torch_tests()

    try:
        run_basic_mind_tests()
        run_define_mind_tests()
    except ModuleNotFoundError as exc:
        print("\n=== MindSpore tests skipped ===")
        print(f"Reason: {exc}")


if __name__ == "__main__":
    main()
