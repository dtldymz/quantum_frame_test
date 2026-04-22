# 注意：此文件依赖于 basic_torch.txt 转换后的 Ascend 兼容 MindSpore 版本。
# 请将 basic_torch.txt 转换后的代码（所有函数名保持不变，数据类型统一为 complex64，并适配 Ascend）保存为 'basic_mind.py' 并放在此文件同目录下。
from basic_mind import * # 导入所有转换后的函数和常量
import numpy as np # For np.pi
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, ops, context
from mindspore import value_and_grad # For automatic differentiation

# --- 设置运行环境 ---
# 指定在 Ascend 设备上运行
context.set_context(device_target="Ascend", device_id=0) # device_id 选择具体的 NPU ID

def phi_0(num_qubits=1):
    """
    生成全零态 |0...0> 的量子态向量
    参数:
    num_qubits (int): 量子比特的数量
    返回:
    Tensor: 形状为 (2^num_qubits, 1) 的量子态向量
    """
    # 从单个量子比特的 |0> 态开始
    state = KET_0
    # 通过张量积构造多量子比特的 |0...0> 态
    for _ in range(1, num_qubits):
        state = ops.kron(state, KET_0)
    return state

def expectation(state, hamiltonian):
    """
    计算量子态在给定哈密顿量下的期望值
    Args:
        state (Tensor): 量子态向量或者密度矩阵
                         如果是态矢量，形状应为 (2^N, 1) 或 (2^N,)
                         如果是密度矩阵，形状应为 (2^N, 2^N)
        hamiltonian (Tensor): 哈密顿量矩阵，形状应为 (2^N, 2^N)
    Returns:
        Tensor: 哈密顿量的期望值 <ψ|H|ψ> 或 Tr(ρH) (标量张量)
    """
    # 检查输入维度
    if state.ndim == 1:
        # 状态向量 |ψ>，形状 (2^N,) -> (2^N, 1)
        state = state.expand_dims(1) # Use expand_dims instead of unsqueeze
    if state.ndim == 2 and state.shape[1] == 1:
        # 纯态情况 |ψ>，计算 <ψ|H|ψ>
        # dagger(state) 是 (1, 2^N), hamiltonian 是 (2^N, 2^N), state 是 (2^N, 1)
        # 结果是 (1, 1) 的张量
        expectation = matrix_product(dagger(state), hamiltonian, state)
        # 取 (1,1) 张量中的标量值
        expectation = expectation.squeeze() # Or expectation[0, 0]
    elif state.ndim == 2 and state.shape[0] == state.shape[1]:
        # 混合态情况 ρ，计算 Tr(ρH)
        expectation = ops.trace(ops.matmul(state, hamiltonian))
    else:
        raise ValueError("state必须是态矢量(2^N,)或(2^N, 1)或密度矩阵(2^N, 2^N)")

    # 返回期望值的实部（理论上期望值应该是实数）
    # 如果 H 是厄米算符，<ψ|H|ψ> 或 Tr(ρH) 是实数
    # ops.real 对复数张量返回实部，梯度可以继续传播
    # 如果输入的 state 或 hamiltonian 是可微的，expectation 也是可微的
    # MindSpore 会自动处理复数张量的梯度（Wittrick规则或类似方法）
    return ops.real(expectation)

def circuit(*gates, num_qubits=1):
    """
    量子电路函数，用于计算量子线路的矩阵表示
    参数:
    *gates: 可变数量的门，每个门是一个字典，包含 'type', 'target_qubit', 'control_qubit', 'parameter' 等键
            例如: {'type': 'cnot', 'target_qubit': 1, 'control_qubit': 0, 'parameter': None}
                 {'type': 'toffoli', 'target_qubit': 2, 'control_qubits': [0, 1], 'parameter': None}
    num_qubits: 电路中的总量子比特数（可选），如果未指定，则根据门信息自动确定。注意：必做通过关键字显示传入！！！
    返回:
    Tensor: 整个电路的矩阵表示
    使用示例:
    # 创建贝尔态电路
    circuit_matrix = circuit(
        {'type': 'hadamard', 'target_qubit': 0, 'parameter': None},
        {'type': 'cnot', 'target_qubit': 1, 'control_qubit': 0, 'parameter': None}
    )
    # 创建包含toffoli门的电路
    circuit_matrix = circuit(
        {'type': 'pauli_x', 'target_qubit': 0, 'parameter': None},
        {'type': 'pauli_x', 'target_qubit': 1, 'parameter': None},
        {'type': 'toffoli', 'target_qubit': 2, 'control_qubits': [0, 1], 'parameter': None}
    )
    """
    if not gates:
        # 如果没有门，则返回单位矩阵（默认1量子比特）
        return IDENTITY_2
    # 确定所有门中的最大量子比特数）
    gate_qubits = 0
    for gate in gates:
        gate_type = gate['type']
        if gate_type in ['pauli_x', 'X', 'pauli_y', 'Y', 'pauli_z', 'Z', 'hadamard', 'H', 's_gate', 'S', 't_gate', 'T', 'rx', 'ry', 'rz', 'u3', 'u2']:
            gate_qubits = max(gate_qubits, gate['target_qubit'] + 1)
        elif gate_type in ['cnot', 'cx', 'cz', 'cy', 'crx', 'cry', 'crz']:
            gate_qubits = max(gate_qubits, gate['target_qubit'] + 1, max(gate['control_qubits']) + 1)
        elif gate_type == 'toffoli':
            gate_qubits = max(gate_qubits, gate['target_qubit'] + 1, max(gate['control_qubits']) + 1)
        elif gate_type == 'swap':
            gate_qubits = max(gate_qubits, gate['qubit_1'] + 1, gate['qubit_2'] + 1)
        elif gate_type in ['identity', 'I']:
            gate_qubits = max(gate_qubits, gate['num_qubits'])
        else:
            gate_qubits = max(gate_qubits, gate['target_qubit'] + 1)
    # 依次计算每个门的矩阵并相乘
    if gate_qubits > num_qubits:
        raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {num_qubits}")
    else:
        circuit_matrix = identity(num_qubits)
        for gate in gates:
            gate_matrix = gate_to_matrix(gate, num_qubits)
            circuit_matrix = ops.matmul(gate_matrix, circuit_matrix)
    return circuit_matrix 

SINGLE_QUBIT_GATES = [
    'pauli_x', 'X',
    'pauli_y', 'Y', 
    'pauli_z', 'Z',
    'hadamard', 'H',
    's_gate', 'S',
    't_gate', 'T',
    'rx', 'ry', 'rz',
    'u3', 'u2'
]

TWO_QUBIT_GATES = [
    'cnot', 'cx',
    'cy', 'cz',
    'swap'
]

THREE_QUBIT_GATES = [
    'toffoli', 'ccnot'
]

def pauli_x(target_qubit=0):
    return {'type': 'pauli_x', 'target_qubit': target_qubit}

def pauli_y(target_qubit=0):
    return {'type': 'pauli_y', 'target_qubit': target_qubit}

def pauli_z(target_qubit=0):
    return {'type': 'pauli_z', 'target_qubit': target_qubit}

def hadamard(target_qubit=0):
    return {'type': 'hadamard', 'target_qubit': target_qubit}

def rx(theta, target_qubit=0):
    # 确保 theta 是实数类型，但矩阵元素是复数
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64) # 用于计算 cos/sin
    return {'type': 'rx', 'target_qubit': target_qubit, 'parameter': theta}

def ry(theta, target_qubit=0):
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'ry', 'target_qubit': target_qubit, 'parameter': theta}

def rz(theta, target_qubit=0):
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'rz', 'target_qubit': target_qubit, 'parameter': theta}

def s_gate(target_qubit=0):
    return {'type': 's_gate', 'target_qubit': target_qubit}

def t_gate(target_qubit=0):
    return {'type': 't_gate', 'target_qubit': target_qubit}

def cx(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'cx', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

cnot = cx

def cy(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'cy', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

def cz(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'cz', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

def crx(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'crx', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def cry(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'cry', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def crz(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'crz', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def swap(qubit_1=0, qubit_2=1):
    return {'type': 'swap', 'qubit_1': qubit_1, 'qubit_2': qubit_2}

def toffoli(target_qubit=2, control_qubits=[0,1]):
    return {'type': 'toffoli', 'target_qubit': target_qubit, 'control_qubits': control_qubits}

ccnot = toffoli

def u3(theta, phi, lam, target_qubit=0):
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    if not isinstance(phi, Tensor):
        phi = Tensor(phi, dtype=ms.float64)
    if not isinstance(lam, Tensor):
        lam = Tensor(lam, dtype=ms.float64)
    return {'type': 'u3', 'target_qubit': target_qubit, 'parameter': [theta, phi, lam]}

def u2(phi, lam, target_qubit=0):
    # Use np.pi
    return u3(Tensor(np.pi, dtype=ms.float64)/2, phi, lam, target_qubit)

# --- 示例：使用 MindSpore 的自动微分 ---
if __name__ == "__main__":
    # 创建一个可求导的旋转角度
    theta = Tensor(1.5, dtype=ms.float64)
    # 创建一个简单的量子态 |psi> = |0>
    psi = KET_0 # No need to clone in MindSpore like PyTorch

    # 定义一个计算概率的函数（需要对 theta 求导）
    def compute_prob(theta_val):
        # 应用 rz(theta) 门 (使用 rz 作为示例，因为原始 PyTorch 示例中是 rx_gate = _rz(theta))
        rz_gate = _rz(theta_val)
        # 假设这是一个简单的演化 U|psi>
        evolved_state = ops.matmul(rz_gate, psi)

        # 计算一个简单的实数输出，例如 <psi_out|Z|psi_out> (Z 是泡利Z算符)
        # 这里我们用 <0| evolved_state 来模拟一个简单的期望值计算
        bra_0 = dagger(psi) # <0|
        overlap = ops.matmul(bra_0, evolved_state) # <0|rz(theta)|0>
        # 取模长平方 |<0|rz(theta)|0>|^2
        prob_0 = ops.real(ops.abs(overlap)**2) # This is a real scalar
        # MindSpore 的自动微分需要标量输出
        # prob_0 是一个 0 维张量
        return prob_0

    # 使用 value_and_grad 获取函数值和梯度
    prob_and_grad_func = ms.value_and_grad(compute_prob, grad_position=0)
    prob_0_val, grad_theta = prob_and_grad_func(theta)

    print(f"Input theta: {theta.asnumpy()}")
    print(f"Probability of |0> after rz({theta.asnumpy()}): {prob_0_val.asnumpy()}")
    print(f"Gradient of probability w.r.t. theta: {grad_theta.asnumpy()}")

    # --- 测试矩阵乘积和张量积 ---
    print("\n--- Matrix Product Test ---")
    i2 = identity(1) # 2x2 identity
    x = _pauli_x()
    ix = tensor_product(i2, x) # I \otimes X
    xx = _pauli_x(1) # X on qubit 1 (with I on qubit 0)
    print(f"I kron X:\n{ix}")
    print(f"X on qubit 1:\n{xx}")
    print(f"Are they equal? {ops.allclose(ix, xx).asnumpy()}")

    print("\n--- Gate Matrix Conversion Test ---")
    gate_info = {'type': 'rx', 'target_qubit': 0, 'parameter': Tensor(0.5, dtype=ms.float64)}
    rx_mat = gate_to_matrix(gate_info, cir_qubits=2) # Should expand to 2 qubits
    print(f"rx(0.5) gate matrix (2 qubits):\n{rx_mat}")
    print(f"Shape: {rx_mat.shape}")

    # --- Test _rz with stack ---
    print("\n--- _rz with stack Test ---")
    rz_gate = _rz(Tensor(0.7))
    print(f"rz(0.7) gate matrix:\n{rz_gate}")

    # --- Test _s_gate with stack ---
    print("\n--- _s_gate with stack Test ---")
    s_gate = _s_gate()
    print(f"S gate matrix:\n{s_gate}")

    # --- Test _t_gate with stack ---
    print("\n--- _t_gate with stack Test ---")
    t_gate = _t_gate()
    print(f"T gate matrix:\n{t_gate}")

    # --- Test _u3 with stack ---
    print("\n--- _u3 with stack Test ---")
    u3_gate = _u3(Tensor(0.1), Tensor(0.2), Tensor(0.3))
    print(f"u3(0.1, 0.2, 0.3) gate matrix:\n{u3_gate}")

    # --- Test RZZ with stack ---
    print("\n--- RZZ with stack Test ---")
    rzz_gate = _rzz(Tensor(0.8))
    print(f"RZZ(0.8) gate matrix:\n{rzz_gate}")
