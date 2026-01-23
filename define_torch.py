# 注意：此文件依赖于 basic_mind.txt 转换后的 PyTorch 版本。
# 请将 basic_mind.txt 转换后的代码（所有函数名保持不变）保存为 'basic_torch.py' 并放在此文件同目录下。
from basic_torch import * # 导入所有转换后的函数和常量
import torch
import math # For math.pi if torch.pi is not available

def Phi_0(num_qubits=1):
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
        state = torch.kron(state, KET_0)
    return state

def Expectation(state, hamiltonian):
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
    if state.dim() == 1:
        # 状态向量 |ψ>，形状 (2^N,) -> (2^N, 1)
        state = state.unsqueeze(1) # Use unsqueeze instead of expand_dims
    if state.dim() == 2 and state.shape[1] == 1:
        # 纯态情况 |ψ>，计算 <ψ|H|ψ>
        # Dagger(state) 是 (1, 2^N), hamiltonian 是 (2^N, 2^N), state 是 (2^N, 1)
        # 结果是 (1, 1) 的张量
        expectation = Matrix_Product(Dagger(state), hamiltonian, state)
        # 取 (1,1) 张量中的标量值
        expectation = expectation.squeeze() # Or expectation[0, 0]
    elif state.dim() == 2 and state.shape[0] == state.shape[1]:
        # 混合态情况 ρ，计算 Tr(ρH)
        expectation = torch.trace(torch.matmul(state, hamiltonian))
    else:
        raise ValueError("state必须是态矢量(2^N,)或(2^N, 1)或密度矩阵(2^N, 2^N)")

    # 返回期望值的实部（理论上期望值应该是实数）
    # 如果 H 是厄米算符，<ψ|H|ψ> 或 Tr(ρH) 是实数
    # torch.real 对复数张量返回实部，梯度可以继续传播
    # 如果输入的 state 或 hamiltonian 是可微的，expectation 也是可微的
    # PyTorch 会自动处理复数张量的梯度（Wittrick规则或类似方法）
    return expectation.real

def Circuit(*gates, num_qubits=1):
    """
    量子电路函数，用于计算量子线路的矩阵表示
    参数:
    *gates: 可变数量的门，每个门是一个字典，包含 'type', 'target_qubit', 'control_qubit', 'parameter' 等键
            例如: {'type': 'CNOT', 'target_qubit': 1, 'control_qubit': 0, 'parameter': None}
                 {'type': 'TOFFOLI', 'target_qubit': 2, 'control_qubits': [0, 1], 'parameter': None}
    num_qubits: 电路中的总量子比特数（可选），如果未指定，则根据门信息自动确定。注意：必做通过关键字显示传入！！！
    返回:
    Tensor: 整个电路的矩阵表示
    使用示例:
    # 创建贝尔态电路
    circuit_matrix = Circuit(
        {'type': 'HADAMARD', 'target_qubit': 0, 'parameter': None},
        {'type': 'CNOT', 'target_qubit': 1, 'control_qubit': 0, 'parameter': None}
    )
    # 创建包含TOFFOLI门的电路
    circuit_matrix = Circuit(
        {'type': 'PAULI_X', 'target_qubit': 0, 'parameter': None},
        {'type': 'PAULI_X', 'target_qubit': 1, 'parameter': None},
        {'type': 'TOFFOLI', 'target_qubit': 2, 'control_qubits': [0, 1], 'parameter': None}
    )
    """
    if not gates:
        # 如果没有门，则返回单位矩阵（默认1量子比特）
        return IDENTITY_2
    # 确定所有门中的最大量子比特数）
    gate_qubits = 0
    for gate in gates:
        gate_type = gate['type']
        if gate_type in ['PAULI_X', 'X', 'PAULI_Y', 'Y', 'PAULI_Z', 'Z', 'HADAMARD', 'H', 'S_GATE', 'S', 'T_GATE', 'T', 'RX', 'RY', 'RZ', 'U3', 'U2']:
            gate_qubits = max(gate_qubits, gate['target_qubit'] + 1)
        elif gate_type in ['CNOT', 'CX', 'CZ', 'CY', 'CRX', 'CRY', 'CRZ']:
            gate_qubits = max(gate_qubits, gate['target_qubit'] + 1, max(gate['control_qubits']) + 1)
        elif gate_type == 'TOFFOLI':
            gate_qubits = max(gate_qubits, gate['target_qubit'] + 1, max(gate['control_qubits']) + 1)
        elif gate_type == 'SWAP':
            gate_qubits = max(gate_qubits, gate['qubit_1'] + 1, gate['qubit_2'] + 1)
        elif gate_type in ['IDENTITY', 'I']:
            gate_qubits = max(gate_qubits, gate['num_qubits'])
        else:
            gate_qubits = max(gate_qubits, gate['target_qubit'] + 1)
    # 依次计算每个门的矩阵并相乘
    if gate_qubits > num_qubits:
        raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {num_qubits}")
    else:
        circuit_matrix = IDENTITY(num_qubits)
        for gate in gates:
            gate_matrix = Gate_To_Matrix(gate, num_qubits)
            circuit_matrix = torch.matmul(gate_matrix, circuit_matrix)
    return circuit_matrix 

Single_Qubit_Gates = [
    'PAULI_X', 'X',
    'PAULI_Y', 'Y', 
    'PAULI_Z', 'Z',
    'HADAMARD', 'H',
    'S_GATE', 'S',
    'T_GATE', 'T',
    'RX', 'RY', 'RZ',
    'U3', 'U2'
]

Two_Qubits_Gates = [
    'CNOT', 'CX',
    'CY', 'CZ',
    'SWAP'
]

Three_Qubits_Gates = [
    'TOFFOLI', 'CCNOT'
]

def PAULI_X(target_qubit=0):
    return {'type': 'PAULI_X', 'target_qubit': target_qubit}

def PAULI_Y(target_qubit=0):
    return {'type': 'PAULI_Y', 'target_qubit': target_qubit}

def PAULI_Z(target_qubit=0):
    return {'type': 'PAULI_Z', 'target_qubit': target_qubit}

def HADAMARD(target_qubit=0):
    return {'type': 'HADAMARD', 'target_qubit': target_qubit}

def RX(theta, target_qubit=0):
    return {'type': 'RX', 'target_qubit': target_qubit, 'parameter': theta}

def RY(theta, target_qubit=0):
    return {'type': 'RY', 'target_qubit': target_qubit, 'parameter': theta}

def RZ(theta, target_qubit=0):
    return {'type': 'RZ', 'target_qubit': target_qubit, 'parameter': theta}

def S_GATE(target_qubit=0):
    return {'type': 'S_GATE', 'target_qubit': target_qubit}

def T_GATE(target_qubit=0):
    return {'type': 'T_GATE', 'target_qubit': target_qubit}

def CX(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'CX', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

CNOT = CX

def CY(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'CY', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

def CZ(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'CZ', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

def CRX(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'CRX', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def CRY(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'CRY', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def CRZ(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'CRZ', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def SWAP(qubit_1=0, qubit_2=1):
    return {'type': 'SWAP', 'qubit_1': qubit_1, 'qubit_2': qubit_2}

def TOFFOLI(target_qubit=2, control_qubits=[0,1]):
    return {'type': 'TOFFOLI', 'target_qubit': target_qubit, 'control_qubits': control_qubits}

CCNOT = TOFFOLI

def U3(theta, phi, lam, target_qubit=0):
    return {'type': 'U3', 'target_qubit': target_qubit, 'parameter': [theta, phi, lam]}

def U2(phi, lam, target_qubit=0):
    # Use torch.pi if available (PyTorch 2.0+), otherwise math.pi
    return U3(torch.pi / 2 if hasattr(torch, 'pi') else math.pi, phi, lam, target_qubit)
