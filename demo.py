from nexq.circuit import *


cir1 = Circuit(hadamard(0), cnot(1, [3]))
save_path = "cir1.qasm"
save_circuit_qasm3(cir1, save_path)
print(cir1)
print(f"Saved circuit as QASM to {save_path}")