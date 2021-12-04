from quantumGAN.functions import Partial_Trace
import qiskit
from qiskit.aqua.utils import tensorproduct
import matplotlib.pyplot as plt

"""
Test to verify the equation to get the post measurement state
 of a partial measurement
"""
import numpy as np


def create_partial_POVM(num_qubits_to_measure):
	ket_0 = np.array([1, 0])
	qubit = num_qubits_to_measure
	bra0ket0 = np.outer(ket_0, ket_0)
	while qubit > 1:
		bra0ket0 = np.kron(bra0ket0, np.outer(ket_0, ket_0))
		qubit -= 1

	return bra0ket0


state_1 = np.array(
	[ 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0.707+0j, 0.707+0j ])
#print(np.outer(state,state))
#print(np.linalg.norm(np.outer(state, state)))
#exit()
post_M_stat = np.array(
	[ 0+0j, 0+0j, 0+0j, 1+0j ])


def partial_measurament(state, trace_qubits):
	num_qubits = int(np.log2(state.shape[0]))
	pt = qiskit.quantum_info.partial_trace(state, trace_qubits)

	POVM = create_partial_POVM(num_qubits-len(trace_qubits))
	top = np.dot(POVM, pt)

	POVM_I = np.kron(POVM, np.identity(2**(num_qubits - (num_qubits-len(trace_qubits)))))
	print(POVM_I)
	bot_no_trace = np.dot(POVM_I, np.outer(state, state))
	bot = np.trace(bot_no_trace)
	print(bot_no_trace)

	return top / bot


state = np.array([ 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0.707+0j, 0.707+0j ])
output = partial_measurament(state, [1,2,3])
print(output)



