from line_profiler import LineProfiler

import numpy as np
import qiskit
from qiskit.circuit.library import TwoLocal

from quantumGAN.quantum_generator import QuantumGenerator

num_qubits = [4]
k = len(num_qubits)

# Set number of training epochs
num_epochs = 200
# Batch size
batch_size = 10

# Set entangler map
entangler_map = [[0, 1]]

randoms = np.random.normal(-np.pi * .01, np.pi * .01, 4)

init_dist = qiskit.QuantumCircuit(4)
init_dist.ry(randoms[0], 0)
init_dist.ry(randoms[1], 1)

ansatz = TwoLocal(int(np.sum(num_qubits)), 'rx', 'cz', entanglement=entangler_map, reps=2, insert_barriers=True)

init_params = np.random.rand(ansatz.num_parameters_settable)
print(init_params)

train_data = []
for _ in range(500):
	x2 = np.random.uniform(1, .95, (2,))
	fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (4,))
	real_datapoint = np.array([1., 0., 0., 0.])
	train_data.append((real_datapoint, fake_datapoint))

g_circuit = ansatz.compose(init_dist, front=True)
print(g_circuit)

generator = QuantumGenerator(training_data=train_data,
                             mini_batch_size=batch_size,
                             num_qubits=num_qubits,
                             generator_circuit=g_circuit,
                             shots=2048)

lp = LineProfiler()
lp_wrapper_imagev2 = lp(generator.get_output_pixels)

lp_wrapper_imagev2()
lp.print_stats()
