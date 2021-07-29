from typing import cast
from line_profiler import LineProfiler

import numpy as np
import qiskit
from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

shots = 2048
num_qubits = [2]
entangler_map = [[0, 1]]

randoms = np.random.normal(-np.pi * .01, np.pi * .01, 2)

init_dist = qiskit.QuantumCircuit(2)
init_dist.ry(randoms[0], 0)
init_dist.ry(randoms[1], 1)

ansatz = TwoLocal(int(np.sum(num_qubits)), 'rx', 'cz', entanglement=entangler_map, reps=2, insert_barriers=True)

g_circuit = ansatz.compose(init_dist, front=True)
parameter_values = np.random.rand(g_circuit.num_parameters)

batch_noise = []
for _ in range(100):
	batch_noise.append(np.random.uniform(-np.pi * .01, np.pi * .01, (2,)))


def construct_circuit(params):
	return g_circuit.assign_parameters(params)


def get_output_V1():
	for noise in batch_noise:
		real_keys = {"00", "10", "01", "11"}

		quantum = QuantumRegister(sum(num_qubits), name="q")
		qc = QuantumCircuit(sum(num_qubits))

		init_dist = qiskit.QuantumCircuit(sum(num_qubits))
		assert noise.shape[0] == sum(num_qubits)

		for num_qubit in range(sum(num_qubits)):
			init_dist.ry(noise[num_qubit], num_qubit)

		params = cast(np.ndarray, parameter_values)

		qc.append(construct_circuit(params), quantum)
		final_circuit = qc.compose(init_dist, front=True)
		final_circuit.measure_all()

		simulator_1 = qiskit.Aer.get_backend("aer_simulator")
		final_circuit = qiskit.transpile(final_circuit, simulator_1)
		result = simulator_1.run(final_circuit, shots=shots).result()
		counts = result.get_counts(final_circuit)

		try:
			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		except KeyError:
			# dealing with the keys that qiskit doesn't include in the
			# dictionary because they don't get any measurements

			keys = counts.keys()
			missing_keys = real_keys.difference(keys)
			# we use sets to get the missing keys
			for key_missing in missing_keys:
				counts[key_missing] = 0

			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		pixels = pixels / shots
		print(pixels)


from qiskit.providers.aer import AerSimulator


def get_output_V2_5():
	aersim = AerSimulator()

	for noise in batch_noise:
		real_keys = {"00", "10", "01", "11"}

		quantum = QuantumRegister(sum(num_qubits), name="q")
		qc = QuantumCircuit(sum(num_qubits))

		init_dist = qiskit.QuantumCircuit(sum(num_qubits))
		assert noise.shape[0] == sum(num_qubits)

		for num_qubit in range(sum(num_qubits)):
			init_dist.ry(noise[num_qubit], num_qubit)

		params = cast(np.ndarray, parameter_values)

		qc.append(construct_circuit(params), quantum)
		final_circuit = qc.compose(init_dist, front=True)
		final_circuit.measure_all()
		# print(final_circuit)

		result_ideal = qiskit.execute(final_circuit, aersim).result()
		counts = result_ideal.get_counts()

		# final_circuit = qiskit.transpile(final_circuit, simulator)
		# result = simulator.run(final_circuit, shots=shots).result()
		# counts = result.get_counts(final_circuit)

		try:
			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		except KeyError:
			# dealing with the keys that qiskit doesn't include in the
			# dictionary because they don't get any measurements

			keys = counts.keys()
			missing_keys = real_keys.difference(keys)
			# we use sets to get the missing keys
			for key_missing in missing_keys:
				counts[key_missing] = 0

			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		pixels = pixels / shots
		print(pixels)


def get_output_V3():
	simulator = qiskit.Aer.get_backend("aer_simulator")
	real_keys = {"00", "10", "01", "11"}

	for noise in batch_noise:
		init_dist = qiskit.QuantumCircuit(sum(num_qubits))
		assert noise.shape[0] == sum(num_qubits)

		for num_qubit in range(sum(num_qubits)):
			init_dist.ry(noise[num_qubit], num_qubit)

		params = cast(np.ndarray, parameter_values)

		quantum = QuantumRegister(sum(num_qubits), name="q")

		qc = QuantumCircuit(sum(num_qubits))
		qc.append(construct_circuit(params), quantum)
		final_circuit = qc.compose(init_dist, front=True)
		final_circuit.measure_all()

		final_circuit = qiskit.compiler.transpile(final_circuit, simulator)
		result = simulator.run(final_circuit, shots=shots).result()
		counts = result.get_counts(final_circuit)

		try:
			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		except KeyError:
			# dealing with the keys that qiskit doesn't include in the
			# dictionary because they don't get any measurements

			keys = counts.keys()
			missing_keys = real_keys.difference(keys)
			# we use sets to get the missing keys
			for key_missing in missing_keys:
				counts[key_missing] = 0

			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		pixels = pixels / shots
		print(pixels)


lp = LineProfiler()

lp_wrapper_v1 = lp(get_output_V1)
lp_wrapper_v1()

lp_wrapper_v2 = lp(get_output_V2_5)
lp_wrapper_v2()

lp_wrapper_v3 = lp(get_output_V3)
lp_wrapper_v3()

lp.print_stats()
