from ctypes import cast

from qiskit import IBMQ, QuantumCircuit, QuantumRegister, assemble, transpile
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.random import random_circuit
import numpy as np
import qiskit

import logging

from qiskit.providers.aer import AerSimulator

from quantumGAN.performance_testing.performance_get_output_generator import construct_circuit

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
for _ in range(1):
	batch_noise.append(np.random.uniform(-np.pi * .01, np.pi * .01, (2,)))

aersim = AerSimulator()
for noise in batch_noise:
	real_keys = {"00", "10", "01", "11"}

	quantum = QuantumRegister(sum(num_qubits), name="q")
	qc = QuantumCircuit(sum(num_qubits))
	init_dist = qiskit.QuantumCircuit(sum(num_qubits))

	assert noise.shape[0] == sum(num_qubits)
	for num_qubit in range(sum(num_qubits)):
		init_dist.ry(noise[num_qubit], num_qubit)

	qc.append(construct_circuit(parameter_values), quantum)
	final_circuit = qc.compose(init_dist, front=True)
	final_circuit.measure_all()

	# print(final_circuit)
	result_ideal = qiskit.execute(final_circuit,
	                              aersim,
	                              shots=2048,
	                              optimization_level=0).result()

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

exit()
provider = IBMQ.enable_account(
	"c89511dc555c9ee925932a44194e53b04a08dce77de669c241155012fae696106b98047242f6ab0a94348c6598a7338abe4a642ece374241adea384614c29f79")
backend = provider.backend.ibmq_santiago
qx = random_circuit(n_qubits=5, depth=4)

transpiled = transpile(qx, backend=backend)
job = backend.run(transpiled)
retrieved_job = backend.retrieve_job(job.job_id())

status = backend.status()
is_operational = status.operational
jobs_in_queue = status.pending_jobs
