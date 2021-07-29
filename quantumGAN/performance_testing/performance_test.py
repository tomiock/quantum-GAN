from line_profiler import LineProfiler

import numpy as np
import qiskit
from qiskit.circuit.library import TwoLocal

from quantumGAN.performance_testing.performance_quantum_generator import PerformanceQuantumGeneratorV1
from quantumGAN.performance_testing.performance_quantum_generator import PerformanceQuantumGeneratorV2

from quantumGAN.discriminator import Network

lp = LineProfiler()


def mainV1():
	seed = 71
	np.random.seed = seed

	num_qubits = [2]
	batch_size = 10
	entangler_map = [[0, 1]]

	randoms = np.random.normal(-np.pi * .01, np.pi * .01, 2)

	init_dist = qiskit.QuantumCircuit(2)
	init_dist.ry(randoms[0], 0)
	init_dist.ry(randoms[1], 1)

	ansatz = TwoLocal(int(np.sum(num_qubits)), 'rx', 'cz', entanglement=entangler_map, reps=2, insert_barriers=True)

	train_data = []
	for _ in range(15):
		x2 = np.random.uniform(.5, .4, (2,))
		fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (2,))
		real_datapoint = np.array([x2[1], 0., x2[0], 0])
		train_data.append((real_datapoint, fake_datapoint))

	g_circuit = ansatz.compose(init_dist, front=True)

	discriminator = Network(training_data=train_data,
	                        mini_batch_size=batch_size,
	                        sizes=[4, 16, 8, 1],
	                        loss_BCE=True)
	generator = PerformanceQuantumGeneratorV1(training_data=train_data,
	                                          mini_batch_size=batch_size,
	                                          num_qubits=num_qubits,
	                                          generator_circuit=g_circuit,
	                                          shots=2048,
	                                          learning_rate=.1)
	generator.set_discriminator(discriminator)

	for o in range(num_epochs):
		mini_batches = discriminator.create_mini_batches()
		for mini_batch in mini_batches:
			output_real = mini_batch[0][0]
			output_fake = generator.get_output(latent_space_noise=mini_batch[0][1],
			                                   params=None)
			generator.set_mini_batch(mini_batch)
			generator.shots = 2048
			generator.train_mini_batch()

			discriminator.train_mini_batch(generator.mini_batch, .1, o)
		print("Epoch {}: Loss: {}".format(o, discriminator.ret["loss"][-1]), output_real, output_fake)
		print(discriminator.ret["label real"][-1], discriminator.ret["label fake"][-1])


num_epochs = 10

lp_wrapper_main_V1 = lp(mainV1)
lp_wrapper_main_V1()

lp.print_stats()
