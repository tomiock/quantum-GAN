import multiprocessing
import time

import numpy as np

from quantumGAN.discriminator import ClassicalDiscriminator
from quantumGAN.qgan import QuantumGAN
from quantumGAN.quantum_generator import QuantumGenerator

BATCH_SIZE = 10

train_data = []
for _ in range(800):
    x2 = np.random.uniform(.55, .46, (2,))
    fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (4,))
    real_datapoint = np.array([x2[0], 0, x2[0], 0])
    train_data.append((real_datapoint, fake_datapoint))


def to_train(arguments):
	num_epochs, num_qubits, num_qubits_ancilla = arguments

	print(num_qubits, num_qubits_ancilla)

	discriminator = ClassicalDiscriminator(sizes=[4, 16, 8, 1],
	                                       type_loss="minimax"
	                                       )
	generator = QuantumGenerator(num_qubits=num_qubits,
	                             generator_circuit=None,
	                             num_qubits_ancilla=num_qubits_ancilla,
	                             shots=4096)

	quantum_gan = QuantumGAN(generator, discriminator)
	print(quantum_gan)
	print(num_epochs)
	quantum_gan.discriminator.init_parameters()
	quantum_gan.train(num_epochs, train_data, batch_size, .1, .1, True)

	quantum_gan.plot()
	quantum_gan.create_gif()


list_processes = [(700, 4, 2), (700, 2, 0)]


def main():
	jobs = []
	for arguments in list_processes:
		simulate = multiprocessing.Process(None, to_train, args=(arguments,))
		jobs.append(simulate)
		time.sleep(2)
		simulate.start()


if __name__ == '__main__':
	main()
