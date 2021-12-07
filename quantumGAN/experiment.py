import time

import numpy as np

from quantumGAN.discriminator import ClassicalDiscriminator
from quantumGAN.qgan import Quantum_GAN
from quantumGAN.quantum_generator import QuantumGenerator

# nombre d'imatges en el batch
batch_size = 10

train_data = []
# generació de les dades
for _ in range(800):
	x2 = np.random.uniform(.55, .46, (2,))
	fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (4,))
	real_datapoint = np.array([x2[0], 0, x2[0], 0])
	train_data.append((real_datapoint, fake_datapoint))
list_epochs = [100, 200, 300, 400]

for num_epochs in list_epochs:
	# definició d'un discriminador pel model amb la funció no-lineal i una altre pel model que no la té
	discriminator_ancilla = \
		ClassicalDiscriminator(sizes=[4, 16, 8, 1], type_loss="minimax")
	discriminator_not_ancilla = \
		ClassicalDiscriminator(sizes=[4, 16, 8, 1], type_loss="minimax")

	# es defineix un generador que té la funció no-lienal integrada en el circuit
	generator_ancilla = \
		QuantumGenerator(num_qubits=4,
		                 generator_circuit=None,
		                 num_qubits_ancilla=2,  # IMPORTANT
		                 shots=4096)
	# i un altre generador que no té la funció no-lineal
	generator_not_ancilla = \
		QuantumGenerator(num_qubits=2,
		                 generator_circuit=None,
		                 num_qubits_ancilla=0,  # IMPORTANT
		                 shots=4096)

	# generem els paràmetres inicials d'acord amb el circuit amb un mayor nombre de paràmetres
	circuit_ancilla = generator_ancilla.construct_circuit(None, False)
	circuit_not_ancilla = generator_not_ancilla.construct_circuit(None, False)

	init_parameters_ancilla = np.random.normal(np.pi / 2, .1, circuit_ancilla.num_parameters)

	# canviem les dimensions dels paràmetres perquè coincideixin amb les dimensions necessàries pel circuit
	# sense qubits ancilla
	not_ancilla_list = init_parameters_ancilla.tolist()[circuit_ancilla.num_parameters - circuit_not_ancilla.num_parameters:]
	init_parameters_not_ancilla = np.array(not_ancilla_list)

	quantum_gan_ancilla = \
		Quantum_GAN(generator_ancilla, discriminator_ancilla)
	time.sleep(1)
	quantum_gan_not_ancilla = \
		Quantum_GAN(generator_not_ancilla, discriminator_not_ancilla)
	print(quantum_gan_not_ancilla.path)
	print(quantum_gan_ancilla.path)


	# abans de començar a entrenar el model s'ha de canviar els paràmetres als definits anteriorment
	generator_ancilla.parameter_values = init_parameters_ancilla
	quantum_gan_ancilla.train(num_epochs, train_data, batch_size, .1, .1, False)

	generator_not_ancilla.parameter_values = init_parameters_not_ancilla
	quantum_gan_not_ancilla.train(num_epochs, train_data, batch_size, .1, .1, False)

	# una vegada ha acabat l'optimització es creen els gràfics per compara l'eficiència del models
	quantum_gan_ancilla.plot()
	quantum_gan_not_ancilla.plot()
