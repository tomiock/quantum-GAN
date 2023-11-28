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
list_epochs = [200]

for num_epochs in list_epochs:
    discriminator_ancilla = \
        ClassicalDiscriminator(sizes=[4, 16, 8, 1], type_loss="minimax")
    discriminator_not_ancilla = \
        ClassicalDiscriminator(sizes=[4, 16, 8, 1], type_loss="minimax")

    generator_ancilla = \
        QuantumGenerator(num_qubits=4,
                         generator_circuit=None,
                         num_qubits_ancilla=2,
                         shots=4096)
    generator_not_ancilla = \
        QuantumGenerator(num_qubits=2,
                         generator_circuit=None,
                         num_qubits_ancilla=0,
                         shots=4096)

    circuit_ancilla = generator_ancilla.construct_circuit(None, False)
    circuit_not_ancilla = generator_not_ancilla.construct_circuit(None, False)

    init_parameters_ancilla = np.random.normal(np.pi / 2, .1, circuit_ancilla.num_parameters)

    not_ancilla_list = init_parameters_ancilla.tolist()[
                       circuit_ancilla.num_parameters - circuit_not_ancilla.num_parameters:]
    init_parameters_not_ancilla = np.array(not_ancilla_list)

    quantum_gan_ancilla = \
        QuantumGAN(generator_ancilla, discriminator_ancilla)
    time.sleep(1)
    quantum_gan_not_ancilla = \
        QuantumGAN(generator_not_ancilla, discriminator_not_ancilla)

    print(quantum_gan_not_ancilla.path)
    print(quantum_gan_ancilla.path)

    generator_ancilla.parameter_values = init_parameters_ancilla
    quantum_gan_ancilla.train(num_epochs, train_data, BATCH_SIZE, .1, .1, False)

    generator_not_ancilla.parameter_values = init_parameters_not_ancilla
    quantum_gan_not_ancilla.train(num_epochs, train_data, BATCH_SIZE, .1, .1, False)

    quantum_gan_ancilla.plot()
    quantum_gan_not_ancilla.plot()
