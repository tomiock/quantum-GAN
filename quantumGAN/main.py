import numpy as np

from quantumGAN.discriminator_functional import ClassicalDiscriminator_that_works
from quantumGAN.qgan import Quantum_GAN
from quantumGAN.quantum_generator import QuantumGenerator

num_qubits: int = 3

# Set number of training epochs
num_epochs = 150
# Batch size
batch_size = 10

train_data = []
for _ in range(800):
	x2 = np.random.uniform(.55, .46, (2,))
	fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (num_qubits,))
	real_datapoint = np.array([x2[0], 0, x2[0], 0])
	train_data.append((real_datapoint, fake_datapoint))

discriminator = ClassicalDiscriminator_that_works(sizes=[4, 16, 8, 1],
                                                  type_loss="minimax"  # ,functions=["relu", "relu", "sigmoid" ]
                                                  )
generator = QuantumGenerator(num_qubits=num_qubits,
                             generator_circuit=None,
                             num_qubits_ancilla=1,
                             shots=4096)

quantum_gan = Quantum_GAN(generator, discriminator)
print(quantum_gan)
print(num_epochs)
quantum_gan.generator.get_output(fake_datapoint[0], None)
#quantum_gan.train(num_epochs, train_data, batch_size, .1, .1, True)

#quantum_gan.plot()
#quantum_gan.create_gif()
#quantum_gan.save()
