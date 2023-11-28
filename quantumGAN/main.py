import numpy as np

from quantumGAN.discriminator import ClassicalDiscriminator
from quantumGAN.qgan import QuantumGAN
from quantumGAN.quantum_generator import QuantumGenerator

NUM_QUBITS: int = 2

NUM_EPOCH = 10
BATCH_SIZE = 10

train_data = []
for _ in range(800):
    x2 = np.random.uniform(.55, .46, (2,))
    fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (NUM_QUBITS,))
    real_datapoint = np.array([x2[0], 0, x2[0], 0])
    train_data.append((real_datapoint, fake_datapoint))

discriminator = ClassicalDiscriminator(sizes=[4, 16, 8, 1],
                                       type_loss="minimax"  # ,functions=["relu", "relu", "sigmoid" ]
                                       )
generator = QuantumGenerator(num_qubits=NUM_QUBITS,
                             generator_circuit=None,
                             num_qubits_ancilla=0,
                             shots=4096)

quantum_gan = QuantumGAN(generator, discriminator)

print(quantum_gan)
print(NUM_EPOCH)

quantum_gan.discriminator.init_parameters()
quantum_gan.train(NUM_EPOCH, train_data, BATCH_SIZE, .1, .1, False)

quantum_gan.plot()
quantum_gan.save()
