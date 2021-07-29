import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit.circuit.library import TwoLocal

from quantumGAN.quantum_generator import QuantumGenerator
from quantumGAN.discriminator import Network

seed = 71
np.random.seed = seed

# Set the data resolution
# Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
bounds = np.array([0., 3.])
# Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
num_qubits = [2]
k = len(num_qubits)

# Set number of training epochs
num_epochs = 200
# Batch size
batch_size = 10

# Set entangler map
# entangler_map = [[0, 1], [1, 2], [2, 3]
entangler_map = [[0, 1]]

# Set an initial state for the generator circuit

randoms = np.random.normal(-np.pi * .01, np.pi * .01, 2)

init_dist = qiskit.QuantumCircuit(2)
init_dist.ry(randoms[0], 0)
init_dist.ry(randoms[1], 1)

ansatz = TwoLocal(int(np.sum(num_qubits)), 'rx', 'cz', entanglement=entangler_map, reps=2, insert_barriers=True)

init_params = np.random.rand(ansatz.num_parameters_settable)
print(init_params)

train_data = []
for _ in range(15):
    x2 = np.random.uniform(.5, .4, (2,))
    fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (2,))
    real_datapoint = np.array([x2[1], 0., x2[0], 0])
    train_data.append((real_datapoint, fake_datapoint))

g_circuit = ansatz.compose(init_dist, front=True)
print(g_circuit)

discriminator = Network(training_data=train_data,
                        mini_batch_size=batch_size,
                        sizes=[4, 16, 8, 1],
                        loss_BCE=True)
generator = QuantumGenerator(training_data=train_data,
                             mini_batch_size=batch_size,
                             num_qubits=num_qubits,
                             generator_circuit=g_circuit,
                             shots=2048)
generator.set_discriminator(discriminator)

for o in range(num_epochs):
    mini_batches = discriminator.create_mini_batches()
    for mini_batch in mini_batches:
        output_real = mini_batch[0][0]
        output_fake = generator.get_output(latent_space_noise=mini_batch[0][1],
                                           params=None)
        mini_batch = generator.train_mini_batch(mini_batch, 2048)
        discriminator.train_mini_batch(mini_batch, .1, o)
    print("Epoch {}: Loss: {}".format(o, discriminator.ret["loss"][-1]), output_real, output_fake)
    print(discriminator.ret["label real"][-1], discriminator.ret["label fake"][-1])

# Plot progress w.r.t the generator's and the discriminator's loss function
t_steps = np.arange(num_epochs * 2)
plt.figure(figsize=(6, 5))
plt.title("Progress in the loss function")
# plt.plot(t_steps, generator.ret["loss"], label='Generator loss function', color='mediumvioletred', linewidth=2)
plt.plot(t_steps, discriminator.ret["loss"], label='Discriminator loss function', color='rebeccapurple', linewidth=2)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('loss')
plt.show()

t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in labels")
plt.scatter(discriminator.ret["label real time"], discriminator.ret["label real"], label='Label for real images',
            color='mediumvioletred',
            linewidth=.1)
plt.scatter(discriminator.ret["label fake time"], discriminator.ret["label fake"], label='Label for fake images',
            color='rebeccapurple',
            linewidth=.1)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('label')
plt.show()
