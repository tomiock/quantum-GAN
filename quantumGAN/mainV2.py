##
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit.circuit.library import TwoLocal

from quantumGAN.quantum_generator import QuantumGenerator
from quantumGAN.discrimintorV2 import DiscriminatorV2


##
def gen_real_data_FCNN(a, b, num_samples: int):
	data = []
	x0 = np.random.uniform(a, b, num_samples)
	x1 = np.random.uniform(a, b, num_samples)

	for i in range(len(x1)):
		array = [[x0[i], 0], [x1[i], 0]]
		data.append(array)

	return np.array(data).flatten()


seed = 71
np.random.seed = seed
##
# Set the data resolution
# Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
bounds = np.array([0., 3.])
# Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
num_qubits = [2]
k = len(num_qubits)

# Set number of training epochs
num_epochs = 10000
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

g_circuit = ansatz.compose(init_dist, front=True)
print(g_circuit)
discriminator = DiscriminatorV2(4, 1)
generator = QuantumGenerator(num_qubits=num_qubits, generator_circuit=g_circuit)
generator.set_discriminator(discriminator)

##
for o in range(num_epochs):
	output_real = gen_real_data_FCNN(.4, .6, 1)
	output_fake = generator.get_output(params=None, shots=2048)

	print(output_real, output_fake)
	discriminator.step(output_real, output_fake, learning_rate=.1)
	generator.step(.1, 2048)

##
# Plot progress w.r.t the generator's and the discriminator's loss function
t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in the loss function")
plt.plot(t_steps, generator._ret["loss"], label='Generator loss function', color='mediumvioletred', linewidth=2)
plt.plot(t_steps, discriminator._ret["loss"], label='Discriminator loss function', color='rebeccapurple', linewidth=2)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('loss')
plt.show()
