##
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal
from qiskit_finance.circuit.library import UniformDistribution

from qiskit.utils import QuantumInstance, algorithm_globals
from quantumGAN.qgan import QGAN
from quantumGAN.numpy_discriminator import NumPyDiscriminator
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
algorithm_globals.random_seed = seed

N = 400
e1 = .8
e2 = .6
real_data = gen_real_data_FCNN(e1, e2, N)
print(len(real_data))
##
from quantumGAN.qgan import QGAN

# Set the data resolution
# Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
bounds = np.array([0., 3.])
# Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
num_qubits = [4]
k = len(num_qubits)

# Set number of training epochs
# Note: The algorithm's runtime can be shortened by reducing the number of training epochs.
num_epochs = 500
# Batch size
batch_size = 10

# Set entangler map
entangler_map = [[0, 1], [1, 2], [2, 3]]

# Set an initial state for the generator circuit
init_dist = UniformDistribution(sum(num_qubits))

q_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                             seed_transpiler=seed,
                             seed_simulator=seed)

# Set the ansatz circuit
ansatz = TwoLocal(int(np.sum(num_qubits)), 'ry', 'cz', entanglement=entangler_map, reps=3, insert_barriers=True)

# Set generator's initial parameters - in order to reduce the training time and hence the
# total running time for this notebook
# init_params = [3., 1., 0.6, 1.6]

# You can increase the number of training epochs and use random initial parameters.
init_params = np.random.rand(ansatz.num_parameters_settable) * 2 * np.pi
print(init_params)

# Set generator circuit by adding the initial distribution infront of the ansatz
g_circuit = ansatz  # .compose(init_dist, front=True)
print(g_circuit)
discriminator = DiscriminatorV2(4, 1)
generator = QuantumGenerator(num_qubits=[4], generator_circuit=g_circuit)
generator.set_discriminator(discriminator)

##
for o in range(num_epochs):
	output_real = gen_real_data_FCNN(.9, .8, 1)
	output_fake = generator.get_output(quantum_instance=q_instance, params=None, shots=None)

	print(output_real, output_fake)
	discriminator.step(fake_image=output_fake, real_image=output_real, learning_rate=.1)
	generator.step(q_instance, 0.1, 2024)
##
exit()
# Plot progress w.r.t the generator's and the discriminator's loss function
t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in the loss function")
plt.plot(t_steps, g_loss, label='Generator loss function', color='mediumvioletred', linewidth=2)
plt.plot(t_steps, d_loss, label='Discriminator loss function', color='rebeccapurple', linewidth=2)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('loss')
plt.show()
#
