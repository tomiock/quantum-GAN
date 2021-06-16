##
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal
from qiskit_finance.circuit.library import UniformDistribution

from qiskit.utils import QuantumInstance, algorithm_globals
from quantumGAN.qgan import QGAN
from quantumGAN.numpy_discriminator import NumPyDiscriminator

##

N = 1000

# Load data samples from log-normal distribution with mean=1 and standard deviation=1
mu = 1
sigma = 1
real_data = np.random.lognormal(mean=mu, sigma=sigma, size=N)


##

def gen_real_data_FCNN(a, b, num_samples: int):
	real_data = []
	real_labels = []
	x0 = np.random.default_rng().uniform(a, b, num_samples)
	x1 = np.random.default_rng().uniform(a, b, num_samples)

	for i in range(len(x1)):
		array = np.array([[x0[i], 0], [x1[i], 0]]).reshape((2, 2))
		real_data.append(array)

	return real_data


seed = 71
np.random.seed = seed
algorithm_globals.random_seed = seed

N = 400
e1 = .8
e2 = .6
real_data = np.asarray(gen_real_data_FCNN(e1, e2, N))
real_data = real_data.reshape(1600, )
real_data = real_data.flatten()
print(real_data)
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
num_epochs = 20
# Batch size
batch_size = 100
quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                   seed_transpiler=seed,
                                   seed_simulator=seed)

# Initialize qGAN
qgan = QGAN(real_data,
            bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None, num_shots=4048,
            quantum_instance=quantum_instance)
qgan.seed = 1
# Set quantum instance to run the quantum generator


# Set entangler map
entangler_map = [[0, 1], [1, 2], [2, 3]]

# Set an initial state for the generator circuit
init_dist = UniformDistribution(sum(num_qubits))

# Set the ansatz circuit
ansatz = TwoLocal(int(np.sum(num_qubits)), 'ry', 'cz', entanglement=entangler_map, reps=2, insert_barriers=True)

# Set generator's initial parameters - in order to reduce the training time and hence the
# total running time for this notebook
# init_params = [3., 1., 0.6, 1.6]

# You can increase the number of training epochs and use random initial parameters.
init_params = np.random.rand(ansatz.num_parameters_settable) * 2 * np.pi
print(init_params)

# Set generator circuit by adding the initial distribution infront of the ansatz
g_circuit = ansatz.compose(init_dist, front=True)
print(g_circuit)

# Set quantum generator
qgan.set_generator(generator_circuit=g_circuit, generator_init_params=init_params)
# The parameters have an order issue that following is a temp. workaround
qgan._generator._free_parameters = sorted(g_circuit.parameters, key=lambda p: p.name)
# Set classical discriminator neural network
discriminator = NumPyDiscriminator(len(num_qubits))
qgan.set_discriminator(discriminator)

##
result = qgan.get_output(quantum_instance, shots=4048)
print(result)

##
train_results = qgan.run()

##
print('Training results:')
for key, value in train_results.items():
	print(f'  {key} : {value}')

##
# Plot progress w.r.t the generator's and the discriminator's loss function
t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in the loss function")
plt.plot(t_steps, qgan.g_loss, label='Generator loss function', color='mediumvioletred', linewidth=2)
plt.plot(t_steps, qgan.d_loss, label='Discriminator loss function', color='rebeccapurple', linewidth=2)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('loss')
plt.show()
exit()
##
# Plot progress w.r.t relative entropy
plt.figure(figsize=(6, 5))
plt.title('Relative Entropy')
plt.plot(np.linspace(0, num_epochs, len(qgan.rel_entr)), qgan.rel_entr, color='mediumblue', lw=4, ls=':')
plt.grid()
plt.xlabel('time steps')
plt.ylabel('relative entropy')
plt.show()

##
# Plot the CDF of the resulting distribution against the target distribution, i.e. log-normal
log_normal = np.random.lognormal(mean=1, sigma=1, size=100000)
log_normal = np.round(log_normal)
log_normal = log_normal[log_normal <= bounds[1]]
temp = []
for i in range(int(bounds[1] + 1)):
	temp += [np.sum(log_normal == i)]
log_normal = np.array(temp / sum(temp))

plt.figure(figsize=(6, 5))
plt.title('CDF (Cumulative Distribution Function)')
samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
samples_g = np.array(samples_g)
samples_g = samples_g.flatten()
num_bins = len(prob_g)
plt.bar(samples_g, np.cumsum(prob_g), color='royalblue', width=0.8, label='simulation')
plt.plot(np.cumsum(log_normal), '-o', label='log-normal', color='deepskyblue', linewidth=4, markersize=12)
plt.xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
plt.grid()
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend(loc='best')
plt.show()
