import itertools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import qiskit
import imageio
import glob
from qiskit.circuit.library import TwoLocal

from quantumGAN.functions import BCE, create_entangler_map, minimax, save_images
from quantumGAN.quantum_generator import QuantumGenerator
from quantumGAN.discriminator import ClassicalDiscriminator
from discriminato_minimax import ClassicalDiscriminatorMINIMAX

num_qubits = [5]
k = len(num_qubits)

# Set number of training epochs
num_epochs = 500
# Batch size
batch_size = 10

# Set entangler map

entangler_map = create_entangler_map(sum(num_qubits))

randoms = np.random.normal(-np.pi * .01, np.pi * .01, 4)

init_dist = qiskit.QuantumCircuit(4)
init_dist.ry(randoms[0], 0)
init_dist.ry(randoms[1], 1)
init_dist.ry(randoms[2], 2)
init_dist.ry(randoms[3], 3)

ansatz = TwoLocal(int(np.sum(num_qubits)), 'ry', 'cz', entanglement=entangler_map, reps=2, insert_barriers=True)

init_params = np.random.rand(ansatz.num_parameters_settable)
print(init_params)

train_data = []
for _ in range(800):
	x2 = np.random.uniform(.55, .46, (4,))
	fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (4,))
	real_datapoint = np.array([0.14285714, 0.14285714, 0.14285714, 0., 0.14285714, 0.,
	                           0.14285714, 0., 0., 0., 0., 0.14285714,
	                           0.14285714, 0., 0., 0., ])
	train_data.append((real_datapoint, fake_datapoint))

g_circuit = ansatz.compose(init_dist, front=True)
print(g_circuit)

discriminator = ClassicalDiscriminator(training_data=train_data,
                                       mini_batch_size=batch_size,
                                       sizes=[16, 64, 8, 1],
                                       type_loss="minimax")
generator = QuantumGenerator(training_data=train_data,
                             mini_batch_size=batch_size,
                             num_qubits=4,
                             generator_circuit=g_circuit,
                             shots=16384)
generator.set_discriminator(discriminator)

df_data = pd.DataFrame({"loss": [], "label real": [], "label fake": []})

loss_series, label_real_series, label_fake_series = [], [], []
noise = train_data[0][1]
print(noise)
fake_images = []

for o in range(num_epochs):
	mini_batches = discriminator.create_mini_batches()
	output_fake = generator.get_output(latent_space_noise=mini_batches[0][0][1], params=None)
	print(output_fake)
	exit()

	for mini_batch in mini_batches:
		mini_batch = generator.train_mini_batch(mini_batch, .1)
		discriminator.train_mini_batch(mini_batch, .1)

	output_real = mini_batches[0][0][0]
	save_images(generator.get_output(latent_space_noise=noise, params=None), o)

	label_real, label_fake = discriminator.predict(output_real), discriminator.predict(output_fake)
	loss_final = 1 / 2 * (minimax(label_real, label_fake) + minimax(label_real, label_fake))

	loss_series.append(loss_final)
	label_real_series.append(label_real)
	label_fake_series.append(label_fake)

	print("Epoch {}: Loss: {}".format(o, loss_final), output_real, output_fake)
	print(label_real[-1], label_fake[-1])

loss = pd.Series(loss_series)
label_real = pd.Series(label_real_series)
label_fake = pd.Series(label_fake_series)

# Plot progress w.r.t the generator's and the discriminator's loss function
t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in the loss function")
plt.plot(t_steps, loss.to_numpy(), label='Discriminator loss function', color='rebeccapurple', linewidth=2)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('loss')
plt.show()

t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in labels")
plt.scatter(t_steps, label_real.to_numpy(), label='Label for real images',
            color='mediumvioletred',
            linewidth=.1)
plt.scatter(t_steps, label_fake.to_numpy(), label='Label for fake images',
            color='rebeccapurple',
            linewidth=.1)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('label')
plt.show()

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
	filenames = glob.glob('images/image*.png')
	filenames = sorted(filenames)
	for filename in filenames:
		image = imageio.imread(filename)
		writer.append_data(image)
	image = imageio.imread(filename)
	writer.append_data(image)
