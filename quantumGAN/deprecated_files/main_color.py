import glob

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from qiskit.circuit.library import TwoLocal

from quantumGAN.discriminator import ClassicalDiscriminator
from quantumGAN.functions import create_entangler_map, minimax, save_images_color
from quantumGAN.quantum_generator import QuantumGenerator
from quantumGAN.deprecated_files.rgb_functions import qcolor_to_image

num_qubits = 2

# Set number of training epochs
num_epochs = 500
# Batch size
batch_size = 1

# Set entangler map
entangler_map = create_entangler_map(num_qubits)

randoms = np.random.normal(-np.pi * .01, np.pi * .01, num_qubits)

init_dist = qiskit.QuantumCircuit(num_qubits)
for index in range(num_qubits):
    init_dist.ry(randoms[index], index)

ansatz = TwoLocal(num_qubits, 'ry', 'cx', entanglement=entangler_map, reps=1, insert_barriers=True)

init_params = np.random.rand(ansatz.num_parameters_settable)

train_data = []
for _ in range(800):
    x2 = np.random.uniform(255, 253, (6,))
    fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (num_qubits,))
    real_datapoint = np.array([50, 168, 82]).flatten()
    train_data.append((real_datapoint.flatten(), fake_datapoint))

g_circuit = ansatz.compose(init_dist, front=True)
print(g_circuit)

discriminator = ClassicalDiscriminator(training_data=train_data,
                                       mini_batch_size=batch_size,
                                       sizes=[3, 64, 16, 1],
                                       type_loss="minimax")
#                                       functions=["relu", "relu", "sigmoid" ])
generator = QuantumGenerator(training_data=train_data,
                             mini_batch_size=batch_size,
                             num_qubits=num_qubits,
                             num_qubits_ancilla=0,
                             generator_circuit=g_circuit,
                             shots=2048)
generator.set_discriminator(discriminator)

df_data = pd.DataFrame({"loss": [], "label real": [], "label fake": []})

loss_series, label_real_series, label_fake_series = [], [], []
noise = train_data[0][1]
print(noise)
fake_images = []

for o in range(num_epochs):
    mini_batches = discriminator.create_mini_batches()

    for mini_batch in mini_batches:
        mini_batch = generator.train_mini_batch_color(mini_batch, .1)
        index = 0
        for image in mini_batch:
            mini_batch[index] = (mini_batch[index][0], qcolor_to_image(image[1]).flatten())
            index += 1

        discriminator.train_mini_batch(mini_batch, .1)

    output_real = mini_batches[0][0][0]
    output_fake = generator.get_output(latent_space_noise=mini_batches[0][0][1], params=None)
    output_fake = qcolor_to_image(output_fake).flatten()
    save_images_color(output_fake, 0)

    label_real, label_fake = discriminator.predict(output_real), discriminator.predict(output_fake)
    loss_final = 1 / 2 * (minimax(label_real, label_fake) + minimax(label_real, label_fake))

    loss_series.append(loss_final)
    label_real_series.append(label_real)
    label_fake_series.append(label_fake)

    print("Epoch {}: Loss: {}".format(o, loss_final), output_fake)
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
