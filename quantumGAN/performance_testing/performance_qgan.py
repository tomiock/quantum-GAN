import glob
import json
import os
import random
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np

from quantumGAN.discriminator import ClassicalDiscriminator
from quantumGAN.functions import minimax
from quantumGAN.quantum_generator import QuantumGenerator


class Quantum_GAN:

    def __init__(self,
                 generator: QuantumGenerator,
                 discriminator: ClassicalDiscriminator
                 ):

        now = datetime.now()
        init_time = now.strftime("%d_%m_%Y__%H_%M_%S")
        self.path = "data/run{}".format(init_time)
        self.path_images = self.path + "/images"
        self.filename = "run.txt"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(self.path_images):
            os.makedirs(self.path_images)

        with open(os.path.join(self.path, self.filename), "w") as file:
            file.write("RUN {} \n".format(init_time))
            file.close()

        self.generator = generator
        self.discriminator = discriminator
        self.loss_series, self.label_real_series, self.label_fake_series = [], [], []

        self.generator.init_parameters()
        self.example_g_circuit = self.generator.construct_circuit(latent_space_noise=None,
                                                                  to_measure=False)
        self.generator.set_discriminator(self.discriminator)
        self.num_epochs = None
        self.training_data = None
        self.batch_size = None
        self.batch_size = None
        self.generator_lr = None
        self.discriminator_lr = None

    def __repr__(self):
        return "Discriminator Architecture: {} \n Generator Example Circuit: \n{}" \
            .format(self.discriminator.sizes, self.example_g_circuit)

    def store_info(self, epoch, loss, real_label, fake_label):
        file = open(os.path.join(self.path, self.filename), "a")
        file.write("{} epoch LOSS {} Parameters {} REAL {} FAKE {}\n"
                   .format(epoch,
                           loss,
                           self.generator.parameter_values,
                           real_label,
                           fake_label))
        file.close()

    def plot(self):
        # save data for plotting
        t_steps = np.arange(self.num_epochs)
        plt.figure(figsize=(6, 5))
        plt.title("Progress in the loss function")
        plt.plot(t_steps, self.loss_series, label='Discriminator loss function', color='rebeccapurple', linewidth=2)
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('time steps')
        plt.ylabel('loss')
        plt.show()

        t_steps = np.arange(self.num_epochs)
        plt.figure(figsize=(6, 5))
        plt.title("Progress in labels")
        plt.scatter(t_steps, self.label_real_series, label='Label for real images',
                    color='mediumvioletred',
                    linewidth=.1)
        plt.scatter(t_steps, self.label_fake_series, label='Label for fake images',
                    color='rebeccapurple',
                    linewidth=.1)
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('time steps')
        plt.ylabel('label')
        plt.show()

    def train(self):

        noise = self.training_data[0][1]

        for o in range(self.num_epochs):
            mini_batches = create_mini_batches(self.training_data, self.batch_size)
            output_fake = self.generator.get_output(latent_space_noise=mini_batches[0][0][1], parameters=None)

            for mini_batch in mini_batches:
                mini_batch = self.generator.train_mini_batch(mini_batch, self.generator_lr)
                self.discriminator.train_mini_batch(mini_batch, self.discriminator_lr)

            output_real = mini_batches[0][0][0]
            #image = self.generator.get_output(latent_space_noise=noise, parameters=None)
#
            #image_shape = int(image.shape[0] / 2)
            #image = image.reshape(image_shape, image_shape)
#
            #plt.imshow(image, cmap='gray', vmax=1., vmin=0.)
            #plt.axis('off')
            #plt.savefig(self.path_images + '/image_at_epoch_{}.png'.format(o))

            label_real, label_fake = self.discriminator.predict(output_real), self.discriminator.predict(output_fake)
            loss_final = 1 / 2 * (minimax(label_real, label_fake) + minimax(label_real, label_fake))

            self.loss_series.append(loss_final)
            self.label_real_series.append(label_real)
            self.label_fake_series.append(label_fake)

            #print("Epoch {}: Loss: {}".format(o, loss_final), output_real, output_fake)
            #print(label_real[-1], label_fake[-1])
            self.store_info(o, loss_final, label_real, label_fake)

    def create_gif(self):
        anim_file = self.path + '/dcgan.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(self.path_images + '/image*.png')
            filenames = sorted(filenames)

            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

            image = imageio.imread(filename)
            writer.append_data(image)

    def save(self):
        """Save the neural network to the file ``filename``."""
        data = {"batch_size": self.batch_size,
                "D_sizes": self.discriminator.sizes,
                "D_weights": [w.tolist() for w in self.discriminator.weights],
                "D_biases": [b.tolist() for b in self.discriminator.biases],
                "D_loss": self.discriminator.type_loss,
                "Q_parameters": [theta for theta in self.generator.parameter_values],
                "Q_shots": self.generator.shots,
                "Q_num_qubits": self.generator.num_qubits_total,
                "Q_num_qubits_ancilla": self.generator.num_qubits_ancilla,
                "real_labels": self.label_real_series,
                "fake_labels": self.label_fake_series,
                "loss_series": self.loss_series
                }
        f = open(os.path.join(self.path, "data.txt"), "a")
        json.dump(data, f)
        f.close()

def create_mini_batches(training_data, mini_batch_size):
    n = len(training_data)
    random.shuffle(training_data)
    mini_batches = [
        training_data[k:k + mini_batch_size]
        for k in range(0, n, mini_batch_size)]
    return [mini_batches[0]]

def load_gan(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    discriminator = ClassicalDiscriminator(data["D_sizes"], data["D_loss"])

    generator = QuantumGenerator(num_qubits=data["Q_num_qubits"],
                                 generator_circuit=None,
                                 num_qubits_ancilla=data["Q_num_qubits_ancilla"],
                                 shots=data["Q_shots"])

    quantum_gan = Quantum_GAN(generator, discriminator)

    quantum_gan.discriminator.weights = [np.array(w) for w in data["D_weights"]]
    quantum_gan.discriminator.biases = [np.array(b) for b in data["D_biases"]]
    quantum_gan.generator.parameter_values = np.array(data["Q_parameters"])

    return quantum_gan, data["batch_size"]