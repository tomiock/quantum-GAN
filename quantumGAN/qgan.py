import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import qiskit
from qiskit.circuit.library import TwoLocal

from quantumGAN.functions import BCE
from quantumGAN.quantum_generator import QuantumGenerator
from quantumGAN.discriminator import ClassicalDiscriminator


class Quantum_GAN:

	def __init__(self,
	             generator_learning_rate: float,
	             discriminator_learning_rate: float,
	             mini_batch_size: int,
	             generator: QuantumGenerator,
	             discriminator: ClassicalDiscriminator
	             ):
		self.generator = generator
		self.discriminator = discriminator

	def __repr__(self):
		print("Discriminator architecture: {}")

	def save(self):
		# save parameters, accuracy and sample of images (for reference)
		pass

	def plot(self):
		# save data for plotting
		pass

	def train(self, epochs, training_data):
		pass
