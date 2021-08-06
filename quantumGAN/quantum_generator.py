"""QUANTUM GENERATOR"""
import itertools
from typing import Any, Dict, List, Optional, cast

import numpy as np
import qiskit
from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer import AerSimulator

from quantumGAN.functions import create_real_keys, minimax_generator
from quantumGAN.rgb_functions import qcolor_to_image


class QuantumGenerator:

	def __init__(
			self,
			training_data: List,
			mini_batch_size: int,
			shots: int,
			num_qubits: int,
			generator_circuit: Optional[QuantumCircuit] = None,
			snapshot_dir: Optional[str] = None
	) -> None:

		super().__init__()
		self.training_data = training_data
		self.mini_batch_size = mini_batch_size
		self.num_qubits = num_qubits
		self.generator_circuit = generator_circuit

		if generator_circuit is None:
			circuit = QuantumCircuit(num_qubits)
			randoms = np.random.normal(-np.pi * .01, np.pi * .01, num_qubits)

			for index in range(len(randoms)):
				circuit.ry(randoms[index], index)

			ansatz = TwoLocal(sum([num_qubits]), "ry", "cz", reps=1, entanglement="circular")
			circuit.compose(ansatz, inplace=True)

			# Set generator circuit
			self.generator_circuit = circuit

		self.parameter_values = np.random.rand(self.generator_circuit.num_parameters)
		print(self.parameter_values)

		self.snapshot_dir = snapshot_dir
		self.shots = shots
		self.discriminator = None
		self.ret: Dict[str, Any] = {"loss": []}
		self.simulator = AerSimulator()

	def set_discriminator(self, discriminator) -> None:
		self.discriminator = discriminator

	def construct_circuit(self, params):
		return self.generator_circuit.assign_parameters(params)

	def get_output(
			self,
			latent_space_noise,
			params: Optional[np.ndarray] = None
	):
		real_keys_set, real_keys_list = create_real_keys(self.num_qubits)

		quantum = QuantumRegister(self.num_qubits, name="q")
		qc = QuantumCircuit(self.num_qubits)

		init_dist = qiskit.QuantumCircuit(self.num_qubits)
		assert latent_space_noise.shape[0] == self.num_qubits

		for num_qubit in range(self.num_qubits):
			init_dist.ry(latent_space_noise[num_qubit], num_qubit)

		if params is None:
			params = cast(np.ndarray, self.parameter_values)

		qc.append(self.construct_circuit(params), quantum)
		final_circuit = qc.compose(init_dist, front=True)
		final_circuit.measure_all()

		result_ideal = qiskit.execute(experiments=final_circuit,
		                              backend=self.simulator,
		                              shots=self.shots,
		                              optimization_level=0).result()
		counts = result_ideal.get_counts()

		try:
			pixels = np.array([counts[index] for index in list(real_keys_list)])

		except KeyError:
			# dealing with the keys that qiskit doesn't include in the
			# dictionary because they don't get any measurements
			keys = counts.keys()
			missing_keys = real_keys_set.difference(keys)
			# we use sets to get the missing keys
			for key_missing in missing_keys:
				counts[key_missing] = 0

			pixels = np.array([counts[index] for index in list(real_keys_list)])

		pixels = pixels / self.shots
		return pixels

	def get_output_pixels(
			self,
			latent_space_noise,
			params: Optional[np.ndarray] = None
	):
		quantum = QuantumRegister(self.num_qubits, name="q")
		qc = QuantumCircuit(self.num_qubits)

		init_dist = qiskit.QuantumCircuit(self.num_qubits)
		assert latent_space_noise.shape[0] == self.num_qubits

		for num_qubit in range(self.num_qubits):
			init_dist.ry(latent_space_noise[num_qubit], num_qubit)

		if params is None:
			params = cast(np.ndarray, self.parameter_values)

		qc.append(self.construct_circuit(params), quantum)

		state_vector = qiskit.quantum_info.Statevector.from_instruction(qc)
		pixels = []
		for qubit in range(self.num_qubits):
			pixels.append(state_vector.probabilities([qubit])[0])

		generated_samples = np.array(pixels)
		generated_samples.flatten()

		return generated_samples

	def accuracy(self):
		pass

	def train_mini_batch(self, mini_batch, learning_rate):
		nabla_theta = np.zeros(self.parameter_values.shape)
		new_images = []

		for _, noise in mini_batch:
			for index in range(len(self.parameter_values)):
				perturbation_vector = np.zeros(len(self.parameter_values))
				perturbation_vector[index] = 1

				pos_params = self.parameter_values + (np.pi / 4) * perturbation_vector
				neg_params = self.parameter_values - (np.pi / 4) * perturbation_vector

				pos_result = self.get_output(noise, params=pos_params)
				neg_result = self.get_output(noise, params=neg_params)

				pos_result = self.discriminator.predict(pos_result)
				neg_result = self.discriminator.predict(neg_result)
				gradient = minimax_generator(pos_result) - minimax_generator(neg_result)
				nabla_theta[index] += gradient
			new_images.append(self.get_output(noise))

		for index in range(len(self.parameter_values)):
			self.parameter_values[index] += (learning_rate / self.mini_batch_size) * nabla_theta[index]

		mini_batch = [(datapoint[0], fake_image) for datapoint, fake_image in zip(mini_batch, new_images)]

		return mini_batch

	def train_mini_batch_color(self, mini_batch, learning_rate):
		nabla_theta = np.zeros(self.parameter_values.shape)
		new_images = []

		for _, noise in mini_batch:
			for index in range(len(self.parameter_values)):
				perturbation_vector = np.zeros(len(self.parameter_values))
				perturbation_vector[index] = 1

				pos_params = self.parameter_values + (np.pi / 4) * perturbation_vector
				neg_params = self.parameter_values - (np.pi / 4) * perturbation_vector

				pos_result = self.get_output(noise, params=pos_params)
				neg_result = self.get_output(noise, params=neg_params)

				pos_result = qcolor_to_image(pos_result)
				neg_result = qcolor_to_image(neg_result)

				print(pos_result)

				pos_result = self.discriminator.predict(pos_result.flatten())
				neg_result = self.discriminator.predict(neg_result.flatten())
				gradient = minimax_generator(pos_result) - minimax_generator(neg_result)
				nabla_theta[index] += gradient
			new_images.append(self.get_output(noise))

		for index in range(len(self.parameter_values)):
			self.parameter_values[index] += (learning_rate / self.mini_batch_size) * nabla_theta[index]

		mini_batch = [(datapoint[0], fake_image) for datapoint, fake_image in zip(mini_batch, new_images)]

		return mini_batch
