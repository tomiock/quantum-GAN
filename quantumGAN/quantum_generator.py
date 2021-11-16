"""QUANTUM GENERATOR"""

from typing import Any, Dict, List, Optional, cast

import numpy as np
import qiskit
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer import AerSimulator

from quantumGAN.functions import create_entangler_map, create_real_keys, minimax_generator
from quantumGAN.rgb_functions import qcolor_to_image


class QuantumGenerator:

	def __init__(
			self,
			shots: int,
			num_qubits: int,
			num_qubits_ancilla: int,
			generator_circuit: Optional[QuantumCircuit] = None,
			snapshot_dir: Optional[str] = None
	) -> None:

		super().__init__()
		self.num_qubits_total = num_qubits
		self.num_qubits_ancilla = num_qubits_ancilla
		self.generator_circuit = generator_circuit
		self.snapshot_dir = snapshot_dir
		self.shots = shots
		self.discriminator = None
		self.ret: Dict[str, Any] = {"loss": []}
		self.simulator = AerSimulator()

	def init_parameters(self):
		self.generator_circuit = self.construct_circuit(latent_space_noise=None,
		                                                to_measure=False)
		self.parameter_values = np.random.normal(np.pi / 2, .1, self.generator_circuit.num_parameters)

	def construct_circuit(self,
	                      latent_space_noise,
	                      to_measure: bool):
		if self.num_qubits_ancilla == 0:
			qr = QuantumRegister(self.num_qubits_total, 'q')
			cr = ClassicalRegister(self.num_qubits_total, 'c')
			qc = QuantumCircuit(qr, cr)
		else:
			qr = QuantumRegister(self.num_qubits_total - self.num_qubits_ancilla, 'q')
			anc = QuantumRegister(self.num_qubits_ancilla, 'ancilla')
			cr = ClassicalRegister(self.num_qubits_total - self.num_qubits_ancilla, 'c')
			qc = QuantumCircuit(anc, qr, cr)

		if latent_space_noise is None:
			randoms = np.random.normal(-np.pi * .01, np.pi * .01, self.num_qubits_total)
			init_dist = qiskit.QuantumCircuit(self.num_qubits_total)

			for index in range(self.num_qubits_total):
				init_dist.ry(randoms[index], index)
		else:
			init_dist = qiskit.QuantumCircuit(self.num_qubits_total)

			for index in range(self.num_qubits_total):
				init_dist.ry(latent_space_noise[index], index)

		if self.num_qubits_ancilla == 0:
			entangler_map = create_entangler_map(self.num_qubits_total)
		else:
			entangler_map = create_entangler_map(self.num_qubits_total - self.num_qubits_ancilla)

		ansatz = TwoLocal(int(self.num_qubits_total), 'ry', 'cz', entanglement=entangler_map, reps=1,
		                  insert_barriers=True)

		qc = qc.compose(init_dist, front=True)
		qc = qc.compose(ansatz, front=False)

		if to_measure:
			qc.measure(qr, cr)

		return qc

	def set_discriminator(self, discriminator) -> None:
		self.discriminator = discriminator

	def get_output(
			self,
			latent_space_noise,
			parameters: Optional[np.ndarray] = None
	):
		real_keys_set, real_keys_list = create_real_keys(self.num_qubits_total - self.num_qubits_ancilla)

		if parameters is None:
			parameters = cast(np.ndarray, self.parameter_values)

		qc = self.construct_circuit(latent_space_noise, True)

		parameter_binds = {parameter_id: parameter_value for parameter_id, parameter_value in
		                   zip(qc.parameters, parameters)}

		qc = qc.bind_parameters(parameter_binds)

		result_ideal = qiskit.execute(experiments=qc,
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
		qc = QuantumCircuit(self.num_qubits_total)

		init_dist = qiskit.QuantumCircuit(self.num_qubits_total)
		assert latent_space_noise.shape[0] == self.num_qubits_total

		for num_qubit in range(self.num_qubits_total):
			init_dist.ry(latent_space_noise[num_qubit], num_qubit)

		if params is None:
			params = cast(np.ndarray, self.parameter_values)

		qc.assign_parameters(params)

		state_vector = qiskit.quantum_info.Statevector.from_instruction(qc)
		pixels = []
		for qubit in range(self.num_qubits_total):
			pixels.append(state_vector.probabilities([qubit])[0])

		generated_samples = np.array(pixels)
		generated_samples.flatten()

		return generated_samples

	def train_mini_batch(self, mini_batch, learning_rate):
		nabla_theta = np.zeros(self.parameter_values.shape)
		new_images = []

		for _, noise in mini_batch:
			for index in range(len(self.parameter_values)):
				perturbation_vector = np.zeros(len(self.parameter_values))
				perturbation_vector[index] = 1

				pos_params = self.parameter_values + (np.pi / 4) * perturbation_vector
				neg_params = self.parameter_values - (np.pi / 4) * perturbation_vector

				pos_result = self.get_output(noise, parameters=pos_params)
				neg_result = self.get_output(noise, parameters=neg_params)

				pos_result = self.discriminator.predict(pos_result)
				neg_result = self.discriminator.predict(neg_result)

				gradient = minimax_generator(pos_result) - minimax_generator(neg_result)
				nabla_theta[index] += gradient
			new_images.append(self.get_output(noise))

		for index in range(len(self.parameter_values)):
			self.parameter_values[index] += (learning_rate / len(mini_batch)) * nabla_theta[index]

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

				pos_result = self.get_output(noise, parameters=pos_params)
				neg_result = self.get_output(noise, parameters=neg_params)

				pos_result = qcolor_to_image(pos_result)
				neg_result = qcolor_to_image(neg_result)

				pos_result = self.discriminator.predict(pos_result.flatten())
				neg_result = self.discriminator.predict(neg_result.flatten())
				gradient = minimax_generator(pos_result) - minimax_generator(neg_result)
				nabla_theta[index] += gradient
			new_images.append(self.get_output(noise))

		for index in range(len(self.parameter_values)):
			self.parameter_values[index] += (learning_rate / len(mini_batch)) * nabla_theta[index]

		mini_batch = [(datapoint[0], fake_image) for datapoint, fake_image in zip(mini_batch, new_images)]

		return mini_batch
