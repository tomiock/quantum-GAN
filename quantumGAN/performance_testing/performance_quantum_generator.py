"""Quantum Generator for performance testing"""
import random
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import qiskit
from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer import AerSimulator


class PerformanceQuantumGeneratorV1:

	def __init__(
			self,
			training_data: List,
			mini_batch_size: int,
			shots: int,
			learning_rate: float,
			num_qubits: Union[List[int], np.ndarray],
			generator_circuit: Optional[QuantumCircuit] = None,
			snapshot_dir: Optional[str] = None
	) -> None:

		super().__init__()
		self.training_data = training_data
		self.mini_batch_size = mini_batch_size
		self.num_qubits = num_qubits
		self.learning_rate = learning_rate
		self.generator_circuit = generator_circuit

		if generator_circuit is None:
			circuit = QuantumCircuit(sum(num_qubits))
			circuit.h(circuit.qubits)
			ansatz = TwoLocal(sum(num_qubits), "ry", "cz", reps=1, entanglement="circular")
			circuit.compose(ansatz, inplace=True)

			# Set generator circuit
			self.generator_circuit = circuit

		self.parameter_values = np.random.rand(self.generator_circuit.num_parameters)
		print(self.parameter_values)

		self.snapshot_dir = snapshot_dir
		self.shots = shots
		self.discriminator = None
		self.ret: Dict[str, Any] = {"loss": []}

	def set_discriminator(self, discriminator) -> None:
		self.discriminator = discriminator

	def set_mini_batch(self, mini_batch) -> None:
		self.mini_batch = mini_batch

	def construct_circuit(self, params):
		return self.generator_circuit.assign_parameters(params)

	def get_output(
			self,
			latent_space_noise,
			params: Optional[np.ndarray] = None
	):
		real_keys = {"00", "10", "01", "11"}

		quantum = QuantumRegister(sum(self.num_qubits), name="q")
		qc = QuantumCircuit(sum(self.num_qubits))

		init_dist = qiskit.QuantumCircuit(sum(self.num_qubits))
		assert latent_space_noise.shape[0] == sum(self.num_qubits)

		for num_qubit in range(sum(self.num_qubits)):
			init_dist.ry(latent_space_noise[num_qubit], num_qubit)

		if params is None:
			params = cast(np.ndarray, self.parameter_values)

		qc.append(self.construct_circuit(params), quantum)
		final_circuit = qc.compose(init_dist, front=True)
		final_circuit.measure_all()

		simulator = qiskit.Aer.get_backend("aer_simulator")
		final_circuit = qiskit.transpile(final_circuit, simulator)
		result = simulator.run(final_circuit, shots=self.shots).result()
		counts = result.get_counts(final_circuit)

		try:
			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		except KeyError:
			# dealing with the keys that qiskit doesn't include in the
			# dictionary because they don't get any measurements

			keys = counts.keys()
			missing_keys = real_keys.difference(keys)
			# we use sets to get the missing keys
			for key_missing in missing_keys:
				counts[key_missing] = 0

			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		pixels = pixels / self.shots
		return pixels

	def loss(self, prediction_fake):
		return np.log10(1 - prediction_fake)

	def BCE(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
		return targets * np.log10(predictions) + (1 - targets) * np.log10(1 - predictions)

	def create_mini_batches(self):
		n = len(self.training_data)
		random.shuffle(self.training_data)
		mini_batches = [
			self.training_data[k:k + self.mini_batch_size]
			for k in range(0, n, self.mini_batch_size)]
		return mini_batches

	def train_mini_batch(self):
		nabla_theta = np.zeros(self.parameter_values.shape)
		new_images = []

		for _, noise in self.mini_batch:
			for index in range(len(self.parameter_values)):
				perturbation_vector = np.zeros(len(self.parameter_values))
				perturbation_vector[index] = 1

				pos_params = self.parameter_values + (np.pi / 4) * perturbation_vector
				neg_params = self.parameter_values - (np.pi / 4) * perturbation_vector

				pos_result = self.get_output(noise, params=pos_params)
				neg_result = self.get_output(noise, params=neg_params)

				pos_result = self.discriminator.predict(pos_result)
				neg_result = self.discriminator.predict(neg_result)
				gradient = self.BCE(pos_result, np.array([1.])) - self.BCE(neg_result, np.array([1.]))
				nabla_theta[index] += gradient
			new_images.append(self.get_output(noise))

		for index in range(len(self.parameter_values)):
			self.parameter_values[index] -= (self.learning_rate / self.mini_batch_size) * nabla_theta[index]

		self.mini_batch = [(datapoint[0], fake_image) for datapoint, fake_image in zip(self.mini_batch, new_images)]
	# result_final, _ = self._discriminator.forward(result_final, self._discriminator.params_values)
	# loss_final = self.loss(result_final)
	# self.ret["loss"].append(loss_final.flatten())

# return mini_batch


class PerformanceQuantumGeneratorV3:

	def __init__(
			self,
			training_data: List,
			mini_batch_size: int,
			shots: int,
			learning_rate: float,
			num_qubits: Union[List[int], np.ndarray],
			generator_circuit: Optional[QuantumCircuit] = None,
			snapshot_dir: Optional[str] = None
	) -> None:

		super().__init__()
		self.training_data = training_data
		self.mini_batch_size = mini_batch_size
		self.num_qubits = num_qubits
		self.learning_rate = learning_rate
		self.generator_circuit = generator_circuit

		if generator_circuit is None:
			circuit = QuantumCircuit(sum(num_qubits))
			circuit.h(circuit.qubits)
			ansatz = TwoLocal(sum(num_qubits), "ry", "cz", reps=1, entanglement="circular")
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

	def set_mini_batch(self, mini_batch) -> None:
		self.mini_batch = mini_batch

	def construct_circuit(self, params):
		return self.generator_circuit.assign_parameters(params)

	def get_output(
			self,
			latent_space_noise,
			params: Optional[np.ndarray] = None
	):
		real_keys = {"00", "10", "01", "11"}

		quantum = QuantumRegister(sum(self.num_qubits), name="q")
		init_dist = qiskit.QuantumCircuit(sum(self.num_qubits))
		qc = QuantumCircuit(sum(self.num_qubits))

		assert latent_space_noise.shape[0] == sum(self.num_qubits)

		for num_qubit in range(sum(self.num_qubits)):
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
			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		except KeyError:
			# dealing with the keys that qiskit doesn't include in the
			# dictionary because they don't get any measurements

			keys = counts.keys()
			missing_keys = real_keys.difference(keys)
			# we use sets to get the missing keys
			for key_missing in missing_keys:
				counts[key_missing] = 0

			pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

		pixels = pixels / self.shots
		return pixels

	def loss(self, prediction_fake):
		return np.log10(1 - prediction_fake)

	def BCE(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
		return targets * np.log10(predictions) + (1 - targets) * np.log10(1 - predictions)

	def create_mini_batches(self):
		n = len(self.training_data)
		random.shuffle(self.training_data)
		mini_batches = [
			self.training_data[k:k + self.mini_batch_size]
			for k in range(0, n, self.mini_batch_size)]
		return mini_batches

	def train_mini_batch(self):
		nabla_theta = np.zeros(self.parameter_values.shape)
		new_images = []

		for _, noise in self.mini_batch:
			for index in range(len(self.parameter_values)):
				perturbation_vector = np.zeros(len(self.parameter_values))
				perturbation_vector[index] = 1

				pos_params = self.parameter_values + (np.pi / 4) * perturbation_vector
				neg_params = self.parameter_values - (np.pi / 4) * perturbation_vector

				pos_result = self.get_output(noise, params=pos_params)
				neg_result = self.get_output(noise, params=neg_params)

				pos_result = self.discriminator.predict(pos_result)
				neg_result = self.discriminator.predict(neg_result)
				gradient = self.BCE(pos_result, np.array([1.])) - self.BCE(neg_result, np.array([1.]))
				nabla_theta[index] += gradient
			new_images.append(self.get_output(noise))

		for index in range(len(self.parameter_values)):
			self.parameter_values[index] -= (self.learning_rate / self.mini_batch_size) * nabla_theta[index]

		self.mini_batch = [(datapoint[0], fake_image) for datapoint, fake_image in zip(self.mini_batch, new_images)]
