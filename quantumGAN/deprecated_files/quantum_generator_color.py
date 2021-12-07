"""QUANTUM GENERATOR"""

from typing import Any, Dict, Optional, cast

import numpy as np
import qiskit
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.providers.aer import AerSimulator

from quantumGAN.functions import create_entangler_map, create_real_keys, minimax_generator
from quantumGAN.deprecated_files.rgb_functions import qcolor_to_image


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
		# passar els arguments de la classe a metodes en de classes
		# d'aquesta manera son accessibles per qualsevol funció dintre de la classe
		self.num_qubits_total = num_qubits
		self.num_qubits_ancilla = num_qubits_ancilla
		self.generator_circuit = generator_circuit
		self.snapshot_dir = snapshot_dir
		self.shots = shots
		self.discriminator = None
		self.ret: Dict[str, Any] = {"loss": []}
		self.simulator = AerSimulator()

	def init_parameters(self):
		""" Funció que inicia el parametres inicial i crea """
		# iniciació dels parameters inicials i del circuits al qual posar aquests parameters
		self.generator_circuit = self.construct_circuit(latent_space_noise=None,
		                                                to_measure=False)
		self.parameter_values = np.random.normal(np.pi / 2, .1, self.generator_circuit.num_parameters)

	def construct_circuit(self,
	                      latent_space_noise,
	                      to_measure: bool):
		""" Funció que crea el circuit quàntic des de zero a partir de diversos registres de qubits"""
		if self.num_qubits_ancilla is 0:
			qr = QuantumRegister(self.num_qubits_total, 'q')
			cr = ClassicalRegister(self.num_qubits_total, 'c')
			qc = QuantumCircuit(qr, cr)
		else:
			qr = QuantumRegister(self.num_qubits_total - self.num_qubits_ancilla, 'q')
			anc = QuantumRegister(self.num_qubits_ancilla, 'ancilla')
			cr = ClassicalRegister(self.num_qubits_total - self.num_qubits_ancilla, 'c')
			qc = QuantumCircuit(anc, qr, cr)

		# creació de la part del circuit que conté l'implantació dels parameters d'input. En cas de que no es donin
		# aquests parameters es creen automàticament
		if latent_space_noise is None:
			randoms = np.random.normal(-np.pi * .01, np.pi * .01, self.num_qubits_total)
			init_dist = qiskit.QuantumCircuit(self.num_qubits_total)

			# es col·loca una porta RY en cada qubits i amb un paràmetre diferent cadascuna
			for index in range(self.num_qubits_total):
				init_dist.ry(randoms[index], index)
		else:
			init_dist = qiskit.QuantumCircuit(self.num_qubits_total)

			for index in range(self.num_qubits_total):
				init_dist.ry(latent_space_noise[index], index)

		# la funció create_entagler_map crea les parelles de qubits a les qual col·locar les portes CZ
		# en funció del nombre de qubits
		if self.num_qubits_ancilla == 0:
			entangler_map = create_entangler_map(self.num_qubits_total)
		else:
			entangler_map = create_entangler_map(self.num_qubits_total - self.num_qubits_ancilla)

		# creació final dels circuits a partir una funció integrada a Qiskit que va repetint les operacions
		# que se li especifiquen
		ansatz = TwoLocal(int(self.num_qubits_total), 'ry', 'cz', entanglement=entangler_map, reps=1,
		                  insert_barriers=True)

		# aqui s'ajunten el circuit que funciona com a input amb el circuit que consisteix en la repetició
		# de les portes RY i CZ
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
		"""Funció que retorna un output del generador quan se li dona un estat d'input i opcionalment un parameters en
		especific. Els pixels estan compostos per la probabilitat de que un qubits resulti en ket_0 en cada base. Per tant,
		els pixels de l'imatge estan normalitzats amb la norma l-1."""
		real_keys_set, real_keys_list = create_real_keys(self.num_qubits_total - self.num_qubits_ancilla)

		# en cas de que de que no es donin parameters com a input, es treuen els parameters de la variable
		# self.parameter_values. Es a dir els parameters que es creen automàticament al principi i que es van
		# actualitzant al mateix temps que el model s'optimitza
		if parameters is None:
			parameters = cast(np.ndarray, self.parameter_values)

		qc = self.construct_circuit(latent_space_noise, True)

		parameter_binds = {parameter_id: parameter_value for parameter_id, parameter_value in
		                   zip(qc.parameters, parameters)}

		# el mètode bind_parameters del circuit quàntic
		qc = qc.bind_parameters(parameter_binds)

		# simulació dels circuits mitjçant el simulador Aer de Qiskit. El nivell d'optimització es zero, perquè al ser
		# circuits petits i que simulen una vegada, no es necessari. Al optimitzar el procés acaba sent més lent.
		result_ideal = qiskit.execute(experiments=qc,
		                              backend=self.simulator,
		                              shots=self.shots,
		                              optimization_level=0).result()
		counts = result_ideal.get_counts()

		try:
			# creació de l'imatge resultant
			pixels = np.array([counts[index] for index in list(real_keys_list)])

		except KeyError:
			# aquesta excepció sorgeix quan en el diccionari dels resultats no estan totes les keys degut a que qiskit
			# en cas de que no hi hagi un mesurament en una base, no inclou aquesta base en el diccionari
			keys = counts.keys()
			missing_keys = real_keys_set.difference(keys)
			# s'utilitza un la resta entre dos sets per poder veure quina es la key que falta en el diccionari
			for key_missing in missing_keys:
				counts[key_missing] = 0

			# una vegada es troba les keys que faltaven es crea l'imatge resultant
			pixels = np.array([counts[index] for index in list(real_keys_list)])

		pixels = pixels / self.shots
		return pixels

	def get_output_pixels(
			self,
			latent_space_noise,
			params: Optional[np.ndarray] = None
	):
		"""Funció que retorna un output del generador quan se li dona un estat d'input i opcionalment un parameters en
		especific. Cada pixel es la probabilitat de que un qubits result en l'estat ket_0, per tant, els valors cada
		pixel (que son independents entre si) es troba en l'interval (0, 1) """
		qc = QuantumCircuit(self.num_qubits_total)

		init_dist = qiskit.QuantumCircuit(self.num_qubits_total)
		assert latent_space_noise.shape[0] == self.num_qubits_total

		for num_qubit in range(self.num_qubits_total):
			init_dist.ry(latent_space_noise[num_qubit], num_qubit)

		if params is None:
			params = cast(np.ndarray, self.parameter_values)

		qc.assign_parameters(params)

		# comptes de simular els valors que donarà cada qubits, es simula l'estat final del circuit i d'aquest
		# s'extreuen els valors que es mesuraran per a cada qubit
		state_vector = qiskit.quantum_info.Statevector.from_instruction(qc)
		pixels = []
		for qubit in range(self.num_qubits_total):
			# per treure la probabilitat s'utilitza una funció implementada en Qiskit
			pixels.append(state_vector.probabilities([qubit])[0])

		# creació de l'imatge resultat a partir de la list que conté el valor per a cada pixel
		generated_samples = np.array(pixels)
		generated_samples.flatten()

		return generated_samples

	def train_mini_batch_color(self, mini_batch, learning_rate):
		nabla_theta = np.zeros_like(self.parameter_values)
		new_images = []

		for _, noise in mini_batch:
			for index, _ in enumerate(self.parameter_values):
				perturbation_vector = np.zeros_like(self.parameter_values)
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

		for index, _ in enumerate(self.parameter_values):
			self.parameter_values[index] += (learning_rate / len(mini_batch)) * nabla_theta[index]

		mini_batch = [(datapoint[0], fake_image) for datapoint, fake_image in zip(mini_batch, new_images)]
		return mini_batch
