"""Quantum Generator."""

from typing import Optional, List, Union, Dict, Any, Callable, cast, Tuple

import numpy as np
import qiskit
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import QuantumCircuit

from qiskit.utils import QuantumInstance, algorithm_globals
from quantumGAN.discrimintorV2 import DiscriminatorV2


def gen_real_data_FCNN(a, b, num_samples: int):
    data = []
    x0 = np.random.uniform(a, b, num_samples)
    x1 = np.random.uniform(a, b, num_samples)

    for i in range(len(x1)):
        array = [[x0[i], 0], [x1[i], 0]]
        data.append(array)

    return np.array(data).flatten()


class QuantumGenerator:

    def __init__(
            self,
            num_qubits: Union[List[int], np.ndarray],
            generator_circuit: Optional[QuantumCircuit] = None,
            snapshot_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._num_qubits = num_qubits
        self.generator_circuit = generator_circuit
        if generator_circuit is None:
            circuit = QuantumCircuit(sum(num_qubits))
            circuit.h(circuit.qubits)
            ansatz = TwoLocal(sum(num_qubits), "ry", "cz", reps=1, entanglement="circular")
            circuit.compose(ansatz, inplace=True)

            # Set generator circuit
            self.generator_circuit = circuit

        self.parameter_values = np.random.rand(self.generator_circuit.num_parameters)

        # Set optimizer for updating the generator network
        self._snapshot_dir = snapshot_dir

        self._seed = 7
        self._shots = None
        self._discriminator: Optional[DiscriminatorV2] = None
        self._ret: Dict[str, Any] = {"loss": []}

    @property
    def seed(self) -> int:
        """
        Get seed.
        """
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        """
        Set seed.
        Args:
            seed (int): seed to use.
        """
        self._seed = seed
        algorithm_globals.random_seed = seed

    def set_discriminator(self, discriminator: DiscriminatorV2) -> None:
        self._discriminator = discriminator

    def construct_circuit(self, params):
        return self.generator_circuit.assign_parameters(params)

    def get_output(
            self,
            params: Optional[np.ndarray] = None,
            shots: Optional[int] = None,
    ):

        quantum = QuantumRegister(sum(self._num_qubits), name="q")
        qc = QuantumCircuit(quantum)

        if params is None:
            params = cast(np.ndarray, self.parameter_values)
        qc.append(self.construct_circuit(params), quantum)
        qc.measure_all()

        simulator = qiskit.Aer.get_backend("aer_simulator")
        qc = qiskit.transpile(qc, simulator)

        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts(qc)

        try:
            pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

        except KeyError as error:
            counts[error.args[0]] = 0

            try:
                pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])

            except KeyError as error:
                counts[error.args[0]] = 0
                pixels = np.array([counts["00"], counts["10"], counts["01"], counts["11"]])


        pixels = pixels/shots
        return pixels

        #state_vector = qiskit.quantum_info.Statevector.from_instruction(qc)
        #pixels = []
        #for qubit in range(sum(self._num_qubits)):
        #    pixels.append(state_vector.probabilities([qubit])[0])
#
        #generated_samples = np.array(pixels)
        #generated_samples.flatten()
        #return np.array(generated_samples)

    def loss(self, prediction_real, prediction_fake):
        return np.log10(prediction_real) + np.log10(1 - prediction_fake)

    def step(self, learning_rate, shots):

        global real_prediction
        for index in range(len(self.parameter_values)):
            perturbation_vector = np.zeros(len(self.parameter_values))
            perturbation_vector[index] = 1

            pos_params = self.parameter_values + (np.pi / 4) * perturbation_vector
            neg_params = self.parameter_values - (np.pi / 4) * perturbation_vector

            pos_result = self.get_output(params=pos_params, shots=shots)
            neg_result = self.get_output(params=neg_params, shots=shots)

            pos_result = self._discriminator.get_label(pos_result, self._discriminator.params_values)[0]
            neg_result = self._discriminator.get_label(neg_result, self._discriminator.params_values)[0]

            real_image = gen_real_data_FCNN(.6, .4, 1)
            real_prediction = self._discriminator.forward(real_image, self._discriminator.params_values)[0]

            gradient = self.loss(real_prediction, pos_result) - self.loss(real_prediction, neg_result)

            self.parameter_values[index] -= learning_rate * gradient

        result_final = self.get_output(self.parameter_values, shots)
        result_final, _ = self._discriminator.forward(result_final, self._discriminator.params_values)
        loss_final = self.loss(real_prediction, result_final)

        self._ret["loss"].append(loss_final.flatten())
        # self._ret["params"] = self._bound_parameters
        # self._ret["output"] = self.get_output(quantum_instance, self._bound_parameters, 4048)[0]

        return self._ret
