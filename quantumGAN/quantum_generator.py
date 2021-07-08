"""Quantum Generator."""

from typing import Optional, List, Union, Dict, Any, Callable, cast, Tuple

import numpy as np
import qiskit
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import QuantumCircuit

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow.gradients import Gradient
from quantumGAN.discrimintorV2 import DiscriminatorV2

class QuantumGenerator:

    def __init__(
            self,
            # bounds: np.ndarray,
            num_qubits: Union[List[int], np.ndarray],
            generator_circuit: Optional[QuantumCircuit] = None,
            init_params: Optional[Union[List[float], np.ndarray]] = None,
            optimizer: Optional[Optimizer] = None,
            gradient_function: Optional[Union[Callable, Gradient]] = None,
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

        self._free_parameters = sorted(self.generator_circuit.parameters, key=lambda p: p.name)

        if init_params is None:
            init_params = (
                    algorithm_globals.random.random(self.generator_circuit.num_parameters) * 2e-1
            )

        self._bound_parameters = init_params

        # Set optimizer for updating the generator network
        self._snapshot_dir = snapshot_dir
        self.optimizer = optimizer

        self._gradient_function = gradient_function

        self._seed = 7
        self._shots = None
        self._discriminator: Optional[DiscriminatorV2] = None
        self._ret: Dict[str, Any] = {}

    @property
    def parameter_values(self) -> Union[List, np.ndarray]:
        """
        Get parameter values from the quantum generator
        Returns:
            Current parameter values
        """
        return self._bound_parameters

    @parameter_values.setter
    def parameter_values(self, p_values: Union[List, np.ndarray]) -> None:
        """
        Set parameter values for the quantum generator
        Args:
            p_values: Parameter values
        """
        self._bound_parameters = p_values

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

    def construct_circuit(self, params=Optional[None]):
        """
        Construct generator circuit.
        Args:
            params (list | dict): parameters which should be used to run the generator.
        Returns:
            Instruction: construct the quantum circuit and return as gate
        """
        if params is None:
            return self.generator_circuit

        if isinstance(params, (list, np.ndarray)):
            params = dict(zip(self._free_parameters, params))

        return self.generator_circuit.assign_parameters(params)
        #     self.generator_circuit.build(qc=qc, q=q)
        # else:
        #     generator_circuit_copy = deepcopy(self.generator_circuit)
        #     generator_circuit_copy.params = params
        #     generator_circuit_copy.build(qc=qc, q=q)

        # # return qc.copy(name='qc')
        # return qc.to_instruction()

    def get_output(
            self,
            quantum_instance: QuantumInstance,
            params: Optional[np.ndarray] = None,
            shots: Optional[int] = None,
    ):

        instance_shots = shots
        quantum = QuantumRegister(sum(self._num_qubits), name="q")
        qc = QuantumCircuit(quantum)

        if params is None:
            params = cast(np.ndarray, self._bound_parameters)
        qc.append(self.construct_circuit(params), quantum)
        if quantum_instance.is_statevector:
            pass
        else:
            classical = ClassicalRegister(sum(self._num_qubits), name="c")
            qc.add_register(classical)
            qc.measure(quantum, classical)

        if shots is not None:
            quantum_instance.set_config(shots=shots)

        state_vector = qiskit.quantum_info.Statevector.from_instruction(qc)
        pixels = []
        for qubit in range(sum(self._num_qubits)):
            pixels.append(state_vector.probabilities([qubit])[0])

        generated_samples = np.array(pixels)
        generated_samples.flatten()
        return np.array(generated_samples)

    def loss(self, prediction):  # pylint: disable=arguments-differ
        try:
            # pylint: disable=no-member
            loss = np.log10(1 - prediction).transpose()
        except Exception:  # pylint: disable=broad-except
            loss = np.log10(1 - prediction)
        return loss.flatten()

    def step(self, quantum_instance, learning_rate, shots):

        for index in range(len(self.parameter_values)):
            perturbation_vector = np.zeros(len(self.parameter_values))
            perturbation_vector[index] = 1

            pos_params = self.parameter_values + (np.pi / 4) * perturbation_vector
            neg_params = self.parameter_values - (np.pi / 4) * perturbation_vector

            pos_result = self.get_output(quantum_instance, params=pos_params, shots=shots)
            neg_result = self.get_output(quantum_instance, params=neg_params, shots=shots)

            pos_result = self._discriminator.get_label(pos_result, self._discriminator.params_values)[0]
            neg_result = self._discriminator.get_label(neg_result, self._discriminator.params_values)[0]

            gradient = self.loss(pos_result) - self.loss(neg_result)
            self.parameter_values[index] -= learning_rate * gradient
            # self.update(index, gradient, learning_rate)

        result_final = self.get_output(quantum_instance, self.parameter_values, shots)
        loss_final = self.loss(result_final)

        self._ret["loss"] = loss_final
        self._ret["params"] = self._bound_parameters
        self._ret["output"] = self.get_output(quantum_instance, self._bound_parameters, 4048)[0]

        return self._ret
