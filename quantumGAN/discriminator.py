"""DISCRIMINATOR"""
import json
from typing import Dict, List

import numpy as np

from quantumGAN.functions import BCE_derivative, minimax_derivative_fake, minimax_derivative_real, sigmoid, \
    sigmoid_prime


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    # cost = getattr(sys.modules[__name__], data["cost"])
    net = ClassicalDiscriminator(data["sizes"], data["loss"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


class ClassicalDiscriminator:
    """
	A simple fully connected neural network based on code from Michael Nielsen.
	"""

    def __init__(self,
                 sizes: List[int],
                 type_loss: str) -> None:

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.type_loss = type_loss
        self.data_loss = {"real": [],
                          "fake": []}
        self.ret: Dict[str, any] = {"loss": [],
                                    "label real": [],
                                    "label fake": [],
                                    "label fake time": [],
                                    "label real time": []}
        self.biases = None
        self.weights = None

    def init_parameters(self):
        """Return random parameters based on the network's architecture"""
        self.biases = [np.random.randn(y, ) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        return self.biases, self.weights

    def predict(self, x):
        """Return a label for a given image input"""
        activation = x
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
        return activation

    def forwardprop(self, x: np.ndarray):
        """Return label and all activations over the hole network"""
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activation, activations, zs

    def backprop_bce(self, image, label):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of ``self.biases`` and ``self.weights`` using Binary Cross Entropy as loss function."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation, activations, zs = self.forwardprop(image)
        delta = BCE_derivative(activations[-1], label) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].reshape(1, activations[-2].shape[0]))

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta.reshape(delta.shape[0], 1),
                                 activations[-l - 1].reshape(1, activations[-l - 1].shape[0]))
        return nabla_b, nabla_w, activations[-1]

    def backprop_minimax(self, real_image, fake_image, is_real):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for ``self.biases`` and ``self.weights`` using Minimax as loss function."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation_real, activations_real, zs_real = self.forwardprop(real_image)
        activation_fake, activations_fake, zs_fake = self.forwardprop(fake_image)

        if is_real:
            delta = minimax_derivative_real(activations_real[-1]) * sigmoid_prime(zs_real[-1])
            activations, zs = activations_real, zs_real
        else:
            delta = minimax_derivative_fake(activations_fake[-1]) * sigmoid_prime(zs_fake[-1])
            activations, zs = activations_fake, zs_fake

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].reshape(1, activations[-2].shape[0]))

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta.reshape(delta.shape[0], 1),
                                 activations[-l - 1].reshape(1, activations[-l - 1].shape[0]))
        return nabla_b, nabla_w, activations[-1]

    def train_mini_batch(self, mini_batch, learning_rate):
        """Update the network's parameters with a mini batch of data. They can be updated using two diferent loss functions.
		It doesn't give any output because this function updates the parameters updating the methods of the class that store them.

		Inputs:
			- ``mini_batch``: [np.ndarray]
			list of images
			- ``learning_rate``: int
			number indicating the learning rate to train the network
		"""
        global label_real, label_fake
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        if self.type_loss == "binary cross entropy":
            for real_image, fake_image in mini_batch:
                delta_nabla_b, delta_nabla_w, label_real = self.backprop_bce(real_image, np.array([1.]))
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                delta_nabla_b, delta_nabla_w, label_fake = self.backprop_bce(fake_image, np.array([0.]))
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        elif self.type_loss == "minimax":
            for real_image, fake_image in mini_batch:
                delta_nabla_b, delta_nabla_w, label_real = self.backprop_minimax(real_image, fake_image, True)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                delta_nabla_b, delta_nabla_w, label_fake = self.backprop_minimax(real_image, fake_image, False)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        else:
            raise Exception("type of loss function not valid")

        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
