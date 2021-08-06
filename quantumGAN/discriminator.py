"""DISCRIMINATOR"""
import json
import random
from typing import Dict, List

import numpy as np

from quantumGAN.functions import BCE_derivative, minimax_derivative_fake, minimax_derivative_real, sigmoid, \
	sigmoid_prime


def load(filename):
	f = open(filename, "r")
	data = json.load(f)
	f.close()
	# cost = getattr(sys.modules[__name__], data["cost"])
	net = ClassicalDiscriminator(None, None, data["sizes"], data["loss"])
	net.weights = [np.array(w) for w in data["weights"]]
	net.biases = [np.array(b) for b in data["biases"]]
    return net


class ClassicalDiscriminator:

	def __init__(self,
	             training_data: List or None,
	             mini_batch_size: int or None,
	             sizes: List[int],
	             type_loss: str) -> None:

		self.training_data = training_data
		self.mini_batch_size: int = mini_batch_size
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
		self.biases = [np.random.randn(y, ) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x)
		                for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def predict(self, x):
        # feedforward
        activation = x
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
        return activation

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "loss": self.type_loss  # ,
                # "cost": str(self..__name__)
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def forwardprop(self, x: np.ndarray):
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activation, activations, zs

	def backprop_bce(self, image, label):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# feedforward and back error calculation depending on type of image
		activation, activations, zs = self.forwardprop(image)
		delta = BCE_derivative(activations[-1], label) * sigmoid_prime(zs[-1])

        # backward pass
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
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# feedforward and back error calculation depending on type of image
		activation_real, activations_real, zs_real = self.forwardprop(real_image)
		activation_fake, activations_fake, zs_fake = self.forwardprop(fake_image)

		if is_real:
			delta = minimax_derivative_real(activations_real[-1]) * sigmoid_prime(zs_real[-1])
			activations, zs = activations_real, zs_real
		else:
			delta = minimax_derivative_fake(activations_fake[-1]) * sigmoid_prime(zs_fake[-1])
			activations, zs = activations_fake, zs_fake

		# backward pass
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

		# gradient descent
		# nabla_w and nabla_b are multiplied by the learning rate
		# and taken the mean of (dividing by the mini batch size)
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def create_mini_batches(self):
        n = len(self.training_data)
        random.shuffle(self.training_data)
        mini_batches = [
            self.training_data[k:k + self.mini_batch_size]
            for k in range(0, n, self.mini_batch_size)]
        return [mini_batches[0]]
