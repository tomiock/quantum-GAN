"""
DISCRIMINATOR FOR BINARY CROSS ENTROPY (LABELED IMAGES)
"""

# Libraries
# Standard library
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple


def load(filename):
	"""Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
	f = open(filename, "r")
	data = json.load(f)
	f.close()
	# cost = getattr(sys.modules[__name__], data["cost"])
	net = DiscriminatorBCE(data["sizes"])
	net.weights = [np.array(w) for w in data["weights"]]
	net.biases = [np.array(b) for b in data["biases"]]
	return net


class DiscriminatorBCE:

	def __init__(self, sizes: List[int]) -> None:
		"""The list ``sizes`` contains the number of neuronfs in the
        respective layers of the network.  For example, iff the list
        was [2, 3, 1] then it would be a three-layer netwfork, with the
        first layer containing 2 neurons, the second layffer 3 neurons,
        and the third layer 1 neuron."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self._ret: Dict[str, any] = {"loss": [],
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
		activations = [x]  # list to store all the activations, layer by layer
		zs = []  # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		return activation

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), y)
		                for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def save(self, filename):
		"""Save the neural network to the file ``filename``."""
		data = {"sizes": self.sizes,
		        "weights": [w.tolist() for w in self.weights],
		        "biases": [b.tolist() for b in self.biases]  # ,
		        # "cost": str(self..__name__)
		        }
		f = open(filename, "w")
		json.dump(data, f)
		f.close()

	def MSE_derivative(self, prediction, y):
		"""Return the vector of partial derivatives \partial{C_x,a}
         for the output activations."""
		return 2 * (y - prediction)

	def MSE(self, prediction, y):
		return (y - prediction)**2

	def BCE_derivative(self, prediction, target):
		return -target / prediction + (1 - target) / (1 - prediction)

	def BCE(self, predictions: np.ndarray, targets: np.ndarray) -> float:
		return targets * np.log10(predictions) + (1 - targets) * np.log10(1 - predictions).mean()

	def minimax_derivative(self, real_prediction, fake_prediction):
		real_prediction = np.array(real_prediction)
		fake_prediction = np.array(fake_prediction)

		return 1 / (real_prediction * np.log10(10)) + 1 / ((fake_prediction - 1) * np.log10(10))

	@property
	def ret(self):
		return self._ret

	def backprop(self, x, y) -> Tuple:
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_w_prev = nabla_w

		# feedforward
		activation = x
		activations = [x]  # list to store all the activations, layer by layer
		zs = []  # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		self.data_for_loss = (activation, y)

		# backward pass
		delta = self.BCE_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].reshape(1, activations[-2].shape[0]))

		for l in range(2, self.num_layers):
			z = zs[-l]
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta.reshape(delta.shape[0], 1),
			                     activations[-l - 1].reshape(1, activations[-l - 1].shape[0]))
		return (nabla_b, nabla_w)

	def train_mini_batch(self, mini_batch, learning_rate):
		"""Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		# gradient descent
		self.weights = [w - (learning_rate / len(mini_batch)) * nw
		                for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (learning_rate / len(mini_batch)) * nb
		               for b, nb in zip(self.biases, nabla_b)]

	def train_SGD(self,
	              training_data: List[Tuple],
	              epochs: int,
	              mini_batch_size: int,
	              learning_rate: float,
	              test_data=None):
		"""Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

		global n_test
		if test_data:
			n_test = len(test_data)

		n = len(training_data)
		for j in range(epochs):
			time1 = time.time()
			random.shuffle(training_data)

			mini_batches = [
				training_data[k:k + mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			for mini_batch in mini_batches:
				self.train_mini_batch(mini_batch, learning_rate)
			time2 = time.time()
			loss_final = self.BCE(self.data_for_loss[0], self.data_for_loss[1])

			if test_data:
				print("Epoch {0}: {1} / {2}, xtook {3:.2f} seconds".format(
					j, self.evaluate(test_data), n_test, time2 - time1))
			else:
				print("Epoch {0} complete in {1:.2f} seconds".format(j, time2 - time1))

			self._ret["loss"].append(loss_final)

			if self.data_for_loss[1] == np.array([1.]):
				self._ret["label real"].append(self.data_for_loss[0].flatten())
				self._ret["label real time"].append(j)
			else:
				self._ret["label fake"].append(self.data_for_loss[0].flatten())
				self._ret["label fake time"].append(j)

		return self._ret


def sigmoid(z):
	"""The sigmoid function."""
	return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z) * (1 - sigmoid(z))


nn = DiscriminatorBCE([4, 16, 4, 1])

train_data = []
train_data_fake, train_data_real = [], []
for _ in range(70):
	x1 = np.random.uniform(.8, .9, (4,))
	x2 = np.random.uniform(.5, .4, (2,))

	train_data.append((x1, np.array([0.])))
	train_data.append((np.array([x2[1], 0., x2[0], 0]), np.array([1.])))

num_epochs = 100
nn.train_SGD(train_data, num_epochs, 10, 0.1)
nn.save("C:/Users/usuario/qGAN/quantumGAN/nns")

t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in the loss function")
plt.plot(t_steps, nn.ret["loss"], label='Discriminator loss function', color='rebeccapurple', linewidth=2)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('loss')
plt.show()

t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in labels")
plt.scatter(nn.ret["label real time"], nn.ret["label real"], label='Label for real images', color='mediumvioletred',
            linewidth=.1)
plt.scatter(nn.ret["label fake time"], nn.ret["label fake"], label='Label for fake images', color='rebeccapurple',
            linewidth=.1)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('label')
plt.show()

label_1 = nn.predict(np.array([0.80417561, 0.87188567, 0.88118713, 0.82633528]))
label_0 = nn.predict(np.array([0.40396169, 0., 0.43473614, 0.]))

print(label_1, label_0)

score = nn.evaluate(train_data[23:45])
print(score)

#######################################################################
from quantumGAN.deprecated_files.discriminator_V4 import ClassicalDiscriminator

train_data = []
train_data_fake, train_data_real = [], []
for _ in range(150):
	x2 = np.random.uniform(.5, .4, (2,))
	fake_datapoint = (np.random.uniform(.8, .9, (4,)), False)
	real_datapoint = (np.array([x2[1], 0., x2[0], 0]), True)
	data_point = [real_datapoint, fake_datapoint]
	train_data.append(random.sample(data_point, 2))

nn = ClassicalDiscriminator(train_data, 10, [4, 128, 32, 16, 1], loss_BCE=True)
print(train_data)

num_epochs = 400
nn.train_SGD(num_epochs, 0.1)
nn.save("C:/Users/usuario/qGAN/quantumGAN/nns")

t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in the loss function")
plt.plot(t_steps, nn.ret["loss"], label='Discriminator loss function', color='rebeccapurple', linewidth=2)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('loss')
plt.show()

t_steps = np.arange(num_epochs)
plt.figure(figsize=(6, 5))
plt.title("Progress in labels")
plt.scatter(nn.ret["label real time"], nn.ret["label real"],
            label='Label for real images',
            color='mediumvioletred',
            linewidth=.1)
plt.scatter(nn.ret["label fake time"], nn.ret["label fake"],
            label='Label for fake images',
            color='rebeccapurple',
            linewidth=.1)
plt.grid()
plt.legend(loc='best')
plt.xlabel('time steps')
plt.ylabel('label')
plt.show()

label_1 = nn.predict(np.array([0.80417561, 0.87188567, 0.88118713, 0.82633528]))
label_0 = nn.predict(np.array([0.40396169, 0., 0.43473614, 0.]))

print(label_1, label_0)

# score = nn.evaluate(train_data[23:45])
# print(score)
