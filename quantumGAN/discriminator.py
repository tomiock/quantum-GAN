"""DISCRIMINATOR WITH MINIMAX"""
import json
import random
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    # cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(None, None, data["sizes"], data["loss"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


class Network:

    def __init__(self,
                 training_data: List or None,
                 mini_batch_size: int or None,
                 sizes: List[int],
                 loss_BCE: bool) -> None:

        self.training_data = training_data
        self.mini_batch_size: int = mini_batch_size
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.loss_BCE = loss_BCE
        self.data_loss = {"real": [],
                          "fake": []}
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
                "loss": self.loss_BCE  # ,
                # "cost": str(self..__name__)
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def MSE_derivative(self, prediction, y):
        return 2 * (y - prediction)

    def MSE(self, prediction, y):
        return (y - prediction)**2

    def BCE_derivative(self, prediction, target):
        return -target / prediction + (1 - target) / (1 - prediction)

    def BCE(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return targets * np.log10(predictions) + (1 - targets) * np.log10(1 - predictions).mean()

    def minimax_derivative(self, real_prediction, fake_prediction):
        real_prediction = np.array(real_prediction)
        fake_prediction = np.array(fake_prediction)
        return np.nan_to_num(1 / (real_prediction * np.log(10)) + 1 / ((fake_prediction - 1) * np.log(10)))

    def minimax(self, real_prediction, fake_prediction):
        return np.nan_to_num(np.log10(real_prediction) + np.log10(1 - fake_prediction))

    @property
    def ret(self):
        return self._ret

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

    def backprop(self, image, label):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward and back error calculation depending on type of image
        activation, activations, zs = self.forwardprop(image)
        delta = self.BCE_derivative(activations[-1], label) * sigmoid_prime(zs[-1])

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

    def train_mini_batch(self, mini_batch, learning_rate, epoch):
        global label_real, label_fake
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for real_image, fake_image in mini_batch:
            delta_nabla_b, delta_nabla_w, label_real = self.backprop(real_image, np.array([1.]))
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            delta_nabla_b, delta_nabla_w, label_fake = self.backprop(fake_image, np.array([0.]))
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        loss_real = self.BCE(label_real, np.array([1.]))
        loss_fake = self.BCE(label_fake, np.array([0.]))
        final_loss = 1 / 2 * (loss_real + loss_fake)

        # gradient descent
        # nabla_w and nabla_b are multiplied by the learning rate
        # and taken the mean of (dividing by the mini batch size)
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

        self.ret["loss"].append(final_loss)
        self._ret["label real"].append(label_real)
        self._ret["label real time"].append(epoch)
        self._ret["label fake"].append(label_fake)
        self._ret["label fake time"].append(epoch)

    def create_mini_batches(self):
        n = len(self.training_data)
        random.shuffle(self.training_data)
        mini_batches = [
            self.training_data[k:k + self.mini_batch_size]
            for k in range(0, n, self.mini_batch_size)]

        return mini_batches

    def train_SGD(self,
                  epochs: int,
                  learning_rate: float,
                  test_data=None):

        global n_test
        if test_data:
            n_test = len(test_data)

        for j in range(epochs):
            time1 = time.time()
            mini_batches = self.create_mini_batches()

            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch, learning_rate, j)
            time2 = time.time()

            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, self.evaluate(test_data), n_test, time2 - time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, time2 - time1))

            self._ret["label real"].append(1)
            self._ret["label real time"].append(j)
            self._ret["label fake"].append(self.data_loss["fake"])
            self._ret["label fake time"].append(j)

        return self._ret


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
