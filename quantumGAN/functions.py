import itertools

import numpy as np
from matplotlib import pyplot as plt


# DATA PROCESSING
def save_images(image, epoch):
	image_shape = int(image.shape[0] / 2)
	image = image.reshape(image_shape, image_shape)
	plt.imshow(image, cmap='gray', vmax=1., vmin=0.)
	plt.axis('off')
	plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))


def save_images_color(image, epoch):
	plt.imshow(image.reshape(int(image.shape[0] / 3), 1, 3))
	plt.axis('off')
	plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))


def create_real_keys(num_qubits):
	lst = [[str(a) for a in i] for i in itertools.product([0, 1], repeat=num_qubits)]
	new_lst = []
	for element in lst:
		word = str()
		for number in element:
			word = word + number
		new_lst.append(word)
	return set(new_lst), new_lst


def create_entangler_map(num_qubits: int):
	lst = [list(i) for i in itertools.combinations(range(num_qubits), 2)]
	index = 0
	entangler_map = []
	for i in reversed(range(num_qubits)):
		try:
			entangler_map.append(lst[index])
			index += i

		except IndexError:
			return entangler_map


# ACTIVATION FUNCTIONS

def sigmoid(z):
	"""The sigmoid function."""
	return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z) * (1 - sigmoid(z))


# LOSSES
def MSE_derivative(prediction, y):
	return 2 * (y - prediction)


def MSE(prediction, y):
	return (y - prediction)**2


def BCE_derivative(prediction, target):
	# return prediction - target
	return -target / prediction + (1 - target) / (1 - prediction)


def BCE(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
	return targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions).mean()


def minimax_derivative_real(real_prediction):
	real_prediction = np.array(real_prediction)
	return np.nan_to_num((-1) * (1 / real_prediction))


def minimax_derivative_fake(fake_prediction):
	fake_prediction = np.array(fake_prediction)
	return np.nan_to_num(1 / (1 - fake_prediction))


def minimax(real_prediction, fake_prediction):
	return np.nan_to_num(np.log(real_prediction) + np.log(1 - fake_prediction))


def minimax_generator(prediction_fake):
	return (-1) * np.log(1 - prediction_fake)
