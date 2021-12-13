import itertools
import math

import numpy as np
from matplotlib import pyplot as plt


# DATA PROCESSING
from scipy import linalg


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


def images_to_distribution(batch_image):
    num_images = len(batch_image)
    sum_result = 0
    for image in batch_image:
        sum_result += image
    average_result = sum_result / num_images
    keys = create_real_keys(int(math.sqrt(batch_image[0].shape[0])))[1]
    return keys, average_result


def images_to_scatter(batch_image):
    keys = create_real_keys(int(math.sqrt(batch_image[0].shape[0])))[1]
    x_axis = []
    y_axis = []

    for image in batch_image:
        pixel_count = 0
        for pixel in image:
            x_axis.append(pixel)
            y_axis.append(keys[pixel_count])
            pixel_count += 1

    return y_axis, x_axis


# ACTIVATION FUNCTIONS

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


# LOSSES
def MSE_derivative(prediction, y):
    return 2 * (y - prediction)


def MSE(prediction, y):
    return (y - prediction) ** 2


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


class Partial_Trace:
    def __init__(self, state: np.array, qubits_out: int, side: str):

        self.state = state
        self.qubits_out = qubits_out
        self.side = side

        if self.state.ndim == 1:
            self.state = np.outer(self.state, self.state)

        self.total_dim = self.state.shape[0]

        self.num_qubits = int(np.log2(self.total_dim))
        self.a_dim = 2 ** (self.num_qubits - self.qubits_out)
        self.b_dim = 2 ** self.qubits_out

        # if self.side == "bot":
        self.basis_b = [_ for _ in np.identity(int(self.b_dim))]
        self.basis_a = [_ for _ in np.identity(int(self.a_dim))]

        # elif self.side == "top":
        #	self.basis_a = [_ for _ in np.identity(int(self.b_dim))]
        #	self.basis_b = [_ for _ in np.identity(int(self.a_dim))]
        #
        # else:
        #	raise NameError("invalid side argument, enter \"bot\" or \"top\"")
        print(self.basis_a, self.basis_b)

    def get_entry(self, index_i, index_j):
        sigma = 0

        if self.side == "bot":
            for k in range(self.qubits_out + 1):
                ab_l = np.kron(self.basis_a[index_i],
                               self.basis_b[k])
                ab_r = np.kron(self.basis_a[index_j],
                               self.basis_b[k])

                print(ab_r, ab_l)

                right_side = np.dot(self.state, ab_r)
                sigma += np.inner(ab_l, right_side)

        if self.side == "top":
            for k in range(self.qubits_out + 1):
                ba_l = np.kron(self.basis_b[index_i],
                               self.basis_a[k])
                ba_r = np.kron(self.basis_b[index_j],
                               self.basis_a[k])

                print(ba_r, ba_l)

                right_side = np.dot(self.state, ba_r)
                sigma += np.inner(ba_l, right_side)

        return sigma

    def compute_matrix(self):
        a = [_ for _ in range(self.a_dim)]
        b = [__ for __ in range(self.a_dim)]

        entries_pre = [(x, y) for x in a for y in b]
        entries = []

        for i_index, j_index in entries_pre:
            entries.append(self.get_entry(i_index, j_index))

        entries = np.array(entries)
        return entries.reshape(self.a_dim, self.a_dim)
    
    
def fechet_distance(image1: np.array, 
                    image2: np.array):
    assert image1.shape == image2.shape
    y = np.arange(0, image1.flatten().shape[0])

    matrix_a_cov = np.cov(np.stack((y, image1.flatten()), axis=0))
    matrix_b_cov = np.cov(np.stack((y, image2.flatten()), axis=0))

    to_trace = matrix_a_cov + matrix_b_cov - 2*(linalg.fractional_matrix_power(np.dot(matrix_a_cov, matrix_b_cov), .5))
    return np.abs(image1.mean() - image2.mean())**2 + np.trace(to_trace)
