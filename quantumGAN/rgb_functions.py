import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio
from skimage import data


def l1_norm(vector: np.ndarray):
	assert vector.ndim == 1
	return np.sum(vector)


def rgb_to_quantum_color(rgb_values: np.ndarray):
	rgb_values = np.append(rgb_values, np.array([255]))
	norm = np.linalg.norm(rgb_values)
	return rgb_values / norm


def quantum_color_to_rgb(quantum_color: np.ndarray):
	return quantum_color * (255 / quantum_color[3])


def image_to_qcolor(image: np.ndarray):
	assert image.ndim == 3
	final_shape = list(image.shape)
	final_shape[-1] = 4

	result = []
	for row in image:
		for pixel in row:
			new_pixel = np.append(pixel, np.array([255]))
			print(new_pixel)
			result.append(new_pixel / l1_norm(new_pixel))

	image = np.array(result)
	return image.reshape(tuple(final_shape))


def image_to_qcolor_v2(image: np.ndarray):
	assert image.ndim == 3
	final_shape = list(image.shape)
	final_shape[-1] = 4

	image.flatten()
	image_result = np.append(image, np.array([255]))

	print(np.log2(image_result.shape[0]))

	if np.log2(image_result.shape[0]) % 2 != 0:
		number_add = 2**np.ceil(np.log2(image_result.shape[0])) - image_result.shape[0]
		image_result = np.append(image_result, np.zeros(int(number_add)))
	return image_result / l1_norm(image_result)


def qcolor_to_image(image: np.ndarray):
	assert image.ndim == 3
	final_shape = list(image.shape)
	final_shape[-1] = 3

	result = []
	for row in image:
		for pixel in row:
			new_pixel = pixel * (255 / pixel[3])
			result.append(np.delete(new_pixel, 3, 0))

	result_image = np.array(result)
	return result_image.reshape(tuple(final_shape))


pic = imageio.imread("images/image_at_epoch_0000.png")
arr = np.array(pic)

image = np.array([[[255, 255, 255], [0, 255, 0]],
                  [[255, 0, 0], [0, 0, 255]]])
print(image)
result = image_to_qcolor_v2(image)
# result = qcolor_to_image(result)
print(result)
