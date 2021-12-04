import numpy as np


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
	final_shape = list(image.shape)
	final_shape[-1] = 4

	image_result = np.append(image, np.array([255]))
	print(l1_norm(image_result))

	# print(np.log2(image_result.shape[0]))
	#
	# if np.log2(image_result.shape[0]) % 2 != 0:
	#	number_add = 2**np.ceil(np.log2(image_result.shape[0])) - image_result.shape[0]
	#	image_result = np.append(image_result, np.zeros(int(number_add)))
	return image_result / l1_norm(image_result)


def qcolor_to_image(image: np.ndarray):
	assert image.ndim == 1
	result_image = image * (255 / np.amax(image))
	result_image = np.delete(result_image, -1, 0)
	return result_image.astype(int)
