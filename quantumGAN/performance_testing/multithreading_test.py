import random

import numpy as np
import pandas as pd
import time

train_data = []
d = {"real datapoint": [], "fake datapoint": []}
for _ in range(5000):
	x2 = np.random.uniform(.5, .4, (2,))
	fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (2,))
	real_datapoint = np.array([x2[1], 0., x2[0], 0])
	index = ["real datapoint", "fake datapoint"]
	d["real datapoint"].append(real_datapoint)
	d["fake datapoint"].append(fake_datapoint)
df = pd.DataFrame(d)

train_data = []
for _ in range(5000):
	x2 = np.random.uniform(.5, .4, (2,))
	fake_datapoint = np.random.uniform(-np.pi * .01, np.pi * .01, (2,))
	real_datapoint = np.array([x2[1], 0., x2[0], 0])
	train_data.append((real_datapoint, fake_datapoint))

time1 = time.time()


def create_mini_batches(training_data, mini_batch_size):
	n = len(training_data)
	random.shuffle(training_data)
	mini_batches = [
		training_data[k:k + mini_batch_size]
		for k in range(0, n, mini_batch_size)]

	return mini_batches[0]


mini = create_mini_batches(train_data, 10)
print(mini)
