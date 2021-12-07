import numpy as np
from matplotlib import pyplot as plt
import matplotlib

font = {'family': 'Helvetica',
        'size': 18}

matplotlib.rc('font', **font)

# generate the data
np.random.seed(222)
X = np.random.normal(0, 1, (200, 1))
w_target = np.random.normal(0, 1, (1, 1))
# data + white noise
y = X @ w_target + np.random.normal(0, 1, (200, 1))

# least squares
w_estimate = np.linalg.inv(X.T @ X) @ X.T @ y
y_estimate = X @ w_estimate

# plot the data
plt.figure(figsize=(15, 10))
plt.scatter(X.flat, y_estimate.flat, label="Predicció")
plt.scatter(X.flat, y.flat, color='red', alpha=0.4, label="Dades")
plt.tight_layout()
plt.title("Regressió per diferencia de quadrats")
plt.legend()
plt.savefig("least_squares.png")
plt.show()
