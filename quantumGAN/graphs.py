import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

lst_images = np.array([104, 262, 498, 1600, 2991, 6575])
lst_no_images = np.array([93, 203, 400, 788, 1032, 1781])
x_axis = np.array([20, 50, 100, 200, 300, 500])

plt.title("saving the images vs not saving them")
plt.scatter(x_axis, lst_images, label="with saving images")
plt.scatter(x_axis, lst_no_images, label="without saving images")
plt.legend(loc='best')
plt.xlabel('seconds')
plt.ylabel('epochs')
plt.yscale("log")
plt.xticks(np.arange(0, 525, step=25))
plt.show()
