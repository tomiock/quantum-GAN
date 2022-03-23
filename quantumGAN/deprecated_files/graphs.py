import matplotlib.pyplot as plt
import numpy as np

lst_images = np.array([104, 262, 498, 1600, 2991, 6575])
lst_no_images = np.array([93, 203, 400, 788, 1032, 1781])
x_axis = np.array([20, 50, 100, 200, 300, 500])

plt.title("Guardar les imatges VS No Guardar-les")
plt.scatter(x_axis, lst_images, label="Guardant")
plt.scatter(x_axis, lst_no_images, label="No Guardant")
plt.legend(loc='best')
plt.xlabel('segons')
plt.ylabel('iteracions')
plt.xticks(np.arange(0, 525, step=25))
plt.show()
