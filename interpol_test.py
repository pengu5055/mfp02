import numpy as np
import matplotlib.pyplot as plt

x_data = [0, 1, 2, 3, 4, 5, 6, 7]
y_data = [0, 1, 2, 3, 3, 2, 1, 0]
intr = [i + 0.5 for i in range(0, 6)]
print(x_data, y_data)
interpol = np.interp(intr, x_data, y_data)
print(intr, interpol)

plt.scatter(x_data, y_data)
plt.plot(x_data, y_data)
plt.scatter(intr, interpol)
plt.show()