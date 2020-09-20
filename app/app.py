# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()
