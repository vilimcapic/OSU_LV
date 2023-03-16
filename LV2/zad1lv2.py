import numpy as np
import matplotlib.pyplot as plt 

x = np.array([1, 2 , 3, 3, 1], float)
y = np.array([1, 2, 2, 1, 1], float)
plt.plot(x, y, 'r', marker='.', markersize=5,)
plt.axis([0,4,0,4])
plt.title ( 'LV2 Zadatak 1')
plt.xlabel ('X')
plt.ylabel ('Y')
plt.show ()
