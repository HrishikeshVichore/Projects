import numpy as np
a = np.array([[1,6,12],
              [3,12,24],
              [5,18,72],
              [7,48,36]])

b = np.mean(a)
c = (a - b) / np.sqrt(12)
print(c/np.std(c))
