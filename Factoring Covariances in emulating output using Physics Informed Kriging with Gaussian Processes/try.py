import numpy as np
A = np.array([i for i in range(1,11)])
B = np.ones((10,10))
print(A,B)
C = np.kron(A,B)
C = np.reshape(C,(10,10,10))
print(C, C.shape)