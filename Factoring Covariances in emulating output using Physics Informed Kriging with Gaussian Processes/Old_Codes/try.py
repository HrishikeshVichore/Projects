import numpy as np
a = [[1,2],[2,3],[3,4],[4,5],[5,6]]
p1 = np.array(a)
m = np.zeros((p1.shape[0]))
mew = np.mean(p1,axis=0)
for i,xi in enumerate(p1):
    k = 1
    for j,x in enumerate(xi):
        k *= np.sum((x - mew[j])*(x - mew[j]))/2
    m[i] = k
    
print(m)
m = np.zeros((p1.shape[0]))
for i,xi in enumerate(p1):
    print(xi-mew)
    m[i] = np.sum((xi - mew)*(xi - mew))/2
    
print(m)
