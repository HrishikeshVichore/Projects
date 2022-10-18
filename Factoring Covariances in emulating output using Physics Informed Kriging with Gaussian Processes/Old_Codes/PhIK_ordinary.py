from sympy import var, solve, symbols
from sympy.plotting import plot
import numpy as np
from sympy.stats import covariance
# Assuming a function
x = symbols('x')
a = symbols('a')
f_x = x**2 + a*x
# let a range from 1 to 5.
val_a = list(range(1,6)) 
val_f_x = np.array([f_x.subs(a,i) for i in val_a])
roots = [solve(i) for i in val_f_x]
print('Roots of all equations = ',roots)
#plot(*val_f_x)
# for i in val_a:
#     plot(f_x.subs(a,i))

mean = np.mean(val_f_x)
print('Mean = ',mean)
temp = val_f_x-mean
temp = temp.reshape(temp.shape[0],1)
cov_matrix = np.dot(temp,temp.T)
print('Covariance Matrix =\n',cov_matrix)
