from scipy.io import loadmat, savemat
from PhIK_Top_Down_Approach2 import PhIK_1D
# from PhIK_Bottom_Up_Approach3 import PhIK_1D
import numpy as np


'train1,z,E0,k0,u'
file = loadmat('train1')

space_points = file['zu']
param_points = file['params']
output_points = file['u']

# print(space_points.shape)
# print(param_points.shape)
# print(output_points.shape)

test_file = loadmat('test1')
x_star = test_file['test_zu']
q_star = test_file['test_params']
original = test_file['u1']

# print(q_star.shape)
# exit()
print('Kernel_Mean')
p = PhIK_1D(space_points=space_points, param_points = param_points, output_points = output_points,kernel=1)
y_cap_kernel_mean = p.fit(q_star=q_star,x_star=None)
print('Kernel_Sq_Exp')
p = PhIK_1D(space_points=space_points, param_points = param_points, output_points = output_points,kernel=0)
y_cap_kernel_sq_exp = p.fit(q_star=q_star,x_star=None,length_scale=[0.0005,0.0005])
# exit()
y_cap_kernel_mean = np.squeeze(np.array([np.float32(i) for i in y_cap_kernel_mean]))
y_cap_kernel_sq_exp = np.squeeze(np.array([np.float32(i) for i in y_cap_kernel_sq_exp]))

# print(y_cap_kernel_1.shape)
savemat('y_cap3D.mat',{'y_cap_kernel_sq_exp':y_cap_kernel_sq_exp, 'y_cap_kernel_mean':y_cap_kernel_mean})
print('Saved!')
predicted = y_cap_kernel_mean
# print(original.shape)
# print(predicted.shape)
l = p.get_loss(original, predicted, type='l1')
print(np.sum(l))
