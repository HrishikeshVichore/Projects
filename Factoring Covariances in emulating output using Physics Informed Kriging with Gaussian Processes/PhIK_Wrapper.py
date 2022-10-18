from scipy.io import loadmat, savemat
from Working_Codes_Copy.PhIK_Top_Down_Approach2_Copy import PhIK_1D
# from PhIK_Bottom_Up_Approach3 import PhIK_1D
import numpy as np
# from lengthscale_calc import lengthscale
# import lengthscale_calc


'train1,z,E0,k0,u'
file = loadmat('Data/train_hetero.mat')
print('For train file')
# s = input('Variable name for space points -> ')
# p = input('Variable name for param points -> ')
# u = input('Variable name for output points -> ')
s = 'x'
p = 'P'
u = 'u'
space_points = file[s] #(m,k) m-> number of pts. k-> sets of points
param_points = file[p] #(n,l) n-> number of pts. l-> sets of points
output_points = file[u] #(n,m)

print(space_points.shape)
print(param_points.shape)
print(output_points.shape)

test_file = loadmat('Data/test_hetero.mat')
print('For test file')
# s = input('Variable name for space points -> ')
# p = input('Variable name for param points -> ')
# u = input('Variable name for output points -> ')
s = 'x_test'
p = 'p_test'
u = 'u_test'
x_star = test_file[s]
q_star = test_file[p]
original = test_file[u]


# print(q_star.shape)
# exit()
print('Kernel_Mean')
p = PhIK_1D(space_points=space_points, param_points = param_points, output_points = output_points,kernel=1)
y_cap_kernel_mean = p.fit(q_star=q_star,x_star=x_star)
y_cap_kernel_mean = np.squeeze(np.array([np.float32(i) for i in y_cap_kernel_mean]))
# savemat('Data/y_cap3D.mat',{'y_cap_kernel_mean':y_cap_kernel_mean})

# print('Kernel_Sq_Exp')
# p = PhIK_1D(space_points=space_points, param_points = param_points, output_points = output_points,kernel=0)
# y_cap_kernel_sq_exp = p.fit(q_star=q_star,x_star=None,length_scale=[0.0005,0.0005])
# # exit()
# y_cap_kernel_sq_exp = np.squeeze(np.array([np.float32(i) for i in y_cap_kernel_sq_exp]))
# print(y_cap_kernel_1.shape)
# savemat('y_cap3D.mat',{'y_cap_kernel_sq_exp':y_cap_kernel_sq_exp})
# predicted = y_cap_kernel_sq_exp

print('Saved!')
predicted = y_cap_kernel_mean
# print(original.shape)
# print(predicted.shape)
l = p.get_loss(original, predicted, type='l1')
print(f'L1 loss = {np.sum(l)}')
l = p.get_loss(original, predicted, type='l2')
print(f'L2 loss = {np.sum(l)}')
