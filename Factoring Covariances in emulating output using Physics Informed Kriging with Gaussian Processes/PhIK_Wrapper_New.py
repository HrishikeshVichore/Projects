import pandas as pd
from Working_Codes_Copy.PhIK_Top_Down_Approach2_Copy import PhIK_1D
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

file = pd.read_csv('Data/UT.csv')
number_of_space_points = 51

space_points = file.iloc[:,:number_of_space_points].to_numpy()
param_points = file['Eray'].to_numpy()
time_points = file['time'].to_numpy()
output_points = file.iloc[:,number_of_space_points+2:].to_numpy()
print('For train')
print(space_points.shape)
print(param_points.shape)
print(time_points.shape)
print(output_points.shape)

file = pd.read_csv('Data/UT_Test.csv')

q_star = file['Eray'].to_numpy()
original = file.iloc[:,2:].to_numpy()

print('For test')
print(q_star.shape)
print(original.shape)

for_plot = []
for i in range(time_points.shape[0]):
    x = np.expand_dims(space_points[i],1)
    u = np.expand_dims(output_points[i],1).T
    q = np.expand_dims(np.array([param_points[i]]),1)
    q_1 = np.expand_dims(np.array([q_star[i]]),1)

    p = PhIK_1D(space_points=x, param_points = q, output_points = u,kernel=1)
    y_cap_kernel_mean = p.fit(q_star=q_1,x_star=None)
    y_cap_kernel_mean = np.squeeze(np.array([np.float32(i) for i in y_cap_kernel_mean]))
    for_plot.append(y_cap_kernel_mean)

for_plot = np.array(for_plot)
print(for_plot.shape)

np.savetxt("for_plot_new.txt", for_plot, delimiter=",")

'''for_plot = np.load('for_plot_new.npy')

# print(time_points)
xline,yline = np.meshgrid(space_points[0],time_points)
HA=1e8
K=1e14
h=5e-3
phi0=0.8
gamma = np.sqrt(HA*((1-phi0)**2)/K)
yline *= h**2/gamma**2
xline *= h
zline = for_plot * h

# print(yline)

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xline, yline, zline, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
surf = ax.plot_surface(xline, yline, original, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
plt.show()'''

'''x_star = space_points[-1]
q_star = param_points[-1]
original = output_points[-1]
t_star = time_points[-1]
# print(original.shape)
i = 38
space_points = np.expand_dims(space_points[i],1)
param_points = np.expand_dims(np.array([param_points[i]]),1)
output_points = np.expand_dims(output_points[i],1).T
time_points = time_points[i]



q_1 = np.expand_dims(np.array([q_star]),1)
x_1 = np.expand_dims(x_star,1)

print(q_1.shape)
print(x_1.shape)


p = PhIK_1D(space_points=space_points, param_points = param_points, output_points = output_points,kernel=1)
y_cap_kernel_mean = p.fit(q_star=q_1,x_star=None)
y_cap_kernel_mean = np.squeeze(np.array([np.float32(i) for i in y_cap_kernel_mean]))

plt.plot(x_1, original,label='Original')
plt.plot(x_1, y_cap_kernel_mean,label='Predicted')
plt.legend()
plt.show()
predicted = y_cap_kernel_mean
l = p.get_loss(original, predicted, type='l1')
print(f'L1 loss = {np.sum(l)}')
l = p.get_loss(original, predicted, type='l2')
print(f'L2 loss = {np.sum(l)}')'''
