load('test1.mat')
% load('y_cap6_copy.mat')
load('y_cap3D.mat')
i=3;
hold on
plot(u1(i,1:end))
plot(y_cap_kernel_sq_exp(i,1:end))
plot(y_cap_kernel_mean(i,1:end))
legend('original', 'kernel_sq_exp', 'kernel_mean')