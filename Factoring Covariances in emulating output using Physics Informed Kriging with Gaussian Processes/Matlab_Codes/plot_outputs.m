load('test_hetero.mat')
% load('y_cap6_copy.mat')
load('y_cap3D.mat')
i=1;
hold on
plot(u_test(i,1:end))
% plot(y_cap_kernel_sq_exp(i,1:end))
plot(y_cap_kernel_mean(i,1:end))
legend('original', 'kernel\_mean')