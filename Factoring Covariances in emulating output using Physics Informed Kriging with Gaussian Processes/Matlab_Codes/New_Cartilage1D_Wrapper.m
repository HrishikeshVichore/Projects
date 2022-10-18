function New_Cartilage1D_Wrapper
%CARTILAGE1D_WRAPPER Summary of this function goes here
%   Detailed explanation goes here
nz = 101; np=5; 

%for space/time point
% nz=13; number of space/time points

%for parameter phi0
% np=12; number of paramtere points

xmin=0.5;
xmax=10;
E0=xmin+rand(1,np)*(xmax-xmin);
k0=E0;
u = zeros(np^2,nz);
params = zeros(np^2,2);
count = 1;
for i=1:length(E0)
    for j=1:length(k0)
        [zu,k] = New_Cartilage1D(E0(i), k0(j),nz);
        u(count,:) = k';
        disp(size(k));
        params(count,:) = [E0(i), k0(j)];
        count = count + 1;
    end
end

% zu = zu';
% disp(zu);
save('train1.mat', 'zu', 'params', 'u')


% For testing


% nz = 130; 
np=6;

% xmin=zu(1);
% xmax=zu(2);
% test_zu = xmin+rand(1,nz)*(xmax-xmin);
test_zu = zu;
% xmin=phi0(1);
% xmax=phi0(2);
test_E0 = xmin+rand(1,np)*(xmax-xmin);
% test_E0 = [2.3];
test_k0 = test_E0;
% test_phi0 = [0.8,0.88,0.2,0.73,0.15];
% test_phi0 = [0.5];
u1 = zeros(length(test_E0)*length(test_k0),length(test_zu));
test_params = zeros(length(test_E0)*length(test_k0),2);
count = 1;
for i=1:length(test_E0)
    for j=1:length(test_k0)
        [test_zu,k] = New_Cartilage1D(test_E0(i), test_k0(j),nz);
        u1(count,:) = k';
        test_params(count,:) = [test_E0(i), test_k0(j)];
        count = count + 1;
    end
end

test_zu = test_zu';
save('test1.mat', 'u1','test_zu', 'test_params')

commandStr = 'python "D:\Documents\Liclipse Workspace\Bruce_Pitman_Project\PhIK_Wrapper.py"';
[status, commandOut] = system(commandStr);
if status==0
 fprintf('squared result is %d\n',str2num(commandOut));
end
plot_outputs
end

