function Cartilage1D_Wrapper
%CARTILAGE1D_WRAPPER Summary of this function goes here
%   Detailed explanation goes here
rng(1); nz = 13; np=12; 

%for space/time point
% nz=13; number of space/time points

%for parameter phi0
% np=12; number of paramtere points

xmin=0.1;
xmax=0.9;
phi0=xmin+rand(1,np)*(xmax-xmin);
u = zeros(np,nz);
for i=1:length(phi0)
[zu,k] = cartilage1D(phi0(i), nz);
u(i,:) = k';
end

params = phi0';


save('Data/train.mat', 'zu', 'params', 'u')


% For testing


% nz = 130; np=5;

xmin=zu(1);
xmax=zu(2);
test_zu = xmin+rand(1,nz)*(xmax-xmin);
% test_zu = zu;
% xmin=phi0(1);
% xmax=phi0(2);
% test_phi0 = xmin+rand(1,np)*(xmax-xmin);
% test_phi0 = [0.8,0.88,0.2,0.73,0.15];
test_params = [0.5];
u1 = zeros(length(test_params),length(test_zu));

for i=1:length(test_params)   
[zu,k] = Copy_of_cartilage1D(test_params(i), test_zu, nz);
u1(i,:) = k';

end

save('Data/test.mat', 'u1','test_zu', 'test_params')

end

