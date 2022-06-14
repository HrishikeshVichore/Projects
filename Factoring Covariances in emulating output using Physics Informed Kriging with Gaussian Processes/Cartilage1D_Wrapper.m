function Cartilage1D_Wrapper
%CARTILAGE1D_WRAPPER Summary of this function goes here
%   Detailed explanation goes here

%for space/time point
nz=5; % number of space/time points

%for parameter phi0
xmin=0.1;
xmax=0.9;
n=5;
phi0=xmin+rand(1,n)*(xmax-xmin);
u = zeros(n,nz);
for i=1:length(phi0)
[zu,k] = cartilage1D(phi0(i), nz);
u(i,:) = k';
end
phi0 = phi0';
% zu = zu';
%histogram(phi0,100);
%hold on
save('test.mat', 'zu', 'phi0', 'u')
end

