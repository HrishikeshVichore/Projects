function [z,u] = New_Cartilage1D(E0,k0,nz)
%solve loosly coupled, weakly nonlinear fluid and mechanics biomechanics
% problem from Mansoor
% E d^2u/dz^2 = dPdz
% dP/dz = (K/(1-phis)^2) du/dt ==> du/dt = (((1-phis)^2)/K) dP/dz
% with phis = phi0 (1-du/dz)
% u and P at cell edges
% u(1)=0 dPdz=0 j=1
% u(nz) = f(t), P=0 j=nz
% update u given the boundary forcing. with new u, compute P by inverting
% du/dz matrix
% constants
% E0=10; k0=1; nz=101; 
phi0=.75; P0=1;

dz=1/(nz-1); dz2=dz*dz; dt=0.0125*dz*dz; dtdz=dt/dz;
z=zeros(nz,1);
%totaltimestep= 250000+1; samplerate=1000;
totaltimestep= 400000+1; samplerate=2000;
sampletime=(totaltimestep-1)/samplerate;
for j=1:nz 
z(j)=(j-1)*dz ; 
end
% initializing fields
P=P0*ones(nz,1); dPdz=zeros(nz,1);
u=zeros(nz,1); dudz=zeros(nz,1);
udiff=zeros(nz,1); phis=zeros(nz,1);
k=k0*ones(nz,1); E = E0*ones(nz,1);
D=zeros(nz,nz); rhs=zeros(nz,1);
Ptime=zeros(nz,sampletime+1); Utime=zeros(nz,sampletime+1);
%for j=(nz-1)/2+1:nz
% k(j)=5*k0;
% E(j)=1*E0;
%end
for j=1:nz
phi(j)=phi0*(1-dudz(j));
end
time=zeros(sampletime+1,1);
t=0; time(1)=t;
displacement=zeros(nz,totaltimestep+1);
% boundary forcing
rate=0.001;
forcing = @(s) -rate*s;
%forcing = @(s) max(-rate*s,-rate*0.25);
for timestep=1:sampletime
for tt=1:samplerate
t=t+dt; 
for j=2:nz-1
dudz(j)=(u(j+1)-u(j-1))/(2*dz);
end
for j=1:nz-1
phis(j)=phi0*(1-dudz(j));
end
u(1) = 0;
u(nz)= forcing(t); 
for j=2:nz-1
u(j)=u(j)+0.5*dtdz*((1-phis(j))^2/k(j))*(P(j+1)-P(j-1));
end
for j=2:nz-1
rhs(j)=2*dz*( 0.5*(E(j+1)+E(j))*(u(j+1)-u(j))/dz - 0.5*(E(j)+E(j-1))*(u(j)-u(j-1))/dz)/dz ;
end
rhs(1)=0;
rhs(nz)=0;
for j=2:nz-1
D(j,j+1)=1;
D(j,j-1)=-1;
end
% 2nd order dP/dz=0 at z=0
D(1,1)=1;
D(1,2)=-4;
D(1,3)=3;
D(nz,nz)=1;
P=D\rhs;
end
time(timestep+1)=t;
for j=1:nz
Ptime(j,timestep+1)=P(j);
Utime(j,timestep+1)=u(j);
end
end
% t;
% size(z)
% size(time)
% size(Ptime)
% size(Utime)
%size(meshgrid(z,time))
% [Tg,Zg]=meshgrid(time,z);
% figure
% surf(Zg,Tg,Ptime);
% xlabel('Z'), ylabel('Time'), zlabel('Pressure')
% figure
% surf(Zg,Tg,Utime);
% xlabel('Z'), ylabel('Time'), zlabel('Displacement')
% figure
% plot(u,z)
% xlabel('u')
% ylabel('z')
%figure
%plot(P,z,'r*')
%xlabel('P')
%ylabel('z')
end