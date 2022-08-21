function [zu,u] = Copy_of_cartilage1D(phi0,zu,nz)
%solve loosly coupled, weakly nonlinear fluid and mechanics biomechanics 
% problem from Mansoor
%  E d^2u/dz^2 =   dPdz
%  dP/dz = (K0/(1-phis)^2) du/dt ==> du/dt = (((1-phis)^2)/K0) dP/dz
% with phis = phi0 (1-du/dz) 
% u at cell centers;  P at cell edges 
% u(0)=0    dPdz=0 at bottom  
% u(1) = f(t), P=0 at top 
% update u given the boundary forcing. with new u, compute P
 

% constants
E=10;      k0=1;      P0=1;
 

dz=1/(nz-2);  dz2=dz*dz;  dt=0.0125*dz*dz;   dtdz=dt/dz;
totaltimestep=2500;
% index j=1 is at -dz/2 for u, dPdz. 
% for j=1:nz                                      
%       zu(j)=(j-1.5)*dz ;                    
% end

% index j=1 is at z=0,1, etc for P, dudz
for j=1:nz-1                                  
      zP(j)=(j-1)*dz ;                     
end
 

% initializing fields
P=P0*ones(nz-1,1);    dPdz=zeros(nz,1);
u=zeros(nz,1);        dudz=zeros(nz-1,1);  
udiff=zeros(nz,1);
k=k0*ones(nz,1);  
D=zeros(nz-1,nz-1);
%for j=(nz-2)/2+1:nz+2
%    k(j)=1*k0;
%end
for j=1:nz-1
  phi(j)=phi0*(1-dudz(j));
end
D=zeros(nz-1,nz-1);   rhs=zeros(nz-1,1);  
time=zeros(totaltimestep+1);
t=0; time(1)=t;
displacement=zeros(nz,totaltimestep+1);
% boundary forcing
rate=0.001;
forcing = @(s) -rate*s;
 

for timestep=1:totaltimestep
  t=t+dt; time(timestep+1)=t ; 
 

  for j=2:nz-1
    phih=(phi(j)+phi(j-1))/2;  %phi = phi0(1-dudz) so runs j=1:nz-1
    u(j)=u(j)+dtdz*((1-phih)^2/k(j))*(P(j)-P(j-1));
  end
  u(1) =-u(2);
  u(nz)=forcing(t);
  

  for j=1:nz-1
      dudz(j)=(u(j+1)-u(j))/dz;
  end
  for j=1:nz-1
  phi(j)=phi0*(1-dudz(j));
  end
  

  for j=2:nz-1
    rhs(j)=E*(u(j+1)-2*u(j)+u(j-1))/dz;
  end
    rhs(1)=0;
  for j=2:nz-1
    D(j,j)=1+0.0000001;
    D(j,j-1)=-1;
  end
% 2nd order dP/dz=0 at z=0
  D(1,1)=-3/2;
  D(1,2)=4/2;
  D(1,3)=-1/2;
  

  P=D\rhs;
  

  for j=1:nz
      displacement(j)=displacement(j)+dt*u(j);
  end
  

  

 %nonlinear diffusion version for comparison 
  for j=2:nz-1
    phih=(phi(j)+phi(j-1))/2;  %phi = phi0(1-dudz) so runs j=1:nz-1
    udiff(j)=udiff(j)+(dt/dz2)*((1-phih)^2/k(j))*E* ...
        (udiff(j+1)-2*udiff(j)+udiff(j-1));
  end
  udiff(1) =-udiff(2);
  udiff(nz)=forcing(t);
  

  

end
 
% zu
% phi
% u

%save('test.mat', 'zu', 'phi', 'u')

% plot(zu,u,'--')
%hold on
%plot(zu,udiff,'r*')
for j=1:nz
    displacement(j);
end
 

 

end