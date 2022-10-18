function Burger_1D
%E = burger1D;
% Simulating the 1-D Burgers' equation by the Finite Difference
% Numerical scheme used is a second order in space and time,
% upwind Godunov for the nonlinear term, Davis' method
%%
%Specifying Parameters
nx=102; 
%Number of steps in space(x)
vis0=0.1; 
%mumax=max(vis, 1.0); %Diffusion coefficient/viscosity
nt=501; 
%Number of time steps 
time=zeros(nt,1); time(1)=0.0;
dx=1/(nx-2); 
%Width of space step
dx2=dx*dx;
%dt = 0.9*dx2/(2*mumax); %timestep explicit scheme
dt=0.1*dx;
Tfinal = 0.5;
x=0-(dx/2):dx:1+(dx/2);
%Range of x (0,1) 
u=zeros(1,nx); 
%Preallocating u
un=zeros(1,nx); 
%Preallocating un
du=zeros(1,nx);
ip=zeros(1,nx); 
%Auxillary variable
im=zeros(1,nx); 
...same as above
phi=zeros(1,nx); 
...same as above
dphi=zeros(1,nx); 
...same as above
%int=zeros(nt,1);
A=zeros(nx-2,nx-2);
rhs=zeros(1,nx-2);
w=zeros(1,nx-2);
%data=zeros(nx-2,nt);
ntrial=15;
solndata=zeros(ntrial,nx);
%%
%Setting up auxillary variables
for i=1:nx
ip(i)=i+1;
im(i)=i-1;
end
ip(nx)=1;
im(1)=nx;
for nd=1:ntrial
nx=102; 
%Number of steps in space(x)
vis0=0.01; 
%mumax=max(vis, 1.0); %Diffusion coefficient/viscosity
nt=501; 
%Number of time steps 
time=zeros(nt,1); time(1)=0.0;
dx=1/(nx-2); 
%Width of space step
dx2=dx*dx;
%dt = 0.9*dx2/(2*mumax); %timestep explicit scheme
dt=0.1*dx;
vis=vis0*nd;
mufact=(vis*dt)/(2*dx2);
% initial condition
u=zeros(1,nx); 
%Preallocating u
un=zeros(1,nx); 
%Preallocating un
unew=zeros(1,nx);
% nd
% dt
% for i=(.4*(nx-2)):(.6*(nx-2))
for i=1:nx
if(x(i) >= .4) && (x(i) <= .6)
u(i)=u(i)+1.0;
%% u(i)=(x(i)-0.4)*(-x(i)+0.6)*100;
end
end
% data(:,1)=u(2:nx-1);
% area=0;
% for i=2:nx-1
% area=area+dx*(u(im(i))+2*u(i)+u(ip(i)))/4;
% end
% int(1)=area;
for j=2:nx-3
A(j,j)=1+2*mufact;
A(j,j+1)=-mufact;
A(j,j-1)=-mufact;
end
A(1,1)=1+2*mufact;
A(1,2)=-mufact;
A(1,nx-2)=-mufact;
A(nx-2,nx-2)=1+2*mufact;
A(nx-2,nx-3)=-mufact;
A(nx-2,1)=-mufact;
%%
%Implicit-Explicit scheme 2nd order space and time
for it=2:nt
ttemp=time(it-1)+dt;
if (ttemp > Tfinal)
dt=Tfinal-time(it-1);
ttemp=time(it-1)+dt;
end
time(it)=ttemp;
for i=2:nx-1
up=u(ip(i))-u(i);
um=u(i)-u(im(i));
uc=u(ip(i))-u(im(i));
minmod= [2*abs(up) 2*abs(um) 0.5*abs(uc)];
du(i)=((sign(up)+sign(um))/2)*min(minmod);
end
du(nx)=du(2);
du(1)=du(nx-1);
un=u;
h=plot(x,u); 
%plotting the velocity profile
axis([0 1 -.5 1])
title({['1-D Burgers'' equation (\nu = ',num2str(vis),')'];['time(\itt) = ',num2str(time(it))]})
xlabel('x')
ylabel('u')
drawnow;
refreshdata(h)
for i=1:nx % mid-time prediction implicit
un(i)=u(i)-(0.5*(u(i)*dt*du(i))/dx);
end
for j=1:nx-2
rhs(j)=un(j+1);
end
w=A\rhs';
for j=2:nx-1
un(j)=w(j-1);
end
un(1)=un(nx-1);
un(nx)=un(2);
% Solve Riemann Problem
for i=1:nx-1
ul=un(i)+0.5*du(i);
ur=un(ip(i))-0.5*du(ip(i));
urp(i)=ul; 
%%%%max(ul,ur);
end
urp(nx)=urp(2);
% Corrector
for i=2:nx-1
unew(i)=u(i) - (dt/(2*dx))*(urp(i)^2 - urp(i-1)^2)
...
+(0.5*dt*vis/dx2)*(u(ip(i))-2*u(i)+u(im(i)));
end
unew(1)=unew(nx-1);
unew(nx)=unew(2);
for j=1:nx-2
rhs(j)=unew(j+1);
end
w=A\rhs';
for j=2:nx-1
unew(j)=w(j-1);
end
unew(1)=unew(nx-1);
unew(nx)=unew(2);
for i=1:nx
u(i)=unew(i);
end
% data(:,it)=u(2:nx-1);
solndata(nd,:)=u(:);
area=0;
for i=2:nx-1    
area=area+dx*(u(im(i))+2*u(i)+u(ip(i)))/4;
end
int(it)=area;
if (time(it) == Tfinal)
hold
figure
plot(time(1:it),int(1:it))
break
end
end
%save('solndata');
%save('solndata.txt','solndata','-ascii');
%save('data');
end
%hold
%figure
% plot(time,int)
%end