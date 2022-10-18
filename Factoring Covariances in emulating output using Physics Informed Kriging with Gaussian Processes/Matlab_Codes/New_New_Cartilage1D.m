function New_New_Cartilage1D
%solve loosly coupled, weakly nonlinear fluid and mechanics biomechanics
% problem from Mansoor
% E d^2u/dz^2 = dPdz
% dP/dz = (K/(1-phis)^2) du/dt ==> du/dt = (((1-phis)^2)/K) dP/dz
% with phis = phi0 (1-du/dz)
% u and P at cell edges
% u = 0 dPdz=0 at z=1
% u = f(t), P=0 at z=0
% update u given the boundary forcing. with new u, compute P by inverting
% du/dz matrix
% constants
E0=1; k0=1; phi0=.8; P0=1;
nz=51; 
dz=1/(nz-1); dz2=dz*dz; dt=0.005*dz*dz; dtdz=dt/dz;
z=zeros(nz,1);
%totaltimestep= 250000+1; samplerate=1000;
%% totaltimestep= 4000+1; samplerate=200;
%totaltimestep=50000+1; samplerate=250;
%%%totaltimestep=75001; samplerate=7500;
totaltimestep=75001; samplerate=7500;
sampletime=(totaltimestep-1)/samplerate;
for j=1:nz 
z(j)=(j-1)*dz ; 
end
% initializing fields
P=zeros(nz,1); dPdz=zeros(nz,1);
u=zeros(nz,1); dudz=zeros(nz,1);
phi=phi0*ones(nz,1); k=k0*ones(nz,1);
E = E0*ones(nz,1);
D=zeros(nz,nz); rhs=zeros(nz,1);
Ptime=zeros(nz,sampletime+1); Utime=zeros(nz,sampletime+1);
Phitime=phi0*ones(nz,sampletime+1);
%for j=(nz-1)/2+1:nz
% k(j)=10*k0;
% E(j)=10*E0;
%end
for j=1:nz
phi(j)=phi0*(1-dudz(j));
end
time=zeros(sampletime+1,1);
t=0; time(1)=t;
% boundary forcing
%rate2=2;
rate=0.75;
u0bar=0.5; eps=5.0;
%forcing = @(s) rate*s;
forcing = @(s) min(max(rate*max((s-0.01),0.0)), .05);
%forcing = @(s) -ubar*(1/2)*tanh(eps*(s-1)+1);
%forcing = @(s) u0bar*(1/2)*( tanh(eps*((s/3)-3)) +1 );
%%U0Bar=0.05;
%eps=5.0; % controls ramp rate
%for k=1:N+1
% uBar(1,k)=U0Bar*1/2*(tanh(eps*(tBar(k)-1))+1);
%end
for j=1:nz
Ptime(j,1)=P(j);
Utime(j,1)=u(j);
Phitime(j,1)=phi(j);
end
counter=0;
%kvalues=[0.25,0.5,1.0,2.0];
%PT=cell(length(kvalues),11,3);
% Evalues=[0.25,0.5,1.0,2.0];
Evalues=[0.35,0.75,1.5,2.5];

UT=cell(length(Evalues),11,3);
for Eloop=Evalues
%for kloop=kvalues
counter=counter+1;
% initializing fields
t=0;
P=zeros(nz,1); dPdz=zeros(nz,1); u=zeros(nz,1); dudz=zeros(nz,1);
k=k0*ones(nz,1); 
%k=kloop*ones(nz,1); 
E = Eloop*ones(nz,1);
%E=E0*ones(nz,1);
phi=phi0*ones(nz,1);
Ptime=zeros(nz,sampletime+1); Utime=zeros(nz,sampletime+1);
Phitime=phi0*ones(nz,sampletime+1);
for j=(nz-1)/2+1:nz
% k(j)=1.5*k(j);
E(j)=2.5*E(j);
end
for timestep=2:sampletime+1
for tt=1:samplerate
t=t+dt;
for j=2:nz-1
dudz(j)=(u(j+1)-u(j-1))/(2*dz);
end
dudz(nz)=(u(nz)-u(nz-1))/dz;
dudz(1) =(u(2)-u(1))/dz;
for j=1:nz
phi(j)=phi0*(1-dudz(j));
end
% if timestep < 20
% u(1)=0;
% else
u(1) = forcing(t);
% end
u(nz)= 0; 
for j=2:nz-1
u(j)=u(j)+0.5*dtdz*((1-phi(j)/(1-phi0))^2/k(j))*(P(j+1)-P(j-1));
end
for j=2:nz-1
rhs(j)= 2*dz*( 0.5*(E(j+1)+E(j))*(u(j+1)-u(j))/dz - 0.5*(E(j)+E(j-1))*(u(j)-u(j-1))/dz )/dz ;
end
rhs(1)=0;
rhs(nz)=0;
for j=2:nz-1
D(j,j+1)=1;
D(j,j-1)=-1;
end
% 2nd order dP/dz=0 at z=1
% D(1,1)=1;
% D(1,2)=-4;
% D(1,3)=3;
% D(nz,nz)=1;
D(1,1)=1; 
%P=0
D(nz,nz)= 3; 
%dP/dz=0
D(nz,nz-1)=-4;
D(nz,nz-2)= 1;
% D(nz,nz)=1;
% D(nz,nz-1)=-1;
P=D\rhs;
end
time(timestep)=t;
for j=1:nz
Ptime(j,timestep)=P(j);
Utime(j,timestep)=u(j);
Phitime(j,timestep)=phi(j);
end
end

%kray=kloop*ones((sampletime+1),1);
%PTemp{counter}=table(kray,time,transpose(Ptime))
Eray=Eloop*ones((sampletime+1),1);
UTemp{counter}=table(Eray,time,transpose(Utime));
%PT(counter,:,:)=table2cell(PTemp);
%writetable(PT, 'PT.txt');
%save('Pout.txt','time','Ptime','-ascii');
end
%PToutput=vertcat(PTemp{:});
%writetable(PToutput, 'PT.txt');
UToutput=vertcat(UTemp{:});
% size(UToutput)
% writetable(UToutput,'UT_Test.csv');
%size(z)
%size(time)
%size(Ptime)
% size(Utime)

%size(meshgrid(z,time))
opts = detectImportOptions('for_plot_new.csv','NumHeaderLines',33);
Pred = table2array(readtable('for_plot_new.csv', opts))';
opts = detectImportOptions('UT_Test.csv','NumHeaderLines',34);
opts.SelectedVariableNames = (2:52);
Og = table2array(readtable('UT_Test.csv', opts))';
% size(Pred)
% size(Og)
% Redimensionalize
HA=1e8;
K=1e14;
h=5e-3;
gamma = sqrt(HA*((1-phi0)^2)/K);
%uPlot=uPlot*h;
%tPlot=tPlot*h^2/gamma^2;
%pPlot=pPlot*HA;
% [z,t]=meshgrid(0:dzBar*h:h,0:dtBar*h^2/gamma^2*skip:tBarMax*h^2/gamma^2);
% [Tg,Zg]=meshgrid(time,z);
[Tg,Zg]=meshgrid(time*h^2/gamma^2, z*h );
% size(Zg)
% figure
%surf(Zg,Tg,Ptime);
% surf(Zg*h,Tg*h^2/gamma^2,Ptime*HA);
% xlabel('Z'), ylabel('Time'), zlabel('Pressure')
% figure
%surf(Zg,Tg,Utime);
surf(Zg*h,Tg*h^2/gamma^2,Utime*h);
xlabel('Z'), ylabel('Time'), zlabel('Displacement\_Og')
figure
surf(Zg*h,Tg*h^2/gamma^2,Pred*h);
xlabel('Z'), ylabel('Time'), zlabel('Displacement\_Pred')
% surf(Zg*h,Tg*h^2/gamma^2,Og);
% xlabel('Z'), ylabel('Time'), zlabel('Displacement\_Og')
% figure
%surf(Zg,Tg,Phitime);
% surf(Zg*h,Tg*h^2/gamma^2,Phitime);
% xlabel('Z'), ylabel('Time'), zlabel('PhiS')
%figure
%plot(u,z,'k*')
%xlabel('u')
%ylabel('z')
%figure
%plot(P,z,'r*')
%xlabel('P')
%ylabel('z')
end