clear; close all; clc;
%------------------------------------------------------------------
% THIS CODE SOLVES A HEAT EXCHANGER MODEL USING CRANK-NICOLSON METHOD. THE
% BOUNDARY CONDITIONS CAN EASILY BE MODIFIED TO MODEL COUNTERFLOW OR
% CROSSFLOW HEAT EXCHANGERS.
%   WRITTEN BY : SAM YANG
%   DATE : 2017.09
%------------------------------------------------------------------
set(0,'defaulttextinterpreter','latex')

%------------------------------------------------------------------
% HEAT EXCHANGER PARAMETERS
%------------------------------------------------------------------
R = 1;
Vc = 1;
Vh = 1;

%------------------------------------------------------------------
% NUMERICAL PARAMETERS
%------------------------------------------------------------------
Ns = 30;        % number of space steps
Nt = 1000;       % number of time steps
L = 10;          % length
dx = L/Ns;
x = dx:dx:L;
tau = 10;            % final time
dt = tau/Nt;        % time step
T0 = 0;             % initial condition
time = dt:dt:tau;


% coefficients
alpha_h = R*dt/(Vh);
alpha_c = dt/Vc;
beta_h = alpha_h/(dx);
beta_c = alpha_c/(dx);

if( beta_h + beta_c >= 1 ) 
    beta_h + beta_c
   error('need to decrease the step size for stability!') 
end

%------------------------------------------------------------------
% load coefficient matrices
%------------------------------------------------------------------
% 1. u^(j+1) terms (next time step)
Tnext = zeros(Ns*3,1);
% 2. u^j terms
Tcurr = T0*ones(Ns*3,1);
% 3. coefficient matrix
A11 = diag((1-alpha_h-beta_h)*ones(Ns,1));
A11 = A11 + diag(beta_h*ones(Ns-1,1),-1);
A12 = zeros(Ns);
A13 = diag(alpha_h*ones(Ns,1));
A21 = zeros(Ns);
A22 = diag((1-alpha_c-beta_c)*ones(Ns,1));
A22 = A22 + diag(beta_c*ones(Ns-1,1),-1);
A23 = fliplr(diag(alpha_c*ones(Ns,1)));
A31 = diag(dt*R*ones(Ns,1));
A32 = fliplr(diag(dt*ones(Ns,1)));
A33 = diag((1-dt*(1+R))*ones(Ns,1));
A = [A11 A12 A13; A21 A22 A23; A31 A32 A33];

% 3. create boundary vector
d = zeros(Ns*3,1);
d(1) = beta_h;
% d(Ns+1) = -2*alpha_c+2*beta_c;
for i=1:1:Nt
%     d(1) = beta_h*(1-sin(-0.5*time));
    d(1) = beta_h;
    Tnext = (A)*Tcurr+d;
    sol(:,i) = Tnext;
    Tcurr = Tnext;
end

theta_h = sol(1:Ns,:);
theta_c = sol(Ns+1:2*Ns,:)';
theta_w = sol(2*Ns+1:3*Ns,:);

% nx = 100; nt = 100;
% [X,Y] = meshgrid(linspace(min(x),max(x),nx),linspace(min(time),max(time),nt)) ;
% z = theta_h';
% Z = griddata(x,time,z,X,Y) ;
% figure(1)
% subplot(1,3,1)
% [p1,p2] = contourf(X,Y,Z);
% hold on
% colormap jet
% axis equal 
% title('$\theta_h$')
% xlabel('$x$')
% ylabel('$\tau$')
% subplot(1,3,2)
% z = theta_c';
% Z = griddata(x,time,z,X,Y) ;
% [p1,p2] = contourf(X,Y,Z);
% hold on
% axis equal 
% title('$\theta_c$')
% xlabel('$x$')
% subplot(1,3,3)
% z = theta_w';
% Z = griddata(x,time,z,X,Y) ;
% [p1,p2] = contourf(X,Y,Z);
% hold on
% axis equal 
% title('$\theta_w$')
% xlabel('$x$')

% subplot(3,1,1)
% surf(time,x,Th)
% xlabel('t (s)')
% ylabel('x (m)')
% zlabel('\tau')
% 
% subplot(3,1,2)
% plot(time,dT)
% xlabel('t (s)')
% ylabel('\tau_{h,o}-\tau_{c,o}')
% 
figure(1)
plot(x,sol(1:Ns,Nt),'-r')
hold on
plot(x,sol(Ns+1:2*Ns,Nt),'-b')
hold on
plot(x,sol(2*Ns+1:3*Ns,Nt))

dlmwrite('sol_true.dat',sol,'delimiter',' ')



% 
% x = 0:dx:L-dx;
% % animated graph
% figure(1)
% time = 0;
% 
% T_hin = 293.15;
% T_cin = 279.15;
% for i=1:1:Nt
%     %TnH = Tcurr(1:10,1);
%     %TnC = Tcurr(11:20,1);
%     %TnW = Tcurr(21:30,1);
%    % TnC = flipud(TnC);
%     %Tcurr = [TnH;TnC;TnW];
%     Tnext = (A)*Tcurr+d;
%     sol(:,i) = Tnext;
%     Tcurr = Tnext;
%     
%     T_hin = Tnext(Ns)*(T_hin-T_cin)+T_cin
%     Tplot = [Tnext(1:Ns); Tnext(Ns+1:Ns*2); Tnext(Ns*2+1:end)];
%     flipud(Tplot(Ns+2:2*Ns+2));
%     plot(x,Tplot(1:Ns),'-r',x,flipud(Tplot(Ns+1:2*Ns)),'-b',x,Tplot(2*Ns+1:3*Ns),'-k')   
%     time = time + dt;
%     legend('hot stream','cold stream','wall')
%     xlabel('$\tilde{x}$')
%     ylabel('$\tau$')
%     title(['$\tilde{t}=$' num2str(time)])
%     xlim([0 L])
%     ylim([0 1])
%     drawnow                                   % refresh the image on screen
%     pause(0.01)                                % control animation speed
% %     snapnow 
% end

