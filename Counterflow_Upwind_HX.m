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
R = 5;
Vc = 1;
Vh = 1;

%------------------------------------------------------------------
% NUMERICAL PARAMETERS
%------------------------------------------------------------------
Ns = 50;        % number of space steps
Nt = 500;       % number of time steps
L = 1;          % length
dx = L/Ns;
x = 0:dx:L;
tau = 1;            % final time
dt = tau/Nt;        % time step
T0 = 0;             % initial condition
time = 0:dt:tau;


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
for i=1:1:Nt+1
%     d(1) = beta_h*(1-sin(-0.5*time));
    x_f(i,:) = x;
    sol(:,i) = Tcurr;        
    d(1) = beta_h;
    Tnext = (A)*Tcurr+d;
    Tcurr = Tnext;
end

% add theta_h BC
sol = [ones(1,width(sol)); sol];
% add theta_c BC
sol = [sol(1:Ns+1,:); zeros(1,width(sol)); sol(1*Ns+2:end,:)];
% add theta_w BC
sol = [sol(1:2*Ns+2,:); sol(2*Ns+3,:); sol(2*Ns+3:end,:)];


theta_h = sol(1:Ns+1,:)';
theta_c = flipud(sol(Ns+2:2*Ns+2,:))';
theta_w = sol(2*Ns+3:3*Ns+3,:)';

t_f = repmat(time,1,Ns+1); 

figure(1)
plot(x,sol(1:Ns+1,Nt),'-r')
hold on
plot(x,flipud(sol(Ns+2:2*Ns+2,Nt)),'-b')
hold on
plot(x,sol(2*Ns+3:3*Ns+3,Nt))

data = [x_f(:), t_f',theta_w(:),theta_h(:),theta_c(:)];
dlmwrite('sol_true.dat',data,'delimiter',' ')


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

