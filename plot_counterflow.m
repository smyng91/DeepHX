clear; clc; close all
% plot 2D CFD data


sol = importdata('test.dat');
data = sol.data;

x = data(:,1);
t = data(:,2);
theta_w = data(:,3);
theta_h = data(:,4);
theta_c = data(:,5);

figure(2)
n = 10;
plot(data(abs(data(:,2))==n,1),data(abs(data(:,2))==n,3),'-k')
hold on
plot(data(abs(data(:,2))==n,1),data(abs(data(:,2))==n,4),'-r')
plot(data(abs(data(:,2))==n,1),data(abs(data(:,2))==n,5),'-b')
% 
% figure(1)
% subplot(3,1,1)
% nx = 50 ; nt = 50;
% [X,Y] = meshgrid(linspace(min(x),max(x),nx),linspace(min(t),max(t),nt)) ;
% z = theta_h;
% Z =griddata(x,t,z,X,Y) ;
% [p1,p2] = contourf(X,Y,Z);
% title('$\theta_h$')
% hold on
% colormap jet
% colorbar;
% axis equal 
% 
% subplot(3,1,2)
% [X,Y] = meshgrid(linspace(min(x),max(x),nx),linspace(min(t),max(t),nt)) ;
% z = theta_c;
% Z =griddata(x,t,z,X,Y) ;
% [p1,p2] = contourf(X,Y,Z);
% title('$\theta_c$')
% hold on
% colormap jet
% colorbar;
% axis equal 
% 
% subplot(3,1,3)
% [X,Y] = meshgrid(linspace(min(x),max(x),nx),linspace(min(t),max(t),nt)) ;
% z = theta_w;
% Z =griddata(x,t,z,X,Y) ;
% [p1,p2] = contourf(X,Y,Z);
% title('$\theta_w$')
% hold on
% colormap jet
% colorbar;
% axis equal 

% 
% yp = 0.5;
% [idy,~]=find(abs(Y(:,1)-yp)<1e-4);
% figure(2)
% plot(X(1,:),Z(idy,:))
% hold on
% axis equal
% 
% 
% z = u;
% nx = 501 ; ny = 501;
% [X,Y] = meshgrid(linspace(min(x),max(x),nx),linspace(min(y),max(y),ny)) ;
% Z =griddata(x,y,z,X,Y) ;
% 
% xp = 0.5;
% [~,idx]=find(abs(X(1,:)-xp)<1e-4);
% figure(3)
% plot(Z(:,idx),Y(:,idx))
% hold on
% axis equal