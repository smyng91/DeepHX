clear; clc; close all
% plot 2D CFD data


sol = importdata('test.dat');
data = sol.data;

x = data(:,1);
y = data(:,2);
u = data(:,3);
v = data(:,4);
p = data(:,5);

umag = sqrt(u.^2+v.^2);
z = v;

nx = 501 ; ny = 501;
[X,Y] = meshgrid(linspace(min(x),max(x),nx),linspace(min(y),max(y),ny)) ;
Z =griddata(x,y,z,X,Y) ;
figure(1)
[p1,p2] = contourf(X,Y,Z);
hold on
colormap jet
colorbar;
axis equal 

yp = 0.5;
[idy,~]=find(abs(Y(:,1)-yp)<1e-4);
figure(2)
plot(X(1,:),Z(idy,:))
hold on
axis equal


z = u;
nx = 501 ; ny = 501;
[X,Y] = meshgrid(linspace(min(x),max(x),nx),linspace(min(y),max(y),ny)) ;
Z =griddata(x,y,z,X,Y) ;

xp = 0.5;
[~,idx]=find(abs(X(1,:)-xp)<1e-4);
figure(3)
plot(Z(:,idx),Y(:,idx))
hold on
axis equal