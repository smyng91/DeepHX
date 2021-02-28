import numpy as np 
import sciann as sn 
import matplotlib.pyplot as plt 
import scipy.io

from numpy import pi
from sciann.utils.math import diff, sign, sin

L_hx = 5
tau = 10

U = 1
Vh = 1
Vc = 1

x = sn.Variable('x',dtype='float64')
t = sn.Variable('t',dtype='float64')

theta_h = sn.Functional('theta_h', [t,x], 4*[20], 'tanh')
theta_c = sn.Functional('theta_c', [t,x], 4*[20], 'tanh')
theta_w = sn.Functional('theta_w', [t,x], 4*[20], 'tanh')
dtheta_h = diff(theta_h,x, order = 1)
dtheta_c = diff(theta_c,x, order = 1)

# governing equations
L1 = diff(theta_h, t) - U/Vh*(theta_w-theta_h - dtheta_h) 
L2 = diff(theta_c, t) - 1/Vc*(theta_w-theta_c - dtheta_c) 
L3 = diff(theta_w, t) - theta_c - U*theta_h + (1-U)*theta_w 

TOL = 0.001
# initial conditions
C1 = (1-sign(t - TOL)) * theta_h
C2 = (1-sign(t - TOL)) * theta_c
C3 = (1-sign(t - TOL)) * theta_w
# BCs
C4 = (1-sign(x - TOL)) * theta_c
C5 = (1-sign(x - TOL)) * dtheta_h
C6 = (1-sign(x - ( 1-TOL))) * dtheta_c
C7 = (1-sign(x - ( 1-TOL))) * (theta_h - 1)
C8 = theta_w*0.0

m = sn.SciModel( 
    inputs = [x, t],
    targets = [L1, L2, L3, C1, C2, C3, C4, C5, C6, C7, C8],
    loss_func = 'mse', optimizer = 'Adam')

x_data, t_data = np.meshgrid(
    np.linspace(0, L_hx, 200), 
    np.linspace(0, tau, 200)
)

h = m.train([x_data, t_data], 11*['zero'], learning_rate=0.001, epochs=50000, batch_size=100, shuffle=True, adaptive_weights = True, verbose=1)
plt.semilogy(h.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('loss')
m.save_weights('trained_HX.hdf5')


x_test, t_test = np.meshgrid(
    np.linspace(0, L_hx, 200), 
    np.linspace(0, tau, 200)
)
pred_theta_h = theta_h.eval(m, [x_test, t_test])
pred_theta_c = theta_c.eval(m, [x_test, t_test])
pred_theta_w = theta_w.eval(m, [x_test, t_test])

sol = [pred_theta_h,pred_theta_c,pred_theta_h]
np.save('prediction.npy',sol)
