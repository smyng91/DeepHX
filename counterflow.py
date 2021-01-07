from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.backend import set_floatx
import time
import deepxde as dde
import tensorflow as tf
import numpy as np

dde.config.real.set_float64()

def main():

    def HX(x, y, X):
        """ 
        dtheta_w_dt = theta_h + R*theta_c - (1+R)*theta_w + lambda_h*N_h*d^2theta_w/dx^2 + lambda_c*N_c*R*d^2theta_w/dy^2
        V_h*dtheta_h_dt = theta_w - theta_h - dtheta_h_dx + N_h/Pe_h*d^2theta_h/dx^2
        V_c*dtheta_c_dt = theta_w - theta_c - dtheta_c_dy + N_c/Pe_c*d^2theta_c/dy^2
        """
        lambda_h = 1.
        lambda_c = 1.
        V_h = 1.
        V_c = 1.
        Pe_h = 1.
        Pe_c = 1.
        N_h = 1.
        N_c = 1.
        R = 1.
        
        theta_w, theta_h, theta_c = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        dtheta_w = tf.gradients(theta_w, x)[0]
        dtheta_h = tf.gradients(theta_h, x)[0]
        dtheta_c = tf.gradients(theta_c, x)[0]

        dtheta_w_x, dtheta_w_y, dtheta_w_t = dtheta_w[:,0:1], dtheta_w[:,1:2], dtheta_w[:,2:]
        dtheta_h_x, dtheta_h_t = dtheta_h[:,0:1], dtheta_h[:,2:]
        dtheta_c_y, dtheta_c_t = dtheta_c[:,1:2], dtheta_c[:,2:]
        
        dtheta_w_xx, dtheta_w_yy = tf.gradients(dtheta_w_x, x)[0][:, 0:1], tf.gradients(dtheta_w_y, x)[0][:, 1:2]
        dtheta_h_xx = tf.gradients(dtheta_h_x, x)[0][:, 0:1]
        dtheta_c_yy = tf.gradients(dtheta_c_y, x)[0][:, 1:2]

        eq_w = dtheta_w_t - theta_h - R*theta_c + (1.+R)*theta_w - lambda_h*N_h*dtheta_w_xx - lambda_c*N_c*R*dtheta_w_yy
        eq_h = V_h*dtheta_h_t - theta_w + theta_h + dtheta_h_x + N_h/Pe_h*dtheta_h_xx
        eq_c = V_c*dtheta_c_t - theta_w + theta_c + dtheta_c_y - N_c/Pe_c*dtheta_c_yy

        return [ eq_w, eq_h, eq_c ]

    def bc_wx_left(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0.0)

    def bc_wx_right(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1.0)

    def bc_wy_left(x, on_boundary):
        return on_boundary and np.isclose(x[1], 0.0)

    def bc_wy_right(x, on_boundary):
        return on_boundary and np.isclose(x[1], 1.0)
        
    def bc_h_inlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0.0)

    def bc_h_outlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1.0)

    def bc_c_inlet(x, on_boundary):
        return on_boundary and np.isclose(x[1], 0.0)

    def bc_c_outlet(x, on_boundary):
        return on_boundary and np.isclose(x[1], 1.0)

    def inlet(x):
        return 1.-np.exp(x[:, 2:])

    geom = dde.geometry.geometry_2d.Rectangle([0,0], [1,1])
    timedomain = dde.geometry.TimeDomain(0, 1.)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    h_inlet = dde.DirichletBC(geomtime, inlet, bc_h_inlet, component=1)
    h_outlet = dde.NeumannBC(geomtime, lambda x: 0, bc_h_outlet, component=1)
    c_inlet = dde.DirichletBC(geomtime, lambda x: 0, bc_c_inlet, component=2)
    c_outlet = dde.NeumannBC(geomtime, lambda x: 0, bc_c_outlet, component=2)
    w_x_left = dde.NeumannBC(geomtime, lambda x: 0, bc_wx_left, component=0)
    w_x_right = dde.NeumannBC(geomtime, lambda x: 0, bc_wx_right, component=0)
    w_y_left = dde.NeumannBC(geomtime, lambda x: 0, bc_wy_left, component=0)
    w_y_right = dde.NeumannBC(geomtime, lambda x: 0, bc_wy_right, component=0)
    ic = dde.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, HX,
        [ h_inlet, h_outlet, 
        c_inlet, c_outlet, 
        w_x_left, w_x_right,
        w_y_left, w_y_right, ic ], 
        num_domain=20, num_boundary=9, num_initial=10
    )
    net = dde.maps.FNN([3] + [40] * 6 + [3], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile( "adam", lr=1e-4 )
    losshistory, train_state = model.train(epochs=1000, display_every=100)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

if __name__ == "__main__":
    main()

   
