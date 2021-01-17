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

    L = 10.
    tend = 10.

    def HX(x, y):
        """ 
        """
        R = 1
        Vc = 1
        Vh = 1
        
        theta_w, theta_h, theta_c = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        dtheta_w = tf.gradients(theta_w, x)[0]
        dtheta_h = tf.gradients(theta_h, x)[0]
        dtheta_c = tf.gradients(theta_c, x)[0]

        dtheta_w_t = dtheta_w[:,1:]
        dtheta_h_x, dtheta_h_t = dtheta_h[:,0:1], dtheta_h[:,1:]
        dtheta_c_x, dtheta_c_t = dtheta_c[:,0:1], dtheta_c[:,1:]
        
        eq_w = dtheta_w_t - theta_c - R*theta_h + (1+R)*theta_w 
        eq_h = dtheta_h_t - R/Vh*(theta_w - theta_h - dtheta_h_x)
        eq_c = dtheta_c_t - 1/Vc*(theta_w - theta_c - dtheta_c_x)

        return [ eq_w, eq_h, eq_c ]

    def bc_inlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

    def bc_outlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], L)

    def inlet(x):
        # return 1.-np.sin(-0.5*x[:, 1:])
        return 1.


    geom = dde.geometry.Interval(0, L)
    timedomain = dde.geometry.TimeDomain(0, tend)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    h_inlet = dde.DirichletBC(geomtime, inlet, bc_inlet, component=1)
    h_outlet = dde.NeumannBC(geomtime, lambda x: 0, bc_outlet, component=1)
    c_inlet = dde.DirichletBC(geomtime, lambda x: 0, bc_inlet, component=2)
    c_outlet = dde.NeumannBC(geomtime, lambda x: 0, bc_outlet, component=2)
    ic = dde.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, HX,
        [ h_inlet, h_outlet, 
        c_inlet, c_outlet, 
        ic ], 
        num_domain=2500, num_boundary=1000, num_initial=1000, num_test=1000,
    )
    layer_size = [2] + [50] * 6 + [3]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)
    # net.apply_output_transform(lambda x, y: x[:,1:] * y)
    model = dde.Model(data, net)
    model.compile( "adam", lr=1e-3 )
    losshistory, train_state = model.train(epochs=10000, display_every=100)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

if __name__ == "__main__":
    main()

   
