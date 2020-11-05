from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deepxde as dde
from deepxde.backend import tf

def main():
        
    def pde(x, y, theta):
        #how to declare values for constants r, lambda_a and lambda_b, na and bn, pe_a and pe_b? 
        tw = y[:, 0:1]
        ta = y[:, 1:2]
        tb = y[:, 2:]

        dtw_theta = tf.gradients(tw, theta)[0]
        dta_theta = tf.gradients(ta, theta)[0]
        dtb_theta = tf.gradients(tb, theta)[0]

        dtw_x = tf.gradients(tw, x)[0]
        dtw_y = tf.gradients(tw, x)[1]
        dta_x = tf.gradients(ta, x)[0]
        dtb_y = tf.gradients(tb, x)[1]

        dtw_xx = tf.gradients(dtw_x, x)[0][:, 0:1]
        dtw_yy = tf.gradients(dtw_y, x)[1][:, 0:1]
        dta_xx = tf.gradients(dta_x, x)[0][:, 0:1]
        dtb_yy = tf.gradients(dtb_y, x)[1][:, 0:1]

        return dtw_theta - ta - (r * tb) + ((1+r) * tw) - (lambdaa * na * dtw_xx) - (lambdab * nb * r * dtw_yy), 
            (va * dta_theta) - tw + ta + dta_x - ((na / pea) * dta_xx), 
            (vb * dtb_theta) - tw + tb + dtb_y - ((nb / peb) * dtb_yy)

    def initial_condition(_, on_initial):
    
        return on_initial

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 10.)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    ic1 = dde.IC(geomtime, lambda _: 0., initial_condition, component=0)
    ic2 = dde.IC(geomtime, lambda _: 0., initial_condition, component=1)
    ic3 = dde.IC(geomtime, lambda _: 0., initial_condition, component=2)

    #not sure how to set the boundary conditions
    bc = dde.DirichletBC(geomtime, lambda x: 0., lambda _, on_boundary: on_boundary)

    data = dde.data.TimePDE(geomtime, pde, [bc, [ic1, ic2, ic3]], num_domain=2500, num_boundary=3, num_initial=160)
    net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1.0e-3)
    model.train(epochs=10000)
    model.compile("L-BFGS-B")
    model.train()

if __name__ == "__main__":
    main()
