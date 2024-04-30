import numpy as np
import gym_quad.utils.geomutils as geom
from numpy.linalg import inv
from math import cos, sin


#REMEMBER THAT A SEPARATE STATESPACE IS DEFINED IN ODE45JIT.PY #TODO make this more elgant/merge stuff
#ENSURE THAT IT MATCHES THE ONE IN THIS FILE (STATE_SPACE.PY)
#PER NOW THEY MATCH 30.04.2024

I3 = np.identity(3)
zero3= 0 * I3
g = 9.81

# Quad parameters
m = 0.5 #kg
W = m*g #N

thrust_min = -6 #N
thrust_max = 6  #N

# Moments of inertia
I_x = 0.005
I_y = I_x
I_z = 0.01
Ig = np.vstack([
    np.hstack([I_x, 0, 0]),
    np.hstack([0, I_y, 0]),
    np.hstack([0, 0, I_z])])

lamb = 0.08 # inflow ratio
l = 0.5 # length from rotors to center of mass

def M_RB():
    M_RB_CG = np.vstack([
        np.hstack([m * I3, zero3]),
        np.hstack([zero3, Ig])
    ])

    #M_RB_CO = geom.move_to_CO(M_RB_CG, r_G)
    return M_RB_CG


def M_inv():
    M = M_RB()
    return inv(M)

def C(nu):
    p = nu[3]
    q = nu[4]
    r = nu[5]
    Cv = np.array([0,
                 0,
                 0,
                 (I_z - I_y) * q * r,
                 (I_x - I_z) * r * p,
                 0])
    return Cv


def B():
    B = np.array([[0,0,0,0],
                [0,0,0,0],
                [1,1,1,1],
                [0,-l,0,l],
                [-l,0,l,0],
                [-lamb,lamb,-lamb,lamb]])
    return B

def G(eta):
    phi = eta[3]
    theta = eta[4]

    G = np.array([
        -(W)*sin(theta),
        (W)*cos(theta)*sin(phi),
        (W)*cos(theta)*cos(phi),
        0,
        0,
        0
    ])
    return G

##############################################################

# Aerodynamic friction Coefficients
d_u = 0.3729
d_v = 0.3729
d_w = 0.3729

# Rotational drag Coefficients
d_p = 5.56e-4
d_q = 5.56e-4
d_r = 5.56e-4

def d(nu):
    u = nu[0]
    v = nu[1]
    w = nu[2]
    p = nu[3]
    q = nu[4]
    r = nu[5]

    d = np.array([
        d_u * u,
        d_v * v,
        d_w * w,
        d_p * p ** 2,
        d_q * q ** 2,
        d_r * r ** 2
    ])

    return d
