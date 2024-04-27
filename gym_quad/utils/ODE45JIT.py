import numpy as np
import numba as nb

#Note Attempted optimization with scipy IVP solver SEEMINGLY NOT FASTER  10% slower than the basic odesolver45 in quad.py

#REMEMBER THAT A SEPARATE STATESPACE IS DEFINED IN STATE_SPACE.PY
#ENSURE THAT IT MATCHES THE ONE IN THIS FILE (ODE45JIT.PY)

#STATE SPACE AS JIT FUNCTIONS
I3 = np.identity(3)
zero3= 0 * I3
g = np.float64(9.81)

# Quad parameters
m = np.float64(0.5) #kg #IF YOU CHANGE THIS YOU NEED TO CHANGE THE INVERSE MASS MATRIX AT LINE 65ish
W = np.float64(m*g) #N

thrust_min = np.float64(-6) #N
thrust_max = np.float64(6)  #N

# Moments of inertia
I_x = np.float64(0.005) #IF YOU CHANGE THIS YOU NEED TO CHANGE THE INVERSE MASS MATRIX AT LINE 65ish
I_y = I_x               #IF YOU CHANGE THIS YOU NEED TO CHANGE THE INVERSE MASS MATRIX AT LINE 65ish
I_z = np.float64(0.01)  #IF YOU CHANGE THIS YOU NEED TO CHANGE THE INVERSE MASS MATRIX AT LINE 65ish
Ig = np.vstack([        #IF YOU CHANGE THIS YOU NEED TO CHANGE THE INVERSE MASS MATRIX AT LINE 65ish
    np.hstack([I_x, 0, 0]),
    np.hstack([0, I_y, 0]),
    np.hstack([0, 0, I_z])],dtype=np.float64)


lamb = np.float64(0.08) # inflow ratio
l = np.float64(0.5) # length from rotors to center of mass

B = np.array([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, -l, 0, l],
                [-l, 0, l, 0],
                [-lamb, lamb, -lamb, lamb]],dtype=np.float64)

@nb.jit
def j_M_RB()->np.ndarray:
    M_RB_CG = np.zeros((6, 6))
    M_RB_CG[:3, :3] = m * I3
    M_RB_CG[3:, 3:] = Ig
    return M_RB_CG


@nb.jit
def j_M_inv()->np.ndarray:
    M = j_M_RB()
    M_inv = np.linalg.inv(M)
    return M_inv

###### THIS IS WHERE YOU CHANGE THE INVERSE MASS MATRIX AFTER CHANGING MASS OR INERTIA ######

#RUN THIS SCRIPT 

#PRINT IT 
# print("\nInverse of M:",j_M_inv())

#AND ASK E.G. GPT TO WRITE IT AS np.array OR MANUALLY WRITE IT
M_inv = np.array([
    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 200.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 200.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
], dtype=np.float64)

# print("Determinant of invM:", np.linalg.det(M_inv))
# print("Condition number of invM:", np.linalg.cond(M_inv))


@nb.jit
def j_C(nu:np.ndarray)->np.ndarray:
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


@nb.jit
def j_G(eta:np.ndarray)->np.ndarray:
    phi = eta[3]
    theta = eta[4]

    G = np.array([
        -(W) * np.sin(theta),
        (W) * np.cos(theta) * np.sin(phi),
        (W) * np.cos(theta) * np.cos(phi),
        0,
        0,
        0
    ])
    return G


# Aerodynamic friction Coefficients
d_u = np.float64(0.3729)
d_v = np.float64(0.3729)
d_w = np.float64(0.3729)

# Rotational drag Coefficients
d_p = np.float64(5.56e-4)
d_q = np.float64(5.56e-4)
d_r = np.float64(5.56e-4)

@nb.jit
def j_d(nu:np.ndarray)->np.ndarray:
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


#THE NEEDED GEOMETRY FUNCTIONS AS JIT FUNCTIONS
@nb.jit
def j_Rzyx(phi, theta, psi):
    '''
    input: phi, theta, psi of the body frame relative to the world frame
    Rotation matrix from the body frame to the world frame.
    '''
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.array([[cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth],
                     [spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi],
                     [-sth, cth*sphi, cth*cphi]])
@nb.jit
def j_Tzyx(phi, theta, psi):
    sphi = np.sin(phi)
    tth = np.tan(theta)
    cphi = np.cos(phi)
    cth = np.cos(theta)

    return np.array([[1, sphi*tth, cphi*tth], 
                     [0, cphi, -sphi],
                     [0, sphi/cth, cphi/cth]])

@nb.jit
def j_J(eta):
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    R = j_Rzyx(phi, theta, psi)
    T = j_Tzyx(phi, theta, psi)

    # Create the combined matrix fully initialized to zero
    J_mat = np.zeros((6, 6))
    J_mat[:3, :3] = R
    J_mat[3:, 3:] = T

    # If necessary, initialize the off-diagonal blocks (assumed to be zeros here)
    J_mat[:3, 3:] = 0  # Top-right block
    J_mat[3:, :3] = 0  # Bottom-left block

    return J_mat


#THE THREE FUNCTIONS FOR DOING STEP IN QUAD.PY

@nb.jit
def j_ssa(angle: np.float64) -> np.float64:
    """
    Saturate the angle to the range [-pi, pi].
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

@nb.jit
def j_state_dot(state: np.ndarray, input_: np.ndarray) -> np.ndarray:
    """
    The right hand side of the 12 ODEs governing the Quad dynamics.
    """
    eta = state[:6]
    nu = state[6:]
    J = j_J(eta)
    eta_dot = J.dot(nu)

    B_dot_input: np.ndarray = B.dot(input_)

    G = j_G(eta)
    C = j_C(nu)
    d = j_d(nu)
    # assert_no_nans_infs(J, "J") #Cant be used apparently with numba
    # assert_no_nans_infs(G, "G")
    # assert_no_nans_infs(C, "C")
    # assert_no_nans_infs(d, "d")
    nu_dot: np.ndarray = M_inv.dot(B_dot_input - G - C - d)

    # state_dot = np.array([eta_dot, nu_dot])
    state_dot = np.concatenate((eta_dot, nu_dot))
    return state_dot

a1 = None

a2_0_d = np.float64(4.0)

a3_0_m = np.float64(3.0)
a3_1_d = np.float64(32.0)
a3_2_m = np.float64(9.0)
a3_3_d = np.float64(32.0)

a4_0_m = np.float64(1932.0)
a4_1_d = np.float64(2197.0)
a4_2_m = np.float64(7200.0)
a4_3_d = np.float64(2197.0)
a4_4_m = np.float64(7296.0)
a4_5_d = np.float64(2197.0)

a5_0_m = np.float64(439.0)
a5_1_d = np.float64(216.0)
a5_2_m = np.float64(8.0)
a5_3_m = np.float64(3680.0)
a5_4_d = np.float64(513.0)
a5_5_m = np.float64(845.0)
a5_6_d = np.float64(4104.0)

a6_0_m = np.float64(8.0)
a6_1_d = np.float64(27.0)
a6_2_m = np.float64(2.0)
a6_3_m = np.float64(3544.0)
a6_4_d = np.float64(2565.0)
a6_5_m = np.float64(1859.0)
a6_6_d = np.float64(4104.0)
a6_7_m = np.float64(11.0)
a6_8_d = np.float64(40.0)

w_0_m = np.float64(25.0)
w_1_d = np.float64(216.0)
w_2_m = np.float64(1408.0)
w_3_d = np.float64(2565.0)
w_4_m = np.float64(2197.0)
w_5_d = np.float64(4104.0)
w_6_d = np.float64(5.0)

#The ODE solver with coefficients defined outside the function
@nb.jit
def j_odesolver45(y:np.ndarray, input_: np.ndarray , h:float):
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 4 approx.
        w: float. Order 5 approx.
    """
    s1 = j_state_dot(y, input_)
    s2 = j_state_dot(y + h*s1/a2_0_d, input_)
    s3 = j_state_dot(y + a3_0_m*h*s1/a3_1_d + a3_2_m*h*s2/a3_3_d, input_)
    s4 = j_state_dot(y + a4_0_m*h*s1/a4_1_d - a4_2_m*h*s2/a4_3_d + a4_4_m*h*s3/a4_5_d, input_)
    s5 = j_state_dot(y + a5_0_m*h*s1/a5_1_d - a5_2_m*h*s2 + a5_3_m*h*s3/a5_4_d - a5_5_m*h*s4/a5_6_d, input_)
    # s6 = state_dot_for_ode(y - a6_0_m*h*s1/a6_1_d + a6_2_m*h*s2 - a6_3_m*h*s3/a6_4_d + a6_5_m*h*s4/a6_6_d - a6_7_m*h*s5/a6_8_d, input_)
    w = y + h*(w_0_m*s1/w_1_d + w_2_m*s3/w_3_d + w_4_m*s4/w_5_d - s5/w_6_d)
    # q = y + h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w
#Uncomment s6 and q and return q if we want to use the 5th order approximation




#OLD CODE KEPT AROUND TO 2x CHECK THE COEFFICIENTS
#The ODE solver with coefficients defined inside the function
# @nb.jit
# def j_odesolver45(y:np.ndarray, input_: np.ndarray , h:float):
#     """Calculate the next step of an IVP of a time-invariant ODE with a RHS
#     described by f, with an order 4 approx. and an order 5 approx.
#     Parameters:
#         f: function. RHS of ODE.
#         y: float. Current position.
#         h: float. Step length.
#     Returns:
#         q: float. Order 4 approx.
#         w: float. Order 5 approx.
#     """
#     s1 = state_dot_for_ode(y, input_)
#     s2 = state_dot_for_ode(y + h*s1/4.0, input_)
#     s3 = state_dot_for_ode(y + 3.0*h*s1/32.0 + 9.0*h*s2/32.0, input_)
#     s4 = state_dot_for_ode(y + 1932.0*h*s1/2197.0 - 7200.0*h*s2/2197.0 + 7296.0*h*s3/2197.0, input_)
#     s5 = state_dot_for_ode(y + 439.0*h*s1/216.0 - 8.0*h*s2 + 3680.0*h*s3/513.0 - 845.0*h*s4/4104.0, input_)
#     s6 = state_dot_for_ode(y - 8.0*h*s1/27.0 + 2*h*s2 - 3544.0*h*s3/2565 + 1859.0*h*s4/4104.0 - 11.0*h*s5/40.0, input_)
#     w = y + h*(25.0*s1/216.0 + 1408.0*s3/2565.0 + 2197.0*s4/4104.0 - s5/5.0)
#     # q = y + h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
#     return w#, q
