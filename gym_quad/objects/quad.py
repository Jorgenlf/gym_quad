# import numpy as np
# import gym_quad.utils.state_space as ss
# import gym_quad.utils.geomutils as geom

# # optimizaion attempts
# # from scipy.integrate import solve_ivp #Worse than our own odesolver45
# import numba as nb

# # TODO see if jit can be used or if it messes up the update of the state
# # The code runs with the stuff below, but i dont trust it so use old version of quad.py below per now

# # ## BEWARE AFTER THIS OPTIMIZATION WAS IMPLEMENTED A PRINTOUT SOMETIMES HAPPEN ###
# # eigen: too many iterations in Jacobi transform
# # # We might want to revert back although step now only takes 5% of runtime instead of 20%

# #For jit to work this must be defined here i think its about J() having no input args
# lamb = 0.08 # inflow ratio
# l = 0.5 # length from rotors to center of mass
# B = np.array([[0, 0, 0, 0],
#                 [0, 0, 0, 0],
#                 [1, 1, 1, 1],
#                 [0, -l, 0, l],
#                 [-l, 0, l, 0],
#                 [-lamb, lamb, -lamb, lamb]],dtype=np.float64)

# # Define state_dot_for_ode function outside the Quad class
# @nb.jit
# def state_dot_for_ode(state: np.ndarray, input_: np.ndarray) -> np.ndarray:
#     """
#     The right hand side of the 12 ODEs governing the Quad dynamics.
#     """
#     eta = state[:6]
#     nu = state[6:]

#     eta_dot = geom.j_J(eta).dot(nu)

#     M_inv = ss.j_M_inv()

#     B_dot_input: np.ndarray = B.dot(input_)
#     G = ss.j_G(eta)
#     C = ss.j_C(nu)
#     d = ss.j_d(nu)
#     # nu_dot = M_inv.dot(B_dot_input - G - C - d)
#     nu_dot: np.ndarray = M_inv.dot(B_dot_input - G - C - d)

#     # state_dot = np.array([eta_dot, nu_dot])
#     state_dot = np.concatenate((eta_dot, nu_dot))
#     return state_dot


# @nb.jit
# def odesolver45(y:float, input_: np.ndarray , h:float):
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
#     q = y + h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
#     return w, q


# class Quad():
#     """
#     Implementation of quadcopter dynamics. 
#     """
#     def __init__(self, step_size, init_eta, safety_radius=1):
#         self.step_size = step_size
#         self.state = np.hstack([init_eta, np.zeros((6,))])
#         self.safety_radius = safety_radius
#         self.input = np.zeros(4)
#         self.position_dot = np.zeros(3)


#     def step(self, thrust):
#         self.input = thrust
#         self._sim()


#     def _sim(self):
#         # w, q = odesolver45(f=state_dot_for_ode, input_=self.input, y=self.state, h=self.step_size)
        
#         w, q = odesolver45(y=self.state, input_=self.input, h=self.step_size)

#         self.state = w
#         self.state[3] = geom.ssa(self.state[3])
#         self.state[4] = geom.ssa(self.state[4])
#         self.state[5] = geom.ssa(self.state[5])
#         self.position_dot = self.state_dot(self.state)[0:3]
    

#     def state_dot(self, state:np.ndarray) -> np.ndarray:
#         """
#         The right hand side of the 12 ODEs governing the Quad dyanmics.
#         """
#         eta = state[:6]
#         nu = state[6:]

#         eta_dot = geom.J(eta).dot(nu)
#         nu_dot = ss.M_inv().dot(
#             ss.B().dot(self.input)
#             - ss.G(eta)
#             - ss.C(nu)
#             - ss.d(nu)
#         )
#         state_dot = np.hstack([eta_dot, nu_dot])
#         return state_dot

#     @property
#     def position(self):
#         """
#         Returns an array holding the position of the Quadcopter in world frame.
#         coordinates.
#         """
#         return self.state[0:3]

#     @property
#     def attitude(self):
#         """
#         Returns an array holding the attitude of the Quadcopter wrt. to world coordinates.
#         """
#         return self.state[3:6]

#     @property
#     def roll(self):
#         """
#         Returns the roll of the Quadcopter wrt world frame.
#         """
#         return geom.ssa(self.state[3])
    
#     @property
#     def pitch(self):
#         """
#         Returns the pitch of the Quadcopter wrt world frame.
#         """
#         return geom.ssa(self.state[4])

#     @property
#     def heading(self):
#         """
#         Returns the heading of the Quadcopter wrt true north.
#         """
#         return geom.ssa(self.state[5])

#     @property
#     def velocity(self):
#         """
#         Returns the surge, sway and heave velocity of the Quadcopter.
#         """
#         return self.state[6:9]
    
#     @property
#     def surge(self):
#         """
#         Returns the surge velocity of the Quadcopter.
#         """
#         return self.state[6]
    
#     @property
#     def sway(self):
#         """
#         Returns the sway velocity of the Quadcopter.
#         """
#         return self.state[7]
    
#     @property
#     def heave(self):
#         """
#         Returns the heave velocity of the Quadcopter.
#         """
#         return self.state[8]

#     @property
#     def relative_speed(self):
#         """
#         Returns the length of the velocity vector of the Quadcopter.
#         """
#         return np.linalg.norm(self.velocity)

#     @property
#     def angular_velocity(self):
#         """
#         Returns the rate of rotation about the world frame.
#         """
#         return self.state[9:12]
    
#     @property
#     def chi(self):
#         """
#         Returns the angle between the velocity vector and the world xy plane.
#         i.e heading angle. (?)
#         """
#         [x_dot, y_dot, z_dot] = self.position_dot
#         return np.arctan2(y_dot, x_dot)
        
#     @property
#     def upsilon(self):
#         """
#         Returns the angle between the world velocity vector and the world z-axis.
#         """
#         [x_dot, y_dot, z_dot] = self.position_dot
#         return np.arctan2(z_dot, np.sqrt(x_dot**2 + y_dot**2))
    
#     @property
#     def aoa(self):
#         """
#         Returns the angle between the velocity vector and the body xy plane. (angle of attack)
#         """
#         [u, v, w] = self.velocity
#         return np.arctan2(w, u)

#     @property
#     def beta(self):
#         """
#         Returns the angle between the velocity vector and the body x-axis about the z-axis. (sideslip angle)
#         """
#         [u, v, w] = self.velocity
#         return np.arctan2(v, u)        

# def _thrust(force):
#     force = np.clip(force, -1, 1)
#     return force*ss.thrust_max

# Attempted optimization with scipy IVP solver SEEMINGLY NOT FASTER  10% slower than odesolver45







# OLD VERSION OF quad.py WITH NO ATTEMPTS AT OPTIMIZING 

import numpy as np
import gym_quad.utils.state_space as ss
import gym_quad.utils.geomutils as geom

def odesolver45(f, y, h):
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
    s1 = f(y)
    s2 = f(y + h*s1/4.0)
    s3 = f(y + 3.0*h*s1/32.0 + 9.0*h*s2/32.0)
    s4 = f(y + 1932.0*h*s1/2197.0 - 7200.0*h*s2/2197.0 + 7296.0*h*s3/2197.0)
    s5 = f(y + 439.0*h*s1/216.0 - 8.0*h*s2 + 3680.0*h*s3/513.0 - 845.0*h*s4/4104.0)
    s6 = f(y - 8.0*h*s1/27.0 + 2*h*s2 - 3544.0*h*s3/2565 + 1859.0*h*s4/4104.0 - 11.0*h*s5/40.0)
    w = y + h*(25.0*s1/216.0 + 1408.0*s3/2565.0 + 2197.0*s4/4104.0 - s5/5.0)
    q = y + h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q


class Quad():
    """
    Implementation of quadcopter dynamics. 
    """
    def __init__(self, step_size, init_eta, safety_radius=1):
        self.step_size = step_size
        self.state = np.hstack([init_eta, np.zeros((6,))])
        self.safety_radius = safety_radius
        self.input = np.zeros(4)
        self.position_dot = np.zeros(3)


    def step(self, thrust):
        self.input = thrust
        self._sim()


    def _sim(self):
        w, q = odesolver45(self.state_dot, self.state, self.step_size)

        self.state = w
        self.state[3] = geom.ssa(self.state[3])
        self.state[4] = geom.ssa(self.state[4])
        self.state[5] = geom.ssa(self.state[5])
        self.position_dot = self.state_dot(self.state)[0:3]


    def state_dot(self, state):
        """
        The right hand side of the 12 ODEs governing the Quad dyanmics.
        """
        eta = state[:6]
        nu = state[6:]

        eta_dot = geom.J(eta).dot(nu)
        nu_dot = ss.M_inv().dot(
            ss.B().dot(self.input)
            - ss.G(eta)
            - ss.C(nu)
            - ss.d(nu)
        )
        state_dot = np.hstack([eta_dot, nu_dot])
        return state_dot

    @property
    def position(self):
        """
        Returns an array holding the position of the Quadcopter in world frame.
        coordinates.
        """
        return self.state[0:3]

    @property
    def attitude(self):
        """
        Returns an array holding the attitude of the Quadcopter wrt. to world coordinates.
        """
        return self.state[3:6]

    @property
    def roll(self):
        """
        Returns the roll of the Quadcopter wrt world frame.
        """
        return geom.ssa(self.state[3])
    
    @property
    def pitch(self):
        """
        Returns the pitch of the Quadcopter wrt world frame.
        """
        return geom.ssa(self.state[4])

    @property
    def heading(self):
        """
        Returns the heading of the Quadcopter wrt true north.
        """
        return geom.ssa(self.state[5])

    @property
    def velocity(self):
        """
        Returns the surge, sway and heave velocity of the Quadcopter.
        """
        return self.state[6:9]
    
    @property
    def surge(self):
        """
        Returns the surge velocity of the Quadcopter.
        """
        return self.state[6]
    
    @property
    def sway(self):
        """
        Returns the sway velocity of the Quadcopter.
        """
        return self.state[7]
    
    @property
    def heave(self):
        """
        Returns the heave velocity of the Quadcopter.
        """
        return self.state[8]

    @property
    def relative_speed(self):
        """
        Returns the length of the velocity vector of the Quadcopter.
        """
        return np.linalg.norm(self.velocity)

    @property
    def angular_velocity(self):
        """
        Returns the rate of rotation about the world frame.
        """
        return self.state[9:12]
    
    @property
    def chi(self):
        """
        Returns the angle between the velocity vector and the world xy plane.
        i.e heading angle. (?)
        """
        [x_dot, y_dot, z_dot] = self.position_dot
        return np.arctan2(y_dot, x_dot)
        
    @property
    def upsilon(self):
        """
        Returns the angle between the world velocity vector and the world z-axis.
        """
        [x_dot, y_dot, z_dot] = self.position_dot
        return np.arctan2(z_dot, np.sqrt(x_dot**2 + y_dot**2))
    
    @property
    def aoa(self):
        """
        Returns the angle between the velocity vector and the body xy plane. (angle of attack)
        """
        [u, v, w] = self.velocity
        return np.arctan2(w, u)

    @property
    def beta(self):
        """
        Returns the angle between the velocity vector and the body x-axis about the z-axis. (sideslip angle)
        """
        [u, v, w] = self.velocity
        return np.arctan2(v, u)        

def _thrust(force):
    force = np.clip(force, -1, 1)
    return force*ss.thrust_max