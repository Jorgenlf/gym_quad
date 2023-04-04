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


    def step(self, action):
        # Un-normalize actions from neural network
        # F_1 = _thrust(action[0])
        # F_2 = _thrust(action[1])
        # F_3 = _thrust(action[2])
        # F_4 = _thrust(action[3])

        # self.input=np.array([F_1,F_2,F_3,F_4])
        self.input = action
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
        Returns the rate of rotation about the world frame.
        """
        [N_dot, E_dot, D_dot] = self.position_dot
        return np.arctan2(E_dot, N_dot)
        
    @property
    def upsilon(self):
        """
        Returns the rate of rotation about the world frame.
        """
        [N_dot, E_dot, D_dot] = self.position_dot
        return np.arctan2(D_dot, np.sqrt(N_dot**2 + E_dot**2))

def _thrust(force):
    force = np.clip(force, -1, 1)
    return force*ss.thrust_max