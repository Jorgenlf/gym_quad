import numpy as np
import pandas as pd
import gymnasium as gym

import gym_quad.utils.geomutils as geom
import gym_quad.utils.state_space as ss
import gym_quad.objects.scenarios as scen
from gym_quad.objects.quad import Quad
from gym_quad.objects.current3d import Current
from gym_quad.objects.QPMI import QPMI, generate_random_waypoints
from gym_quad.objects.obstacle3d import Obstacle
from gym_quad.utils.controllers import PI, PID
 

class CTBR_VAE(gym.Env):
    '''Creates an environment where the actionspace consists of Collective thrust and body rates which will be passed to a PD or PID controller, 
    while the observationspace uses a Varial AutoEncoder pluss more for observations to  environment.'''
    def __init__(self, env_config, init_scenario="line", seed=None):
        np.random.seed(0) 

        for key in env_config:
            setattr(self, key, env_config[key]) # set all the parameters from GYM_QUAD/qym_quad/__init__.py as attributes of the class

        self.action_space = gym.spaces.Box(
            low = np.array([-1,-1,-1], dtype=np.float32),
            high = np.array([1, 1, 1], dtype=np.float32),
            dtype = np.float32
        )

        self.perception_space = gym.spaces.Box(
            low = 0,
            high = 1,
            shape = (1, self.sensor_suite[0], self.sensor_suite[1]),
            dtype = np.float64
        )

        self.observation_space = gym.spaces.Dict({
        'perception': self.perception_space
        })

        self.n_sensor_readings = self.sensor_suite[0]*self.sensor_suite[1]
        max_horizontal_angle = self.sensor_span[0]/2
        max_vertical_angle = self.sensor_span[1]/2
        self.sectors_horizontal = np.linspace(-max_horizontal_angle*np.pi/180, max_horizontal_angle*np.pi/180, self.sensor_suite[0])
        self.sectors_vertical =  np.linspace(-max_vertical_angle*np.pi/180, max_vertical_angle*np.pi/180, self.sensor_suite[1])
        
        self.scenario = init_scenario
        self.scenario_switch = {
            # Training scenarios
            "line": scen.scenario_line(self), #TODO check if this works, if not must make the scenarios in the same file or give nwaypoints and path as input
            "line_new": self.scenario_line_new,
            "horizontal": self.scenario_horizontal,
            "horizontal_new": self.scenario_horizontal_new,
            "3d": self.scenario_3d,
            "3d_new": self.scenario_3d_new,
            "helix": self.scenario_helix,
            "intermediate": self.scenario_intermediate,
            "proficient": self.scenario_proficient,
            # "advanced": self.scenario_advanced,
            "expert": self.scenario_expert,
            # Testing scenarios
            "test_path": self.scenario_test_path,
            "test": self.scenario_test,
            "test_current": self.scenario_test_current,
            "horizontal": self.scenario_horizontal_test,
            "vertical": self.scenario_vertical_test,
            "deadend": self.scenario_deadend_test
        }
        self.reset()
    

